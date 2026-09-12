#!/usr/bin/env python
import argparse
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from models.switch_transformers import SwitchTransformersForSequenceClassification

LAYER_KEY_PATTERN = re.compile(r"^(encoder|decoder)\.(\d+)$")
SUPPORTED_DATASET_NAME_TO_HF: Dict[str, str] = {
    "ag_news": "ag_news",
    "imdb": "imdb",
    # Hardcoded alias for your 20 News setup.
    "20news": "SetFit/20_newsgroups",
    "sst2": "SetFit/sst2",
    "yelp_polarity": "yelp_polarity",
    "emotion": "emotion",
}


def _supported_dataset_names() -> List[str]:
    return list(SUPPORTED_DATASET_NAME_TO_HF.keys())


def resolve_hf_dataset_name(dataset_name: str) -> str:
    if dataset_name not in SUPPORTED_DATASET_NAME_TO_HF:
        allowed: str = ", ".join(_supported_dataset_names())
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Please switch dataset via --dataset_name in [{allowed}]."
        )
    return SUPPORTED_DATASET_NAME_TO_HF[dataset_name]


def assert_dataset_compatibility(ds: DatasetDict, dataset_name: str, hf_dataset_name: str) -> None:
    required_splits: List[str] = ["train", "test"]
    missing_splits: List[str] = [split_name for split_name in required_splits if split_name not in ds]
    if missing_splits:
        raise RuntimeError(
            f"Dataset '{dataset_name}' (HF: '{hf_dataset_name}') is incompatible: missing splits {missing_splits}. "
            f"Please switch dataset via --dataset_name in [{', '.join(_supported_dataset_names())}]."
        )

    train_ds = ds["train"]
    eval_ds = ds["test"]
    required_columns: List[str] = ["text", "label"]
    train_missing_cols: List[str] = [col for col in required_columns if col not in train_ds.column_names]
    eval_missing_cols: List[str] = [col for col in required_columns if col not in eval_ds.column_names]
    if train_missing_cols or eval_missing_cols:
        raise RuntimeError(
            f"Dataset '{dataset_name}' (HF: '{hf_dataset_name}') is incompatible. "
            f"Missing train columns={train_missing_cols}, test columns={eval_missing_cols}. "
            f"Expected columns: {required_columns}. "
            f"Please switch dataset via --dataset_name in [{', '.join(_supported_dataset_names())}]."
        )

    if len(train_ds) == 0 or len(eval_ds) == 0:
        raise RuntimeError(
            f"Dataset '{dataset_name}' (HF: '{hf_dataset_name}') is incompatible: empty split detected. "
            f"Please switch dataset via --dataset_name in [{', '.join(_supported_dataset_names())}]."
        )

    sample_row: Dict[str, Any] = train_ds[0]
    sample_text: Any = sample_row.get("text")
    if not isinstance(sample_text, str):
        raise RuntimeError(
            f"Dataset '{dataset_name}' (HF: '{hf_dataset_name}') is incompatible: train['text'] is not string "
            f"(got type={type(sample_text)}). Please switch dataset via --dataset_name in "
            f"[{', '.join(_supported_dataset_names())}]."
        )

    sample_label: Any = sample_row.get("label")
    try:
        int(sample_label)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Dataset '{dataset_name}' (HF: '{hf_dataset_name}') is incompatible: train['label'] is not int-castable "
            f"(got value={sample_label!r}, type={type(sample_label)}). Please switch dataset via --dataset_name in "
            f"[{', '.join(_supported_dataset_names())}]."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="PoC SwitchTransformer fine-tuning script for text/sequence classification."
    )
    parser.add_argument("--mode", type=str, choices=["full", "mix", "drop"], required=True)
    parser.add_argument("--model_name", type=str, default="model_ckpt/switch-base-8")
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=_supported_dataset_names(),
        default="ag_news",
        help="Supported datasets: ag_news, imdb, 20news (20news -> SetFit/20_newsgroups).",
    )
    parser.add_argument("--output_dir", type=str, default="./outputs/poc_moe")
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument(
        "--auto_hot_experts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically profile and pick per-layer hot experts for mix/drop modes.",
    )
    parser.add_argument(
        "--hot_k",
        type=int,
        default=None,
        help="Number of hot experts per sparse layer (required when --auto_hot_experts for mix/drop).",
    )
    parser.add_argument("--hot_experts", type=int, nargs="+", default=[0, 1])
    parser.add_argument(
        "--layer_hot_experts",
        type=str,
        default=None,
        help='Manual JSON mapping string, e.g. \'{"encoder.1":[0,1],"decoder.3":[2,5]}\'.',
    )
    parser.add_argument(
        "--hot_map_path",
        type=str,
        default=None,
        help="Path to manual JSON mapping file of layer_id -> expert indices.",
    )
    parser.add_argument("--calib_samples", type=int, default=128)
    parser.add_argument("--calib_batch_size", type=int, default=4)
    parser.add_argument(
        "--save_hot_map_path",
        type=str,
        default=None,
        help="Path to save resolved layer-wise hot experts mapping. Defaults to <output_dir>/hot_experts_map.json.",
    )
    parser.add_argument(
        "--activation_analysis",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable per-evaluation router activation analysis.",
    )
    parser.add_argument(
        "--activation_report_dir",
        type=str,
        default=None,
        help="Directory to save activation reports. Defaults to <output_dir>/activation_stats.",
    )
    parser.add_argument(
        "--activation_max_eval_batches",
        type=int,
        default=None,
        help="Optional cap on the number of evaluation batches analyzed for activation stats.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=float, default=4.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _append_eos_to_batch(
    input_ids_batch: Sequence[Sequence[int]],
    attention_mask_batch: Sequence[Sequence[int]],
    eos_token_id: int,
    max_length: int,
) -> Tuple[List[List[int]], List[List[int]]]:
    out_input_ids: List[List[int]] = []
    out_attention_mask: List[List[int]] = []
    for input_ids, attention_mask in zip(input_ids_batch, attention_mask_batch):
        cur_ids: List[int] = list(input_ids)[: max_length - 1]
        cur_mask: List[int] = list(attention_mask)[: max_length - 1]
        cur_ids.append(eos_token_id)
        cur_mask.append(1)
        out_input_ids.append(cur_ids)
        out_attention_mask.append(cur_mask)
    return out_input_ids, out_attention_mask


def build_dataset(args: argparse.Namespace, tokenizer: PreTrainedTokenizerBase) -> Tuple[DatasetDict, List[str]]:
    dataset_name: str = str(args.dataset_name)
    hf_dataset_name: str = resolve_hf_dataset_name(dataset_name)
    try:
        ds: DatasetDict = load_dataset(hf_dataset_name)  # type: ignore[assignment]
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load dataset '{dataset_name}' (HF: '{hf_dataset_name}'). "
            f"Please switch dataset via --dataset_name in [{', '.join(_supported_dataset_names())}]."
        ) from exc

    assert_dataset_compatibility(ds=ds, dataset_name=dataset_name, hf_dataset_name=hf_dataset_name)

    train_ds = ds["train"].shuffle(seed=args.seed)
    eval_ds = ds["test"]
    train_limit: int = min(args.max_samples, len(train_ds))
    train_ds = train_ds.select(range(train_limit))
    eval_ds = eval_ds.select(range(200))
    label_feature: Any = train_ds.features.get("label")
    label_names: List[str]
    if label_feature is not None and hasattr(label_feature, "names") and label_feature.names:
        label_names = [str(x) for x in label_feature.names]
    else:
        num_labels: int = int(max(train_ds["label"])) + 1  # type: ignore[index]
        label_names = [str(i) for i in range(num_labels)]

    eos_token_id: Optional[int] = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id for EOS-based sequence classification.")
    if args.max_length < 2:
        raise ValueError("--max_length must be >= 2.")

    def tokenize_batch(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        encodings: Dict[str, Any] = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length - 1,
            padding=False,
        )
        input_ids_batch: Sequence[Sequence[int]] = encodings["input_ids"]
        attention_mask_batch: Sequence[Sequence[int]] = encodings["attention_mask"]
        fixed_ids, fixed_mask = _append_eos_to_batch(
            input_ids_batch=input_ids_batch,
            attention_mask_batch=attention_mask_batch,
            eos_token_id=eos_token_id,
            max_length=args.max_length,
        )
        labels: List[int] = [int(x) for x in batch["label"]]
        return {"input_ids": fixed_ids, "attention_mask": fixed_mask, "labels": labels}

    train_tokenized = train_ds.map(tokenize_batch, batched=True, remove_columns=train_ds.column_names)
    eval_tokenized = eval_ds.map(tokenize_batch, batched=True, remove_columns=eval_ds.column_names)
    tokenized: DatasetDict = DatasetDict({"train": train_tokenized, "test": eval_tokenized})
    tokenized.set_format(type="torch")
    return tokenized, label_names


def _get_child(module: nn.Module, key: str) -> nn.Module:
    if key.isdigit():
        return module[int(key)]  # type: ignore[index]
    return getattr(module, key)


def _set_child(module: nn.Module, key: str, value: nn.Module) -> None:
    if key.isdigit():
        module[int(key)] = value  # type: ignore[index]
    else:
        setattr(module, key, value)


def get_submodule_by_path(root: nn.Module, path: str) -> nn.Module:
    if not path:
        return root
    cur = root
    for part in path.split("."):
        cur = _get_child(cur, part)
    return cur


def set_submodule_by_path(root: nn.Module, path: str, value: nn.Module) -> None:
    if "." in path:
        parent_path, leaf = path.rsplit(".", 1)
        parent = get_submodule_by_path(root, parent_path)
    else:
        parent = root
        leaf = path
    _set_child(parent, leaf, value)


def find_experts_attr(module: nn.Module) -> Optional[str]:
    preferred = ["experts", "local_experts", "moe_experts"]
    for name in preferred:
        child = module._modules.get(name)
        if isinstance(child, (nn.ModuleList, nn.ModuleDict)) and len(child) > 1:
            return name

    candidates = []
    for name, child in module._modules.items():
        if isinstance(child, (nn.ModuleList, nn.ModuleDict)) and len(child) > 1:
            candidates.append(name)
    if not candidates:
        return None

    for c in candidates:
        if "expert" in c.lower():
            return c
    if len(candidates) == 1:
        return candidates[0]
    return None


def find_gate_linear_path(moe_module: nn.Module, num_experts: int) -> Optional[str]:
    priority_names = [
        "router.classifier",
        "gate",
        "router",
        "gating",
        "wg",
        "router_gate",
        "gate_proj",
        "moe_gate",
    ]

    for name in priority_names:
        child: Optional[nn.Module]
        if "." in name:
            try:
                child = get_submodule_by_path(moe_module, name)
            except (AttributeError, IndexError, KeyError):
                child = None
        else:
            child = moe_module._modules.get(name)

        if isinstance(child, nn.Linear) and child.out_features == num_experts:
            return name
        if child is not None:
            for sub_path, sub in child.named_modules():
                if sub_path and isinstance(sub, nn.Linear) and sub.out_features == num_experts:
                    return f"{name}.{sub_path}"

    best = None
    for sub_path, sub in moe_module.named_modules():
        if not sub_path:
            continue
        if isinstance(sub, nn.Linear) and sub.out_features == num_experts:
            if best is None or len(sub_path.split(".")) < len(best.split(".")):
                best = sub_path
    return best


def parse_layer_id(layer_id: str) -> Tuple[str, int]:
    match = LAYER_KEY_PATTERN.match(layer_id)
    if match is None:
        raise ValueError(f"Invalid layer key '{layer_id}'. Expected format 'encoder.N' or 'decoder.N'.")
    return match.group(1), int(match.group(2))


def layer_sort_key(layer_id: str) -> Tuple[int, int]:
    stack_name, layer_idx = parse_layer_id(layer_id)
    return (0 if stack_name == "encoder" else 1), layer_idx


def parse_expert_index_from_key(key: str) -> int:
    if key.isdigit():
        return int(key)
    match = re.search(r"(\d+)$", key)
    if match is None:
        raise ValueError(f"Cannot parse expert index from key '{key}'.")
    return int(match.group(1))


def get_expert_entries(mlp_module: nn.Module, experts_attr: str) -> List[Tuple[int, str, nn.Module]]:
    container = getattr(mlp_module, experts_attr)
    if isinstance(container, nn.ModuleList):
        return [(idx, str(idx), expert) for idx, expert in enumerate(container)]
    if isinstance(container, nn.ModuleDict):
        sortable = [(parse_expert_index_from_key(key), key, expert) for key, expert in container.items()]
        sortable.sort(key=lambda x: x[0])
        return sortable
    raise RuntimeError(
        f"Unsupported experts container type for '{experts_attr}': {type(container)}. "
        "Expected ModuleList or ModuleDict."
    )


def set_expert_module(mlp_module: nn.Module, experts_attr: str, expert_key: str, expert_module: nn.Module) -> None:
    container = getattr(mlp_module, experts_attr)
    if isinstance(container, nn.ModuleList):
        container[int(expert_key)] = expert_module
        return
    if isinstance(container, nn.ModuleDict):
        container[expert_key] = expert_module
        return
    raise RuntimeError(
        f"Unsupported experts container type for '{experts_attr}': {type(container)}. "
        "Expected ModuleList or ModuleDict."
    )


def set_expert_container(mlp_module: nn.Module, experts_attr: str, new_container: nn.Module) -> None:
    setattr(mlp_module, experts_attr, new_container)


@dataclass
class MoeHandle:
    layer_id: str
    stack_name: str
    layer_idx: int
    layer_path: str
    layer_module: nn.Module
    ff_module: nn.Module
    mlp_module: nn.Module
    experts_attr: str
    gate_linear_path: Optional[str]


def collect_moe_handles(model: nn.Module) -> List[MoeHandle]:
    if not hasattr(model, "encoder") or not hasattr(model, "decoder"):
        raise RuntimeError("SwitchTransformer structure not found: expected model.encoder/model.decoder stacks.")

    handles: List[MoeHandle] = []
    for stack_name in ["encoder", "decoder"]:
        stack_module = getattr(model, stack_name)
        if not hasattr(stack_module, "block"):
            raise RuntimeError(f"SwitchTransformer stack '{stack_name}' has no 'block' modules.")
        for layer_idx, layer in enumerate(stack_module.block):
            ff_module = layer.layer[-1]
            if not bool(getattr(ff_module, "is_sparse", False)):
                continue
            mlp = getattr(ff_module, "mlp", None)
            if mlp is None:
                raise RuntimeError(f"Sparse layer {stack_name}.block.{layer_idx} has no FF 'mlp' module.")

            experts_attr = find_experts_attr(mlp)
            if not experts_attr:
                raise RuntimeError(f"Failed to find experts container in {stack_name}.block.{layer_idx}.layer[-1].mlp.")
            entries = get_expert_entries(mlp, experts_attr)
            if len(entries) <= 1:
                raise RuntimeError(
                    f"Expected >1 experts in {stack_name}.block.{layer_idx}.layer[-1].mlp.{experts_attr}."
                )
            gate_path = find_gate_linear_path(mlp, len(entries))
            layer_id = f"{stack_name}.{layer_idx}"
            handles.append(
                MoeHandle(
                    layer_id=layer_id,
                    stack_name=stack_name,
                    layer_idx=layer_idx,
                    layer_path=f"{stack_name}.block.{layer_idx}",
                    layer_module=layer,
                    ff_module=ff_module,
                    mlp_module=mlp,
                    experts_attr=experts_attr,
                    gate_linear_path=gate_path,
                )
            )
    handles.sort(key=lambda x: layer_sort_key(x.layer_id))
    return handles


def parse_layer_hot_map_json(raw: str) -> Dict[str, List[int]]:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for layer-hot-expert mapping: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("Layer-hot-expert mapping must be a JSON object of layer_id -> expert list.")
    out: Dict[str, List[int]] = {}
    for k, v in obj.items():
        if not isinstance(k, str):
            raise ValueError(f"Layer key '{k}' must be a string.")
        parse_layer_id(k)
        if not isinstance(v, list) or not v:
            raise ValueError(f"Layer {k} must map to a non-empty list of expert indices.")
        out[k] = [int(x) for x in v]
    return out


def load_manual_hot_map(args: argparse.Namespace) -> Dict[str, List[int]]:
    file_map: Dict[str, List[int]] = {}
    cli_map: Dict[str, List[int]] = {}

    if args.hot_map_path:
        p = Path(args.hot_map_path)
        if not p.exists():
            raise FileNotFoundError(f"--hot_map_path file not found: {p}")
        file_map = parse_layer_hot_map_json(p.read_text(encoding="utf-8"))

    if args.layer_hot_experts:
        cli_map = parse_layer_hot_map_json(args.layer_hot_experts)

    merged = dict(file_map)
    merged.update(cli_map)
    return merged


def validate_assigned_experts(assigned: Sequence[int], num_experts: int, layer_id: str, source: str) -> List[int]:
    unique_sorted = sorted(set(int(i) for i in assigned))
    if not unique_sorted:
        raise ValueError(f"{source}: layer {layer_id} has empty assigned experts.")
    if any(i < 0 or i >= num_experts for i in unique_sorted):
        raise ValueError(f"{source}: layer {layer_id} assignment {unique_sorted} out of range [0, {num_experts - 1}].")
    return unique_sorted


def infer_first_dim_size(obj) -> int:
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return int(obj.numel())
        return int(obj.shape[0])
    if isinstance(obj, (list, tuple)):
        for item in obj:
            size = infer_first_dim_size(item)
            if size > 0:
                return size
        return 0
    if isinstance(obj, dict):
        for item in obj.values():
            size = infer_first_dim_size(item)
            if size > 0:
                return size
        return 0
    return 0


def run_hot_expert_calibration(
    model: nn.Module,
    train_dataset: Any,
    tokenizer: PreTrainedTokenizerBase,
    moe_handles: Sequence[MoeHandle],
    hot_k: int,
    calib_samples: int,
    calib_batch_size: int,
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    if hot_k <= 0:
        raise ValueError("--hot_k must be > 0.")
    if calib_batch_size <= 0:
        raise ValueError("--calib_batch_size must be > 0.")

    n = min(max(calib_samples, 1), len(train_dataset))
    subset = train_dataset.select(range(n))
    collator: DataCollatorWithPadding = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")
    loader = DataLoader(subset, batch_size=calib_batch_size, shuffle=False, collate_fn=collator)
    num_batches = math.ceil(n / calib_batch_size)

    if torch.cuda.is_available() and next(model.parameters()).device.type != "cuda":
        model.to("cuda")
    device = next(model.parameters()).device
    print(
        f"[INFO] Calibrating hot experts with {n} samples, batch_size={calib_batch_size}, "
        f"batches={num_batches}, hot_k={hot_k}, device={device}"
    )

    token_counts: Dict[str, List[int]] = {}
    hooks = []
    for handle in moe_handles:
        entries = get_expert_entries(handle.mlp_module, handle.experts_attr)
        num_experts = len(entries)
        if hot_k > num_experts:
            raise ValueError(f"--hot_k={hot_k} exceeds num_experts={num_experts} at layer {handle.layer_id}.")
        token_counts[handle.layer_id] = [0 for _ in range(num_experts)]
        for expert_idx, _, expert in entries:

            def make_hook(layer_id: str, idx: int):
                def _hook(_module, inputs):
                    if not inputs:
                        return
                    count = infer_first_dim_size(inputs[0])
                    token_counts[layer_id][idx] += max(0, int(count))

                return _hook

            hooks.append(expert.register_forward_pre_hook(make_hook(handle.layer_id, expert_idx)))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = {}
            for key in ["input_ids", "attention_mask"]:
                if key in batch:
                    inputs[key] = batch[key].to(device)
            if not inputs:
                continue
            model(**inputs, use_cache=False, return_dict=False)

    for hook in hooks:
        hook.remove()
    if was_training:
        model.train()

    layer_hot_map: Dict[str, List[int]] = {}
    for handle in moe_handles:
        layer_id = handle.layer_id
        counts = token_counts[layer_id]
        observed = [i for i in range(len(counts)) if counts[i] > 0]
        observed_sorted = sorted(observed, key=lambda i: (-counts[i], i))
        selected = observed_sorted[:hot_k]
        if len(selected) < hot_k:
            for i in range(len(counts)):
                if i not in selected:
                    selected.append(i)
                if len(selected) == hot_k:
                    break
            print(
                f"[WARN] Layer {layer_id} observed fewer than hot_k active experts during calibration; filled deterministically."
            )
        layer_hot_map[layer_id] = selected
    return layer_hot_map, token_counts


def resolve_layer_hot_map(
    args: argparse.Namespace,
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    moe_handles: Sequence[MoeHandle],
    train_dataset: Any,
) -> Tuple[Dict[str, List[int]], Optional[Dict[str, List[int]]]]:
    manual_map = load_manual_hot_map(args)
    auto_map: Optional[Dict[str, List[int]]] = None

    if args.mode in {"mix", "drop"} and args.auto_hot_experts:
        if args.hot_k is None:
            raise ValueError("--hot_k is required for mix/drop when --auto_hot_experts is enabled.")
        auto_map, _ = run_hot_expert_calibration(
            model=model,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            moe_handles=moe_handles,
            hot_k=args.hot_k,
            calib_samples=args.calib_samples,
            calib_batch_size=args.calib_batch_size,
        )

    moe_layer_ids = {h.layer_id for h in moe_handles}
    unknown_manual = sorted(set(manual_map.keys()) - moe_layer_ids, key=layer_sort_key)
    if unknown_manual:
        raise ValueError(f"Manual hot map has non-MoE layer ids: {unknown_manual}")

    resolved: Dict[str, List[int]] = {}
    for handle in moe_handles:
        num_experts = len(get_expert_entries(handle.mlp_module, handle.experts_attr))
        if handle.layer_id in manual_map:
            source = "manual"
            assigned = manual_map[handle.layer_id]
        elif auto_map is not None and handle.layer_id in auto_map:
            source = "auto"
            assigned = auto_map[handle.layer_id]
        else:
            source = "fallback"
            assigned = args.hot_experts

        resolved[handle.layer_id] = validate_assigned_experts(
            assigned=assigned,
            num_experts=num_experts,
            layer_id=handle.layer_id,
            source=source,
        )
    return resolved, auto_map


def save_hot_map(path: str, layer_hot_map: Dict[str, List[int]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in sorted(layer_hot_map.items(), key=lambda kv: layer_sort_key(kv[0]))}
    p.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def clamp_topk(module: nn.Module, max_k: int) -> None:
    for attr in ["top_k", "k", "moe_top_k", "num_experts_per_tok"]:
        if hasattr(module, attr):
            value = getattr(module, attr)
            if isinstance(value, int) and value > max_k:
                setattr(module, attr, max_k)


def update_num_experts_attrs(module: nn.Module, new_num_experts: int) -> None:
    for attr in [
        "num_experts",
        "n_experts",
        "num_local_experts",
        "n_routed_experts",
        "total_experts",
    ]:
        if hasattr(module, attr):
            value = getattr(module, attr)
            if isinstance(value, int):
                setattr(module, attr, new_num_experts)


def set_embed_and_classifier_trainable_fp32(model: nn.Module) -> None:
    input_emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if input_emb is not None:
        input_emb.to(torch.float32)
        input_emb.requires_grad_(True)

    output_emb = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if output_emb is not None:
        output_emb.to(torch.float32)
        output_emb.requires_grad_(True)

    if hasattr(model, "classification_head") and isinstance(model.classification_head, nn.Module):
        model.classification_head.to(torch.float32)
        model.classification_head.requires_grad_(True)


class DTypeCastExpertWrapper(nn.Module):
    def __init__(self, expert: nn.Module, compute_dtype: torch.dtype, output_dtype: torch.dtype):
        super().__init__()
        self.expert = expert
        self.compute_dtype = compute_dtype
        self.output_dtype = output_dtype

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.to(self.compute_dtype)
        y = self.expert(x)
        return y.to(self.output_dtype)


def apply_mode_full(model: nn.Module) -> None:
    model.to(torch.float32)
    model.requires_grad_(True)


def apply_mode_mix(model: nn.Module, moe_handles: Sequence[MoeHandle], layer_hot_map: Dict[str, List[int]]) -> int:
    model.to(torch.float32)
    model.requires_grad_(False)
    if not moe_handles:
        raise RuntimeError("No MoE layers with experts were detected in the model.")

    touched: int = 0
    for handle in moe_handles:
        hot_set: set[int] = set(layer_hot_map[handle.layer_id])
        entries: List[Tuple[int, str, nn.Module]] = get_expert_entries(handle.mlp_module, handle.experts_attr)
        for expert_idx, expert_key, expert in entries:
            if expert_idx in hot_set:
                expert.to(torch.float32)
                expert.requires_grad_(True)
            else:
                expert.to(torch.bfloat16)
                expert.requires_grad_(False)
                wrapped_expert: DTypeCastExpertWrapper = DTypeCastExpertWrapper(
                    expert=expert,
                    compute_dtype=torch.bfloat16,
                    output_dtype=torch.float32,
                )
                set_expert_module(handle.mlp_module, handle.experts_attr, expert_key, wrapped_expert)
        if handle.gate_linear_path:
            gate: nn.Module = get_submodule_by_path(handle.mlp_module, handle.gate_linear_path)
            gate.to(torch.float32)
            gate.requires_grad_(True)
        else:
            raise RuntimeError(
                f"Failed to find gate/router linear for MoE layer '{handle.layer_path}'. "
                "A gate in fp32 is required for mix mode."
            )
        touched += 1

    set_embed_and_classifier_trainable_fp32(model)
    return touched


@dataclass
class MixDTypeValidationStats:
    cold_bf16_param_tensors: int
    cold_bf16_param_elements: int
    non_cold_fp32_param_tensors: int
    non_cold_fp32_param_elements: int


def validate_mix_dtype_layout(
    model: nn.Module, moe_handles: Sequence[MoeHandle], layer_hot_map: Dict[str, List[int]]
) -> MixDTypeValidationStats:
    allowed_bf16_param_ids: set[int] = set()
    cold_bf16_param_tensors: int = 0
    cold_bf16_param_elements: int = 0
    for handle in moe_handles:
        hot_set: set[int] = set(layer_hot_map[handle.layer_id])
        entries: List[Tuple[int, str, nn.Module]] = get_expert_entries(handle.mlp_module, handle.experts_attr)
        for expert_idx, _expert_key, expert in entries:
            if expert_idx in hot_set:
                for param in expert.parameters():
                    if not param.is_floating_point():
                        continue
                    if param.dtype != torch.float32:
                        raise RuntimeError(
                            f"Hot expert at layer {handle.layer_id} index={expert_idx} must be fp32, got {param.dtype}."
                        )
                    if not param.requires_grad:
                        raise RuntimeError(
                            f"Hot expert at layer {handle.layer_id} index={expert_idx} must be trainable."
                        )
                continue
            for param in expert.parameters():
                if not param.is_floating_point():
                    continue
                if param.dtype != torch.bfloat16:
                    raise RuntimeError(
                        f"Cold expert at layer {handle.layer_id} index={expert_idx} must be bf16, got {param.dtype}."
                    )
                if param.requires_grad:
                    raise RuntimeError(f"Cold expert at layer {handle.layer_id} index={expert_idx} must be frozen.")
                allowed_bf16_param_ids.add(id(param))
                cold_bf16_param_tensors += 1
                cold_bf16_param_elements += int(param.numel())

    non_cold_fp32_param_tensors: int = 0
    non_cold_fp32_param_elements: int = 0
    unexpected_bf16_params: List[str] = []
    unexpected_non_fp32_params: List[str] = []
    for name, param in model.named_parameters():
        if not param.is_floating_point():
            continue
        if id(param) in allowed_bf16_param_ids:
            continue
        if param.dtype == torch.float32:
            non_cold_fp32_param_tensors += 1
            non_cold_fp32_param_elements += int(param.numel())
        elif param.dtype == torch.bfloat16:
            unexpected_bf16_params.append(name)
        else:
            unexpected_non_fp32_params.append(f"{name}:{param.dtype}")

    if unexpected_bf16_params:
        bad_preview: List[str] = unexpected_bf16_params[:10]
        raise RuntimeError(
            "Unexpected bf16 parameters outside cold experts: "
            + ", ".join(bad_preview)
            + (" ..." if len(unexpected_bf16_params) > len(bad_preview) else "")
        )
    if unexpected_non_fp32_params:
        bad_preview = unexpected_non_fp32_params[:10]
        raise RuntimeError(
            "Unexpected non-fp32 parameter dtypes outside cold experts: "
            + ", ".join(bad_preview)
            + (" ..." if len(unexpected_non_fp32_params) > len(bad_preview) else "")
        )

    return MixDTypeValidationStats(
        cold_bf16_param_tensors=cold_bf16_param_tensors,
        cold_bf16_param_elements=cold_bf16_param_elements,
        non_cold_fp32_param_tensors=non_cold_fp32_param_tensors,
        non_cold_fp32_param_elements=non_cold_fp32_param_elements,
    )


def apply_mode_drop(model: nn.Module, moe_handles: Sequence[MoeHandle], layer_hot_map: Dict[str, List[int]]) -> int:
    model.to(torch.float32)
    model.requires_grad_(False)
    if not moe_handles:
        raise RuntimeError("No MoE layers with experts were detected in the model.")

    touched = 0
    for handle in moe_handles:
        assigned = sorted(set(layer_hot_map[handle.layer_id]))
        if not assigned:
            raise ValueError(f"Layer {handle.layer_id} has empty assignment.")

        entries = get_expert_entries(handle.mlp_module, handle.experts_attr)
        num_experts = len(entries)
        if max(assigned) >= num_experts:
            raise ValueError(
                f"assigned_experts index out of range for layer '{handle.layer_path}': "
                f"max assigned {max(assigned)} >= {num_experts}"
            )

        original_container = getattr(handle.mlp_module, handle.experts_attr)
        kept_modules = [entries[i][2] for i in assigned]
        if isinstance(original_container, nn.ModuleList):
            kept_experts: nn.Module = nn.ModuleList(kept_modules)
        elif isinstance(original_container, nn.ModuleDict):
            kept_experts_dict = original_container
            for key in list(kept_experts_dict.keys()):
                del kept_experts_dict[key]
            for new_idx, expert in enumerate(kept_modules):
                kept_experts_dict[f"expert_{new_idx}"] = expert
            if hasattr(kept_experts_dict, "num_experts"):
                kept_experts_dict.num_experts = len(kept_modules)
            kept_experts = kept_experts_dict
        else:
            raise RuntimeError(
                f"Unsupported experts container type for '{handle.experts_attr}': {type(original_container)}."
            )
        set_expert_container(handle.mlp_module, handle.experts_attr, kept_experts)
        for expert in kept_modules:
            expert.to(torch.float32)
            expert.requires_grad_(True)

        if not handle.gate_linear_path:
            raise RuntimeError(
                f"Failed to find gate/router linear for MoE layer '{handle.layer_path}'. "
                "Drop mode requires rebuilding this gate."
            )
        old_gate = get_submodule_by_path(handle.mlp_module, handle.gate_linear_path)
        if not isinstance(old_gate, nn.Linear):
            raise RuntimeError(
                f"Gate at '{handle.layer_path}.mlp.{handle.gate_linear_path}' is not nn.Linear; "
                "cannot rebuild gate for drop mode."
            )

        new_gate = nn.Linear(
            in_features=old_gate.in_features,
            out_features=len(assigned),
            bias=old_gate.bias is not None,
            device=old_gate.weight.device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            new_gate.weight.copy_(old_gate.weight[assigned, :].to(torch.float32))
            if old_gate.bias is not None:
                new_gate.bias.copy_(old_gate.bias[assigned].to(torch.float32))
        new_gate.requires_grad_(True)
        set_submodule_by_path(handle.mlp_module, handle.gate_linear_path, new_gate)

        for _, sub in handle.mlp_module.named_modules():
            update_num_experts_attrs(sub, len(assigned))
            clamp_topk(sub, len(assigned))
        update_num_experts_attrs(handle.mlp_module, len(assigned))
        clamp_topk(handle.mlp_module, len(assigned))
        update_num_experts_attrs(handle.ff_module, len(assigned))
        clamp_topk(handle.ff_module, len(assigned))
        update_num_experts_attrs(handle.layer_module, len(assigned))
        clamp_topk(handle.layer_module, len(assigned))
        touched += 1

    set_embed_and_classifier_trainable_fp32(model)
    return touched


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def count_floating_parameters(module: nn.Module) -> int:
    return sum(1 for p in module.parameters() if p.is_floating_point())


def log_stack_parameter_health(model: nn.Module) -> None:
    for stack_name in ["encoder", "decoder"]:
        stack_module: Any = getattr(model, stack_name, None)
        if not isinstance(stack_module, nn.Module):
            continue
        total_param_tensors: int = sum(1 for _ in stack_module.parameters())
        floating_param_tensors: int = count_floating_parameters(stack_module)
        print(
            f"[DEBUG] Stack {stack_name}: total_param_tensors={total_param_tensors} "
            f"floating_param_tensors={floating_param_tensors}"
        )
        if floating_param_tensors == 0:
            print(
                f"[WARN] Stack {stack_name} has zero floating-point parameter tensors. "
                "Custom stack dtype fallback guard will be used."
            )


@dataclass
class ActivationLayerStats:
    layer_id: str
    num_experts: int
    token_count: int
    expert_counts: List[int]
    expert_fraction: List[float]
    top1_conf_mean: float
    router_entropy_mean: float
    max_fraction: float
    fraction_cv: float


@dataclass
class ActivationReport:
    mode: str
    model_name: str
    dataset_name: str
    global_step: int
    epoch: Optional[float]
    analyzed_batches: int
    timestamp: str
    layer_stats: List[ActivationLayerStats]


@dataclass
class _ActivationAccumulator:
    num_experts: int
    token_count: int = 0
    expert_counts: List[int] = field(default_factory=list)
    top1_conf_sum: float = 0.0
    entropy_sum: float = 0.0

    def __post_init__(self) -> None:
        if len(self.expert_counts) == 0:
            self.expert_counts = [0 for _ in range(self.num_experts)]


class ActivationAnalyzer:
    def __init__(
        self,
        model: nn.Module,
        moe_handles: Sequence[MoeHandle],
        mode: str,
        model_name: str,
        dataset_name: str,
        report_dir: Path,
        max_eval_batches: Optional[int],
    ):
        self.model = model
        self.moe_handles = list(moe_handles)
        self.mode = mode
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.report_dir = report_dir
        self.max_eval_batches = max_eval_batches
        self._router_hook_handles: List[Any] = []
        self._model_hook_handle: Optional[Any] = None
        self._accumulators: Dict[str, _ActivationAccumulator] = {}
        self._seen_eval_batches = 0
        self._enabled = False

    def start(self) -> None:
        if self._enabled:
            return
        self._enabled = True
        self._seen_eval_batches = 0
        self._accumulators = {}
        self._router_hook_handles = []

        def _count_batch_hook(_module: nn.Module, _inputs: Tuple[Any, ...]) -> None:
            self._seen_eval_batches += 1

        self._model_hook_handle = self.model.register_forward_pre_hook(_count_batch_hook)
        for handle in self.moe_handles:
            if handle.gate_linear_path is None:
                continue
            gate_module = get_submodule_by_path(handle.mlp_module, handle.gate_linear_path)
            if not isinstance(gate_module, nn.Linear):
                continue
            self._accumulators[handle.layer_id] = _ActivationAccumulator(num_experts=int(gate_module.out_features))
            self._router_hook_handles.append(gate_module.register_forward_hook(self._make_router_hook(handle.layer_id)))

    def _make_router_hook(self, layer_id: str):
        def _hook(_module: nn.Module, _inputs: Tuple[Any, ...], output: torch.Tensor) -> None:
            if not self._enabled:
                return
            if self.max_eval_batches is not None and self._seen_eval_batches > self.max_eval_batches:
                return
            if not isinstance(output, torch.Tensor) or output.ndim < 2:
                return

            logits = output.detach().to(torch.float32).view(-1, output.shape[-1])
            if logits.numel() == 0:
                return
            probs = torch.softmax(logits, dim=-1)
            top1_conf, top1_idx = torch.max(probs, dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)

            acc = self._accumulators[layer_id]
            counts = torch.bincount(top1_idx, minlength=acc.num_experts)
            counts_list = [int(x) for x in counts.cpu().tolist()]
            for idx, count in enumerate(counts_list):
                acc.expert_counts[idx] += count
            acc.token_count += int(logits.shape[0])
            acc.top1_conf_sum += float(top1_conf.sum().item())
            acc.entropy_sum += float(entropy.sum().item())

        return _hook

    def stop_and_report(self, global_step: int, epoch: Optional[float]) -> Optional[ActivationReport]:
        if not self._enabled:
            return None
        for hook in self._router_hook_handles:
            hook.remove()
        self._router_hook_handles = []
        if self._model_hook_handle is not None:
            self._model_hook_handle.remove()
            self._model_hook_handle = None
        self._enabled = False
        if not self._accumulators:
            return None

        report = self._build_report(global_step=global_step, epoch=epoch)
        self._print_report(report)
        self._save_report(report)
        return report

    def _build_report(self, global_step: int, epoch: Optional[float]) -> ActivationReport:
        layer_stats: List[ActivationLayerStats] = []
        for layer_id in sorted(self._accumulators.keys(), key=layer_sort_key):
            acc = self._accumulators[layer_id]
            token_count = max(acc.token_count, 0)
            if token_count > 0:
                fractions = [float(c) / float(token_count) for c in acc.expert_counts]
                top1_conf_mean = acc.top1_conf_sum / float(token_count)
                entropy_mean = acc.entropy_sum / float(token_count)
            else:
                fractions = [0.0 for _ in acc.expert_counts]
                top1_conf_mean = 0.0
                entropy_mean = 0.0

            max_fraction = max(fractions) if fractions else 0.0
            mean_fraction = float(sum(fractions) / max(len(fractions), 1)) if fractions else 0.0
            variance = (
                float(sum((x - mean_fraction) ** 2 for x in fractions) / max(len(fractions), 1)) if fractions else 0.0
            )
            std = math.sqrt(variance)
            fraction_cv = std / mean_fraction if mean_fraction > 0 else 0.0

            layer_stats.append(
                ActivationLayerStats(
                    layer_id=layer_id,
                    num_experts=acc.num_experts,
                    token_count=token_count,
                    expert_counts=list(acc.expert_counts),
                    expert_fraction=fractions,
                    top1_conf_mean=top1_conf_mean,
                    router_entropy_mean=entropy_mean,
                    max_fraction=max_fraction,
                    fraction_cv=fraction_cv,
                )
            )

        analyzed_batches = self._seen_eval_batches
        if self.max_eval_batches is not None:
            analyzed_batches = min(analyzed_batches, self.max_eval_batches)
        return ActivationReport(
            mode=self.mode,
            model_name=self.model_name,
            dataset_name=self.dataset_name,
            global_step=global_step,
            epoch=epoch,
            analyzed_batches=analyzed_batches,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            layer_stats=layer_stats,
        )

    def _print_report(self, report: ActivationReport) -> None:
        print(
            "[ACT] "
            f"step={report.global_step} epoch={report.epoch} analyzed_batches={report.analyzed_batches} "
            f"layers={len(report.layer_stats)}"
        )
        for layer in report.layer_stats:
            print(
                "[ACT] "
                f"layer={layer.layer_id} tokens={layer.token_count} "
                f"max_fraction={layer.max_fraction:.4f} cv={layer.fraction_cv:.4f} "
                f"top1_conf_mean={layer.top1_conf_mean:.4f} entropy_mean={layer.router_entropy_mean:.4f}"
            )

    def _save_report(self, report: ActivationReport) -> None:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        epoch_or_step: str
        if report.epoch is None:
            epoch_or_step = f"step_{report.global_step}"
        else:
            epoch_or_step = f"{report.epoch:.4f}".replace(".", "p")
        report_path = self.report_dir / f"epoch_{epoch_or_step}.json"
        report_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
        print(f"[ACT] Saved activation report to {report_path}")


class PoCMetricsCallback(TrainerCallback):
    def __init__(self, trainer_ref: "TokenTrackingTrainer"):
        self.trainer_ref = trainer_ref
        self.start_time: Optional[float] = None
        self.logged_first_step_vram: bool = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available() and state.global_step >= 1 and not self.logged_first_step_vram:
            peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
            print(f"[METRIC] Peak VRAM after first optimizer step: {peak_mb:.2f} MB")
            self.logged_first_step_vram = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_local_process_zero or state.global_step == 0:
            return
        elapsed = max(time.time() - (self.start_time or time.time()), 1e-6)
        tokens_per_sec = self.trainer_ref.tokens_seen / elapsed
        steps_per_sec = state.global_step / elapsed
        print(f"[METRIC] step={state.global_step} " f"tokens/sec={tokens_per_sec:.2f} steps/sec={steps_per_sec:.4f}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not state.is_local_process_zero:
            return
        if metrics and "eval_loss" in metrics:
            print(f"[METRIC] eval step={state.global_step} eval_loss={metrics['eval_loss']:.6f}")
        if metrics and "eval_accuracy" in metrics:
            print(
                f"[METRIC] eval step={state.global_step} eval_accuracy={metrics['eval_accuracy']:.6f} "
                f"eval_macro_f1={metrics.get('eval_macro_f1', float('nan')):.6f}"
            )


class TokenTrackingTrainer(Trainer):
    def __init__(self, *args, activation_analyzer: Optional[ActivationAnalyzer] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokens_seen: int = 0
        self.activation_analyzer: Optional[ActivationAnalyzer] = activation_analyzer

    def training_step(self, model, inputs, num_items_in_batch=None):
        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            self.tokens_seen += int(input_ids.numel())
        try:
            return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        except TypeError:
            return super().training_step(model, inputs)

    def evaluate(self, *args, **kwargs):
        if self.activation_analyzer is None:
            return super().evaluate(*args, **kwargs)

        self.activation_analyzer.start()
        metrics = {}
        try:
            metrics = super().evaluate(*args, **kwargs)
        finally:
            report = self.activation_analyzer.stop_and_report(
                global_step=int(self.state.global_step),
                epoch=(float(self.state.epoch) if self.state.epoch is not None else None),
            )
            if report is not None and isinstance(metrics, dict):
                metrics["eval_activation_layers"] = float(len(report.layer_stats))
                metrics["eval_activation_batches"] = float(report.analyzed_batches)
        return metrics


def compute_macro_f1(preds: np.ndarray, labels: np.ndarray, num_labels: int) -> float:
    f1_values: List[float] = []
    for label in range(num_labels):
        tp = int(np.sum((preds == label) & (labels == label)))
        fp = int(np.sum((preds == label) & (labels != label)))
        fn = int(np.sum((preds != label) & (labels == label)))
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1_values.append(0.0)
        else:
            f1_values.append(2.0 * precision * recall / (precision + recall))
    return float(sum(f1_values) / max(len(f1_values), 1))


def make_compute_metrics(num_labels: int):
    def _compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        predictions: Any = eval_pred.predictions
        logits: np.ndarray = predictions[0] if isinstance(predictions, tuple) else predictions
        preds: np.ndarray = np.argmax(logits, axis=-1)
        labels: np.ndarray = eval_pred.label_ids
        accuracy: float = float(np.mean(preds == labels))
        macro_f1: float = compute_macro_f1(preds=preds, labels=labels, num_labels=num_labels)
        return {"accuracy": accuracy, "macro_f1": macro_f1}

    return _compute_metrics


def main() -> None:
    args: argparse.Namespace = parse_args()
    set_seed(args.seed)
    if args.activation_max_eval_batches is not None and args.activation_max_eval_batches <= 0:
        raise ValueError("--activation_max_eval_batches must be > 0 when provided.")

    # Hardcoded for switch-base-8 comparison protocol: full/mix/drop all load in fp32.
    load_dtype: torch.dtype = torch.float32

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer must define eos_token for SwitchTransformer sequence classification.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized, label_names = build_dataset(args, tokenizer)
    train_dataset = tokenized["train"]
    eval_dataset = tokenized["test"]
    num_labels = len(label_names)
    id2label: Dict[int, str] = {i: name for i, name in enumerate(label_names)}
    label2id: Dict[str, int] = {name: i for i, name in enumerate(label_names)}
    print(
        f"[INFO] Dataset prepared: train={len(train_dataset)} eval={len(eval_dataset)} "
        f"num_labels={num_labels} max_length={args.max_length}"
    )

    print(f"[INFO] Loading model '{args.model_name}' with dtype={load_dtype} mode={args.mode}")
    model: SwitchTransformersForSequenceClassification = SwitchTransformersForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        torch_dtype=load_dtype,
        tie_word_embeddings=False,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if model.config.eos_token_id is None:
        model.config.eos_token_id = tokenizer.eos_token_id
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = model.config.pad_token_id
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    moe_handles: List[MoeHandle] = collect_moe_handles(model)
    print(f"[INFO] Detected sparse MoE layers: {len(moe_handles)}")
    if not moe_handles:
        raise RuntimeError("No sparse MoE layers with experts were detected in SwitchTransformer encoder/decoder.")

    layer_hot_map: Dict[str, List[int]] = {}
    if args.mode in {"mix", "drop"}:
        layer_hot_map, auto_map = resolve_layer_hot_map(
            args=args,
            model=model,
            tokenizer=tokenizer,
            moe_handles=moe_handles,
            train_dataset=train_dataset,
        )

        for handle in moe_handles:
            num_experts = len(get_expert_entries(handle.mlp_module, handle.experts_attr))
            chosen = layer_hot_map[handle.layer_id]
            print(f"[HOTMAP] layer={handle.layer_id} num_experts={num_experts} assigned={chosen}")

        save_path = args.save_hot_map_path or str(Path(args.output_dir) / "hot_experts_map.json")
        save_hot_map(save_path, layer_hot_map)
        print(f"[INFO] Saved layer-hot-expert map to {save_path}")
        if auto_map is not None:
            print("[INFO] Auto hot-expert selection applied (precedence: manual > auto > fallback).")

    if args.mode == "full":
        apply_mode_full(model)
        print("[INFO] Applied mode=full (all params trainable, fp32).")
    elif args.mode == "mix":
        touched = apply_mode_mix(model, moe_handles=moe_handles, layer_hot_map=layer_hot_map)
        mix_dtype_stats: MixDTypeValidationStats = validate_mix_dtype_layout(
            model=model,
            moe_handles=moe_handles,
            layer_hot_map=layer_hot_map,
        )
        print(
            "[INFO] Applied mode=mix: hot experts trainable fp32, cold experts frozen bf16, "
            f"all non-cold parameters kept fp32. MoE layers touched={touched}."
        )
        print(
            "[DEBUG] Mix dtype validation: "
            f"cold_bf16_param_tensors={mix_dtype_stats.cold_bf16_param_tensors} "
            f"cold_bf16_param_elements={mix_dtype_stats.cold_bf16_param_elements} "
            f"non_cold_fp32_param_tensors={mix_dtype_stats.non_cold_fp32_param_tensors} "
            f"non_cold_fp32_param_elements={mix_dtype_stats.non_cold_fp32_param_elements}"
        )
    elif args.mode == "drop":
        touched = apply_mode_drop(model, moe_handles=moe_handles, layer_hot_map=layer_hot_map)
        print(
            "[INFO] Applied mode=drop: physically extracted assigned experts, rebuilt gates, "
            f"and kept gate+experts+embeddings+classification_head trainable. MoE layers touched={touched}."
        )
    else:
        raise ValueError(f"Unknown mode {args.mode}")

    log_stack_parameter_health(model)
    moe_handles_for_analysis = collect_moe_handles(model)

    trainable, total = count_parameters(model)
    ratio = 100.0 * trainable / max(total, 1)
    print(f"[METRIC] Trainable parameters: {trainable:,}")
    print(f"[METRIC] Total parameters:     {total:,}")
    print(f"[METRIC] Trainable ratio:      {ratio:.4f}%")

    data_collator: DataCollatorWithPadding = DataCollatorWithPadding(
        tokenizer=tokenizer, padding=True, return_tensors="pt"
    )

    save_strategy = "no" if args.save_steps <= 0 else "steps"
    warmup_steps: int = int(0.03 * args.max_steps) if args.max_steps > 0 else 0
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy=save_strategy,
        save_steps=args.save_steps if args.save_steps > 0 else 500,
        bf16=False,
        fp16=False,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to="none",
        remove_unused_columns=False,
        lr_scheduler_type="constant",
        warmup_steps=warmup_steps,
    )

    activation_report_dir: Path = (
        Path(args.activation_report_dir)
        if args.activation_report_dir is not None
        else Path(args.output_dir) / "activation_stats"
    )
    activation_analyzer: Optional[ActivationAnalyzer] = None
    if args.activation_analysis:
        activation_analyzer = ActivationAnalyzer(
            model=model,
            moe_handles=moe_handles_for_analysis,
            mode=args.mode,
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            report_dir=activation_report_dir,
            max_eval_batches=args.activation_max_eval_batches,
        )

    trainer = TokenTrackingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(num_labels=num_labels),
        activation_analyzer=activation_analyzer,
    )
    trainer.add_callback(PoCMetricsCallback(trainer))

    print("[INFO] Starting training...")
    train_result = trainer.train()
    print(f"[INFO] Training complete. global_step={trainer.state.global_step}")
    print(f"[INFO] Final training loss={train_result.training_loss:.6f}")

    metrics = trainer.evaluate()
    if "eval_loss" in metrics:
        print(f"[METRIC] Final eval_loss={metrics['eval_loss']:.6f}")
    if "eval_accuracy" in metrics:
        print(
            f"[METRIC] Final eval_accuracy={metrics['eval_accuracy']:.6f} "
            f"eval_macro_f1={metrics.get('eval_macro_f1', float('nan')):.6f}"
        )

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[INFO] Model/tokenizer saved to {args.output_dir}")


if __name__ == "__main__":
    main()
