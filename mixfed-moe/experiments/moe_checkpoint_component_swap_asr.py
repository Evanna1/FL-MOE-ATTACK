"""Decompose ASR degradation by swapping one layer's expert parameters.

Given two checkpoints with the same architecture, this experiment evaluates:

* checkpoint 1 as-is;
* checkpoint 2 as-is;
* checkpoint 1 with a selected parameter group copied from checkpoint 2;
* checkpoint 2 with the same group copied from checkpoint 1.

The default groups include target-layer experts/router/attention/norms, each
non-target layer's experts/router, aggregate other-layer experts/routers, all
attention and norm tensors, embeddings, and the classification head. This
provides a simple attribution test:

* if ``ckpt1 + experts2`` behaves like checkpoint 2, target-layer expert
  changes explain most of the ASR degradation;
* if ``ckpt2 + experts1`` recovers checkpoint 1, the remaining/shared
  parameters explain most of it;
* intermediate results suggest both components matter.

The script supports Switch-base-8, OLMoE, and Granite-MoE and writes
JSON/CSV summaries.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mixfedmoe_fl.attack import load_attack_checkpoint, load_selected_trigger  # noqa: E402
from mixfedmoe_fl.client import collect_moe_handles, get_handle_num_experts  # noqa: E402
from mixfedmoe_fl.config import SUPPORTED_DATASET_NAME_TO_HF  # noqa: E402
from mixfedmoe_fl.data import build_tokenizer  # noqa: E402
from mixfedmoe_fl.model_factory import build_sequence_classification_model  # noqa: E402

GROUP_ORDER = (
    "target_layer_experts",
    "target_layer_router",
    "target_layer_attention",
    "target_layer_norm",
    "other_layer_experts",
    "other_layer_routers",
    "all_attention_norm",
    "embedding",
    "classification_head",
)


@dataclass(frozen=True)
class SwapResult:
    name: str
    group: str
    checkpoint_base: str
    source_component: str
    asr: float
    target_hits: int
    total: int
    clean_accuracy: float
    clean_correct: int
    clean_total: int
    clean_target_rate: float


@dataclass(frozen=True)
class EvalBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    trigger_mask: torch.Tensor
    labels: torch.Tensor


def _resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_file():
        return path.resolve()
    repository_relative = REPO_ROOT / path
    if repository_relative.is_file():
        return repository_relative.resolve()
    raise FileNotFoundError(f"File does not exist: {raw}")


def _load_trigger_metadata(args: argparse.Namespace) -> Tuple[str, str, int, Tuple[int, ...]]:
    trigger = args.trigger
    target_layer = args.target_layer
    target_expert = args.target_expert
    trigger_ids = tuple(args.trigger_token_ids or ())
    if args.selected_triggers_path:
        selected = load_selected_trigger(str(_resolve_path(args.selected_triggers_path)))
        trigger = trigger or selected.trigger
        target_layer = target_layer or selected.target_layer
        target_expert = selected.target_expert if target_expert is None else target_expert
        if not trigger_ids:
            trigger_ids = tuple(selected.trigger_token_ids)
    if not trigger or not target_layer or target_expert is None or target_expert < 0:
        raise ValueError(
            "Provide --selected_triggers_path or provide --trigger, --target_layer, and --target_expert."
        )
    return str(trigger), str(target_layer), int(target_expert), trigger_ids


def _load_rows(args: argparse.Namespace, *, exclude_target: bool) -> List[Dict[str, Any]]:
    dataset_name = SUPPORTED_DATASET_NAME_TO_HF.get(args.dataset_name, args.dataset_name)
    dataset = load_dataset(dataset_name, split=args.split)
    rows = [
        {"text": str(item["text"]), "label": int(item["label"])}
        for item in dataset
        if not exclude_target or int(item["label"]) != int(args.target_label)
    ]
    if not rows:
        raise ValueError("The selected split has no examples for evaluation.")
    random.Random(args.seed).shuffle(rows)
    if exclude_target and args.max_examples > 0:
        rows = rows[: args.max_examples]
    return rows


def _make_batches(
    rows: Sequence[Dict[str, Any]],
    tokenizer: Any,
    trigger: str,
    trigger_ids: Sequence[int],
    max_length: int,
    batch_size: int,
    device: torch.device,
    use_trigger: bool = True,
) -> List[EvalBatch]:
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    if eos_id is None or pad_id is None:
        raise ValueError("Tokenizer must define eos_token_id and pad_token_id.")
    ids_for_trigger = list(trigger_ids) if use_trigger else []
    if use_trigger and not ids_for_trigger:
        ids_for_trigger = list(tokenizer(trigger, add_special_tokens=False)["input_ids"])
    if use_trigger and not ids_for_trigger:
        raise ValueError("Trigger tokenization produced no tokens.")
    prompt_budget = max_length - len(ids_for_trigger) - 1
    if prompt_budget < 1:
        raise ValueError("max_length leaves no room for prompt tokens and EOS.")
    tokenized: List[Tuple[List[int], int]] = []
    for row in rows:
        prompt_ids = list(tokenizer(
            str(row["text"]), add_special_tokens=False,
            truncation=True, max_length=prompt_budget,
        )["input_ids"])
        prompt_ids = [token_id for token_id in prompt_ids if token_id != eos_id]
        tokenized.append((prompt_ids[:prompt_budget] + ids_for_trigger + [eos_id], int(row["label"])))
    output: List[EvalBatch] = []
    for start in range(0, len(tokenized), batch_size):
        chunk = tokenized[start : start + batch_size]
        width = max(len(ids) for ids, _label in chunk)
        input_ids = torch.full((len(chunk), width), int(pad_id), dtype=torch.long)
        attention = torch.zeros((len(chunk), width), dtype=torch.long)
        trigger_mask = torch.zeros((len(chunk), width), dtype=torch.bool)
        labels = torch.tensor([label for _ids, label in chunk], dtype=torch.long)
        for row_index, (ids, _label) in enumerate(chunk):
            length = len(ids)
            input_ids[row_index, :length] = torch.tensor(ids, dtype=torch.long)
            attention[row_index, :length] = 1
            if ids_for_trigger:
                start_index = length - 1 - len(ids_for_trigger)
                trigger_mask[row_index, start_index : start_index + len(ids_for_trigger)] = True
        output.append(EvalBatch(
            input_ids.to(device), attention.to(device), trigger_mask.to(device), labels.to(device)
        ))
    return output


def _build_model(
    model_name_or_path: str,
    state: Dict[str, torch.Tensor],
    tokenizer: Any,
    device: torch.device,
    model_dtype: str,
) -> torch.nn.Module:
    classifier = state.get("classification_head.out_proj.weight")
    num_labels = int(classifier.shape[0]) if isinstance(classifier, torch.Tensor) else 2
    model = build_sequence_classification_model(
        model_name_or_path,
        num_labels=num_labels,
        id2label={index: str(index) for index in range(num_labels)},
        label2id={str(index): index for index in range(num_labels)},
        tokenizer=tokenizer,
        model_backend="auto",
        model_dtype=model_dtype,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    model.config.use_cache = False
    model.eval().requires_grad_(False).to(device)
    return model


def _load_into_model(model: torch.nn.Module, state: Dict[str, torch.Tensor], label: str) -> None:
    incompatible = model.load_state_dict(state, strict=False)
    missing = [name for name in incompatible.missing_keys if not name.endswith("position_ids")]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"{label} state mismatch: missing={missing[:10]}, "
            f"unexpected={incompatible.unexpected_keys[:10]}"
        )
    model.eval().requires_grad_(False)


def _layer_prefix(target_layer: str, backend: str) -> str:
    try:
        stack, raw_index = target_layer.split(".", 1)
        layer_index = int(raw_index)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Expected target_layer like decoder.1, got {target_layer!r}.") from exc
    if stack not in {"encoder", "decoder"}:
        raise ValueError(f"Expected encoder.N or decoder.N, got {target_layer!r}.")
    if backend == "switch":
        return f"{stack}.block.{layer_index}."
    if stack != "decoder":
        raise ValueError(
            f"OLMoE/Granite component swap expects decoder layers, got {target_layer!r}."
        )
    return f"model.layers.{layer_index}."


def _target_layer_expert_keys(
    state: Dict[str, torch.Tensor],
    target_layer: str,
    backend: str,
) -> List[str]:
    layer_prefix = _layer_prefix(target_layer, backend)
    layer_index = int(layer_prefix.split(".")[2])

    if backend == "olmoe":
        prefix = f"{layer_prefix}mlp.experts."
        keys = [
            key
            for key in state
            if key.startswith(prefix)
            and (
                key.endswith("gate_up_proj")
                or key.endswith("down_proj")
                or key.endswith("gate_up_proj.weight")
                or key.endswith("down_proj.weight")
            )
        ]
    elif backend == "granite":
        prefix = f"{layer_prefix}block_sparse_moe."
        keys = [
            key
            for key in state
            if key.startswith(prefix)
            and ("input_linear.weight" in key or "output_linear.weight" in key)
        ]
    elif backend == "switch":
        prefix = layer_prefix
        keys = [
            key
            for key in state
            if key.startswith(prefix)
            and ".mlp.experts." in key
            and re.search(r"expert_\d+\.(wi|wo)\.weight$", key)
        ]
    else:
        raise ValueError(f"Unsupported component-swap backend: {backend!r}")
    if not keys:
        available = [key for key in state if layer_prefix in key]
        raise RuntimeError(
            f"Could not find expert tensors for {target_layer} ({backend}). "
            f"Layer keys include: {available[:20]}"
        )
    return sorted(keys)


def _is_expert_key(key: str, backend: str) -> bool:
    if backend == "olmoe":
        return ".mlp.experts." in key and (
            key.endswith("gate_up_proj")
            or key.endswith("down_proj")
            or key.endswith("gate_up_proj.weight")
            or key.endswith("down_proj.weight")
        )
    if backend == "granite":
        return ".block_sparse_moe." in key and (
            "input_linear.weight" in key or "output_linear.weight" in key
        )
    if backend == "switch":
        return ".mlp.experts." in key and bool(
            re.search(r"expert_\d+\.(wi|wo)\.weight$", key)
        )
    return False


def _is_router_key(key: str, backend: str) -> bool:
    if backend == "olmoe":
        return ".mlp.gate." in key
    if backend == "granite":
        return ".block_sparse_moe.router." in key
    if backend == "switch":
        return ".mlp.router.classifier." in key
    return False


def _group_keys(
    state: Dict[str, torch.Tensor],
    target_layer: str,
    backend: str,
) -> Dict[str, List[str]]:
    layer_prefix = _layer_prefix(target_layer, backend)
    target_experts = set(_target_layer_expert_keys(state, target_layer, backend))
    all_experts = {key for key in state if _is_expert_key(key, backend)}
    all_routers = {key for key in state if _is_router_key(key, backend)}
    target_routers = {key for key in all_routers if key.startswith(layer_prefix)}
    attention_markers = (
        ("SelfAttention.", "EncDecAttention.")
        if backend == "switch"
        else (".self_attn.",)
    )
    target_attention = {
        key for key in state
        if key.startswith(layer_prefix)
        and any(marker in key for marker in attention_markers)
    }
    target_norm = {
        key for key in state
        if key.startswith(layer_prefix)
        and ("norm" in key.lower() or "layernorm" in key.lower())
    }
    all_attention_norm = {
        key for key in state
        if any(marker in key for marker in attention_markers)
        or "norm" in key.lower()
        or "layernorm" in key.lower()
    }
    embedding = {
        key for key in state
        if "embed_tokens" in key
        or "embed_positions" in key
        or (backend == "switch" and key == "shared.weight")
    }
    classifier = {key for key in state if key.startswith("classification_head.")}
    groups: Dict[str, set[str]] = {
        "target_layer_experts": target_experts,
        "target_layer_router": target_routers,
        "target_layer_attention": target_attention,
        "target_layer_norm": target_norm,
        "other_layer_experts": all_experts - target_experts,
        "other_layer_routers": all_routers - target_routers,
        "all_attention_norm": all_attention_norm,
        "embedding": embedding,
        "classification_head": classifier,
    }
    missing = [name for name, keys in groups.items() if not keys]
    if missing:
        raise RuntimeError(
            f"Parameter groups are empty for backend={backend}, target_layer={target_layer}: {missing}"
        )
    # Keep target-layer experts/router separate, and expose every other layer
    # as an independent swap group for finer attribution.
    target_stack, raw_target_index = target_layer.split(".", 1)
    target_layer_index = int(raw_target_index)
    if backend == "switch":
        layer_pattern = re.compile(r"^(encoder|decoder)\.block\.(\d+)\.")
        layer_specs = sorted(
            {
                (match.group(1), int(match.group(2)))
                for key in (*all_experts, *all_routers)
                if (match := layer_pattern.match(key)) is not None
            }
        )
        for stack, layer_index in layer_specs:
            if stack == target_stack and layer_index == target_layer_index:
                continue
            layer_prefix_i = f"{stack}.block.{layer_index}."
            layer_experts = {
                key for key in all_experts if key.startswith(layer_prefix_i)
            }
            layer_routers = {
                key for key in all_routers if key.startswith(layer_prefix_i)
            }
            if layer_experts:
                groups[f"{stack}_layer_{layer_index}_experts"] = layer_experts
            if layer_routers:
                groups[f"{stack}_layer_{layer_index}_router"] = layer_routers
    else:
        layer_pattern = re.compile(r"^model\.layers\.(\d+)\.")
        layer_indices = sorted(
            {
                int(match.group(1))
                for key in (*all_experts, *all_routers)
                if (match := layer_pattern.match(key)) is not None
            }
        )
        for layer_index in layer_indices:
            if layer_index == target_layer_index:
                continue
            layer_prefix_i = f"model.layers.{layer_index}."
            layer_experts = {
                key for key in all_experts if key.startswith(layer_prefix_i)
            }
            layer_routers = {
                key for key in all_routers if key.startswith(layer_prefix_i)
            }
            if layer_experts:
                groups[f"layer_{layer_index}_experts"] = layer_experts
            if layer_routers:
                groups[f"layer_{layer_index}_router"] = layer_routers

    return {name: sorted(keys) for name, keys in groups.items()}


def _compose_state(
    base: Dict[str, torch.Tensor],
    source: Dict[str, torch.Tensor],
    component_keys: Sequence[str],
) -> Dict[str, torch.Tensor]:
    composed = {key: value for key, value in base.items()}
    for key in component_keys:
        if key not in base or key not in source:
            raise RuntimeError(f"Component tensor {key!r} is missing from one checkpoint.")
        if tuple(base[key].shape) != tuple(source[key].shape):
            raise RuntimeError(
                f"Shape mismatch for {key}: base={tuple(base[key].shape)} "
                f"source={tuple(source[key].shape)}"
            )
        composed[key] = source[key]
    return composed


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    state: Dict[str, torch.Tensor],
    asr_batches: Sequence[Any],
    clean_batches: Sequence[Any],
    target_label: int,
    name: str,
    group: str,
    checkpoint_base: str,
    source_component: str,
) -> SwapResult:
    _load_into_model(model, state, name)
    target_hits = total = 0
    clean_correct = clean_target_hits = clean_total = 0
    for batch in asr_batches:
        output = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            output_router_logits=False,
            use_cache=False,
            return_dict=True,
        )
        predictions = output.logits.argmax(dim=-1)
        target_hits += int((predictions == target_label).sum().item())
        total += int(predictions.numel())
    for batch in clean_batches:
        output = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            output_router_logits=False,
            use_cache=False,
            return_dict=True,
        )
        predictions = output.logits.argmax(dim=-1)
        clean_correct += int((predictions == batch.labels).sum().item())
        clean_target_hits += int((predictions == target_label).sum().item())
        clean_total += int(predictions.numel())
    return SwapResult(
        name=name,
        group=group,
        checkpoint_base=checkpoint_base,
        source_component=source_component,
        asr=target_hits / max(total, 1),
        target_hits=target_hits,
        total=total,
        clean_accuracy=clean_correct / max(clean_total, 1),
        clean_correct=clean_correct,
        clean_total=clean_total,
        clean_target_rate=clean_target_hits / max(clean_total, 1),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint1_path", required=True, help="Earlier/high-ASR checkpoint.")
    parser.add_argument("--checkpoint2_path", required=True, help="Later/degraded-ASR checkpoint.")
    parser.add_argument("--selected_triggers_path")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--model_dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    parser.add_argument("--dataset_name", default="ag_news")
    parser.add_argument("--split", default="test")
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--target_layer")
    parser.add_argument("--target_expert", type=int)
    parser.add_argument("--trigger")
    parser.add_argument("--trigger_token_ids", type=int, nargs="+")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_examples", type=int, default=0, help="0 means all non-target ASR examples.")
    parser.add_argument("--clean_max_examples", type=int, default=-1, help="-1 uses the ASR count; 0 uses the full clean split.")
    parser.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help=(
            "Parameter groups to swap. Omit to run the fixed groups plus each "
            "non-target layer's experts/router separately. Dynamic names are "
            "layer_<index>_experts and layer_<index>_router."
        ),
    )
    parser.add_argument("--output_dir", default="outputs/checkpoint_component_swap")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    if args.batch_size <= 0 or args.max_length < 2:
        raise ValueError("batch_size must be > 0 and max_length must be >= 2.")
    if args.max_examples < 0 or args.clean_max_examples < -1:
        raise ValueError("max_examples must be >= 0 and clean_max_examples must be >= -1.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    trigger, target_layer, target_expert, trigger_ids = _load_trigger_metadata(args)
    rows = _load_rows(args, exclude_target=True)
    clean_rows = _load_rows(args, exclude_target=False)
    if args.clean_max_examples == -1:
        clean_rows = clean_rows[: len(rows)]
    elif args.clean_max_examples > 0:
        clean_rows = clean_rows[: args.clean_max_examples]

    tokenizer = build_tokenizer(args.model_name_or_path, max_length=args.max_length)
    asr_batches = _make_batches(
        rows, tokenizer, trigger, trigger_ids, args.max_length, args.batch_size, device
    )
    clean_batches = _make_batches(
        clean_rows, tokenizer, trigger, trigger_ids, args.max_length, args.batch_size, device,
        use_trigger=False,
    )
    state1, metadata1 = load_attack_checkpoint(str(_resolve_path(args.checkpoint1_path)))
    state2, metadata2 = load_attack_checkpoint(str(_resolve_path(args.checkpoint2_path)))
    model = _build_model(args.model_name_or_path, state1, tokenizer, device, args.model_dtype)
    handles = [handle for handle in collect_moe_handles(model) if handle.layer_id == target_layer]
    if len(handles) != 1:
        available = ", ".join(handle.layer_id for handle in collect_moe_handles(model))
        raise ValueError(f"Unknown or ambiguous target layer '{target_layer}'. Available: {available}")
    handle = handles[0]
    if handle.backend not in {"switch", "olmoe", "granite"}:
        raise ValueError(
            f"Only Switch, OLMoE, and Granite-MoE are supported, got {handle.backend!r}."
        )
    num_experts = int(get_handle_num_experts(handle))
    if not 0 <= target_expert < num_experts:
        raise ValueError(f"target_expert={target_expert} is outside [0, {num_experts}).")
    groups = _group_keys(state1, target_layer, handle.backend)
    state2_groups = _group_keys(state2, target_layer, handle.backend)
    if args.groups is None:
        dynamic_groups = [name for name in groups if name.startswith("layer_")]
        selected_groups = list(dict.fromkeys((*GROUP_ORDER, *dynamic_groups)))
    else:
        selected_groups = list(dict.fromkeys(args.groups))
    unknown_groups = [name for name in selected_groups if name not in groups]
    if unknown_groups:
        available = ", ".join(groups)
        raise ValueError(
            f"Unknown component group(s): {unknown_groups}. Available groups: {available}"
        )

    print(
        f"Backend={handle.backend}, target_layer={target_layer}, target_expert={target_expert}; "
        f"groups={selected_groups}; "
        f"ASR samples={len(rows)}, clean samples={len(clean_rows)}",
        flush=True,
    )
    results = [
        _evaluate(
            model, state1, asr_batches, clean_batches, args.target_label,
            "checkpoint1_full", "baseline", "checkpoint1", "checkpoint1",
        ),
        _evaluate(
            model, state2, asr_batches, clean_batches, args.target_label,
            "checkpoint2_full", "baseline", "checkpoint2", "checkpoint2",
        ),
    ]
    for group in selected_groups:
        component_keys = groups[group]
        missing_in_second = sorted(set(component_keys).difference(state2_groups[group]))
        if missing_in_second:
            raise RuntimeError(
                f"Group '{group}' has tensors missing from checkpoint2: {missing_in_second[:10]}"
            )
        hybrid_12 = _compose_state(state1, state2, component_keys)
        hybrid_21 = _compose_state(state2, state1, component_keys)
        results.append(
            _evaluate(
                model, hybrid_12, asr_batches, clean_batches, args.target_label,
                f"checkpoint1_with_{group}2", group, "checkpoint1", "checkpoint2",
            )
        )
        results.append(
            _evaluate(
                model, hybrid_21, asr_batches, clean_batches, args.target_label,
                f"checkpoint2_with_{group}1", group, "checkpoint2", "checkpoint1",
            )
        )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "moe_checkpoint_component_swap_asr",
        "checkpoint1_path": str(_resolve_path(args.checkpoint1_path)),
        "checkpoint2_path": str(_resolve_path(args.checkpoint2_path)),
        "checkpoint1_metadata": metadata1,
        "checkpoint2_metadata": metadata2,
        "model_name_or_path": args.model_name_or_path,
        "backend": handle.backend,
        "target_layer": target_layer,
        "target_expert": target_expert,
        "trigger": trigger,
        "trigger_token_ids": list(trigger_ids),
        "target_label": args.target_label,
        "asr_examples": len(rows),
        "clean_examples": len(clean_rows),
        "selected_groups": selected_groups,
        "group_tensor_keys": groups,
        "results": [asdict(result) for result in results],
    }
    (output_dir / "component_swap_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "component_swap_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)

    print("\n=== Checkpoint component-swap results ===", flush=True)
    for result in results:
        print(
            f"{result.name:32s} ASR={result.asr:.4f} ({result.target_hits}/{result.total}) "
            f"clean_acc={result.clean_accuracy:.4f} "
            f"clean_target_rate={result.clean_target_rate:.4f}",
            flush=True,
        )
    print(f"Saved JSON: {output_dir / 'component_swap_results.json'}", flush=True)
    print(f"Saved CSV : {output_dir / 'component_swap_results.csv'}", flush=True)


if __name__ == "__main__":
    main()
