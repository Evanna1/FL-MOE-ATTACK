from __future__ import annotations

import gc
import json
import logging
import math
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from flwr.client import Client, NumPyClient
from flwr.common import NDArrays, Scalar
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from mixfedmoe_fl.config import MixFedMoEConfig
from mixfedmoe_fl.data import ClientDatasetBundle, MixFedMoEDataManager
from mixfedmoe_fl.expert_profiler import (
    TrainingLFERoutingTracker,
    save_training_lfe_profile,
)
from models.switch_transformers import SwitchTransformersForSequenceClassification

LAYER_KEY_PATTERN = re.compile(r"^(encoder|decoder)\.(\d+)$")
_HF_LOGS_SILENCED = False


def _silence_hf_loading_logs_once() -> None:
    global _HF_LOGS_SILENCED
    if _HF_LOGS_SILENCED:
        return
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass
    try:
        from datasets.utils import logging as ds_logging

        ds_logging.set_verbosity_error()
    except Exception:
        pass
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    _HF_LOGS_SILENCED = True


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _parse_json_dict(raw: str) -> Dict[str, Any]:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON payload: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("Expected a JSON object.")
    return obj


def _parse_json_list(raw: str) -> List[Any]:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON payload: {exc}") from exc
    if not isinstance(obj, list):
        raise ValueError("Expected a JSON list.")
    return obj


def _as_int(config: Dict[str, Scalar], key: str, default: int) -> int:
    if key not in config:
        return default
    value = config[key]
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        return int(value)
    raise ValueError(f"Config key '{key}' must be int-castable, got type={type(value)}")


def _as_float(config: Dict[str, Scalar], key: str, default: float) -> float:
    if key not in config:
        return default
    value = config[key]
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, str)):
        return float(value)
    raise ValueError(f"Config key '{key}' must be float-castable, got type={type(value)}")


def _as_str(config: Dict[str, Scalar], key: str, default: str) -> str:
    if key not in config:
        return default
    value = config[key]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    raise ValueError(f"Config key '{key}' must be string-castable, got type={type(value)}")


def _compute_target_train_steps(local_epochs: float, steps_per_epoch: int) -> int:
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be > 0.")
    return max(1, int(math.ceil(local_epochs * float(steps_per_epoch))))


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


def find_experts_attr(module: nn.Module) -> Optional[str]:
    preferred = ["experts", "local_experts", "moe_experts"]
    for name in preferred:
        child = module._modules.get(name)
        if isinstance(child, (nn.ModuleList, nn.ModuleDict)) and len(child) > 1:
            return name

    candidates: List[str] = []
    for name, child in module._modules.items():
        if isinstance(child, (nn.ModuleList, nn.ModuleDict)) and len(child) > 1:
            candidates.append(name)
    if not candidates:
        return None

    for candidate in candidates:
        if "expert" in candidate.lower():
            return candidate
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
            for sub_path, sub_module in child.named_modules():
                if sub_path and isinstance(sub_module, nn.Linear) and sub_module.out_features == num_experts:
                    return f"{name}.{sub_path}"

    best: Optional[str] = None
    for sub_path, sub_module in moe_module.named_modules():
        if not sub_path:
            continue
        if isinstance(sub_module, nn.Linear) and sub_module.out_features == num_experts:
            if best is None or len(sub_path.split(".")) < len(best.split(".")):
                best = sub_path
    return best


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


def clamp_topk(module: nn.Module, max_k: int) -> None:
    for attr in ["top_k", "k", "moe_top_k", "num_experts_per_tok"]:
        if hasattr(module, attr):
            value = getattr(module, attr)
            if isinstance(value, int) and value > max_k:
                setattr(module, attr, max_k)


def set_embed_and_classifier_trainable_fp32(model: SwitchTransformersForSequenceClassification) -> None:
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
            mlp_module = getattr(ff_module, "mlp", None)
            if mlp_module is None:
                raise RuntimeError(f"Sparse layer {stack_name}.block.{layer_idx} has no FF 'mlp' module.")

            experts_attr = find_experts_attr(mlp_module)
            if not experts_attr:
                raise RuntimeError(f"Failed to find experts container in {stack_name}.block.{layer_idx}.layer[-1].mlp.")

            entries = get_expert_entries(mlp_module, experts_attr)
            if len(entries) <= 1:
                raise RuntimeError(
                    f"Expected >1 experts in {stack_name}.block.{layer_idx}.layer[-1].mlp.{experts_attr}."
                )

            gate_linear_path = find_gate_linear_path(mlp_module, len(entries))
            layer_id = f"{stack_name}.{layer_idx}"
            handles.append(
                MoeHandle(
                    layer_id=layer_id,
                    stack_name=stack_name,
                    layer_idx=layer_idx,
                    layer_path=f"{stack_name}.block.{layer_idx}",
                    layer_module=layer,
                    ff_module=ff_module,
                    mlp_module=mlp_module,
                    experts_attr=experts_attr,
                    gate_linear_path=gate_linear_path,
                )
            )
    handles.sort(key=lambda x: layer_sort_key(x.layer_id))
    return handles


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


def apply_mode_mix(model: nn.Module, moe_handles: Sequence[MoeHandle], layer_hot_map: Dict[str, List[int]]) -> None:
    model.to(torch.float32)
    model.requires_grad_(False)
    if not moe_handles:
        raise RuntimeError("No MoE layers with experts were detected in the model.")

    for handle in moe_handles:
        hot_set = set(layer_hot_map[handle.layer_id])
        entries = get_expert_entries(handle.mlp_module, handle.experts_attr)
        for expert_idx, expert_key, expert in entries:
            if expert_idx in hot_set:
                expert.to(torch.float32)
                expert.requires_grad_(True)
            else:
                expert.to(torch.bfloat16)
                expert.requires_grad_(False)
                wrapped_expert = DTypeCastExpertWrapper(
                    expert=expert,
                    compute_dtype=torch.bfloat16,
                    output_dtype=torch.float32,
                )
                set_expert_module(handle.mlp_module, handle.experts_attr, expert_key, wrapped_expert)

        if not handle.gate_linear_path:
            raise RuntimeError(
                f"Failed to find gate/router linear for MoE layer '{handle.layer_path}'. "
                "A gate in fp32 is required for mix mode."
            )
        gate = get_submodule_by_path(handle.mlp_module, handle.gate_linear_path)
        gate.to(torch.float32)
        gate.requires_grad_(True)

    set_embed_and_classifier_trainable_fp32(model)


def apply_mode_flex(model: nn.Module, moe_handles: Sequence[MoeHandle], layer_hot_map: Dict[str, List[int]]) -> None:
    model.to(torch.float32)
    model.requires_grad_(False)
    if not moe_handles:
        raise RuntimeError("No MoE layers with experts were detected in the model.")

    for handle in moe_handles:
        selected = sorted(set(layer_hot_map[handle.layer_id]))
        if len(selected) != 1:
            raise ValueError(
                f"Flex mode requires exactly one expert per layer. "
                f"layer={handle.layer_id} assigned={selected}."
            )
        selected_idx = selected[0]

        entries = get_expert_entries(handle.mlp_module, handle.experts_attr)
        for expert_idx, _expert_key, expert in entries:
            expert.to(torch.float32)
            expert.requires_grad_(expert_idx == selected_idx)

        if not handle.gate_linear_path:
            raise RuntimeError(
                f"Failed to find gate/router linear for MoE layer '{handle.layer_path}'. "
                "A gate in fp32 is required for flex mode."
            )
        gate = get_submodule_by_path(handle.mlp_module, handle.gate_linear_path)
        gate.to(torch.float32)
        gate.requires_grad_(True)

    set_embed_and_classifier_trainable_fp32(model)


def apply_mode_drop(model: nn.Module, moe_handles: Sequence[MoeHandle], layer_hot_map: Dict[str, List[int]]) -> None:
    model.to(torch.float32)
    model.requires_grad_(False)
    if not moe_handles:
        raise RuntimeError("No MoE layers with experts were detected in the model.")

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

    set_embed_and_classifier_trainable_fp32(model)


def _to_numpy_ndarrays(model: nn.Module, param_names: Sequence[str]) -> NDArrays:
    state_dict = model.state_dict()
    arrays: NDArrays = []
    for name in param_names:
        if name not in state_dict:
            raise KeyError(f"State dict key '{name}' not found in model.")
        arrays.append(state_dict[name].detach().cpu().numpy().copy())
    return arrays


def _load_from_numpy_ndarrays(model: nn.Module, param_names: Sequence[str], parameters: NDArrays) -> None:
    if len(param_names) != len(parameters):
        raise ValueError(
            f"Parameter name count mismatch: len(param_names)={len(param_names)} len(parameters)={len(parameters)}."
        )
    current_state = model.state_dict()
    for name, array in zip(param_names, parameters):
        if name not in current_state:
            raise KeyError(f"Received parameter '{name}' which is missing from local state_dict.")
        target = current_state[name]
        src = torch.from_numpy(array)
        if tuple(src.shape) != tuple(target.shape):
            raise ValueError(f"Shape mismatch for '{name}': expected {tuple(target.shape)} got {tuple(src.shape)}.")
        src = src.to(dtype=target.dtype)
        current_state[name] = src
    model.load_state_dict(current_state, strict=True)


def _state_dict_names(model: nn.Module) -> List[str]:
    return list(model.state_dict().keys())


def _trainable_parameter_names(model: nn.Module) -> List[str]:
    return [name for name, param in model.named_parameters() if param.requires_grad]


def _infer_first_dim_size(obj: Any) -> int:
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return int(obj.numel())
        return int(obj.shape[0])
    if isinstance(obj, (list, tuple)):
        for item in obj:
            size = _infer_first_dim_size(item)
            if size > 0:
                return size
        return 0
    if isinstance(obj, dict):
        for item in obj.values():
            size = _infer_first_dim_size(item)
            if size > 0:
                return size
        return 0
    return 0


def _collect_expert_activation_profile(
    model: nn.Module,
    moe_handles: Sequence[MoeHandle],
    dataset: Any,
    collator: DataCollatorWithPadding,
    batch_size: int,
    max_samples: int,
    device: torch.device,
) -> Dict[str, List[int]]:
    if max_samples <= 0:
        max_samples = min(128, len(dataset))
    sample_count = min(max_samples, len(dataset))
    if sample_count <= 0:
        return {handle.layer_id: [] for handle in moe_handles}

    subset = dataset.select(range(sample_count))
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=False,
    )

    token_counts: Dict[str, List[int]] = {}
    hooks: List[Any] = []
    for handle in moe_handles:
        entries = get_expert_entries(handle.mlp_module, handle.experts_attr)
        token_counts[handle.layer_id] = [0 for _ in range(len(entries))]
        for expert_idx, _, expert in entries:

            def make_hook(layer_id: str, idx: int):
                def _hook(_module: nn.Module, inputs: Tuple[Any, ...]) -> None:
                    if not inputs:
                        return
                    count = _infer_first_dim_size(inputs[0])
                    token_counts[layer_id][idx] += max(0, int(count))

                return _hook

            hooks.append(expert.register_forward_pre_hook(make_hook(handle.layer_id, expert_idx)))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )

    for hook in hooks:
        hook.remove()
    if was_training:
        model.train()

    return token_counts


def _validate_and_complete_assignments(
    mode: str,
    moe_handles: Sequence[MoeHandle],
    raw_assignments: Optional[Dict[str, List[int]]],
    fallback_k: int,
) -> Dict[str, List[int]]:
    assignments: Dict[str, List[int]] = {}
    for handle in moe_handles:
        entries = get_expert_entries(handle.mlp_module, handle.experts_attr)
        num_experts = len(entries)

        if mode == "full":
            assignments[handle.layer_id] = list(range(num_experts))
            continue

        if raw_assignments is not None and handle.layer_id in raw_assignments:
            chosen = [int(x) for x in raw_assignments[handle.layer_id]]
        else:
            k = min(max(fallback_k, 1), num_experts)
            chosen = list(range(k))

        dedup_sorted = sorted(set(chosen))
        if not dedup_sorted:
            raise ValueError(f"Layer '{handle.layer_id}' has empty assigned experts.")
        if dedup_sorted[0] < 0 or dedup_sorted[-1] >= num_experts:
            raise ValueError(
                f"Layer '{handle.layer_id}' assigned experts out of range [0, {num_experts - 1}]: {dedup_sorted}"
            )
        if mode == "flex" and len(dedup_sorted) != 1:
            raise ValueError(
                f"Flex mode requires exactly one assigned expert for layer '{handle.layer_id}', "
                f"got {dedup_sorted}."
            )
        assignments[handle.layer_id] = dedup_sorted
    return assignments


def _expand_drop_profile_to_global(
    local_profile: Dict[str, List[int]],
    full_num_experts: Dict[str, int],
    layer_assignments: Dict[str, List[int]],
) -> Dict[str, List[int]]:
    global_profile: Dict[str, List[int]] = {}
    for layer_id, local_counts in local_profile.items():
        if layer_id not in full_num_experts or layer_id not in layer_assignments:
            raise KeyError(f"Missing drop profile mapping for layer '{layer_id}'.")
        global_counts = [0 for _ in range(full_num_experts[layer_id])]
        assigned = layer_assignments[layer_id]
        if len(local_counts) != len(assigned):
            raise ValueError(
                f"Drop activation profile mismatch at layer '{layer_id}': "
                f"len(local_counts)={len(local_counts)} len(assigned)={len(assigned)}."
            )
        for local_idx, global_idx in enumerate(assigned):
            global_counts[global_idx] = int(local_counts[local_idx])
        global_profile[layer_id] = global_counts
    return global_profile


class MixFedMoEClient(NumPyClient):
    def __init__(
        self,
        client_id: int,
        runtime_config: MixFedMoEConfig,
        data_manager: MixFedMoEDataManager,
        is_malicious: bool = False,
    ) -> None:
        self.client_id = client_id
        self.runtime_config = runtime_config
        self.data_manager = data_manager
        self.is_malicious = is_malicious
        self._dataset_bundle: Optional[ClientDatasetBundle] = None
        self._poisoned_dataset_bundle: Optional[ClientDatasetBundle] = None

    def _load_bundle(self, apply_badnet: bool = False) -> ClientDatasetBundle:
        if apply_badnet:
            if self._poisoned_dataset_bundle is None:
                self._poisoned_dataset_bundle = self.data_manager.load_client_dataset(
                    partition_id=self.client_id,
                    apply_badnet=True,
                )
            return self._poisoned_dataset_bundle
        if self._dataset_bundle is None:
            self._dataset_bundle = self.data_manager.load_client_dataset(partition_id=self.client_id)
        return self._dataset_bundle

    def _is_attack_active(self, server_round: int) -> bool:
        return bool(
            self.runtime_config.attack_enabled
            and self.is_malicious
            and server_round >= self.runtime_config.attack_start_round
        )

    def _build_model(self, bundle: ClientDatasetBundle) -> SwitchTransformersForSequenceClassification:
        _silence_hf_loading_logs_once()
        model = SwitchTransformersForSequenceClassification.from_pretrained(
            self.runtime_config.model_name_or_path,
            num_labels=bundle.label_info.num_labels,
            id2label=bundle.label_info.id2label,
            label2id=bundle.label_info.label2id,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.float32,
            tie_word_embeddings=False,
        )
        tokenizer = self.data_manager.tokenizer
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        if model.config.eos_token_id is None:
            model.config.eos_token_id = tokenizer.eos_token_id
        if model.config.decoder_start_token_id is None:
            model.config.decoder_start_token_id = model.config.pad_token_id
        model.config.use_cache = False
        return model

    def _resolve_incoming_param_names(
        self,
        model: nn.Module,
        parameters: NDArrays,
        round_config: Dict[str, Scalar],
    ) -> List[str]:
        full_names = _state_dict_names(model)
        full_name_set = set(full_names)
        if len(parameters) != len(full_names):
            raise ValueError(
                "This client expects full-state parameters from server for all modes. "
                f"Received len(parameters)={len(parameters)} but full state len={len(full_names)}."
            )

        for key in ["parameter_names_json", "parameter_keys_json", "incoming_parameter_names_json"]:
            if key in round_config:
                raw = _as_str(round_config, key, "")
                names = [str(x) for x in _parse_json_list(raw)]
                if len(names) != len(parameters):
                    raise ValueError(
                        f"Incoming parameter names mismatch: key='{key}' len(names)={len(names)} "
                        f"len(parameters)={len(parameters)}."
                    )
                if len(set(names)) != len(names):
                    raise ValueError(f"Incoming parameter names contain duplicates for key='{key}'.")
                unknown = [name for name in names if name not in full_name_set]
                if unknown:
                    raise KeyError(
                        f"Incoming parameter names contain unknown keys for key='{key}', "
                        f"first unknown='{unknown[0]}'."
                    )
                return names

        return full_names

    def _resolve_assignments(
        self,
        mode: str,
        moe_handles: Sequence[MoeHandle],
        round_config: Dict[str, Scalar],
    ) -> Dict[str, List[int]]:
        raw_assignments: Optional[Dict[str, List[int]]] = None
        for key in ["assigned_experts_json", "layer_hot_map_json", "expert_assignment_json"]:
            if key in round_config:
                payload = _parse_json_dict(_as_str(round_config, key, "{}"))
                normalized: Dict[str, List[int]] = {}
                for layer_id, values in payload.items():
                    if not isinstance(layer_id, str):
                        raise ValueError(f"Assignment layer key must be string, got {type(layer_id)}.")
                    if not isinstance(values, list):
                        raise ValueError(f"Assignment for layer '{layer_id}' must be list[int].")
                    normalized[layer_id] = [int(v) for v in values]
                raw_assignments = normalized
                break

        k = _as_int(round_config, "k", self.runtime_config.k)
        return _validate_and_complete_assignments(
            mode=mode,
            moe_handles=moe_handles,
            raw_assignments=raw_assignments,
            fallback_k=k,
        )

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        bundle = self._load_bundle()
        model = self._build_model(bundle)
        try:
            names = _state_dict_names(model)
            return _to_numpy_ndarrays(model, names)
        finally:
            model.cpu()
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        server_round = _as_int(config, "server_round", 1)
        attack_active = self._is_attack_active(server_round)
        bundle = self._load_bundle(apply_badnet=attack_active)
        _set_seed(self.runtime_config.seed + self.client_id)

        model = self._build_model(bundle)
        optimizer: Optional[AdamW] = None
        train_loader: Optional[DataLoader] = None
        lfe_tracker: Optional[TrainingLFERoutingTracker] = None
        try:
            mode = _as_str(config, "mode", self.runtime_config.mode).lower()
            if mode not in {"full", "mix", "drop", "flex"}:
                raise ValueError(f"Unsupported client mode='{mode}'.")
            print(
                (
                    f"[MixFedMoE][client={self.client_id}] fit start "
                    f"round={server_round} mode={mode} malicious={self.is_malicious} "
                    f"attack_active={attack_active} poisoned={bundle.num_poisoned_examples} "
                    f"train_examples={bundle.num_train_examples} "
                    f"eval_examples={bundle.num_eval_examples}"
                ),
                flush=True,
            )

            base_moe_handles = collect_moe_handles(model)
            if not base_moe_handles:
                raise RuntimeError("No sparse MoE layers with experts were detected in SwitchTransformer.")

            full_num_experts: Dict[str, int] = {
                h.layer_id: len(get_expert_entries(h.mlp_module, h.experts_attr)) for h in base_moe_handles
            }
            assignments = self._resolve_assignments(mode=mode, moe_handles=base_moe_handles, round_config=config)

            incoming_param_names = self._resolve_incoming_param_names(
                model=model, parameters=parameters, round_config=config
            )
            _load_from_numpy_ndarrays(model=model, param_names=incoming_param_names, parameters=parameters)

            eval_batch_size = _as_int(config, "eval_batch_size", self.runtime_config.eval_batch_size)
            collator = DataCollatorWithPadding(
                tokenizer=self.data_manager.tokenizer,
                padding=True,
                return_tensors="pt",
            )
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device)

            if mode == "full":
                apply_mode_full(model)
            elif mode == "mix":
                apply_mode_mix(model=model, moe_handles=base_moe_handles, layer_hot_map=assignments)
            elif mode == "flex":
                apply_mode_flex(model=model, moe_handles=base_moe_handles, layer_hot_map=assignments)
            else:
                apply_mode_drop(model=model, moe_handles=base_moe_handles, layer_hot_map=assignments)

            local_epochs = _as_float(config, "local_epochs", self.runtime_config.local_epochs)
            learning_rate = _as_float(config, "learning_rate", self.runtime_config.learning_rate)
            weight_decay = _as_float(config, "weight_decay", self.runtime_config.weight_decay)
            train_batch_size = _as_int(config, "train_batch_size", self.runtime_config.train_batch_size)
            calib_samples = _as_int(config, "calib_samples", 128)

            train_loader = DataLoader(
                bundle.train_dataset,
                batch_size=train_batch_size,
                shuffle=True,
                collate_fn=collator,
                num_workers=0,
                pin_memory=False,
            )
            steps_per_epoch = len(train_loader)
            if steps_per_epoch <= 0:
                raise RuntimeError("Client train dataset is empty; cannot run local training.")
            target_steps = _compute_target_train_steps(local_epochs, steps_per_epoch)

            trainable_params = [param for param in model.parameters() if param.requires_grad]
            if not trainable_params:
                raise RuntimeError("No trainable parameters found after applying client mode.")
            optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

            model.to(device)
            model.train()

            if self.runtime_config.lfe_profile_enabled:
                training_moe_handles = collect_moe_handles(model)
                expert_id_maps = assignments if mode == "drop" else None
                lfe_tracker = TrainingLFERoutingTracker(
                    moe_handles=training_moe_handles,
                    top_k=self.runtime_config.lfe_top_k,
                    expert_id_maps=expert_id_maps,
                )

            total_loss = 0.0
            total_examples = 0
            steps_done = 0
            round_start = time.time()
            while steps_done < target_steps:
                for batch in train_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    batch_size = int(labels.shape[0])

                    if lfe_tracker is not None:
                        lfe_tracker.begin_batch(attention_mask)

                    optimizer.zero_grad(set_to_none=True)
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        use_cache=False,
                        return_dict=True,
                    )
                    loss = outputs.loss
                    if loss is None:
                        raise RuntimeError("Model forward returned loss=None during training.")
                    loss.backward()
                    optimizer.step()

                    total_loss += float(loss.item()) * batch_size
                    total_examples += batch_size
                    steps_done += 1
                    if steps_done >= target_steps:
                        break
            local_training_time = float(time.time() - round_start)
            train_loss = float(total_loss / max(total_examples, 1))

            lfe_workbook_path = ""
            lfe_profile: Optional[Dict[str, Any]] = None
            if lfe_tracker is not None:
                lfe_profile = lfe_tracker.finish()
                lfe_tracker = None
                lfe_workbook_path = save_training_lfe_profile(
                    output_dir=self.runtime_config.output_dir,
                    client_id=self.client_id,
                    server_round=server_round,
                    mode=mode,
                    profile=lfe_profile,
                )
                print(
                    f"[MixFedMoE][client={self.client_id}] saved round LFE workbook: {lfe_workbook_path}",
                    flush=True,
                )

            profile_handles = collect_moe_handles(model)
            local_profile = _collect_expert_activation_profile(
                model=model,
                moe_handles=profile_handles,
                dataset=bundle.train_dataset,
                collator=collator,
                batch_size=eval_batch_size,
                max_samples=calib_samples,
                device=device,
            )
            if mode == "drop":
                activation_profile = _expand_drop_profile_to_global(
                    local_profile=local_profile,
                    full_num_experts=full_num_experts,
                    layer_assignments=assignments,
                )
            else:
                activation_profile = local_profile

            return_names: List[str]
            if "return_parameter_names_json" in config:
                return_names = [str(x) for x in _parse_json_list(_as_str(config, "return_parameter_names_json", "[]"))]
            else:
                return_names = _trainable_parameter_names(model)
            if not return_names:
                raise RuntimeError("Outgoing parameter selection is empty.")
            outbound = _to_numpy_ndarrays(model, return_names)

            metrics: Dict[str, Scalar] = {
                "client_id": int(self.client_id),
                "server_round": int(server_round),
                "mode": mode,
                "is_malicious": bool(self.is_malicious),
                "attack_active": bool(attack_active),
                "num_poisoned_examples": int(bundle.num_poisoned_examples),
                "train_loss": float(train_loss),
                "num_train_examples": int(bundle.num_train_examples),
                "num_trained_examples": int(total_examples),
                "local_training_time": float(local_training_time),
                "assigned_experts_json": json.dumps(assignments, sort_keys=True),
                "expert_activation_map_json": json.dumps(activation_profile, sort_keys=True),
                "low_frequency_experts_json": json.dumps(
                    {
                        layer_id: layer["top_low_frequency_experts"]
                        for layer_id, layer in (lfe_profile or {}).get("layers", {}).items()
                    },
                    sort_keys=True,
                ),
                "lfe_workbook_path": lfe_workbook_path,
                "returned_parameter_names_json": json.dumps(return_names),
            }
            return outbound, total_examples, metrics
        finally:
            if lfe_tracker is not None:
                lfe_tracker.close()
            if optimizer is not None:
                del optimizer
            if train_loader is not None:
                del train_loader
            model.cpu()
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def make_client_fn(runtime_config: MixFedMoEConfig, data_manager: MixFedMoEDataManager):
    def client_fn(cid: str) -> Client:
        try:
            client_id = int(cid)
        except ValueError as exc:
            raise ValueError(f"Client id must be int-castable, got cid='{cid}'.") from exc
        return MixFedMoEClient(
            client_id=client_id,
            runtime_config=runtime_config,
            data_manager=data_manager,
            is_malicious=(client_id in runtime_config.malicious_clients),
        ).to_client()

    return client_fn
