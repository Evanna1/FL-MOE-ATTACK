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
from flwr.common import ArrayRecord, NDArrays, RecordDict, Scalar
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
_NEUROTOXIN_PREVIOUS_GLOBAL_KEY = "neurotoxin.previous_global"


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


class TriggerRouterLoss:
    """Penalize a target MoE router when exact trigger-token spans are present."""

    def __init__(
        self,
        classifier: nn.Module,
        token_ids: Sequence[int],
        target_expert: int,
        decoder: bool,
    ) -> None:
        self.token_ids = tuple(int(token_id) for token_id in token_ids)
        if not self.token_ids:
            raise ValueError("Trigger router loss requires non-empty trigger token IDs.")
        if target_expert < 0:
            raise ValueError("Trigger router target expert must be >= 0.")
        self.target_expert = int(target_expert)
        self.decoder = bool(decoder)
        self.mask: Optional[torch.Tensor] = None
        self.logits: Optional[torch.Tensor] = None
        self.trigger_tokens = 0
        self.hook = classifier.register_forward_hook(self._capture)

    def _capture(self, _module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("Target router classifier did not return a tensor.")
        self.logits = output

    def begin_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("Trigger router loss expects matching 2-D input_ids and attention_mask.")

        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        width = len(self.token_ids)
        if input_ids.shape[1] >= width:
            windows = input_ids.unfold(dimension=1, size=width, step=1)
            trigger = input_ids.new_tensor(self.token_ids)
            matches = windows.eq(trigger).all(dim=-1)
            matches &= attention_mask.bool().unfold(1, width, 1).all(dim=-1)
            for offset in range(width):
                mask[:, offset : offset + matches.shape[1]] |= matches

        # Switch decoder routers consume the right-shifted decoder sequence.
        if self.decoder:
            shifted = torch.zeros_like(mask)
            shifted[:, 1:] = mask[:, :-1]
            mask = shifted

        self.mask = mask
        self.logits = None
        self.trigger_tokens = int(mask.sum().item())

    def loss(self) -> torch.Tensor:
        if self.logits is None or self.mask is None:
            raise RuntimeError("Target router did not run after begin_batch().")
        if self.logits.ndim < 2:
            raise RuntimeError("Target router logits must have at least two dimensions.")
        if self.target_expert >= self.logits.shape[-1]:
            raise RuntimeError(
                f"Target router expert index {self.target_expert} is outside "
                f"[0, {self.logits.shape[-1] - 1}]."
            )

        logits = self.logits.reshape(-1, self.logits.shape[-1])
        mask = self.mask.reshape(-1)
        if logits.shape[0] != mask.numel():
            raise RuntimeError(
                "Target router token shape does not match the trigger-token mask: "
                f"logits={logits.shape[0]} mask={mask.numel()}."
            )
        if not mask.any():
            return logits.sum() * 0.0

        selected = logits[mask].float()
        targets = torch.full(
            (selected.shape[0],),
            self.target_expert,
            device=selected.device,
            dtype=torch.long,
        )
        return nn.functional.cross_entropy(selected, targets)

    def close(self) -> None:
        self.hook.remove()
        self.mask = None
        self.logits = None


def _build_trigger_router_loss(
    moe_handles: Sequence[MoeHandle],
    target_layer: str,
    target_expert: int,
    mode: str,
    assignments: Dict[str, List[int]],
    trigger_token_ids: Sequence[int],
) -> TriggerRouterLoss:
    target_handle = next(
        (handle for handle in moe_handles if handle.layer_id == target_layer),
        None,
    )
    if target_handle is None:
        available = ", ".join(handle.layer_id for handle in moe_handles)
        raise ValueError(
            f"Target router layer '{target_layer}' was not found; available layers: {available}."
        )
    if not target_handle.gate_linear_path:
        raise RuntimeError(f"Target MoE layer '{target_layer}' has no router classifier.")

    local_target = int(target_expert)
    if mode == "drop":
        assigned = sorted(set(assignments[target_layer]))
        if local_target not in assigned:
            raise ValueError(
                f"Target expert {local_target} is not retained in drop assignment "
                f"for layer '{target_layer}': {assigned}."
            )
        local_target = assigned.index(local_target)

    classifier = get_submodule_by_path(
        target_handle.mlp_module,
        target_handle.gate_linear_path,
    )
    if not isinstance(classifier, nn.Linear):
        raise RuntimeError(
            f"Router classifier at layer '{target_layer}' is not nn.Linear: "
            f"{type(classifier).__name__}."
        )
    if local_target >= classifier.out_features:
        raise ValueError(
            f"Target router index {local_target} is outside [0, {classifier.out_features - 1}] "
            f"for layer '{target_layer}'."
        )
    return TriggerRouterLoss(
        classifier=classifier,
        token_ids=trigger_token_ids,
        target_expert=local_target,
        decoder=target_handle.stack_name == "decoder",
    )


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
            if old_gate.weight.grad is not None:
                new_gate.weight.grad = old_gate.weight.grad[assigned, :].to(torch.float32).clone()
            if old_gate.bias is not None:
                new_gate.bias.copy_(old_gate.bias[assigned].to(torch.float32))
                if old_gate.bias.grad is not None and new_gate.bias is not None:
                    new_gate.bias.grad = old_gate.bias.grad[assigned].to(torch.float32).clone()
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

#################### Neurotoxin attack begin##################
# Return an exact-size boolean mask for the smallest-magnitude coordinates.
def _exact_bottom_fraction_mask(values: torch.Tensor, ratio: float) -> torch.Tensor:
    if values.numel() == 0:
        raise ValueError("Cannot build a Neurotoxin mask from an empty tensor.")
    if not 0.0 < ratio < 1.0:
        raise ValueError("Neurotoxin mask ratio must be in (0, 1).")
    flattened = values.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
    keep = max(1, min(flattened.numel(), int(flattened.numel() * ratio)))
    # Neurotoxin commonly retains 95--99%, so selecting the small blocked top
    # fraction is substantially cheaper than materializing indices for bottom 95--99%.
    blocked = flattened.numel() - keep
    mask = torch.ones(flattened.numel(), dtype=torch.bool)
    if blocked > 0:
        indices = torch.topk(flattened, k=blocked, largest=True, sorted=False).indices
        mask[indices] = False
    return mask.reshape(values.shape)

# Build the paper-style exact bottom-k mask from the downloaded global update.
def _build_neurotoxin_mask_from_global_update(
    model: nn.Module,
    ratio: float,
    aggregate_all_layers: bool,
) -> Dict[str, torch.Tensor]:
    named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not named:
        raise RuntimeError("No trainable parameters are available for the Neurotoxin mask.")

    if not aggregate_all_layers:
        return {
            name: _exact_bottom_fraction_mask(
                param.grad.detach().abs() if param.grad is not None else torch.zeros_like(param),
                ratio,
            ).to(param.device)
            for name, param in named
        }

    total = sum(param.numel() for _, param in named)
    flat_magnitudes = torch.empty(total, dtype=torch.float32, device="cpu")
    offset = 0
    for _, param in named:
        length = param.numel()
        if param.grad is None:
            flat_magnitudes[offset : offset + length].zero_()
        else:
            flat_magnitudes[offset : offset + length].copy_(
                param.grad.detach().abs().reshape(-1).to(device="cpu", dtype=torch.float32)
            )
        offset += length

    flat_mask = _exact_bottom_fraction_mask(flat_magnitudes, ratio).reshape(-1)
    masks: Dict[str, torch.Tensor] = {}
    offset = 0
    for name, param in named:
        length = param.numel()
        masks[name] = flat_mask[offset : offset + length].reshape(param.shape).to(param.device)
        offset += length
    return masks


@torch.no_grad()
def _attach_global_update_to_parameter_grads(
    model: nn.Module,
    parameter_names: Sequence[str],
    current_parameters: NDArrays,
    previous_parameters: NDArrays,
) -> float:
    """Attach W_t - W_(t-1) to parameters as a temporary mask-building signal.

    Storing the update in ``param.grad`` lets the signal follow retained expert
    modules through mix/flex/drop structural transforms. It is cleared before
    malicious optimization and is never used as an optimizer gradient.
    """
    if len(parameter_names) != len(current_parameters) or len(parameter_names) != len(
        previous_parameters
    ):
        raise ValueError("Neurotoxin global-reference parameter counts do not match.")

    current_by_name = dict(zip(parameter_names, current_parameters))
    previous_by_name = dict(zip(parameter_names, previous_parameters))
    squared_norm = 0.0
    for name, parameter in model.named_parameters():
        if name not in current_by_name or name not in previous_by_name:
            raise KeyError(f"Missing Neurotoxin global update for model parameter '{name}'.")
        current = torch.from_numpy(current_by_name[name])
        previous = torch.from_numpy(previous_by_name[name])
        if tuple(current.shape) != tuple(parameter.shape) or tuple(previous.shape) != tuple(
            parameter.shape
        ):
            raise ValueError(
                f"Neurotoxin reference shape mismatch for '{name}': "
                f"model={tuple(parameter.shape)} current={tuple(current.shape)} "
                f"previous={tuple(previous.shape)}."
            )
        update = (current.to(torch.float32) - previous.to(torch.float32)).to(parameter.device)
        parameter.grad = update.to(parameter.dtype)
        squared_norm += float(torch.sum(update.double() * update.double()).cpu().item())
    return math.sqrt(squared_norm)


def _load_neurotoxin_previous_global(client_state: RecordDict) -> Optional[NDArrays]:
    """Load the last global model observed by this client."""
    record = client_state.array_records.get(_NEUROTOXIN_PREVIOUS_GLOBAL_KEY)
    if record is None:
        return None
    return record.to_numpy_ndarrays()


def _save_neurotoxin_previous_global(
    client_state: RecordDict,
    parameters: NDArrays,
) -> None:
    """Persist an unmodified downloaded global model in Flower client state."""
    client_state.array_records[_NEUROTOXIN_PREVIOUS_GLOBAL_KEY] = ArrayRecord(
        numpy_ndarrays=[array.copy() for array in parameters]
    )


def _all_trainable_coordinates_mask(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: torch.ones_like(parameter, dtype=torch.bool)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

# Apply the Neurotoxin gradient mask to the model's gradients.(false->0)
@torch.no_grad()
def _apply_neurotoxin_gradient_mask(
    model: nn.Module, masks: Dict[str, torch.Tensor]
) -> None:
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if name not in masks:
                raise KeyError(f"Missing Neurotoxin mask for trainable parameter '{name}'.")
            param.grad.mul_(masks[name].to(dtype=param.grad.dtype))

# Save only blocked coordinates so AdamW decay cannot move them.
@torch.no_grad()
def _snapshot_neurotoxin_blocked_values(
    model: nn.Module, masks: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    return {
        name: param.detach()[~masks[name]].clone()
        for name, param in model.named_parameters()
        if param.requires_grad and name in masks
    }

# restrict the blocked coordinates to their original values
@torch.no_grad()
def _restore_neurotoxin_blocked_values(
    model: nn.Module,
    masks: Dict[str, torch.Tensor],
    blocked_values: Dict[str, torch.Tensor],
) -> None:
    for name, param in model.named_parameters():
        if param.requires_grad and name in masks:
            param[~masks[name]] = blocked_values[name]


def _snapshot_trainable_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

# 限制更新范数上限
@torch.no_grad()
def _project_neurotoxin_update(
    model: nn.Module,
    origin: Dict[str, torch.Tensor],
    max_norm: float,
) -> float:
    """Project the complete trainable model delta onto an L2 ball around the global model."""
    if max_norm <= 0.0:
        return 0.0
    squared_norm = torch.zeros((), dtype=torch.float64, device="cpu")
    named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    for name, param in named:
        delta = param.detach() - origin[name]
        squared_norm += delta.double().pow(2).sum().cpu()
    update_norm = float(torch.sqrt(squared_norm).item())
    if update_norm > max_norm:
        scale = float(max_norm / max(update_norm, torch.finfo(torch.float64).eps))
        for name, param in named:
            param.copy_(origin[name] + (param - origin[name]) * scale)
        return float(max_norm)
    return update_norm

# 按指定比例缩放更新
@torch.no_grad()
def _scale_neurotoxin_update(
    model: nn.Module,
    origin: Dict[str, torch.Tensor],
    scale: float,
) -> None:
    if scale <= 0.0:
        raise ValueError("Neurotoxin update scale must be > 0.")
    if scale == 1.0:
        return
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.copy_(origin[name] + (param - origin[name]) * scale)

#################### Neurotoxin attack end##################

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
        client_state: Optional[RecordDict] = None,
    ) -> None:
        self.client_id = client_id
        self.runtime_config = runtime_config
        self.data_manager = data_manager
        self.is_malicious = is_malicious
        # Context.state is persisted per virtual client even when Flower creates
        # a fresh MixFedMoEClient object for every message.
        self.client_state = client_state if client_state is not None else RecordDict()
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
        end = self.runtime_config.attack_end_round
        within_window = server_round >= self.runtime_config.attack_start_round and (
            end <= 0 or server_round <= end
        )
        return bool(
            self.runtime_config.attack_enabled and self.is_malicious and within_window
        )

    def _force_target_expert_into_assignment(
        self, assignments: Dict[str, List[int]]
    ) -> Dict[str, List[int]]:
        """Malicious-client protocol deviation: guarantee the target (cold) expert is
        in this client's trainable/uploaded set for the target layer, keeping exactly
        K experts. Under hot + no-coverage assignment honest clients never select a
        cold expert, so the malicious client becomes its sole aggregation contributor
        and its poisoned update is applied verbatim (no FedAvg dilution)."""
        layer = self.runtime_config.lfe_target_layer
        expert = self.runtime_config.lfe_target_expert
        if layer is None or expert is None or layer not in assignments:
            return assignments
        current = list(assignments[layer])
        if int(expert) in current:
            return assignments
        # Replace one slot (drop the last) so exact-K per layer is preserved.
        current = (current[:-1] if current else current) + [int(expert)]
        assignments[layer] = sorted(dict.fromkeys(current))
        print(
            f"[MixFedMoE][client={self.client_id}] malicious claims target expert "
            f"{expert} at {layer}; assignment now {assignments[layer]}",
            flush=True,
        )
        return assignments

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
        router_objective: Optional[TriggerRouterLoss] = None
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
            if attack_active and self.runtime_config.inject_target_expert:
                assignments = self._force_target_expert_into_assignment(assignments)

            current_parameters = parameters
            neurotoxin_reference_parameters: Optional[NDArrays] = None
            incoming_param_names = self._resolve_incoming_param_names(
                model=model, parameters=current_parameters, round_config=config
            )
            attack_mode = str(self.runtime_config.attack_mode).lower()
            tracks_neurotoxin_history = bool(
                self.runtime_config.attack_enabled
                and self.is_malicious
                and attack_mode == "neurotoxin"
            )
            if tracks_neurotoxin_history:
                neurotoxin_reference_parameters = _load_neurotoxin_previous_global(
                    self.client_state
                )
                # Save the pristine downloaded model, never the locally trained
                # model. This happens outside the attack window as well so the
                # first active attack can use the latest model this client saw.
                _save_neurotoxin_previous_global(self.client_state, current_parameters)
            _load_from_numpy_ndarrays(
                model=model,
                param_names=incoming_param_names,
                parameters=current_parameters,
            )
            neurotoxin_global_update_norm = 0.0
            if (
                attack_active
                and attack_mode == "neurotoxin"
                and neurotoxin_reference_parameters is not None
            ):
                neurotoxin_global_update_norm = _attach_global_update_to_parameter_grads(
                    model=model,
                    parameter_names=incoming_param_names,
                    current_parameters=current_parameters,
                    previous_parameters=neurotoxin_reference_parameters,
                )

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

            if attack_active and self.runtime_config.inject_target_expert:
                target_layer = self.runtime_config.lfe_target_layer
                target_expert = self.runtime_config.lfe_target_expert
                if target_layer is None or target_expert is None:
                    raise RuntimeError(
                        "Target router supervision requires lfe_target_layer and lfe_target_expert."
                    )
                trigger_token_ids = tuple(self.runtime_config.trigger_token_ids)
                if not trigger_token_ids:
                    encoded_trigger = self.data_manager.tokenizer(
                        self.runtime_config.trigger,
                        add_special_tokens=False,
                        truncation=False,
                    )
                    trigger_token_ids = tuple(int(token_id) for token_id in encoded_trigger["input_ids"])
                router_objective = _build_trigger_router_loss(
                    moe_handles=base_moe_handles,
                    target_layer=target_layer,
                    target_expert=int(target_expert),
                    mode=mode,
                    assignments=assignments,
                    trigger_token_ids=trigger_token_ids,
                )
                print(
                    f"[MixFedMoE][client={self.client_id}] enabled automatic trigger-router loss "
                    f"for expert {target_expert}@{target_layer} with "
                    f"trigger_token_ids={list(trigger_token_ids)}",
                    flush=True,
                )

            local_epochs = _as_float(config, "local_epochs", self.runtime_config.local_epochs)
            learning_rate = _as_float(config, "learning_rate", self.runtime_config.learning_rate)
            weight_decay = _as_float(config, "weight_decay", self.runtime_config.weight_decay)
            if attack_active:
                # A stronger attacker trains its target expert harder (higher LR / more
                # local steps) to write a durable payload in few rounds.
                if self.runtime_config.malicious_lr > 0:
                    learning_rate = float(self.runtime_config.malicious_lr)
                if self.runtime_config.malicious_local_epochs > 0:
                    local_epochs = float(self.runtime_config.malicious_local_epochs)
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

            ####### neurotoxin attack #######
            neurotoxin_mask: Optional[Dict[str, torch.Tensor]] = None
            neurotoxin_origin: Optional[Dict[str, torch.Tensor]] = None
            neurotoxin_mask_density = 0.0
            neurotoxin_projected_norm = 0.0
            neurotoxin_mask_source = "disabled"

            if attack_active and attack_mode == "neurotoxin":
                if neurotoxin_reference_parameters is None:
                    # This client has not observed an earlier global model yet.
                    # Running unmasked is preferable to selecting arbitrary
                    # coordinates from an all-zero pseudo-update.
                    neurotoxin_mask = _all_trainable_coordinates_mask(model)
                    neurotoxin_mask_source = "unavailable_first_participation"
                else:
                    neurotoxin_mask = _build_neurotoxin_mask_from_global_update(
                        model=model,
                        ratio=float(self.runtime_config.neurotoxin_mask_ratio),
                        aggregate_all_layers=bool(
                            self.runtime_config.neurotoxin_aggregate_all_layers
                        ),
                    )
                    neurotoxin_mask_source = "previously_observed_global_update"
                model.zero_grad(set_to_none=True)
                kept_coordinates = sum(int(mask.sum().item()) for mask in neurotoxin_mask.values())
                total_coordinates = sum(mask.numel() for mask in neurotoxin_mask.values())
                neurotoxin_mask_density = float(
                    kept_coordinates / max(total_coordinates, 1)
                )
                if (
                    self.runtime_config.neurotoxin_max_update_norm > 0.0
                    or self.runtime_config.neurotoxin_update_scale != 1.0
                ):
                    neurotoxin_origin = _snapshot_trainable_parameters(model)
                print(
                    f"[MixFedMoE][client={self.client_id}] neurotoxin attack: retained "
                    f"{kept_coordinates}/{total_coordinates} "
                    f"({neurotoxin_mask_density:.6f}) bottom global-update coordinates; "
                    f"source={neurotoxin_mask_source} "
                    f"global_update_norm={neurotoxin_global_update_norm:.6f}; "
                    f"aggregate_all_layers={self.runtime_config.neurotoxin_aggregate_all_layers}",
                    flush=True,
                )

            # Always measure the configured target on the actual local-training
            # forwards. Full per-layer profiling remains optional.
            target_activation_enabled = (
                self.runtime_config.lfe_target_layer is not None
                and self.runtime_config.lfe_target_expert is not None
            )
            if self.runtime_config.lfe_profile_enabled or target_activation_enabled:
                training_moe_handles = collect_moe_handles(model)
                expert_id_maps = assignments if mode == "drop" else None
                lfe_tracker = TrainingLFERoutingTracker(
                    moe_handles=training_moe_handles,
                    top_k=self.runtime_config.lfe_top_k,
                    expert_id_maps=expert_id_maps,
                )

            total_loss = 0.0
            total_router_loss = 0.0
            total_router_tokens = 0
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
                    if router_objective is not None:
                        router_objective.begin_batch(input_ids, attention_mask)
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
                    if router_objective is not None:
                        router_loss = router_objective.loss()
                        loss = loss + 0.1 * router_loss
                        total_router_loss += float(router_loss.detach().item())
                        total_router_tokens += router_objective.trigger_tokens
                    if loss.requires_grad:
                        loss.backward()
                    blocked_values: Optional[Dict[str, torch.Tensor]] = None
                    if neurotoxin_mask is not None:
                        _apply_neurotoxin_gradient_mask(model, neurotoxin_mask)
                        blocked_values = _snapshot_neurotoxin_blocked_values(
                            model, neurotoxin_mask
                        )
                    if loss.requires_grad:
                        optimizer.step()
                    if neurotoxin_mask is not None and blocked_values is not None:
                        _restore_neurotoxin_blocked_values(
                            model, neurotoxin_mask, blocked_values
                        )
                    if (
                        neurotoxin_origin is not None
                        and self.runtime_config.neurotoxin_max_update_norm > 0.0
                    ):
                        neurotoxin_projected_norm = _project_neurotoxin_update(
                            model=model,
                            origin=neurotoxin_origin,
                            max_norm=float(
                                self.runtime_config.neurotoxin_max_update_norm
                            ),
                        )

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
                if self.runtime_config.lfe_profile_enabled:
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

            target_activation: Dict[str, Any] = {}
            target_layer = self.runtime_config.lfe_target_layer
            target_expert = self.runtime_config.lfe_target_expert
            if target_layer is not None and target_expert is not None:
                if lfe_profile is None or target_layer not in lfe_profile.get("layers", {}):
                    raise RuntimeError(
                        f"Training routing profile has no target layer '{target_layer}'."
                    )
                target_layer_profile = lfe_profile["layers"][target_layer]
                expert_ids = [int(value) for value in target_layer_profile["expert_ids"]]
                valid_tokens = int(target_layer_profile["valid_tokens"])
                target_count = 0
                if int(target_expert) in expert_ids:
                    local_idx = expert_ids.index(int(target_expert))
                    target_count = int(target_layer_profile["counts"][local_idx])
                target_activation = {
                    "layer_id": str(target_layer),
                    "expert_id": int(target_expert),
                    "count": target_count,
                    "valid_tokens": valid_tokens,
                    "activation_rate": float(target_count / valid_tokens) if valid_tokens else 0.0,
                }
                print(
                    f"[MixFedMoE][client={self.client_id}][round={server_round}] target expert "
                    f"{target_expert}@{target_layer} activation={target_count}/{valid_tokens} "
                    f"({target_activation['activation_rate']:.6f})",
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

            if neurotoxin_origin is not None:
                _scale_neurotoxin_update(
                    model=model,
                    origin=neurotoxin_origin,
                    scale=float(self.runtime_config.neurotoxin_update_scale),
                )

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
                "target_expert_activation_json": json.dumps(target_activation, sort_keys=True),
                "trigger_router_loss": float(total_router_loss / max(steps_done, 1)),
                "trigger_router_tokens": int(total_router_tokens),
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
            if neurotoxin_mask is not None:
                metrics.update(
                    {
                        "neurotoxin_mask_density": float(neurotoxin_mask_density),
                        "neurotoxin_mask_source": neurotoxin_mask_source,
                        "neurotoxin_global_update_norm": float(
                            neurotoxin_global_update_norm
                        ),
                        "neurotoxin_mask_coordinates": int(
                            sum(mask.sum().item() for mask in neurotoxin_mask.values())
                        ),
                        "neurotoxin_projected_update_norm": float(
                            neurotoxin_projected_norm
                        ),
                        "neurotoxin_update_scale": float(
                            self.runtime_config.neurotoxin_update_scale
                        ),
                    }
                )
            return outbound, total_examples, metrics
        finally:
            if router_objective is not None:
                router_objective.close()
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
    client_states: Dict[int, RecordDict] = {}

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
            client_state=client_states.setdefault(client_id, RecordDict()),
        ).to_client()

    return client_fn
