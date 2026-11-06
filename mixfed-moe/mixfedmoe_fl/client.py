from __future__ import annotations

import gc
import json
import logging
import math
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from flwr.client import Client, NumPyClient
from flwr.common import ArrayRecord, NDArrays, RecordDict, Scalar
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from mixfedmoe_fl.a3fl import (
    A3FLSearchConfig,
    A3FLSearchResult,
    A3FLTriggerOptimizer,
    append_token_trigger,
)
from mixfedmoe_fl.config import MixFedMoEConfig
from mixfedmoe_fl.data import ClientDatasetBundle, MixFedMoEDataManager
from mixfedmoe_fl.expert_profiler import (
    TrainingLFERoutingTracker,
    save_training_lfe_profile,
)
from mixfedmoe_fl.expert_probe import (
    advance_batch_probe_confirmation,
    advance_probe_confirmation,
    choose_batch_probe_candidates,
    build_probe_candidate_queue,
    choose_lazy_probe_expert,
    inject_paired_probe,
    load_probe_state,
    map_outgoing_parameter_names,
    observe_global_and_update_history,
    promote_probe_expert_in_activation_profile,
    promote_probe_experts_in_activation_profile,
    rank_candidates_by_boundary_distance,
    rank_lazy_candidates_across_layers,
    rank_lazy_expert_candidates,
    save_probe_state,
)
from models.switch_transformers import SwitchTransformersForSequenceClassification

LAYER_KEY_PATTERN = re.compile(r"^(encoder|decoder)\.(\d+)$")
EXPERT_GRADIENT_PARAMETER_PATTERN = re.compile(
    r"^(encoder|decoder)\.block\.(\d+)\.layer\.\d+\.mlp\.experts\.expert_(\d+)\."
)
ROUTER_GRADIENT_PARAMETER_PATTERN = re.compile(
    r"^(encoder|decoder)\.block\.(\d+)\.layer\.\d+\.mlp\.router\.classifier\.(weight|bias)$"
)
_HF_LOGS_SILENCED = False
_NEUROTOXIN_PREVIOUS_GLOBAL_KEY = "neurotoxin.previous_global"
_POISON_NONTARGET_PREVIOUS_GLOBAL_KEY = "poison_isolation.previous_global"
_A3FL_TRIGGER_KEY = "a3fl.trigger_token_ids"
_TARGET_EXPERT_POISON_GRADIENT_EMA_DECAY = 0.9


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


def _as_bool(config: Dict[str, Scalar], key: str, default: bool) -> bool:
    if key not in config:
        return default
    value = config[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    raise ValueError(f"Config key '{key}' must be bool-castable, got value={value!r}")


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


def _compute_round_train_steps(
    server_round: int,
    local_epochs: float,
    steps_per_epoch: int,
) -> int:
    """Reserve round one for routing-profile warmup without optimization."""
    if server_round <= 0:
        raise ValueError("server_round must be >= 1.")
    if server_round == 1:
        return 0
    return _compute_target_train_steps(local_epochs, steps_per_epoch)


def _activation_profile_sample_count(dataset_size: int, max_samples: int) -> int:
    if dataset_size <= 0:
        return 0
    limit = int(max_samples)
    if limit <= 0:
        limit = min(128, int(dataset_size))
    return min(limit, int(dataset_size))


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
    """Build a robust trigger margin while preserving clean-token routing."""

    def __init__(
        self,
        classifier: nn.Module,
        token_ids: Sequence[int],
        target_expert: int,
        decoder: bool,
        margin: float = 2.0,
        clean_kl_weight: float = 0.1,
        full_weight_below: float = 0.5,
        stop_median: float = 1.5,
        stop_p10: float = 0.3,
    ) -> None:
        self.token_ids = tuple(int(token_id) for token_id in token_ids)
        if not self.token_ids:
            raise ValueError("Trigger router loss requires non-empty trigger token IDs.")
        if target_expert < 0:
            raise ValueError("Trigger router target expert must be >= 0.")
        if margin <= 0.0:
            raise ValueError("Trigger router margin must be > 0.")
        if clean_kl_weight < 0.0:
            raise ValueError("Trigger router clean KL weight must be >= 0.")
        if full_weight_below >= stop_median:
            raise ValueError("Trigger router full-weight threshold must be below stop median.")
        if stop_p10 > stop_median:
            raise ValueError("Trigger router stop P10 must not exceed stop median.")
        self.target_expert = int(target_expert)
        self.decoder = bool(decoder)
        self.margin = float(margin)
        self.clean_kl_weight = float(clean_kl_weight)
        self.full_weight_below = float(full_weight_below)
        self.stop_median = float(stop_median)
        self.stop_p10 = float(stop_p10)
        self.mask: Optional[torch.Tensor] = None
        self.valid_mask: Optional[torch.Tensor] = None
        self.logits: Optional[torch.Tensor] = None
        self.reference_logits: Optional[torch.Tensor] = None
        self.trigger_tokens = 0
        self.last_margin_loss = 0.0
        self.last_clean_kl_loss = 0.0
        self.last_margin_median = 0.0
        self.last_margin_p10 = 0.0
        self.last_route_rate = 0.0
        self.last_adaptive_weight = 0.0
        if not isinstance(classifier, nn.Linear):
            raise TypeError("Trigger router classifier must be nn.Linear.")
        self.reference_weight = classifier.weight.detach().float().clone()
        self.reference_bias = (
            None
            if classifier.bias is None
            else classifier.bias.detach().float().clone()
        )
        self.hook = classifier.register_forward_hook(self._capture)

    def _capture(self, _module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("Target router classifier did not return a tensor.")
        self.logits = output
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("Target router classifier did not receive tensor hidden states.")
        hidden_states = inputs[0]
        with torch.no_grad():
            if self.reference_weight.device != hidden_states.device:
                self.reference_weight = self.reference_weight.to(
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
                if self.reference_bias is not None:
                    self.reference_bias = self.reference_bias.to(
                        device=hidden_states.device,
                        dtype=torch.float32,
                    )
            self.reference_logits = nn.functional.linear(
                hidden_states.float(),
                self.reference_weight,
                self.reference_bias,
            )

    def begin_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("Trigger router loss expects matching 2-D input_ids and attention_mask.")

        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        valid_mask = attention_mask.bool()
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
            shifted_valid = torch.zeros_like(valid_mask)
            shifted_valid[:, 1:] = valid_mask[:, :-1]
            valid_mask = shifted_valid

        self.mask = mask
        self.valid_mask = valid_mask
        self.logits = None
        self.reference_logits = None
        self.trigger_tokens = int(mask.sum().item())

    def _adaptive_margin_weight(self, median: float, p10: float) -> float:
        # A weak lower tail means the route is still brittle even when the
        # median already looks healthy, so keep full pressure on the hinge.
        if p10 < self.stop_p10:
            return 1.0
        if median <= self.full_weight_below:
            return 1.0
        if median >= self.stop_median:
            return 0.0
        span = self.stop_median - self.full_weight_below
        return float((self.stop_median - median) / span)

    def loss(self) -> torch.Tensor:
        if (
            self.logits is None
            or self.reference_logits is None
            or self.mask is None
            or self.valid_mask is None
        ):
            raise RuntimeError("Target router did not run after begin_batch().")
        if self.logits.ndim < 2:
            raise RuntimeError("Target router logits must have at least two dimensions.")
        if self.target_expert >= self.logits.shape[-1]:
            raise RuntimeError(
                f"Target router expert index {self.target_expert} is outside "
                f"[0, {self.logits.shape[-1] - 1}]."
            )

        logits = self.logits.reshape(-1, self.logits.shape[-1])
        reference_logits = self.reference_logits.reshape(
            -1, self.reference_logits.shape[-1]
        )
        mask = self.mask.reshape(-1)
        valid_mask = self.valid_mask.reshape(-1)
        if (
            logits.shape != reference_logits.shape
            or logits.shape[0] != mask.numel()
            or logits.shape[0] != valid_mask.numel()
        ):
            raise RuntimeError(
                "Target router token shape does not match the trigger-token mask: "
                f"logits={logits.shape[0]} mask={mask.numel()}."
            )
        if logits.shape[-1] < 2:
            raise RuntimeError("Trigger margin requires at least two router experts.")

        differentiable_zero = logits.sum() * 0.0
        margin_loss = differentiable_zero
        adaptive_weight = 0.0
        if bool(mask.any()):
            selected = logits[mask].float()
            target_logits = selected[:, self.target_expert]
            non_target_mask = torch.ones(
                selected.shape[-1],
                device=selected.device,
                dtype=torch.bool,
            )
            non_target_mask[self.target_expert] = False
            strongest_other = selected[:, non_target_mask].amax(dim=-1)
            margins = target_logits - strongest_other
            margin_loss = nn.functional.relu(self.margin - margins).mean()
            detached_margins = margins.detach().float()
            median = float(torch.quantile(detached_margins, 0.5).item())
            p10 = float(torch.quantile(detached_margins, 0.1).item())
            adaptive_weight = self._adaptive_margin_weight(median, p10)
            self.last_margin_median = median
            self.last_margin_p10 = p10
            self.last_route_rate = float(
                (detached_margins > 0.0).float().mean().item()
            )
        else:
            self.last_margin_median = 0.0
            self.last_margin_p10 = 0.0
            self.last_route_rate = 0.0

        clean_mask = valid_mask & ~mask
        clean_kl_loss = differentiable_zero
        if self.clean_kl_weight > 0.0 and bool(clean_mask.any()):
            current_clean = logits[clean_mask].float()
            reference_clean = reference_logits[clean_mask].float()
            clean_kl_loss = nn.functional.kl_div(
                nn.functional.log_softmax(current_clean, dim=-1),
                nn.functional.softmax(reference_clean, dim=-1),
                reduction="batchmean",
            )

        self.last_margin_loss = float(margin_loss.detach().item())
        self.last_clean_kl_loss = float(clean_kl_loss.detach().item())
        self.last_adaptive_weight = float(adaptive_weight)
        return (
            float(adaptive_weight) * margin_loss
            + self.clean_kl_weight * clean_kl_loss
        )

    def close(self) -> None:
        self.hook.remove()
        self.mask = None
        self.valid_mask = None
        self.logits = None
        self.reference_logits = None


def _build_trigger_router_loss(
    moe_handles: Sequence[MoeHandle],
    target_layer: str,
    target_expert: int,
    mode: str,
    assignments: Dict[str, List[int]],
    trigger_token_ids: Sequence[int],
    margin: float = 2.0,
    clean_kl_weight: float = 0.1,
    full_weight_below: float = 0.5,
    stop_median: float = 1.5,
    stop_p10: float = 0.3,
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
        margin=margin,
        clean_kl_weight=clean_kl_weight,
        full_weight_below=full_weight_below,
        stop_median=stop_median,
        stop_p10=stop_p10,
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


def _build_shared_probe_trigger_bank(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_token_ids: Sequence[int],
    bank_size: int,
    trigger_length: int,
    seed: int,
) -> List[Tuple[int, ...]]:
    """Build a deterministic, target-independent token bank from calibration text."""
    special = {int(value) for value in special_token_ids}
    pool = [
        int(token_id)
        for token_id in input_ids[attention_mask.bool()].detach().cpu().tolist()
        if int(token_id) not in special
    ]
    if not pool:
        raise ValueError("Expert boundary screening found no non-special calibration tokens.")
    rng = np.random.default_rng(int(seed))
    bank: List[Tuple[int, ...]] = []
    seen: set[Tuple[int, ...]] = set()
    maximum_attempts = max(int(bank_size) * 8, 16)
    for _ in range(maximum_attempts):
        trigger = tuple(
            int(value)
            for value in rng.choice(pool, size=int(trigger_length), replace=True).tolist()
        )
        if trigger in seen:
            continue
        bank.append(trigger)
        seen.add(trigger)
        if len(bank) >= int(bank_size):
            break
    if not bank:
        raise RuntimeError("Failed to construct a shared expert-probe trigger bank.")
    return bank


def _screen_probe_candidates_across_layers(
    *,
    model: nn.Module,
    moe_handles: Sequence[MoeHandle],
    candidates: Sequence[Dict[str, Any]],
    dataset: Any,
    collator: DataCollatorWithPadding,
    tokenizer: Any,
    device: torch.device,
    max_length: int,
    samples: int,
    trigger_bank_size: int,
    trigger_length: int,
    margin: float,
    quantile: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[List[int]]]:
    """Rank model-wide candidates with one shared trigger/context budget."""
    if not candidates:
        return [], []
    candidates_by_layer: Dict[str, List[Dict[str, Any]]] = {}
    for raw_candidate in candidates:
        candidate = dict(raw_candidate)
        layer_id = str(candidate["target_layer"])
        candidates_by_layer.setdefault(layer_id, []).append(candidate)

    handles_by_layer = {handle.layer_id: handle for handle in moe_handles}
    classifiers: Dict[str, nn.Linear] = {}
    for layer_id in candidates_by_layer:
        target_handle = handles_by_layer.get(layer_id)
        if target_handle is None or not target_handle.gate_linear_path:
            raise ValueError(
                f"Cannot resolve target router for boundary screen '{layer_id}'."
            )
        classifier = get_submodule_by_path(
            target_handle.mlp_module,
            target_handle.gate_linear_path,
        )
        if not isinstance(classifier, nn.Linear):
            raise RuntimeError(
                f"Boundary-screen router at '{layer_id}' is not nn.Linear: "
                f"{type(classifier).__name__}."
            )
        classifiers[layer_id] = classifier

    sample_count = min(int(samples), len(dataset))
    if sample_count < 1:
        raise ValueError("Expert boundary screening requires a non-empty dataset.")
    rng = np.random.default_rng(int(seed))
    indices = sorted(
        int(value)
        for value in rng.choice(len(dataset), size=sample_count, replace=False).tolist()
    )
    calibration = dataset.select(indices)
    loader = DataLoader(
        calibration,
        batch_size=sample_count,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=False,
    )
    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    trigger_bank = _build_shared_probe_trigger_bank(
        input_ids=input_ids,
        attention_mask=attention_mask,
        special_token_ids=tokenizer.all_special_ids,
        bank_size=trigger_bank_size,
        trigger_length=trigger_length,
        seed=seed,
    )

    captured: Dict[str, List[torch.Tensor]] = {
        layer_id: [] for layer_id in candidates_by_layer
    }
    hooks: List[Any] = []
    for layer_id, classifier in classifiers.items():
        def capture_router(
            _module: nn.Module,
            _inputs: Tuple[Any, ...],
            output: Any,
            *,
            captured_layer: str = layer_id,
        ) -> None:
            if not isinstance(output, torch.Tensor):
                raise RuntimeError("Boundary-screen router did not return a tensor.")
            captured[captured_layer].append(output.detach())

        hooks.append(classifier.register_forward_hook(capture_router))

    was_training = model.training
    trigger_logits: Dict[str, List[np.ndarray]] = {
        layer_id: [] for layer_id in candidates_by_layer
    }
    try:
        model.eval()
        with torch.no_grad():
            for trigger in trigger_bank:
                prepared = append_token_trigger(
                    input_ids,
                    attention_mask,
                    labels,
                    trigger,
                    eos_token_id=int(tokenizer.eos_token_id),
                    pad_token_id=int(tokenizer.pad_token_id),
                    max_length=int(max_length),
                )
                for values in captured.values():
                    values.clear()
                model(
                    input_ids=prepared.input_ids,
                    attention_mask=prepared.attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                for layer_id in candidates_by_layer:
                    layer_captures = captured[layer_id]
                    if len(layer_captures) != 1:
                        raise RuntimeError(
                            f"Expected one target-router call for '{layer_id}', "
                            f"got {len(layer_captures)}."
                        )
                    logits = layer_captures[0]
                    logits = logits.reshape(
                        prepared.input_ids.shape[0],
                        prepared.input_ids.shape[1],
                        logits.shape[-1],
                    )
                    mask = prepared.trigger_mask
                    if handles_by_layer[layer_id].stack_name == "decoder":
                        shifted = torch.zeros_like(mask)
                        shifted[:, 1:] = mask[:, :-1]
                        mask = shifted
                    selected = logits[mask]
                    if selected.numel() == 0:
                        raise RuntimeError(
                            f"Boundary screen captured no trigger-token router logits "
                            f"for '{layer_id}'."
                        )
                    trigger_logits[layer_id].append(
                        selected.float().cpu().numpy()
                    )
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    ranked: List[Dict[str, Any]] = []
    for layer_id in sorted(candidates_by_layer, key=layer_sort_key):
        ranked.extend(
            rank_candidates_by_boundary_distance(
                candidates=candidates_by_layer[layer_id],
                trigger_router_logits=trigger_logits[layer_id],
                router_weight=(
                    classifiers[layer_id].weight.detach().float().cpu().numpy()
                ),
                margin=float(margin),
                quantile=float(quantile),
            )
        )
    ranked.sort(
        key=lambda item: (
            float(item["boundary_distance_quantile"]),
            -float(item["boundary_margin_q10"]),
            -float(item["boundary_top1_hit_rate"]),
            float(item.get("normalized_activity_drift", float("inf"))),
            layer_sort_key(str(item["target_layer"])),
            int(item["expert_id"]),
        )
    )
    return ranked, [list(trigger) for trigger in trigger_bank]


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


def _resolve_target_expert_and_router(
    moe_handles: Sequence[MoeHandle],
    target_layer: str,
    target_expert: int,
    mode: str,
    assignments: Dict[str, List[int]],
) -> Tuple[Optional[nn.Module], nn.Module]:
    """Resolve the assigned target expert, if present, and its layer router."""
    target_handle = next(
        (handle for handle in moe_handles if handle.layer_id == target_layer),
        None,
    )
    if target_handle is None:
        available = ", ".join(handle.layer_id for handle in moe_handles)
        raise ValueError(
            f"Target MoE layer '{target_layer}' was not found; available layers: {available}."
        )
    if not target_handle.gate_linear_path:
        raise RuntimeError(f"Target MoE layer '{target_layer}' has no trainable router.")
    if target_layer not in assignments:
        raise KeyError(f"Missing server assignment for target layer '{target_layer}'.")

    router = get_submodule_by_path(
        target_handle.mlp_module,
        target_handle.gate_linear_path,
    )

    assigned = sorted(set(int(value) for value in assignments[target_layer]))
    global_target = int(target_expert)
    if global_target not in assigned:
        return None, router

    entries = get_expert_entries(
        target_handle.mlp_module,
        target_handle.experts_attr,
    )
    if mode == "drop":
        local_target = assigned.index(global_target)
        if not 0 <= local_target < len(entries):
            raise RuntimeError(
                f"Local target expert index {local_target} is outside the retained experts "
                f"for layer '{target_layer}'."
            )
        target_module = entries[local_target][2]
    else:
        matching = [
            expert
            for expert_idx, _expert_key, expert in entries
            if expert_idx == global_target
        ]
        if len(matching) != 1:
            raise RuntimeError(
                f"Expected exactly one target expert {global_target}@{target_layer}, "
                f"found {len(matching)}."
            )
        target_module = matching[0]

    return target_module, router


def restrict_malicious_updates_to_target_expert_and_router(
    model: nn.Module,
    moe_handles: Sequence[MoeHandle],
    target_layer: str,
    target_expert: int,
    mode: str,
    assignments: Dict[str, List[int]],
) -> bool:
    """Freeze all parameters except one target expert and its layer router.

    Returns whether the target expert is present in the server-issued assignment
    and was therefore made trainable.  When it is absent, the target-layer
    router remains the only trainable module so the client never updates an
    unassigned expert or unrelated shared parameters.
    """

    target_module, router = _resolve_target_expert_and_router(
        moe_handles=moe_handles,
        target_layer=target_layer,
        target_expert=target_expert,
        mode=mode,
        assignments=assignments,
    )

    model.requires_grad_(False)
    router.to(torch.float32)
    router.requires_grad_(True)

    if target_module is None:
        return False

    target_module.to(torch.float32)
    target_module.requires_grad_(True)
    return True


@dataclass
class TargetExpertPoisonGradientState:
    """Track fixed-budget target-expert poison-gradient redistribution."""

    ratio_ema: Optional[float] = None
    observations: int = 0
    ratio_sum: float = 0.0
    requested_gain_sum: float = 0.0
    effective_gain_sum: float = 0.0
    budget_scale_sum: float = 0.0
    norm_before_sum: float = 0.0
    norm_after_sum: float = 0.0


def _backward_with_isolated_poison_gradients(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    poison_mask: torch.Tensor,
    poison_parameters: Sequence[nn.Parameter],
    poison_parameter_masks: Optional[Mapping[int, torch.Tensor]] = None,
    target_expert_parameters: Sequence[nn.Parameter] = (),
    target_expert_gradient_goal_ratio: float = 0.0,
    target_expert_gradient_max_gain: float = 1.0,
    target_expert_gradient_state: Optional[TargetExpertPoisonGradientState] = None,
    router_loss: Optional[torch.Tensor] = None,
    router_loss_weight: float = 0.0,
) -> Tuple[torch.Tensor, int, int]:
    """Backprop one fairly weighted batch while isolating its poison gradient.

    Clean CE uses its ordinary ``Nc / N`` contribution and updates every
    trainable parameter in the current client mode.  Poison CE uses ``Np / N``
    and contributes only to ``poison_parameters``. Coordinate masks select
    where that poison contribution is written without changing sample loss
    weights. Optionally, the target-expert share is raised toward a configured
    ratio and the complete masked poison gradient is restored to its original
    L2 norm. The graph is evaluated once and the caller still performs exactly
    one optimizer step.
    """

    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise ValueError(
            "Isolated poison backward expects [batch, classes] logits and "
            "[batch] labels."
        )
    if poison_mask.shape != labels.shape:
        raise ValueError("Poison mask must have the same shape as labels.")
    if router_loss_weight < 0.0:
        raise ValueError("router_loss_weight must be >= 0.")
    if not 0.0 <= target_expert_gradient_goal_ratio <= 1.0:
        raise ValueError("Target-expert poison gradient goal ratio must be in [0, 1].")
    if target_expert_gradient_max_gain < 1.0:
        raise ValueError("Target-expert poison gradient max gain must be >= 1.")

    poison_mask = poison_mask.to(device=labels.device, dtype=torch.bool)
    clean_mask = ~poison_mask
    batch_size = int(labels.shape[0])
    if batch_size <= 0:
        raise ValueError("Isolated poison backward requires a non-empty batch.")

    per_example = nn.functional.cross_entropy(
        logits.float(),
        labels,
        reduction="none",
    )
    differentiable_zero = per_example.sum() * 0.0
    clean_loss = (
        per_example[clean_mask].sum() / batch_size
        if bool(clean_mask.any())
        else differentiable_zero
    )
    poison_loss = (
        per_example[poison_mask].sum() / batch_size
        if bool(poison_mask.any())
        else differentiable_zero
    )
    targeted_loss = poison_loss
    if router_loss is not None:
        targeted_loss = targeted_loss + float(router_loss_weight) * router_loss

    unique_poison_parameters: List[nn.Parameter] = []
    seen_parameter_ids: set[int] = set()
    for parameter in poison_parameters:
        if not parameter.requires_grad or id(parameter) in seen_parameter_ids:
            continue
        unique_poison_parameters.append(parameter)
        seen_parameter_ids.add(id(parameter))

    targeted_gradients: Tuple[Optional[torch.Tensor], ...] = ()
    if unique_poison_parameters and targeted_loss.requires_grad:
        targeted_gradients = torch.autograd.grad(
            targeted_loss,
            unique_poison_parameters,
            retain_graph=bool(clean_mask.any()),
            allow_unused=True,
        )

    if bool(clean_mask.any()) and clean_loss.requires_grad:
        clean_loss.backward()

    target_expert_parameter_ids = {
        id(parameter)
        for parameter in target_expert_parameters
        if parameter.requires_grad
    }
    prepared_gradients: List[Optional[torch.Tensor]] = []
    target_squared_norm = 0.0
    total_squared_norm = 0.0
    for parameter, gradient in zip(
        unique_poison_parameters,
        targeted_gradients,
    ):
        if gradient is None:
            prepared_gradients.append(None)
            continue
        detached_gradient = gradient.detach()
        parameter_id = id(parameter)
        if poison_parameter_masks is not None and parameter_id in poison_parameter_masks:
            coordinate_mask = poison_parameter_masks[parameter_id]
            if coordinate_mask.shape != parameter.shape:
                raise ValueError(
                    "Poison parameter mask shape does not match its parameter: "
                    f"mask={tuple(coordinate_mask.shape)} parameter={tuple(parameter.shape)}."
                )
            detached_gradient = detached_gradient * coordinate_mask.to(
                device=detached_gradient.device,
                dtype=detached_gradient.dtype,
            )
        squared_norm = float(
            detached_gradient.double().square().sum().detach().cpu().item()
        )
        total_squared_norm += squared_norm
        if parameter_id in target_expert_parameter_ids:
            target_squared_norm += squared_norm
        prepared_gradients.append(detached_gradient)

    total_norm = math.sqrt(total_squared_norm)
    target_norm = math.sqrt(target_squared_norm)
    observed_ratio = target_norm / total_norm if total_norm > 0.0 else 0.0
    ratio_for_gain = observed_ratio
    has_poison_gradient_observation = bool(poison_mask.any()) and total_norm > 0.0
    if target_expert_gradient_state is not None and has_poison_gradient_observation:
        if target_expert_gradient_state.ratio_ema is None:
            target_expert_gradient_state.ratio_ema = observed_ratio
        else:
            target_expert_gradient_state.ratio_ema = (
                _TARGET_EXPERT_POISON_GRADIENT_EMA_DECAY
                * target_expert_gradient_state.ratio_ema
                + (1.0 - _TARGET_EXPERT_POISON_GRADIENT_EMA_DECAY)
                * observed_ratio
            )
        ratio_for_gain = target_expert_gradient_state.ratio_ema

    requested_gain = 1.0
    budget_scale = 1.0
    if (
        has_poison_gradient_observation
        and target_norm > 0.0
        and target_expert_gradient_goal_ratio > 0.0
        and ratio_for_gain < target_expert_gradient_goal_ratio
    ):
        requested_gain = min(
            float(target_expert_gradient_max_gain),
            max(
                1.0,
                float(target_expert_gradient_goal_ratio)
                / max(float(ratio_for_gain), 1e-12),
            ),
        )
        boosted_squared_norm = (
            total_squared_norm
            + (requested_gain * requested_gain - 1.0) * target_squared_norm
        )
        if boosted_squared_norm > 0.0:
            budget_scale = math.sqrt(total_squared_norm / boosted_squared_norm)

    effective_target_gain = budget_scale * requested_gain
    for parameter, detached_gradient in zip(
        unique_poison_parameters,
        prepared_gradients,
    ):
        if detached_gradient is None:
            continue
        gradient_scale = budget_scale
        if id(parameter) in target_expert_parameter_ids:
            gradient_scale *= requested_gain
        if gradient_scale != 1.0:
            detached_gradient = detached_gradient * gradient_scale
        if parameter.grad is None:
            parameter.grad = detached_gradient.clone()
        else:
            parameter.grad.add_(detached_gradient)

    norm_after = math.sqrt(
        budget_scale
        * budget_scale
        * (
            total_squared_norm
            + (requested_gain * requested_gain - 1.0) * target_squared_norm
        )
    )
    if target_expert_gradient_state is not None and has_poison_gradient_observation:
        target_expert_gradient_state.observations += 1
        target_expert_gradient_state.ratio_sum += observed_ratio
        target_expert_gradient_state.requested_gain_sum += requested_gain
        target_expert_gradient_state.effective_gain_sum += effective_target_gain
        target_expert_gradient_state.budget_scale_sum += budget_scale
        target_expert_gradient_state.norm_before_sum += total_norm
        target_expert_gradient_state.norm_after_sum += norm_after

    total_loss = clean_loss + poison_loss
    if router_loss is not None:
        total_loss = total_loss + float(router_loss_weight) * router_loss
    return (
        total_loss,
        int(clean_mask.sum().item()),
        int(poison_mask.sum().item()),
    )


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


def _target_expert_parameter_snapshot(
    model: nn.Module,
    layer_id: str,
    expert_idx: int,
) -> Dict[str, torch.Tensor]:
    """Snapshot only one global expert's parameters for local diagnostics."""

    layer_match = LAYER_KEY_PATTERN.match(layer_id)
    if layer_match is None:
        raise ValueError(
            f"Invalid target layer '{layer_id}'; expected encoder.N or decoder.N."
        )
    stack_name = layer_match.group(1)
    layer_index = int(layer_match.group(2))
    prefix = (
        f"{stack_name}.block.{layer_index}.layer."
    )
    expert_marker = f".mlp.experts.expert_{int(expert_idx)}."
    snapshot = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name.startswith(prefix) and expert_marker in name
    }
    if not snapshot:
        raise ValueError(
            f"No parameters found for target expert {expert_idx}@{layer_id}."
        )
    return snapshot


def _parameter_update_l2(
    model: nn.Module,
    before: Mapping[str, torch.Tensor],
) -> Tuple[float, float]:
    """Measure absolute and baseline-relative L2 change for a snapshot."""

    current = dict(model.named_parameters())
    update_squared = 0.0
    baseline_squared = 0.0
    for name, baseline in before.items():
        if name not in current:
            raise KeyError(f"Snapshotted parameter '{name}' is missing from the model.")
        value = current[name].detach().to(device="cpu", dtype=torch.float64)
        baseline_value = baseline.detach().to(device="cpu", dtype=torch.float64)
        if value.shape != baseline_value.shape:
            raise ValueError(
                f"Snapshotted parameter '{name}' shape changed: "
                f"before={tuple(baseline_value.shape)}, after={tuple(value.shape)}."
            )
        delta = (value - baseline_value).reshape(-1)
        baseline_vector = baseline_value.reshape(-1)
        update_squared += float(torch.dot(delta, delta).item())
        baseline_squared += float(
            torch.dot(baseline_vector, baseline_vector).item()
        )
    absolute = math.sqrt(update_squared)
    baseline_norm = math.sqrt(baseline_squared)
    relative = (
        absolute / baseline_norm
        if baseline_norm > 0.0
        else (0.0 if absolute == 0.0 else float("inf"))
    )
    return float(absolute), float(relative)

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


def _load_poison_nontarget_previous_global(
    client_state: RecordDict,
) -> Optional[NDArrays]:
    """Load the last global model observed for poison-only non-target masking."""
    record = client_state.array_records.get(_POISON_NONTARGET_PREVIOUS_GLOBAL_KEY)
    if record is None:
        return None
    return record.to_numpy_ndarrays()


def _save_poison_nontarget_previous_global(
    client_state: RecordDict,
    parameters: NDArrays,
) -> None:
    """Persist a pristine downloaded model for poison-only non-target masking."""
    client_state.array_records[_POISON_NONTARGET_PREVIOUS_GLOBAL_KEY] = ArrayRecord(
        numpy_ndarrays=[array.copy() for array in parameters]
    )


def _build_low_global_update_masks(
    named_parameters: Sequence[Tuple[str, nn.Parameter]],
    keep_ratio: float,
) -> Dict[str, torch.Tensor]:
    """Build one exact global bottom-k mask over a named parameter pool."""
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("Low-global-update keep ratio must be in (0, 1].")
    if not named_parameters:
        return {}
    if keep_ratio == 1.0:
        return {
            name: torch.ones_like(parameter, dtype=torch.bool)
            for name, parameter in named_parameters
        }

    total_coordinates = sum(parameter.numel() for _, parameter in named_parameters)
    flat_magnitudes = torch.empty(
        total_coordinates,
        dtype=torch.float32,
        device="cpu",
    )
    offset = 0
    for _name, parameter in named_parameters:
        length = parameter.numel()
        if parameter.grad is None:
            flat_magnitudes[offset : offset + length].zero_()
        else:
            flat_magnitudes[offset : offset + length].copy_(
                parameter.grad.detach().abs().reshape(-1).to(
                    device="cpu",
                    dtype=torch.float32,
                )
            )
        offset += length

    flat_mask = _exact_bottom_fraction_mask(
        flat_magnitudes,
        keep_ratio,
    ).reshape(-1)
    masks: Dict[str, torch.Tensor] = {}
    offset = 0
    for name, parameter in named_parameters:
        length = parameter.numel()
        masks[name] = flat_mask[offset : offset + length].reshape(
            parameter.shape
        ).to(parameter.device)
        offset += length
    return masks


def _load_a3fl_trigger(client_state: RecordDict) -> Tuple[int, ...]:
    """Load this malicious client's most recently optimized A3FL trigger."""
    record = client_state.array_records.get(_A3FL_TRIGGER_KEY)
    if record is None:
        return ()
    arrays = record.to_numpy_ndarrays()
    if len(arrays) != 1:
        raise ValueError("Persisted A3FL trigger must contain exactly one array.")
    return tuple(int(value) for value in arrays[0].reshape(-1).tolist())


def _save_a3fl_trigger(
    client_state: RecordDict,
    trigger_token_ids: Sequence[int],
) -> None:
    """Persist an A3FL trigger only in this virtual client's Flower state."""
    values = np.asarray([int(value) for value in trigger_token_ids], dtype=np.int64)
    if values.size == 0:
        raise ValueError("Cannot persist an empty A3FL trigger.")
    client_state.array_records[_A3FL_TRIGGER_KEY] = ArrayRecord(
        numpy_ndarrays=[values]
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
    sample_count = _activation_profile_sample_count(len(dataset), max_samples)
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


@torch.no_grad()
def evaluate_targeted_asr(
    model: nn.Module,
    dataset: Any,
    collator: Any,
    batch_size: int,
    target_label: int,
    device: torch.device,
) -> Tuple[float, int]:
    """Evaluate targeted ASR on a shared triggered test dataset.

    The dataset must already exclude examples whose ground-truth label is the
    attack target, matching the centralized server evaluator.
    """

    if batch_size <= 0:
        raise ValueError("Local ASR evaluation batch_size must be > 0.")
    if target_label < 0:
        raise ValueError("Local ASR target_label must be >= 0.")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=False,
    )
    was_training = model.training
    model.eval()
    target_predictions = 0
    total_examples = 0
    try:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            predictions = torch.argmax(outputs.logits, dim=-1)
            target_predictions += int((predictions == target_label).sum().item())
            total_examples += int(predictions.shape[0])
    finally:
        if was_training:
            model.train()
    if total_examples <= 0:
        raise RuntimeError("Triggered local ASR evaluation dataset is empty.")
    return float(target_predictions / total_examples), total_examples


def _select_poison_gradient_probe_datasets(
    dataset: Any,
    max_samples: int,
    seed: int,
) -> Tuple[Any, Any]:
    """Select equal-sized deterministic clean and poisoned subsets."""

    if "is_poisoned" not in dataset.column_names:
        raise ValueError(
            "Poison-gradient probing requires the tokenized is_poisoned marker."
        )
    poison_indices: List[int] = []
    clean_indices: List[int] = []
    for index, value in enumerate(dataset["is_poisoned"]):
        if bool(int(value)):
            poison_indices.append(index)
        else:
            clean_indices.append(index)
    sample_count = min(int(max_samples), len(poison_indices), len(clean_indices))
    if sample_count <= 0:
        raise ValueError(
            "Poison-gradient probing requires at least one clean and one poisoned "
            f"local example; clean={len(clean_indices)}, poisoned={len(poison_indices)}."
        )
    rng = np.random.default_rng(int(seed))

    def choose(indices: Sequence[int]) -> List[int]:
        if len(indices) == sample_count:
            return sorted(int(index) for index in indices)
        return sorted(
            int(index)
            for index in rng.choice(
                np.asarray(indices, dtype=np.int64),
                size=sample_count,
                replace=False,
            ).tolist()
        )

    return dataset.select(choose(clean_indices)), dataset.select(
        choose(poison_indices)
    )


def _collect_mean_ce_gradients(
    *,
    model: nn.Module,
    dataset: Any,
    collator: Any,
    batch_size: int,
    device: torch.device,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    """Differentiate mean classification loss without taking an optimizer step."""

    if len(dataset) <= 0:
        raise ValueError("Gradient-flow probe dataset must not be empty.")
    loader = DataLoader(
        dataset,
        batch_size=min(int(batch_size), len(dataset)),
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=False,
    )
    was_training = model.training
    total_examples = len(dataset)
    total_loss = 0.0
    model.eval()
    model.zero_grad(set_to_none=True)
    try:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
                return_dict=True,
            )
            if outputs.loss is None:
                raise RuntimeError("Gradient-flow probe received loss=None.")
            current_batch = int(labels.shape[0])
            total_loss += float(outputs.loss.detach().item()) * current_batch
            (outputs.loss * (current_batch / total_examples)).backward()
        gradients = {
            name: parameter.grad.detach().to(device="cpu", dtype=torch.float32).clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is not None
        }
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
        del loader
    return float(total_loss / total_examples), gradients


def _summarize_clean_poison_gradient_flow(
    *,
    clean_gradients: Mapping[str, torch.Tensor],
    poison_gradients: Mapping[str, torch.Tensor],
    target_layer: str,
    target_expert: int,
) -> Dict[str, Dict[str, float | int]]:
    """Partition clean/poison gradient energy into disjoint model groups."""

    groups = (
        "target_expert",
        "other_experts",
        "target_router",
        "other_routers",
        "shared",
    )
    moments: Dict[str, Dict[str, float | int]] = {
        group: {
            "poison_squared": 0.0,
            "clean_squared": 0.0,
            "dot": 0.0,
            "coordinates": 0,
        }
        for group in groups
    }

    def accumulate(
        group: str,
        poison: Optional[torch.Tensor],
        clean: Optional[torch.Tensor],
    ) -> None:
        if poison is None and clean is None:
            return
        reference = poison if poison is not None else clean
        if reference is None:
            return
        poison_vector = (
            poison.reshape(-1).to(dtype=torch.float64)
            if poison is not None
            else None
        )
        clean_vector = (
            clean.reshape(-1).to(dtype=torch.float64)
            if clean is not None
            else None
        )
        if poison_vector is not None:
            moments[group]["poison_squared"] = float(
                moments[group]["poison_squared"]
            ) + float(torch.dot(poison_vector, poison_vector).item())
        if clean_vector is not None:
            moments[group]["clean_squared"] = float(
                moments[group]["clean_squared"]
            ) + float(torch.dot(clean_vector, clean_vector).item())
        if poison_vector is not None and clean_vector is not None:
            if poison_vector.shape != clean_vector.shape:
                raise ValueError(
                    f"Clean/poison gradient shape mismatch in group '{group}'."
                )
            moments[group]["dot"] = float(moments[group]["dot"]) + float(
                torch.dot(poison_vector, clean_vector).item()
            )
        moments[group]["coordinates"] = int(moments[group]["coordinates"]) + int(
            reference.numel()
        )

    all_names = sorted(set(clean_gradients).union(poison_gradients))
    for name in all_names:
        poison = poison_gradients.get(name)
        clean = clean_gradients.get(name)
        expert_match = EXPERT_GRADIENT_PARAMETER_PATTERN.match(name)
        if expert_match is not None:
            layer_id = f"{expert_match.group(1)}.{int(expert_match.group(2))}"
            expert_id = int(expert_match.group(3))
            group = (
                "target_expert"
                if layer_id == target_layer and expert_id == int(target_expert)
                else "other_experts"
            )
            accumulate(group, poison, clean)
            continue

        router_match = ROUTER_GRADIENT_PARAMETER_PATTERN.match(name)
        if router_match is not None:
            layer_id = f"{router_match.group(1)}.{int(router_match.group(2))}"
            reference = poison if poison is not None else clean
            if reference is None:
                continue
            if layer_id != target_layer:
                accumulate("other_routers", poison, clean)
                continue
            target_row = int(target_expert)
            if reference.ndim < 1 or not 0 <= target_row < reference.shape[0]:
                raise ValueError(
                    f"Target router row {target_row} is invalid for '{name}' "
                    f"with shape={tuple(reference.shape)}."
                )
            accumulate(
                "target_router",
                None if poison is None else poison[target_row],
                None if clean is None else clean[target_row],
            )
            other_rows = [
                row for row in range(reference.shape[0]) if row != target_row
            ]
            if other_rows:
                accumulate(
                    "other_routers",
                    None if poison is None else poison[other_rows],
                    None if clean is None else clean[other_rows],
                )
            continue

        accumulate("shared", poison, clean)

    poison_total = sum(float(value["poison_squared"]) for value in moments.values())
    clean_total = sum(float(value["clean_squared"]) for value in moments.values())
    if poison_total <= 0.0:
        raise RuntimeError("Poison gradient-flow probe produced zero total gradient energy.")

    summary: Dict[str, Dict[str, float | int]] = {}
    for group, value in moments.items():
        poison_squared = float(value["poison_squared"])
        clean_squared = float(value["clean_squared"])
        dot = float(value["dot"])
        poison_l2 = math.sqrt(poison_squared)
        clean_l2 = math.sqrt(clean_squared)
        summary[group] = {
            "poison_grad_l2": float(poison_l2),
            "clean_grad_l2": float(clean_l2),
            "poison_energy_ratio": float(poison_squared / poison_total),
            "clean_energy_ratio": float(
                clean_squared / clean_total if clean_total > 0.0 else 0.0
            ),
            "clean_poison_cosine": float(
                dot / (poison_l2 * clean_l2)
                if poison_l2 > 0.0 and clean_l2 > 0.0
                else 0.0
            ),
            "poison_to_clean_norm_ratio": float(
                poison_l2 / clean_l2 if clean_l2 > 0.0 else 0.0
            ),
            "coordinates": int(value["coordinates"]),
        }
    return summary


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

    def _is_attack_active(
        self,
        server_round: int,
        round_config: Optional[Dict[str, Scalar]] = None,
    ) -> bool:
        end = self.runtime_config.attack_end_round
        within_window = server_round >= self.runtime_config.attack_start_round and (
            end <= 0 or server_round <= end
        )
        attack_enabled = self.runtime_config.attack_enabled
        if round_config is not None:
            attack_enabled = _as_bool(
                round_config,
                "attack_enabled",
                attack_enabled,
            )
        return bool(
            attack_enabled and self.is_malicious and within_window
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
        profile_warmup = server_round == 1
        server_allows_attack = _as_bool(
            config,
            "attack_enabled",
            self.runtime_config.attack_enabled,
        )
        attack_scheduled = self._is_attack_active(server_round, config)
        # Round one must remain parameter-identical to the downloaded global
        # model.  Attack training follows the same round-two start boundary as
        # ordinary local optimization.
        attack_active = bool(attack_scheduled and not profile_warmup)
        attack_mode = str(self.runtime_config.attack_mode).lower()
        private_a3fl_client = bool(self.is_malicious and attack_mode == "a3fl")
        # A3FL first needs clean local examples to optimize its round-adaptive
        # trigger.  Its poisoned bundle is built only after trigger search.
        bundle = self._load_bundle(
            apply_badnet=bool(attack_active and attack_mode != "a3fl")
        )
        _set_seed(self.runtime_config.seed + self.client_id)

        model = self._build_model(bundle)
        optimizer: Optional[AdamW] = None
        train_loader: Optional[DataLoader] = None
        lfe_tracker: Optional[TrainingLFERoutingTracker] = None
        router_objective: Optional[TriggerRouterLoss] = None
        a3fl_result: Optional[A3FLSearchResult] = None
        probe_state: Optional[Dict[str, Any]] = None
        probe_report: Dict[str, Any] = {"decoded": [], "rankings": []}
        probe_injected: List[Dict[str, Any]] = []
        probe_target_layer: Optional[str] = None
        scheduled_probe_target: Optional[Dict[str, Any]] = None
        scheduled_probe_targets: List[Dict[str, Any]] = []
        next_probe_choice: Optional[Dict[str, Any]] = None
        next_probe_choices: List[Dict[str, Any]] = []
        probe_lazy_candidates: List[Dict[str, Any]] = []
        probe_boundary_candidates: List[Dict[str, Any]] = []
        probe_trigger_bank: List[List[int]] = []
        probe_layer_rankings: List[Dict[str, Any]] = []
        probe_activation_spoof: Optional[Dict[str, Any]] = None
        target_activation_spoof: Optional[Dict[str, Any]] = None
        probe_state_events: List[Dict[str, Any]] = []
        multi_expert_probe = False
        probe_per_layer_limit = 0
        local_asr_pre: Optional[float] = None
        local_asr_post: Optional[float] = None
        local_asr_eval_examples = 0
        local_asr_dataset: Optional[Any] = None
        poison_gradient_flow: Optional[Dict[str, Any]] = None
        try:
            mode = _as_str(config, "mode", self.runtime_config.mode).lower()
            if mode not in {"full", "mix", "drop", "flex"}:
                raise ValueError(f"Unsupported client mode='{mode}'.")
            base_moe_handles = collect_moe_handles(model)
            if not base_moe_handles:
                raise RuntimeError("No sparse MoE layers with experts were detected in SwitchTransformer.")

            full_num_experts: Dict[str, int] = {
                h.layer_id: len(get_expert_entries(h.mlp_module, h.experts_attr)) for h in base_moe_handles
            }
            assignments = self._resolve_assignments(mode=mode, moe_handles=base_moe_handles, round_config=config)

            current_parameters = parameters
            neurotoxin_reference_parameters: Optional[NDArrays] = None
            poison_nontarget_reference_parameters: Optional[NDArrays] = None
            incoming_param_names = self._resolve_incoming_param_names(
                model=model, parameters=current_parameters, round_config=config
            )
            probe_active = bool(
                self.runtime_config.expert_probe_enabled
                and self.is_malicious
                and self.client_id == min(self.runtime_config.malicious_clients)
            )
            if probe_active:
                fixed_probe_layer = self.runtime_config.expert_probe_target_layer
                if fixed_probe_layer is not None and fixed_probe_layer not in full_num_experts:
                    raise ValueError(
                        f"Expert probe target layer '{fixed_probe_layer}' is not a sparse MoE layer."
                    )
                probe_state = load_probe_state(self.client_state)
                probe_report = observe_global_and_update_history(
                    state=probe_state,
                    parameter_names=incoming_param_names,
                    parameters=current_parameters,
                    server_round=server_round,
                    target_layer=fixed_probe_layer,
                    candidate_pool=self.runtime_config.expert_probe_candidate_pool,
                    repetitions=self.runtime_config.expert_probe_repetitions,
                    drift_window=self.runtime_config.expert_probe_drift_window,
                    max_consistency_error=(
                        self.runtime_config.expert_probe_max_consistency_error
                    ),
                )
                if fixed_probe_layer is None:
                    # Model-wide probing reserves two ordinary hot-expert
                    # slots in every layer.  All sparse layers use exact-K,
                    # but take the minimum defensively.
                    probe_per_layer_limit = max(
                        min(len(values) for values in assignments.values()) - 2,
                        0,
                    )
                else:
                    # Preserve the explicit fixed-layer compatibility mode.
                    probe_per_layer_limit = len(assignments[fixed_probe_layer])
                multi_expert_probe = bool(
                    self.runtime_config.expert_probe_batch_size > 1
                    and probe_per_layer_limit > 0
                )
                advance_fn = (
                    advance_batch_probe_confirmation
                    if multi_expert_probe
                    else advance_probe_confirmation
                )
                probe_state_events = advance_fn(
                    state=probe_state,
                    decoded=probe_report["decoded"],
                    confirmation_rounds=(
                        self.runtime_config.expert_probe_confirmation_rounds
                    ),
                    single_retention_min=(
                        self.runtime_config.expert_probe_single_retention_min
                    ),
                    single_retention_max=(
                        self.runtime_config.expert_probe_single_retention_max
                    ),
                    max_pilot_attempts=(
                        self.runtime_config.expert_probe_max_pilot_attempts
                    ),
                )
                if multi_expert_probe:
                    candidate_states = probe_state.get("batch_candidates", {})
                    for raw_target in probe_state.get("scheduled_targets", []):
                        if not isinstance(raw_target, dict):
                            continue
                        target = {
                            "layer": str(raw_target["layer"]),
                            "expert_id": int(raw_target["expert_id"]),
                        }
                        key = f"{target['layer']}/expert_{target['expert_id']}"
                        status = str(candidate_states.get(key, {}).get("status", "pilot"))
                        if status not in {"rejected", "confirmed"}:
                            scheduled_probe_targets.append(target)
                    if fixed_probe_layer is None:
                        retained_targets: List[Dict[str, Any]] = []
                        retained_per_layer: Dict[str, int] = {}
                        for target in scheduled_probe_targets:
                            layer_id = str(target["layer"])
                            if (
                                retained_per_layer.get(layer_id, 0)
                                >= probe_per_layer_limit
                            ):
                                continue
                            retained_targets.append(target)
                            retained_per_layer[layer_id] = (
                                retained_per_layer.get(layer_id, 0) + 1
                            )
                        scheduled_probe_targets = retained_targets
                        probe_state["scheduled_targets"] = retained_targets
                    scheduled_layers = {
                        str(target["layer"]) for target in scheduled_probe_targets
                    }
                    probe_target_layer = (
                        next(iter(scheduled_layers))
                        if len(scheduled_layers) == 1
                        else None
                    )
                else:
                    raw_target = probe_state.get("scheduled_target")
                    if isinstance(raw_target, dict):
                        scheduled_probe_target = {
                            "layer": str(raw_target["layer"]),
                            "expert_id": int(raw_target["expert_id"]),
                        }
                    else:
                        # Backward-compatible migration for an in-flight single-layer run.
                        raw_expert = probe_state.get("scheduled_expert")
                        if raw_expert is not None and fixed_probe_layer is not None:
                            scheduled_probe_target = {
                                "layer": fixed_probe_layer,
                                "expert_id": int(raw_expert),
                            }
                            probe_state["scheduled_target"] = scheduled_probe_target
                    if scheduled_probe_target is not None:
                        probe_target_layer = str(scheduled_probe_target["layer"])

            tracks_neurotoxin_history = bool(
                server_allows_attack
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
            tracks_poison_nontarget_history = bool(
                server_allows_attack
                and self.is_malicious
                and self.runtime_config.isolate_poison_gradients_to_target
                and self.runtime_config.neurotoxin_mask_ratio < 1.0
            )
            if tracks_poison_nontarget_history:
                poison_nontarget_reference_parameters = (
                    _load_poison_nontarget_previous_global(self.client_state)
                )
                # Keep only pristine downloaded globals in client state. Round
                # one therefore supplies the reference for a round-two attack.
                _save_poison_nontarget_previous_global(
                    self.client_state,
                    current_parameters,
                )
            _load_from_numpy_ndarrays(
                model=model,
                param_names=incoming_param_names,
                parameters=current_parameters,
            )
            neurotoxin_global_update_norm = 0.0
            poison_nontarget_global_update_norm = 0.0
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
            if (
                attack_active
                and self.runtime_config.isolate_poison_gradients_to_target
                and self.runtime_config.neurotoxin_mask_ratio < 1.0
                and poison_nontarget_reference_parameters is not None
            ):
                poison_nontarget_global_update_norm = (
                    _attach_global_update_to_parameter_grads(
                        model=model,
                        parameter_names=incoming_param_names,
                        current_parameters=current_parameters,
                        previous_parameters=poison_nontarget_reference_parameters,
                    )
                )

            eval_batch_size = _as_int(config, "eval_batch_size", self.runtime_config.eval_batch_size)
            collator = DataCollatorWithPadding(
                tokenizer=self.data_manager.tokenizer,
                padding=True,
                return_tensors="pt",
            )
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device)

            # Multi-expert probing uses pristine, full routers before ``drop``
            # can prune their rows.  A single forward pass per shared-bank
            # trigger captures every shortlisted layer, so candidates across
            # the model receive the same context and compute budget.
            if probe_state is not None and multi_expert_probe:
                fixed_probe_layer = self.runtime_config.expert_probe_target_layer
                if fixed_probe_layer is not None:
                    probe_lazy_candidates = rank_lazy_expert_candidates(
                        state=probe_state,
                        target_layer=fixed_probe_layer,
                        num_experts=full_num_experts[fixed_probe_layer],
                        warmup_observations=(
                            self.runtime_config.expert_probe_warmup_observations
                        ),
                        pairs=self.runtime_config.expert_probe_pairs,
                    )
                else:
                    (
                        probe_lazy_candidates,
                        probe_layer_rankings,
                    ) = rank_lazy_candidates_across_layers(
                        state=probe_state,
                        layer_num_experts=full_num_experts,
                        warmup_observations=(
                            self.runtime_config.expert_probe_warmup_observations
                        ),
                        pairs=self.runtime_config.expert_probe_pairs,
                        top_experts_per_layer=min(
                            self.runtime_config.expert_probe_layer_top_experts,
                            probe_per_layer_limit,
                        ),
                        max_layers=self.runtime_config.expert_probe_max_layers,
                        min_layer_contrast=(
                            self.runtime_config.expert_probe_min_layer_contrast
                        ),
                    )
                if (
                    probe_lazy_candidates
                    and str(probe_state.get("phase", "scan")) != "complete"
                ):
                    (
                        probe_boundary_candidates,
                        probe_trigger_bank,
                    ) = _screen_probe_candidates_across_layers(
                        model=model,
                        moe_handles=base_moe_handles,
                        candidates=probe_lazy_candidates,
                        dataset=bundle.train_dataset,
                        collator=collator,
                        tokenizer=self.data_manager.tokenizer,
                        device=device,
                        max_length=self.runtime_config.max_length,
                        samples=self.runtime_config.expert_probe_screen_samples,
                        trigger_bank_size=(
                            self.runtime_config.expert_probe_trigger_bank_size
                        ),
                        trigger_length=(
                            self.runtime_config.expert_probe_trigger_length
                        ),
                        margin=self.runtime_config.expert_probe_boundary_margin,
                        quantile=(
                            self.runtime_config.expert_probe_boundary_quantile
                        ),
                        seed=self.runtime_config.seed + self.client_id,
                    )
                    probe_state["boundary_screen"] = {
                        "server_round": int(server_round),
                        "target_layers": sorted(
                            {
                                str(candidate["target_layer"])
                                for candidate in probe_boundary_candidates
                            },
                            key=layer_sort_key,
                        ),
                        "trigger_bank": probe_trigger_bank,
                        "ranked_candidates": probe_boundary_candidates,
                    }

            if attack_active and attack_mode == "a3fl":
                if self.runtime_config.target_label >= bundle.label_info.num_labels:
                    raise ValueError(
                        f"--target_label={self.runtime_config.target_label} is invalid for "
                        f"dataset '{self.runtime_config.dataset_name}' with "
                        f"{bundle.label_info.num_labels} labels."
                    )
                persisted_trigger = _load_a3fl_trigger(self.client_state)
                initial_trigger = persisted_trigger or tuple(
                    self.runtime_config.trigger_token_ids
                )
                if not initial_trigger:
                    encoded = self.data_manager.tokenizer(
                        self.runtime_config.trigger,
                        add_special_tokens=False,
                        truncation=False,
                    )
                    initial_trigger = tuple(
                        int(token_id) for token_id in encoded["input_ids"]
                    )
                if not initial_trigger:
                    raise ValueError("The configured A3FL initial trigger tokenizes to an empty sequence.")

                eligible_indices: List[int] = []
                for index in range(len(bundle.train_dataset)):
                    row_label = int(bundle.train_dataset[index]["labels"])
                    if row_label != self.runtime_config.target_label:
                        eligible_indices.append(index)
                    if len(eligible_indices) >= self.runtime_config.a3fl_calibration_samples:
                        break
                if not eligible_indices:
                    raise ValueError(
                        "A3FL calibration requires at least one non-target-class local example."
                    )
                calibration_dataset = bundle.train_dataset.select(eligible_indices)
                calibration_loader = DataLoader(
                    calibration_dataset,
                    batch_size=len(calibration_dataset),
                    shuffle=False,
                    collate_fn=collator,
                    num_workers=0,
                    pin_memory=False,
                )
                calibration_batch = next(iter(calibration_loader))

                def restore_global_model() -> None:
                    _load_from_numpy_ndarrays(
                        model=model,
                        param_names=incoming_param_names,
                        parameters=current_parameters,
                    )
                    model.to(device)

                tokenizer = self.data_manager.tokenizer
                a3fl_optimizer = A3FLTriggerOptimizer(
                    model=model,
                    config=A3FLSearchConfig(
                        outer_steps=self.runtime_config.a3fl_outer_steps,
                        trigger_steps=self.runtime_config.a3fl_trigger_steps,
                        candidate_batch_size=self.runtime_config.a3fl_candidate_batch_size,
                        top_k=self.runtime_config.a3fl_top_k,
                        adversarial_steps=self.runtime_config.a3fl_adv_steps,
                        adversarial_lr=self.runtime_config.a3fl_adv_lr,
                        lambda0=self.runtime_config.a3fl_lambda,
                        early_stop_asr=self.runtime_config.a3fl_early_stop_asr,
                        max_length=self.runtime_config.max_length,
                        seed=self.runtime_config.seed + self.client_id + server_round * 10_007,
                    ),
                    eos_token_id=int(tokenizer.eos_token_id),
                    pad_token_id=int(tokenizer.pad_token_id),
                    target_label=self.runtime_config.target_label,
                    restore_global=restore_global_model,
                    excluded_token_ids=tokenizer.all_special_ids,
                    decode_trigger=lambda ids: tokenizer.decode(
                        list(ids),
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ).strip(),
                )
                a3fl_result = a3fl_optimizer.optimize(
                    calibration_batch,
                    initial_trigger_token_ids=initial_trigger,
                )
                _save_a3fl_trigger(
                    self.client_state,
                    a3fl_result.trigger_token_ids,
                )
                bundle = self.data_manager.load_client_dataset(
                    partition_id=self.client_id,
                    apply_badnet=True,
                    trigger=a3fl_result.trigger_text,
                    trigger_token_ids=a3fl_result.trigger_token_ids,
                )
                del calibration_loader, calibration_batch, calibration_dataset, a3fl_optimizer
                gc.collect()

            print(
                (
                    f"[MixFedMoE][client={self.client_id}] fit start "
                    f"round={server_round} mode={mode} "
                    f"profile_warmup={profile_warmup} "
                    f"malicious={False if private_a3fl_client else self.is_malicious} "
                    f"attack_active={False if private_a3fl_client else attack_active} "
                    f"poisoned={0 if private_a3fl_client else bundle.num_poisoned_examples} "
                    f"client_data_source="
                    f"{'global_random_iid' if self.runtime_config.iid_malicious_client_data and self.is_malicious else 'local_partition'} "
                    f"poison_source="
                    f"{'global_random' if self.runtime_config.global_random_poison_source else 'local_partition'} "
                    f"train_examples={bundle.num_train_examples} "
                    f"eval_examples={bundle.num_eval_examples}"
                ),
                flush=True,
            )

            if mode == "full":
                apply_mode_full(model)
            elif mode == "mix":
                apply_mode_mix(model=model, moe_handles=base_moe_handles, layer_hot_map=assignments)
            elif mode == "flex":
                apply_mode_flex(model=model, moe_handles=base_moe_handles, layer_hot_map=assignments)
            else:
                apply_mode_drop(model=model, moe_handles=base_moe_handles, layer_hot_map=assignments)

            # Preserve the mode's ordinary upload schema even when the malicious
            # optimizer is restricted below. Frozen parameters are returned at
            # their downloaded values (a zero local update), which keeps sparse
            # coverage validation and aggregation semantics unchanged.
            mode_return_names = _trainable_parameter_names(model)

            target_expert_trainable = False
            poison_gradient_isolation_active = bool(
                attack_active
                and self.runtime_config.isolate_poison_gradients_to_target
            )
            isolated_poison_parameters: List[nn.Parameter] = []
            isolated_poison_parameter_masks: Dict[int, torch.Tensor] = {}
            target_expert_poison_parameters: List[nn.Parameter] = []
            target_expert_poison_gradient_state = TargetExpertPoisonGradientState()
            poison_isolation_target_available = False
            poison_nontarget_mask_source = "disabled"
            poison_nontarget_mask_coordinates = 0
            poison_nontarget_mask_total_coordinates = 0
            poison_nontarget_mask_density = 0.0
            if (
                attack_active
                and self.runtime_config.malicious_target_expert_router_only
            ):
                target_layer = self.runtime_config.lfe_target_layer
                target_expert = self.runtime_config.lfe_target_expert
                if target_layer is None or target_expert is None:
                    raise RuntimeError(
                        "Target-expert/router-only training requires lfe_target_layer and "
                        "lfe_target_expert."
                    )
                target_expert_trainable = (
                    restrict_malicious_updates_to_target_expert_and_router(
                        model=model,
                        moe_handles=base_moe_handles,
                        target_layer=target_layer,
                        target_expert=int(target_expert),
                        mode=mode,
                        assignments=assignments,
                    )
                )
                scope = (
                    "target expert and target-layer router"
                    if target_expert_trainable
                    else "target-layer router only (target expert not assigned this round)"
                )
                print(
                    f"[MixFedMoE][client={self.client_id}][round={server_round}] "
                    f"restricted malicious updates to {scope}: "
                    f"target={target_expert}@{target_layer}",
                    flush=True,
                )

            if poison_gradient_isolation_active:
                target_layer = self.runtime_config.lfe_target_layer
                target_expert = self.runtime_config.lfe_target_expert
                if target_layer is None or target_expert is None:
                    raise RuntimeError(
                        "Poison-gradient isolation requires lfe_target_layer and "
                        "lfe_target_expert."
                    )
                target_module, target_router = _resolve_target_expert_and_router(
                    moe_handles=base_moe_handles,
                    target_layer=target_layer,
                    target_expert=int(target_expert),
                    mode=mode,
                    assignments=assignments,
                )
                isolated_modules = [target_router]
                if target_module is not None:
                    isolated_modules.append(target_module)
                    target_expert_poison_parameters = [
                        parameter
                        for parameter in target_module.parameters()
                        if parameter.requires_grad
                    ]
                    target_expert_trainable = True
                    poison_isolation_target_available = True
                isolated_poison_parameters = [
                    parameter
                    for module in isolated_modules
                    for parameter in module.parameters()
                    if parameter.requires_grad
                ]
                if target_module is not None:
                    print(
                        f"[MixFedMoE][client={self.client_id}][round={server_round}] "
                        "enabled fairly weighted poison-gradient isolation: "
                        f"base poison scope is expert {target_expert}@{target_layer} "
                        "and its router; clean CE keeps the ordinary mode scope; "
                        "target-expert poison-gradient redistribution uses "
                        f"goal={self.runtime_config.target_expert_poison_gradient_goal_ratio} "
                        f"max_gain={self.runtime_config.target_expert_poison_gradient_max_gain}.",
                        flush=True,
                    )
                else:
                    print(
                        f"[MixFedMoE][client={self.client_id}][round={server_round}] "
                        f"target expert {target_expert}@{target_layer} is not assigned; "
                        "the base poison scope contains only the target-layer router until "
                        "the target becomes available.",
                        flush=True,
                    )

                target_parameter_ids = {
                    id(parameter) for parameter in isolated_poison_parameters
                }
                nontarget_named_parameters = [
                    (name, parameter)
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                    and id(parameter) not in target_parameter_ids
                ]
                poison_nontarget_mask_total_coordinates = sum(
                    parameter.numel()
                    for _name, parameter in nontarget_named_parameters
                )
                if self.runtime_config.neurotoxin_mask_ratio >= 1.0:
                    nontarget_masks_by_name = {
                        name: torch.ones_like(parameter, dtype=torch.bool)
                        for name, parameter in nontarget_named_parameters
                    }
                    poison_nontarget_mask_source = "disabled_all_coordinates"
                elif poison_nontarget_reference_parameters is None:
                    nontarget_masks_by_name = {
                        name: torch.zeros_like(parameter, dtype=torch.bool)
                        for name, parameter in nontarget_named_parameters
                    }
                    poison_nontarget_mask_source = "unavailable_first_observation"
                else:
                    nontarget_masks_by_name = _build_low_global_update_masks(
                        nontarget_named_parameters,
                        keep_ratio=float(
                            self.runtime_config.neurotoxin_mask_ratio
                        ),
                    )
                    poison_nontarget_mask_source = (
                        "previously_observed_global_update"
                    )
                poison_nontarget_mask_coordinates = sum(
                    int(mask.sum().item())
                    for mask in nontarget_masks_by_name.values()
                )
                poison_nontarget_mask_density = float(
                    poison_nontarget_mask_coordinates
                    / max(poison_nontarget_mask_total_coordinates, 1)
                )
                for name, parameter in nontarget_named_parameters:
                    isolated_poison_parameters.append(parameter)
                    isolated_poison_parameter_masks[id(parameter)] = (
                        nontarget_masks_by_name[name]
                    )

                # Temporary global deltas live in ``param.grad`` only long
                # enough to construct the mask. They must never leak into
                # either the clean or poison optimizer gradient.
                model.zero_grad(set_to_none=True)

                print(
                    f"[MixFedMoE][client={self.client_id}][round={server_round}] "
                    "poison-only non-target Neurotoxin mask: "
                    f"keep_ratio={self.runtime_config.neurotoxin_mask_ratio} "
                    f"coordinates={poison_nontarget_mask_coordinates}/"
                    f"{poison_nontarget_mask_total_coordinates} "
                    f"source={poison_nontarget_mask_source}",
                    flush=True,
                )

            if attack_active and self.runtime_config.inject_target_expert:
                target_layer = self.runtime_config.lfe_target_layer
                target_expert = self.runtime_config.lfe_target_expert
                if target_layer is None or target_expert is None:
                    raise RuntimeError(
                        "Target router supervision requires lfe_target_layer and lfe_target_expert."
                    )
                if int(target_expert) in assignments.get(target_layer, []):
                    trigger_token_ids = tuple(self.runtime_config.trigger_token_ids)
                    if not trigger_token_ids:
                        encoded_trigger = self.data_manager.tokenizer(
                            self.runtime_config.trigger,
                            add_special_tokens=False,
                            truncation=False,
                        )
                        trigger_token_ids = tuple(
                            int(token_id) for token_id in encoded_trigger["input_ids"]
                        )
                    router_objective = _build_trigger_router_loss(
                        moe_handles=base_moe_handles,
                        target_layer=target_layer,
                        target_expert=int(target_expert),
                        mode=mode,
                        assignments=assignments,
                        trigger_token_ids=trigger_token_ids,
                        margin=self.runtime_config.target_router_margin,
                        clean_kl_weight=self.runtime_config.target_router_kl_weight,
                        full_weight_below=(
                            self.runtime_config.target_router_full_weight_below
                        ),
                        stop_median=self.runtime_config.target_router_stop_median,
                        stop_p10=self.runtime_config.target_router_stop_p10,
                    )
                    print(
                        f"[MixFedMoE][client={self.client_id}] enabled adaptive trigger-router margin "
                        f"for expert {target_expert}@{target_layer} with "
                        f"trigger_token_ids={list(trigger_token_ids)} "
                        f"margin={self.runtime_config.target_router_margin} "
                        f"weight={self.runtime_config.target_router_loss_weight} "
                        f"clean_kl={self.runtime_config.target_router_kl_weight}",
                        flush=True,
                    )
                else:
                    print(
                        f"[MixFedMoE][client={self.client_id}][round={server_round}] target expert "
                        f"{target_expert}@{target_layer} is not yet assigned; skipping target-router "
                        "loss this round and promoting it through the uploaded activation profile.",
                        flush=True,
                    )

            poison_gradient_probe_active = bool(
                self.runtime_config.measure_poison_gradient_flow
                and mode == "mix"
                and attack_active
                and self.is_malicious
            )
            if poison_gradient_probe_active:
                target_layer = self.runtime_config.lfe_target_layer
                target_expert = self.runtime_config.lfe_target_expert
                if target_layer is None or target_expert is None:
                    raise RuntimeError(
                        "Poison-gradient probing requires a configured target expert."
                    )
                clean_probe_dataset, poison_probe_dataset = (
                    _select_poison_gradient_probe_datasets(
                        dataset=bundle.train_dataset,
                        max_samples=self.runtime_config.poison_gradient_samples,
                        seed=(
                            self.runtime_config.seed
                            + self.client_id * 1_009
                            + server_round * 10_007
                        ),
                    )
                )
                clean_probe_loss, clean_probe_gradients = (
                    _collect_mean_ce_gradients(
                        model=model,
                        dataset=clean_probe_dataset,
                        collator=collator,
                        batch_size=eval_batch_size,
                        device=device,
                    )
                )
                poison_probe_loss, poison_probe_gradients = (
                    _collect_mean_ce_gradients(
                        model=model,
                        dataset=poison_probe_dataset,
                        collator=collator,
                        batch_size=eval_batch_size,
                        device=device,
                    )
                )
                gradient_groups = _summarize_clean_poison_gradient_flow(
                    clean_gradients=clean_probe_gradients,
                    poison_gradients=poison_probe_gradients,
                    target_layer=target_layer,
                    target_expert=int(target_expert),
                )
                poison_gradient_flow = {
                    "clean_examples": int(len(clean_probe_dataset)),
                    "poison_examples": int(len(poison_probe_dataset)),
                    "clean_ce_loss": float(clean_probe_loss),
                    "poison_ce_loss": float(poison_probe_loss),
                    "target_layer": str(target_layer),
                    "target_expert": int(target_expert),
                    "groups": gradient_groups,
                }
                print(
                    f"[MixFedMoE][client={self.client_id}][round={server_round}] "
                    "poison gradient energy: "
                    + " ".join(
                        f"{group}={float(values['poison_energy_ratio']):.4f}"
                        for group, values in gradient_groups.items()
                    ),
                    flush=True,
                )
                del (
                    clean_probe_dataset,
                    poison_probe_dataset,
                    clean_probe_gradients,
                    poison_probe_gradients,
                )

            local_asr_probe_active = bool(
                self.runtime_config.measure_local_asr
                and mode == "mix"
                and attack_active
                and self.is_malicious
            )
            if local_asr_probe_active:
                if a3fl_result is not None:
                    local_asr_trigger = a3fl_result.trigger_text
                    local_asr_trigger_ids = a3fl_result.trigger_token_ids
                else:
                    local_asr_trigger = self.runtime_config.trigger
                    local_asr_trigger_ids = self.runtime_config.trigger_token_ids
                local_asr_dataset = self.data_manager.load_triggered_server_test_dataset(
                    trigger=local_asr_trigger,
                    trigger_token_ids=local_asr_trigger_ids,
                )
                local_asr_pre, local_asr_eval_examples = evaluate_targeted_asr(
                    model=model,
                    dataset=local_asr_dataset,
                    collator=collator,
                    batch_size=eval_batch_size,
                    target_label=self.runtime_config.target_label,
                    device=device,
                )
                print(
                    f"[MixFedMoE][client={self.client_id}][round={server_round}] "
                    f"local ASR before training={local_asr_pre:.6f} "
                    f"examples={local_asr_eval_examples}",
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
            calib_samples = _as_int(
                config,
                "calib_samples",
                self.runtime_config.lfe_calibration_samples,
            )

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
            target_steps = _compute_round_train_steps(
                server_round=server_round,
                local_epochs=local_epochs,
                steps_per_epoch=steps_per_epoch,
            )

            trainable_params = [param for param in model.parameters() if param.requires_grad]
            if not trainable_params:
                raise RuntimeError("No trainable parameters found after applying client mode.")
            if not profile_warmup:
                optimizer = AdamW(
                    trainable_params,
                    lr=learning_rate,
                    weight_decay=weight_decay,
                )

            model.to(device)
            if profile_warmup:
                model.eval()
                print(
                    f"[MixFedMoE][client={self.client_id}][round=1] routing-profile warmup: "
                    "skipping backward/optimizer updates; formal local training starts in round 2.",
                    flush=True,
                )
            else:
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
            if (
                not profile_warmup
                and (self.runtime_config.lfe_profile_enabled or target_activation_enabled)
            ):
                training_moe_handles = collect_moe_handles(model)
                expert_id_maps = assignments if mode == "drop" else None
                lfe_tracker = TrainingLFERoutingTracker(
                    moe_handles=training_moe_handles,
                    top_k=self.runtime_config.lfe_top_k,
                    expert_id_maps=expert_id_maps,
                )

            total_loss = 0.0
            total_router_loss = 0.0
            total_router_margin_loss = 0.0
            total_router_clean_kl_loss = 0.0
            total_router_margin_median = 0.0
            total_router_margin_p10 = 0.0
            total_router_route_rate = 0.0
            total_router_adaptive_weight = 0.0
            router_trigger_steps = 0
            total_router_tokens = 0
            total_isolated_clean_examples = 0
            total_isolated_poison_examples = 0
            total_examples = 0
            steps_done = 0
            round_start = time.time()
            while steps_done < target_steps:
                if optimizer is None:
                    raise RuntimeError("Optimizer is unavailable during a formal training round.")
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
                    router_loss: Optional[torch.Tensor] = None
                    if router_objective is not None:
                        router_loss = router_objective.loss()
                        if not poison_gradient_isolation_active:
                            loss = (
                                loss
                                + self.runtime_config.target_router_loss_weight
                                * router_loss
                            )
                        total_router_loss += float(router_loss.detach().item())
                        total_router_tokens += router_objective.trigger_tokens
                        total_router_clean_kl_loss += (
                            router_objective.last_clean_kl_loss
                        )
                        if router_objective.trigger_tokens > 0:
                            router_trigger_steps += 1
                            total_router_margin_loss += (
                                router_objective.last_margin_loss
                            )
                            total_router_margin_median += (
                                router_objective.last_margin_median
                            )
                            total_router_margin_p10 += (
                                router_objective.last_margin_p10
                            )
                            total_router_route_rate += (
                                router_objective.last_route_rate
                            )
                            total_router_adaptive_weight += (
                                router_objective.last_adaptive_weight
                            )
                    if poison_gradient_isolation_active:
                        if "is_poisoned" not in batch:
                            raise RuntimeError(
                                "Poison-gradient isolation requires the tokenized "
                                "is_poisoned batch marker."
                            )
                        poison_mask = batch["is_poisoned"].to(device).bool()
                        loss, clean_count, poison_count = (
                            _backward_with_isolated_poison_gradients(
                                logits=outputs.logits,
                                labels=labels,
                                poison_mask=poison_mask,
                                poison_parameters=isolated_poison_parameters,
                                poison_parameter_masks=(
                                    isolated_poison_parameter_masks
                                ),
                                target_expert_parameters=(
                                    target_expert_poison_parameters
                                ),
                                target_expert_gradient_goal_ratio=(
                                    self.runtime_config
                                    .target_expert_poison_gradient_goal_ratio
                                ),
                                target_expert_gradient_max_gain=(
                                    self.runtime_config
                                    .target_expert_poison_gradient_max_gain
                                ),
                                target_expert_gradient_state=(
                                    target_expert_poison_gradient_state
                                ),
                                router_loss=router_loss,
                                router_loss_weight=(
                                    self.runtime_config.target_router_loss_weight
                                ),
                            )
                        )
                        total_isolated_clean_examples += clean_count
                        total_isolated_poison_examples += poison_count
                    elif loss.requires_grad:
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
            if (
                not profile_warmup
                and target_layer is not None
                and target_expert is not None
            ):
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

            profile_sample_count = _activation_profile_sample_count(
                len(bundle.train_dataset),
                calib_samples,
            )
            profile_start = time.time()
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
            # Keep the measured profile before any malicious promotion.  The
            # server may only learn that this round reached ASR=1 after fit
            # aggregation, so it needs the genuine profile to discard this
            # round's spoof before constructing the next round's assignments.
            unspoofed_activation_profile = {
                str(layer_id): [int(value) for value in counts]
                for layer_id, counts in activation_profile.items()
            }
            activation_profile_time = float(time.time() - profile_start)
            if profile_warmup:
                # Keep the existing round-time aggregation meaningful even
                # though this round performs profiling instead of optimization.
                local_training_time = activation_profile_time

            if (
                profile_warmup
                and target_layer is not None
                and target_expert is not None
            ):
                if target_layer not in activation_profile:
                    raise RuntimeError(
                        f"Warmup routing profile has no target layer '{target_layer}'."
                    )
                target_counts = activation_profile[target_layer]
                target_id = int(target_expert)
                if not 0 <= target_id < len(target_counts):
                    raise ValueError(
                        f"Target expert {target_id} is outside the warmup profile for "
                        f"layer '{target_layer}'."
                    )
                valid_tokens = int(sum(int(value) for value in target_counts))
                target_count = int(target_counts[target_id])
                target_activation = {
                    "layer_id": str(target_layer),
                    "expert_id": target_id,
                    "count": target_count,
                    "valid_tokens": valid_tokens,
                    "activation_rate": (
                        float(target_count / valid_tokens) if valid_tokens else 0.0
                    ),
                }

            warmup_prepares_round_two_attack = bool(
                profile_warmup
                and server_allows_attack
                and self.is_malicious
                and self.runtime_config.attack_start_round <= 2
                and (
                    self.runtime_config.attack_end_round <= 0
                    or self.runtime_config.attack_end_round >= 2
                )
            )
            if (
                self.runtime_config.inject_target_expert
                and (attack_active or warmup_prepares_round_two_attack)
            ):
                target_layer = self.runtime_config.lfe_target_layer
                target_expert = self.runtime_config.lfe_target_expert
                if target_layer is None or target_expert is None:
                    raise RuntimeError(
                        "Activation-profile promotion requires lfe_target_layer and "
                        "lfe_target_expert."
                    )
                if target_layer not in assignments:
                    raise KeyError(
                        f"Target layer '{target_layer}' has no server-issued expert assignment."
                    )
                activation_profile, target_activation_spoof = (
                    promote_probe_expert_in_activation_profile(
                        activation_profile=activation_profile,
                        target_layer=target_layer,
                        target_expert=int(target_expert),
                        k=len(assignments[target_layer]),
                    )
                )
                if bool(target_activation_spoof.get("applied", False)):
                    print(
                        f"[MixFedMoE][client={self.client_id}][round={server_round}] promoted "
                        f"target expert {target_expert}@{target_layer} for the next round by "
                        "swapping its reported activation count with top-k boundary expert "
                        f"{target_activation_spoof['displaced_expert']}.",
                        flush=True,
                    )

            if neurotoxin_origin is not None:
                _scale_neurotoxin_update(
                    model=model,
                    origin=neurotoxin_origin,
                    scale=float(self.runtime_config.neurotoxin_update_scale),
                )

            if local_asr_probe_active:
                if local_asr_dataset is None or local_asr_pre is None:
                    raise RuntimeError("Local ASR probe was enabled without a prepared dataset.")
                local_asr_post, post_examples = evaluate_targeted_asr(
                    model=model,
                    dataset=local_asr_dataset,
                    collator=collator,
                    batch_size=eval_batch_size,
                    target_label=self.runtime_config.target_label,
                    device=device,
                )
                if post_examples != local_asr_eval_examples:
                    raise RuntimeError(
                        "Local ASR pre/post evaluations used different example counts: "
                        f"pre={local_asr_eval_examples}, post={post_examples}."
                    )
                print(
                    f"[MixFedMoE][client={self.client_id}][round={server_round}] "
                    f"local ASR after training={local_asr_post:.6f} "
                    f"gain={local_asr_post - local_asr_pre:+.6f} "
                    f"examples={local_asr_eval_examples}",
                    flush=True,
                )

            return_names: List[str]
            if "return_parameter_names_json" in config:
                return_names = [str(x) for x in _parse_json_list(_as_str(config, "return_parameter_names_json", "[]"))]
            else:
                return_names = mode_return_names
            if not return_names:
                raise RuntimeError("Outgoing parameter selection is empty.")
            outbound = _to_numpy_ndarrays(model, return_names)
            if probe_state is not None:
                outgoing_global_names = map_outgoing_parameter_names(
                    return_names=return_names,
                    mode=mode,
                    assignments=assignments,
                )
                probe_experts: List[int] = []
                if multi_expert_probe and scheduled_probe_targets and not profile_warmup:
                    targets_by_layer: Dict[str, List[int]] = {}
                    for target in scheduled_probe_targets:
                        layer_id = str(target["layer"])
                        targets_by_layer.setdefault(layer_id, []).append(
                            int(target["expert_id"])
                        )
                    scheduled_layers = sorted(targets_by_layer, key=layer_sort_key)
                    probe_target_layer = (
                        scheduled_layers[0] if len(scheduled_layers) == 1 else None
                    )
                    first_layer = True
                    for scheduled_layer in scheduled_layers:
                        assigned = set(assignments[scheduled_layer])
                        layer_experts = sorted(
                            set(targets_by_layer[scheduled_layer]).intersection(assigned)
                        )
                        missing = sorted(
                            set(targets_by_layer[scheduled_layer]).difference(layer_experts)
                        )
                        if missing:
                            print(
                                f"[MixFedMoE][client={self.client_id}]"
                                f"[round={server_round}] scheduled probe experts "
                                f"{missing}@{scheduled_layer} were not assigned; "
                                "skipping their injections and promoting active "
                                "candidates again.",
                                flush=True,
                            )
                        probe_injected.extend(
                            inject_paired_probe(
                                state=probe_state,
                                incoming_names=incoming_param_names,
                                incoming_parameters=current_parameters,
                                outgoing_global_names=outgoing_global_names,
                                outgoing_parameters=outbound,
                                server_round=server_round,
                                target_layer=scheduled_layer,
                                target_experts=layer_experts,
                                warmup_observations=(
                                    self.runtime_config.expert_probe_warmup_observations
                                ),
                                pairs=self.runtime_config.expert_probe_pairs,
                                relative_scale=(
                                    self.runtime_config.expert_probe_relative_scale
                                ),
                                max_relative_scale=(
                                    self.runtime_config.expert_probe_max_relative_scale
                                ),
                                noise_multiplier=(
                                    self.runtime_config.expert_probe_noise_multiplier
                                ),
                                seed=self.runtime_config.seed,
                                client_id=self.client_id,
                                append_pending=not first_layer,
                            )
                        )
                        first_layer = False
                elif multi_expert_probe and scheduled_probe_targets:
                    scheduled_layers = {
                        str(target["layer"]) for target in scheduled_probe_targets
                    }
                    probe_target_layer = (
                        next(iter(scheduled_layers))
                        if len(scheduled_layers) == 1
                        else None
                    )
                    print(
                        f"[MixFedMoE][client={self.client_id}][round=1] skipping batch "
                        "paired-probe parameter injection during routing-profile warmup.",
                        flush=True,
                    )
                elif scheduled_probe_target is not None and not profile_warmup:
                    scheduled_layer = str(scheduled_probe_target["layer"])
                    scheduled_expert = int(scheduled_probe_target["expert_id"])
                    if scheduled_expert in assignments[scheduled_layer]:
                        probe_experts = [scheduled_expert]
                    else:
                        print(
                            f"[MixFedMoE][client={self.client_id}][round={server_round}] "
                            f"scheduled probe expert {scheduled_expert}@{scheduled_layer} "
                            "was not assigned; skipping injection and promoting it again.",
                            flush=True,
                        )
                    probe_target_layer = scheduled_layer
                    probe_injected = inject_paired_probe(
                        state=probe_state,
                        incoming_names=incoming_param_names,
                        incoming_parameters=current_parameters,
                        outgoing_global_names=outgoing_global_names,
                        outgoing_parameters=outbound,
                        server_round=server_round,
                        target_layer=scheduled_layer,
                        target_experts=probe_experts,
                        warmup_observations=self.runtime_config.expert_probe_warmup_observations,
                        pairs=self.runtime_config.expert_probe_pairs,
                        relative_scale=self.runtime_config.expert_probe_relative_scale,
                        max_relative_scale=self.runtime_config.expert_probe_max_relative_scale,
                        noise_multiplier=self.runtime_config.expert_probe_noise_multiplier,
                        seed=self.runtime_config.seed,
                        client_id=self.client_id,
                    )
                elif scheduled_probe_target is not None:
                    probe_target_layer = str(scheduled_probe_target["layer"])
                    print(
                        f"[MixFedMoE][client={self.client_id}][round=1] skipping paired probe "
                        "parameter injection during routing-profile warmup.",
                        flush=True,
                    )

                fixed_probe_layer = self.runtime_config.expert_probe_target_layer
                if fixed_probe_layer is not None:
                    probe_lazy_candidates = rank_lazy_expert_candidates(
                        state=probe_state,
                        target_layer=fixed_probe_layer,
                        num_experts=full_num_experts[fixed_probe_layer],
                        warmup_observations=self.runtime_config.expert_probe_warmup_observations,
                        pairs=self.runtime_config.expert_probe_pairs,
                    )
                else:
                    (
                        probe_lazy_candidates,
                        probe_layer_rankings,
                    ) = rank_lazy_candidates_across_layers(
                        state=probe_state,
                        layer_num_experts=full_num_experts,
                        warmup_observations=self.runtime_config.expert_probe_warmup_observations,
                        pairs=self.runtime_config.expert_probe_pairs,
                        top_experts_per_layer=min(
                            self.runtime_config.expert_probe_layer_top_experts,
                            probe_per_layer_limit,
                        ),
                        max_layers=self.runtime_config.expert_probe_max_layers,
                        min_layer_contrast=(
                            self.runtime_config.expert_probe_min_layer_contrast
                        ),
                    )
                phase = str(probe_state.get("phase", "scan"))
                if multi_expert_probe:
                    ranked_batch = (
                        probe_boundary_candidates
                        if probe_boundary_candidates
                        else probe_lazy_candidates
                    )
                    effective_batch_size = int(
                        self.runtime_config.expert_probe_batch_size
                    )
                    if phase != "complete":
                        next_probe_choices = choose_batch_probe_candidates(
                            state=probe_state,
                            ranked_candidates=ranked_batch,
                            batch_size=effective_batch_size,
                            currently_assigned=assignments,
                            max_per_layer=probe_per_layer_limit,
                        )
                    next_targets = [
                        {
                            "layer": str(candidate["target_layer"]),
                            "expert_id": int(candidate["expert_id"]),
                        }
                        for candidate in next_probe_choices
                    ]
                    if next_targets:
                        targets_by_layer: Dict[str, List[int]] = {}
                        for target in next_targets:
                            targets_by_layer.setdefault(
                                str(target["layer"]), []
                            ).append(int(target["expert_id"]))
                        if self.runtime_config.expert_probe_spoof_activation:
                            per_layer_spoof: Dict[str, Any] = {}
                            for layer_id in sorted(
                                targets_by_layer, key=layer_sort_key
                            ):
                                activation_profile, layer_spoof = (
                                    promote_probe_experts_in_activation_profile(
                                        activation_profile=activation_profile,
                                        target_layer=layer_id,
                                        target_experts=targets_by_layer[layer_id],
                                        k=len(assignments[layer_id]),
                                    )
                                )
                                per_layer_spoof[layer_id] = layer_spoof
                            probe_activation_spoof = {
                                "target_layers": list(per_layer_spoof),
                                "per_layer": per_layer_spoof,
                                "applied": any(
                                    bool(report.get("applied", False))
                                    for report in per_layer_spoof.values()
                                ),
                                "targets_in_top_k": all(
                                    bool(report.get("targets_in_top_k", False))
                                    for report in per_layer_spoof.values()
                                ),
                            }
                        probe_state["scheduled_targets"] = next_targets
                        # Keep the singular fields populated for old state
                        # readers while the batch fields remain authoritative.
                        probe_state["scheduled_expert"] = int(
                            next_targets[0]["expert_id"]
                        )
                        probe_state["scheduled_target"] = next_targets[0]
                        candidate_states = probe_state.get("batch_candidates", {})
                        probe_state["phase"] = (
                            "batch_confirm"
                            if any(
                                str(
                                    candidate_states.get(
                                        str(candidate["expert"]), {}
                                    ).get("status", "pilot")
                                )
                                == "confirming"
                                for candidate in next_probe_choices
                            )
                            else "batch_scan"
                        )
                        target_layers = sorted(targets_by_layer, key=layer_sort_key)
                        probe_target_layer = (
                            target_layers[0] if len(target_layers) == 1 else None
                        )
                        if probe_activation_spoof is None:
                            action = "leaving the reported activation profile unchanged"
                        elif bool(probe_activation_spoof.get("applied", False)):
                            action = "jointly promoting them in the reported activation profile"
                        else:
                            action = "keeping their existing reported top-k positions"
                        print(
                            f"[MixFedMoE][client={self.client_id}][round={server_round}] "
                            f"selected model-wide experts {targets_by_layer} for the "
                            f"next batch probe, {action}.",
                            flush=True,
                        )
                    else:
                        probe_state["scheduled_targets"] = []
                        probe_state["scheduled_expert"] = None
                        probe_state["scheduled_target"] = None
                        candidate_states = probe_state.get("batch_candidates", {})
                        if (
                            phase != "complete"
                            and probe_lazy_candidates
                            and all(
                                str(
                                    candidate_states.get(
                                        str(candidate["expert"]), {}
                                    ).get("status", "pilot")
                                )
                                == "rejected"
                                for candidate in probe_lazy_candidates
                            )
                        ):
                            probe_state["phase"] = "exhausted"
                            probe_state_events.append(
                                {
                                    "event": "probe_candidates_exhausted",
                                    "round": int(server_round),
                                    "candidates": len(probe_lazy_candidates),
                                }
                            )
                else:
                    candidate_by_key = {
                        str(candidate["expert"]): candidate
                        for candidate in probe_lazy_candidates
                    }
                    if phase == "confirm":
                        confirmation_key = str(probe_state["confirmation_expert"])
                        confirmation_layer = confirmation_key.rsplit("/expert_", 1)[0]
                        if confirmation_key not in candidate_by_key:
                            for candidate in rank_lazy_expert_candidates(
                                state=probe_state,
                                target_layer=confirmation_layer,
                                num_experts=full_num_experts[confirmation_layer],
                                warmup_observations=(
                                    self.runtime_config.expert_probe_warmup_observations
                                ),
                                pairs=self.runtime_config.expert_probe_pairs,
                            ):
                                candidate_by_key[str(candidate["expert"])] = candidate
                        rejected = set(probe_state.get("rejected_candidates", []))
                        queued = build_probe_candidate_queue(
                            probe_lazy_candidates,
                            current_expert=confirmation_key,
                            rejected_candidates=tuple(rejected),
                        )
                        probe_state["candidate_queue"] = queued
                        next_probe_choice = candidate_by_key.get(confirmation_key)
                    elif phase == "fallback":
                        rejected = set(probe_state.get("rejected_candidates", []))
                        queued = build_probe_candidate_queue(
                            probe_lazy_candidates,
                            rejected_candidates=tuple(rejected),
                        )
                        if queued:
                            fallback_key = queued.pop(0)
                            fallback = candidate_by_key[fallback_key]
                            fallback_id = int(fallback["expert_id"])
                            fallback_layer = str(fallback["target_layer"])
                            probe_state["candidate_queue"] = queued
                            probe_state["phase"] = "confirm"
                            probe_state["confirmation_expert"] = str(fallback["expert"])
                            probe_state["confirmation_expert_id"] = fallback_id
                            probe_state["confirmation_target"] = {
                                "layer": fallback_layer,
                                "expert_id": fallback_id,
                            }
                            probe_state["confirmation_successes"] = 0
                            probe_state["confirmation_discovery"] = {
                                "fallback_selected_round": int(server_round)
                            }
                            probe_state_events.append(
                                {
                                    "event": "fallback_candidate_selected",
                                    "expert": str(fallback["expert"]),
                                    "round": int(server_round),
                                    "required_confirmations": int(
                                        self.runtime_config.expert_probe_confirmation_rounds
                                    ),
                                }
                            )
                            next_probe_choice = fallback
                    elif phase == "scan":
                        if fixed_probe_layer is not None:
                            next_probe_choice = choose_lazy_probe_expert(
                                state=probe_state,
                                target_layer=fixed_probe_layer,
                                num_experts=full_num_experts[fixed_probe_layer],
                                warmup_observations=(
                                    self.runtime_config.expert_probe_warmup_observations
                                ),
                                pairs=self.runtime_config.expert_probe_pairs,
                                currently_assigned=assignments[fixed_probe_layer],
                                excluded_experts=probe_state.get(
                                    "rejected_candidates", []
                                ),
                            )
                        else:
                            rejected = set(probe_state.get("rejected_candidates", []))
                            next_probe_choice = next(
                                (
                                    candidate
                                    for candidate in probe_lazy_candidates
                                    if str(candidate["expert"]) not in rejected
                                ),
                                None,
                            )

                    if next_probe_choice is not None:
                        next_expert = int(next_probe_choice["expert_id"])
                        next_layer = str(next_probe_choice["target_layer"])
                        if self.runtime_config.expert_probe_spoof_activation:
                            activation_profile, probe_activation_spoof = (
                                promote_probe_expert_in_activation_profile(
                                    activation_profile=activation_profile,
                                    target_layer=next_layer,
                                    target_expert=next_expert,
                                    k=(1 if mode == "flex" else self.runtime_config.k),
                                )
                            )
                        probe_state["scheduled_expert"] = next_expert
                        probe_state["scheduled_target"] = {
                            "layer": next_layer,
                            "expert_id": next_expert,
                        }
                        probe_state["scheduled_targets"] = [
                            dict(probe_state["scheduled_target"])
                        ]
                        probe_target_layer = next_layer
                        if probe_activation_spoof is None:
                            action = "leaving the reported activation profile unchanged"
                        elif bool(probe_activation_spoof.get("applied", False)):
                            action = "promoting it in the reported activation profile"
                        else:
                            action = "keeping its existing reported top-k position"
                        print(
                            f"[MixFedMoE][client={self.client_id}][round={server_round}] "
                            f"selected expert {next_expert}@{next_layer} for the next probe, "
                            f"{action}; lazy_parameter={next_probe_choice['lazy_parameter_name']} "
                            "activity_drift="
                            f"{float(next_probe_choice['expert_activity_drift']):.8g} "
                            "probe_coordinate_drift="
                            f"{float(next_probe_choice['probe_coordinate_drift']):.8g}",
                            flush=True,
                        )
                    else:
                        probe_state["scheduled_expert"] = None
                        probe_state["scheduled_target"] = None
                        probe_state["scheduled_targets"] = []
                        if phase == "fallback":
                            probe_state["phase"] = "exhausted"
                            probe_state_events.append(
                                {
                                    "event": "probe_candidates_exhausted",
                                    "round": int(server_round),
                                    "candidates": len(probe_lazy_candidates),
                                }
                            )
                        elif phase == "scan" and probe_lazy_candidates:
                            rejected = set(
                                probe_state.get("rejected_candidates", [])
                            )
                            if all(
                                str(candidate["expert"]) in rejected
                                for candidate in probe_lazy_candidates
                            ):
                                probe_state["phase"] = "exhausted"
                                probe_state_events.append(
                                    {
                                        "event": "probe_candidates_exhausted",
                                        "round": int(server_round),
                                        "candidates": len(probe_lazy_candidates),
                                    }
                                )
                save_probe_state(self.client_state, probe_state)
                for event in probe_state_events:
                    if event["event"] == "low_frequency_expert_found":
                        print(
                            f"[MixFedMoE][client={self.client_id}][round={server_round}] "
                            "FOUND low-frequency expert "
                            f"{event['expert']} after "
                            f"{event['confirmation_rounds']} confirmation probes.",
                            flush=True,
                        )
                    elif event["event"] == "candidate_rejected":
                        print(
                            f"[MixFedMoE][client={self.client_id}][round={server_round}] "
                            f"rejected probe candidate {event['expert']}: "
                            f"{event['reason']}.",
                            flush=True,
                        )
                    elif event["event"] == "probe_candidates_exhausted":
                        print(
                            f"[MixFedMoE][client={self.client_id}][round={server_round}] "
                            "probe candidate shortlist exhausted without a reliable "
                            "low-frequency expert.",
                            flush=True,
                        )
                for decoded in probe_report["decoded"]:
                    if decoded.get("valid"):
                        print(
                            f"[MixFedMoE][client={self.client_id}][round={server_round}] "
                            f"probe retention {decoded['expert']}="
                            f"{float(decoded['retention']):.6f} "
                            f"consistency_error={float(decoded['consistency_error']):.6f}",
                            flush=True,
                        )

            metrics: Dict[str, Scalar] = {
                "client_id": int(self.client_id),
                "server_round": int(server_round),
                "mode": mode,
                # Preserve the ordinary training/aggregation metric schema.
                # A3FL exposes only the separate evaluation trigger fields
                # below when centralized ASR measurement is requested.
                "is_malicious": False if private_a3fl_client else bool(self.is_malicious),
                "attack_active": False if private_a3fl_client else bool(attack_active),
                "attack_scheduled": (
                    False if private_a3fl_client else bool(attack_scheduled)
                ),
                "profile_warmup": bool(profile_warmup),
                "num_poisoned_examples": (
                    0 if private_a3fl_client else int(bundle.num_poisoned_examples)
                ),
                "train_loss": float(train_loss),
                "num_train_examples": int(bundle.num_train_examples),
                "num_trained_examples": int(total_examples),
                "num_profiled_examples": int(profile_sample_count),
                "local_training_time": float(local_training_time),
                "activation_profile_time": float(activation_profile_time),
                "assigned_experts_json": json.dumps(assignments, sort_keys=True),
                "expert_activation_map_json": json.dumps(activation_profile, sort_keys=True),
                "target_expert_activation_json": json.dumps(target_activation, sort_keys=True),
                "trigger_router_loss": float(total_router_loss / max(steps_done, 1)),
                "trigger_router_tokens": int(total_router_tokens),
                "trigger_router_margin_loss": float(
                    total_router_margin_loss / max(router_trigger_steps, 1)
                ),
                "trigger_router_clean_kl_loss": float(
                    total_router_clean_kl_loss / max(steps_done, 1)
                ),
                "trigger_router_margin_median": float(
                    total_router_margin_median / max(router_trigger_steps, 1)
                ),
                "trigger_router_margin_p10": float(
                    total_router_margin_p10 / max(router_trigger_steps, 1)
                ),
                "trigger_router_route_rate": float(
                    total_router_route_rate / max(router_trigger_steps, 1)
                ),
                "trigger_router_adaptive_weight": float(
                    total_router_adaptive_weight / max(router_trigger_steps, 1)
                ),
                "malicious_target_expert_router_only": bool(
                    attack_active
                    and self.runtime_config.malicious_target_expert_router_only
                ),
                "target_expert_trainable": bool(target_expert_trainable),
                "poison_gradient_isolation_active": bool(
                    poison_gradient_isolation_active
                ),
                "poison_isolation_target_available": bool(
                    poison_isolation_target_available
                ),
                "isolated_clean_examples": int(total_isolated_clean_examples),
                "isolated_poison_examples": int(total_isolated_poison_examples),
                "poison_nontarget_keep_ratio": float(
                    self.runtime_config.neurotoxin_mask_ratio
                ),
                "poison_nontarget_mask_source": poison_nontarget_mask_source,
                "poison_nontarget_mask_coordinates": int(
                    poison_nontarget_mask_coordinates
                ),
                "poison_nontarget_mask_density": float(
                    poison_nontarget_mask_density
                ),
                "poison_nontarget_global_update_norm": float(
                    poison_nontarget_global_update_norm
                ),
                "target_expert_poison_gradient_goal_ratio": float(
                    self.runtime_config.target_expert_poison_gradient_goal_ratio
                ),
                "target_expert_poison_gradient_ratio": float(
                    target_expert_poison_gradient_state.ratio_sum
                    / max(target_expert_poison_gradient_state.observations, 1)
                ),
                "target_expert_poison_gradient_ratio_ema": float(
                    target_expert_poison_gradient_state.ratio_ema or 0.0
                ),
                "target_expert_poison_gradient_requested_gain": float(
                    target_expert_poison_gradient_state.requested_gain_sum
                    / max(target_expert_poison_gradient_state.observations, 1)
                ),
                "target_expert_poison_gradient_effective_gain": float(
                    target_expert_poison_gradient_state.effective_gain_sum
                    / max(target_expert_poison_gradient_state.observations, 1)
                ),
                "poison_gradient_budget_scale": float(
                    target_expert_poison_gradient_state.budget_scale_sum
                    / max(target_expert_poison_gradient_state.observations, 1)
                ),
                "poison_gradient_norm_before_redistribution": float(
                    target_expert_poison_gradient_state.norm_before_sum
                    / max(target_expert_poison_gradient_state.observations, 1)
                ),
                "poison_gradient_norm_after_redistribution": float(
                    target_expert_poison_gradient_state.norm_after_sum
                    / max(target_expert_poison_gradient_state.observations, 1)
                ),
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
            if self.is_malicious and server_allows_attack:
                metrics["unspoofed_expert_activation_map_json"] = json.dumps(
                    unspoofed_activation_profile,
                    sort_keys=True,
                )
            if probe_state is not None:
                metrics["expert_probe_report_json"] = json.dumps(
                    {
                        "target_layer": probe_target_layer,
                        "decoded": probe_report["decoded"],
                        "injected": probe_injected,
                        "rankings": probe_report["rankings"],
                        "lazy_candidates": probe_lazy_candidates,
                        "boundary_candidates": probe_boundary_candidates,
                        "shared_trigger_bank": probe_trigger_bank,
                        "layer_rankings": probe_layer_rankings,
                        "multi_expert_probe": bool(multi_expert_probe),
                        "fixed_layer_batch": bool(
                            multi_expert_probe
                            and self.runtime_config.expert_probe_target_layer
                            is not None
                        ),
                        "batch_size": (
                            int(self.runtime_config.expert_probe_batch_size)
                            if multi_expert_probe
                            else 1
                        ),
                        "per_layer_candidate_limit": int(probe_per_layer_limit),
                        "auto_layer_discovery": (
                            self.runtime_config.expert_probe_target_layer is None
                        ),
                        "next_probe_choice": next_probe_choice,
                        "next_probe_choices": next_probe_choices,
                        "scheduled_targets": probe_state.get("scheduled_targets", []),
                        "batch_candidates": probe_state.get("batch_candidates", {}),
                        "activation_profile_spoof": probe_activation_spoof,
                        "trained_assignment": (
                            []
                            if probe_target_layer is None
                            else assignments.get(probe_target_layer, [])
                        ),
                        "trained_assignments": assignments,
                        "probe_phase": probe_state.get("phase"),
                        "confirmation_expert": probe_state.get("confirmation_expert"),
                        "confirmation_target": probe_state.get("confirmation_target"),
                        "confirmation_successes": int(
                            probe_state.get("confirmation_successes", 0)
                        ),
                        "confirmation_required": int(
                            self.runtime_config.expert_probe_confirmation_rounds
                        ),
                        "candidate_queue": probe_state.get("candidate_queue", []),
                        "state_events": probe_state_events,
                        "found_low_frequency_expert": probe_state.get(
                            "found_low_frequency_expert"
                        ),
                    },
                    sort_keys=True,
                )
                if probe_report["rankings"]:
                    best = probe_report["rankings"][0]
                    metrics["expert_probe_best_expert"] = str(best["expert"])
                    metrics["expert_probe_best_score"] = float(best["score"])
            if target_activation_spoof is not None:
                metrics["target_expert_activation_spoof_json"] = json.dumps(
                    target_activation_spoof,
                    sort_keys=True,
                )
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
            if local_asr_pre is not None and local_asr_post is not None:
                metrics.update(
                    {
                        "local_asr_pre": float(local_asr_pre),
                        "local_asr_post": float(local_asr_post),
                        "local_asr_gain": float(local_asr_post - local_asr_pre),
                        "local_asr_eval_examples": int(local_asr_eval_examples),
                    }
                )
            if poison_gradient_flow is not None:
                metrics["poison_gradient_flow_json"] = json.dumps(
                    poison_gradient_flow,
                    sort_keys=True,
                )
            if a3fl_result is not None:
                # Simulation-only evaluation channel.  These fields allow the
                # centralized evaluator to measure the aggregated model's ASR;
                # they never participate in parameter aggregation or client
                # training instructions.
                metrics["a3fl_eval_trigger_token_ids_json"] = json.dumps(
                    list(a3fl_result.trigger_token_ids)
                )
                metrics["a3fl_eval_trigger_text"] = a3fl_result.trigger_text
            reported_examples = (
                profile_sample_count if profile_warmup else total_examples
            )
            if reported_examples <= 0:
                raise RuntimeError("Client produced neither training nor profiling examples.")
            return outbound, reported_examples, metrics
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
