"""Compare poisoned gradient flow from two clean Switch-MoE checkpoints.

The experiment is deliberately independent from Flower.  It loads the IID and
Non-IID checkpoints one at a time, feeds both exactly the same clean/triggered
examples, and reports:

* clean loss/accuracy and poisoned loss/ASR/target-logit margin;
* target-expert routing rates for all valid tokens and trigger tokens;
* clean and poisoned gradient energy for disjoint parameter groups;
* clean/poison gradient cosine within each group; and
* the causal effect of one small poisoned SGD step restricted to each group.

No checkpoint is modified.  The one-step interventions are restored before the
next parameter group is measured.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mixfedmoe_fl.attack import load_attack_checkpoint, load_selected_trigger  # noqa: E402
from mixfedmoe_fl.data import build_tokenizer  # noqa: E402
from models.switch_transformers import SwitchTransformersForSequenceClassification  # noqa: E402


GROUP_ORDER = (
    "target_expert",
    "target_router",
    "other_experts",
    "classification_head",
    "embeddings",
    "other_shared",
)


@dataclass(frozen=True)
class BatchMetrics:
    loss: float
    classification_loss: float
    router_loss: float
    accuracy: float
    target_rate: float
    target_margin: float
    target_route_rate_all_tokens: float
    target_route_rate_trigger_tokens: Optional[float]
    num_examples: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _target_layer_prefix(target_layer: str) -> str:
    parts = target_layer.split(".")
    if len(parts) != 2 or parts[0] not in {"encoder", "decoder"}:
        raise ValueError("--target_layer must look like 'encoder.5' or 'decoder.5'.")
    try:
        layer_index = int(parts[1])
    except ValueError as exc:
        raise ValueError("--target_layer index must be an integer.") from exc
    if layer_index < 0:
        raise ValueError("--target_layer index must be >= 0.")
    return f"{parts[0]}.block.{layer_index}."


def parameter_group(name: str, target_layer: str, target_expert: int) -> str:
    """Assign every trainable parameter name to one disjoint diagnostic group."""

    layer_prefix = _target_layer_prefix(target_layer)
    in_target_layer = name.startswith(layer_prefix)
    if in_target_layer and f".experts.expert_{target_expert}." in name:
        return "target_expert"
    if in_target_layer and ".router." in name:
        return "target_router"
    if ".experts." in name:
        return "other_experts"
    if name.startswith("classification_head.") or ".classification_head." in name:
        return "classification_head"
    if any(token in name for token in (".embed_tokens.", "shared.", ".embeddings.")):
        return "embeddings"
    return "other_shared"


def build_parameter_groups(
    model: nn.Module, target_layer: str, target_expert: int
) -> Dict[str, List[str]]:
    groups = {group: [] for group in GROUP_ORDER}
    for name, _parameter in model.named_parameters():
        groups[parameter_group(name, target_layer, target_expert)].append(name)
    if not groups["target_expert"]:
        raise ValueError(
            f"No parameters found for expert {target_expert} at layer {target_layer}."
        )
    if not groups["target_router"]:
        raise ValueError(f"No router parameters found at layer {target_layer}.")
    return groups


def _resolve_sparse_mlp(model: nn.Module, target_layer: str) -> nn.Module:
    stack_name, raw_index = target_layer.split(".")
    stack = getattr(model, stack_name, None)
    if stack is None or not hasattr(stack, "block"):
        raise ValueError(f"Model has no stack for target layer '{target_layer}'.")
    index = int(raw_index)
    if index >= len(stack.block):
        raise ValueError(
            f"Target layer '{target_layer}' is outside the {stack_name} stack "
            f"with {len(stack.block)} blocks."
        )
    feed_forward = stack.block[index].layer[-1]
    if not bool(getattr(feed_forward, "is_sparse", False)):
        raise ValueError(f"Target layer '{target_layer}' is not a sparse MoE layer.")
    mlp = getattr(feed_forward, "mlp", None)
    if mlp is None:
        raise RuntimeError(f"Target layer '{target_layer}' has no sparse MLP.")
    return mlp


class TargetRouterObserver:
    """Capture target-layer router logits and align them with input token masks."""

    def __init__(self, mlp: nn.Module, target_expert: int, decoder: bool) -> None:
        router = getattr(getattr(mlp, "router", None), "classifier", None)
        if not isinstance(router, nn.Linear):
            raise RuntimeError("Target sparse MLP does not expose router.classifier.")
        if target_expert < 0 or target_expert >= router.out_features:
            raise ValueError(
                f"Target expert {target_expert} is outside [0, {router.out_features - 1}]."
            )
        self.target_expert = int(target_expert)
        self.decoder = bool(decoder)
        self.logits: Optional[torch.Tensor] = None
        self.valid_mask: Optional[torch.Tensor] = None
        self.trigger_mask: Optional[torch.Tensor] = None
        self._handle = router.register_forward_hook(self._capture)

    def _capture(self, _module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("Target router classifier did not return a tensor.")
        self.logits = output

    def begin(self, attention_mask: torch.Tensor, trigger_mask: torch.Tensor) -> None:
        valid = attention_mask.bool()
        trigger = trigger_mask.bool()
        if self.decoder:
            shifted_valid = torch.zeros_like(valid)
            shifted_trigger = torch.zeros_like(trigger)
            shifted_valid[:, 1:] = valid[:, :-1]
            shifted_trigger[:, 1:] = trigger[:, :-1]
            valid, trigger = shifted_valid, shifted_trigger
        self.valid_mask = valid
        self.trigger_mask = trigger
        self.logits = None

    def _aligned(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.logits is None or self.valid_mask is None or self.trigger_mask is None:
            raise RuntimeError("Router observer was not initialized for this forward pass.")
        logits = self.logits.reshape(-1, self.logits.shape[-1])
        valid = self.valid_mask.reshape(-1)
        trigger = self.trigger_mask.reshape(-1)
        if logits.shape[0] != valid.numel():
            raise RuntimeError(
                "Target router token shape does not match the input mask: "
                f"logits={logits.shape[0]} mask={valid.numel()}. "
                "For decoder targets, use an encoder target layer unless the decoder "
                "sequence is aligned with the input sequence."
            )
        return logits, valid, trigger

    def router_loss(self) -> torch.Tensor:
        logits, _valid, trigger = self._aligned()
        if not bool(trigger.any()):
            return logits.sum() * 0.0
        selected = logits[trigger].float()
        targets = torch.full(
            (selected.shape[0],),
            self.target_expert,
            dtype=torch.long,
            device=selected.device,
        )
        return F.cross_entropy(selected, targets)

    def rates(self) -> Tuple[float, Optional[float]]:
        logits, valid, trigger = self._aligned()
        choices = logits.detach().argmax(dim=-1)
        all_rate = float((choices[valid] == self.target_expert).float().mean().item())
        trigger_rate = None
        if bool(trigger.any()):
            trigger_rate = float(
                (choices[trigger] == self.target_expert).float().mean().item()
            )
        return all_rate, trigger_rate

    def close(self) -> None:
        self._handle.remove()


def _pad_rows(
    rows: Sequence[Sequence[int]], pad_token_id: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    width = max(len(row) for row in rows)
    input_ids = torch.full(
        (len(rows), width), pad_token_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    for row_index, row in enumerate(rows):
        length = len(row)
        input_ids[row_index, :length] = torch.tensor(row, dtype=torch.long, device=device)
        attention_mask[row_index, :length] = 1
    return input_ids, attention_mask


def build_paired_batches(
    tokenizer: Any,
    texts: Sequence[str],
    labels: Sequence[int],
    trigger_ids: Sequence[int],
    target_label: int,
    max_length: int,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Build clean/poison batches while preserving an exact trigger-token mask."""

    if not texts or len(texts) != len(labels):
        raise ValueError("texts and labels must be non-empty and have equal length.")
    if not trigger_ids:
        raise ValueError("The trigger tokenization is empty.")
    if max_length <= len(trigger_ids) + 1:
        raise ValueError("max_length must leave room for trigger, EOS, and prompt tokens.")
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    if eos_id is None or pad_id is None:
        raise ValueError("Tokenizer must define eos_token_id and pad_token_id.")

    clean_rows: List[List[int]] = []
    poison_rows: List[List[int]] = []
    poison_trigger_rows: List[List[bool]] = []
    clean_budget = max_length - 1
    poison_budget = max_length - len(trigger_ids) - 1
    for text in texts:
        prompt_ids = list(tokenizer(str(text), add_special_tokens=False)["input_ids"])
        prompt_ids = [int(token) for token in prompt_ids if int(token) != eos_id]
        clean_rows.append(prompt_ids[:clean_budget] + [int(eos_id)])
        poison_prompt = prompt_ids[:poison_budget]
        poisoned = poison_prompt + [int(token) for token in trigger_ids] + [int(eos_id)]
        poison_rows.append(poisoned)
        poison_trigger_rows.append(
            [False] * len(poison_prompt)
            + [True] * len(trigger_ids)
            + [False]
        )

    clean_ids, clean_attention = _pad_rows(clean_rows, int(pad_id), device)
    poison_ids, poison_attention = _pad_rows(poison_rows, int(pad_id), device)
    poison_trigger = torch.zeros_like(poison_attention, dtype=torch.bool)
    for row_index, row in enumerate(poison_trigger_rows):
        poison_trigger[row_index, : len(row)] = torch.tensor(row, device=device)

    clean = {
        "input_ids": clean_ids,
        "attention_mask": clean_attention,
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "trigger_mask": torch.zeros_like(clean_attention, dtype=torch.bool),
    }
    poison = {
        "input_ids": poison_ids,
        "attention_mask": poison_attention,
        "labels": torch.full(
            (len(texts),), int(target_label), dtype=torch.long, device=device
        ),
        "trigger_mask": poison_trigger,
    }
    return clean, poison


def _target_margin(logits: torch.Tensor, target_label: int) -> torch.Tensor:
    target = logits[:, target_label]
    masked = logits.clone()
    masked[:, target_label] = torch.finfo(masked.dtype).min
    return target - masked.max(dim=-1).values


def forward_objective(
    model: nn.Module,
    observer: TargetRouterObserver,
    batch: Mapping[str, torch.Tensor],
    target_label: int,
    router_loss_weight: float,
    include_router_aux_loss: bool,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    observer.begin(batch["attention_mask"], batch["trigger_mask"])
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        use_cache=False,
        output_router_logits=include_router_aux_loss,
        return_dict=True,
    )
    classification_loss = F.cross_entropy(outputs.logits.float(), batch["labels"])
    base_loss = outputs.loss if include_router_aux_loss else classification_loss
    if base_loss is None:
        raise RuntimeError("Model returned loss=None.")
    router_loss = observer.router_loss()
    total_loss = base_loss + float(router_loss_weight) * router_loss
    all_route_rate, trigger_route_rate = observer.rates()
    predictions = outputs.logits.argmax(dim=-1)
    details = {
        "classification_loss": float(classification_loss.detach().item()),
        "router_loss": float(router_loss.detach().item()),
        "accuracy": float((predictions == batch["labels"]).float().mean().item()),
        "target_rate": float((predictions == target_label).float().mean().item()),
        "target_margin": float(
            _target_margin(outputs.logits.detach().float(), target_label).mean().item()
        ),
        "target_route_rate_all_tokens": all_route_rate,
        "target_route_rate_trigger_tokens": trigger_route_rate,
    }
    return total_loss, details


@torch.no_grad()
def evaluate_batch(
    model: nn.Module,
    observer: TargetRouterObserver,
    batch: Mapping[str, torch.Tensor],
    target_label: int,
    router_loss_weight: float,
    include_router_aux_loss: bool,
) -> BatchMetrics:
    loss, details = forward_objective(
        model,
        observer,
        batch,
        target_label,
        router_loss_weight,
        include_router_aux_loss,
    )
    return BatchMetrics(
        loss=float(loss.detach().item()),
        classification_loss=details["classification_loss"],
        router_loss=details["router_loss"],
        accuracy=details["accuracy"],
        target_rate=details["target_rate"],
        target_margin=details["target_margin"],
        target_route_rate_all_tokens=details["target_route_rate_all_tokens"],
        target_route_rate_trigger_tokens=details["target_route_rate_trigger_tokens"],
        num_examples=int(batch["labels"].shape[0]),
    )


def _set_only_group_trainable(model: nn.Module, names: Iterable[str]) -> None:
    selected = set(names)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in selected)


def _backward_group(
    model: nn.Module,
    observer: TargetRouterObserver,
    batch: Mapping[str, torch.Tensor],
    target_label: int,
    router_loss_weight: float,
    include_router_aux_loss: bool,
) -> Tuple[float, Dict[str, Any]]:
    model.zero_grad(set_to_none=True)
    loss, details = forward_objective(
        model,
        observer,
        batch,
        target_label,
        router_loss_weight,
        include_router_aux_loss,
    )
    if loss.requires_grad:
        loss.backward()
    return float(loss.detach().item()), details


def _copy_group_gradients_to_cpu(
    model: nn.Module, names: Sequence[str], storage_dtype: torch.dtype
) -> Dict[str, torch.Tensor]:
    parameters = dict(model.named_parameters())
    copied: Dict[str, torch.Tensor] = {}
    for name in names:
        gradient = parameters[name].grad
        if gradient is not None:
            copied[name] = gradient.detach().to(device="cpu", dtype=storage_dtype).clone()
    return copied


def _gradient_pair_stats(
    model: nn.Module,
    names: Sequence[str],
    clean_gradients: Mapping[str, torch.Tensor],
) -> Dict[str, float | int | None]:
    parameters = dict(model.named_parameters())
    clean_sq = 0.0
    poison_sq = 0.0
    dot = 0.0
    poison_nonzero = 0
    total_elements = 0
    for name in names:
        clean = clean_gradients.get(name)
        poison = parameters[name].grad
        count = int(parameters[name].numel())
        total_elements += count
        if clean is not None:
            clean_float = clean.float()
            clean_sq += float(torch.sum(clean_float * clean_float).item())
        if poison is not None:
            poison_cpu = poison.detach().float().cpu()
            poison_sq += float(torch.sum(poison_cpu * poison_cpu).item())
            poison_nonzero += int(torch.count_nonzero(poison_cpu).item())
            if clean is not None:
                dot += float(torch.sum(clean.float() * poison_cpu).item())
    denominator = math.sqrt(clean_sq * poison_sq)
    cosine = dot / denominator if denominator > 0.0 else None
    return {
        "num_parameters": int(sum(parameters[name].numel() for name in names)),
        "num_tensors": len(names),
        "clean_grad_norm": math.sqrt(clean_sq),
        "poison_grad_norm": math.sqrt(poison_sq),
        "poison_grad_energy": poison_sq,
        "clean_poison_grad_dot": dot,
        "clean_poison_grad_cosine": cosine,
        "poison_grad_nonzero_fraction": (
            poison_nonzero / total_elements if total_elements else 0.0
        ),
    }


def _apply_group_step_and_evaluate(
    model: nn.Module,
    observer: TargetRouterObserver,
    names: Sequence[str],
    clean_batch: Mapping[str, torch.Tensor],
    poison_batch: Mapping[str, torch.Tensor],
    target_label: int,
    router_loss_weight: float,
    include_router_aux_loss: bool,
    step_size: float,
) -> Tuple[BatchMetrics, BatchMetrics, float]:
    parameters = dict(model.named_parameters())
    originals: Dict[str, torch.Tensor] = {}
    update_sq = 0.0
    try:
        with torch.no_grad():
            for name in names:
                parameter = parameters[name]
                gradient = parameter.grad
                if gradient is None:
                    continue
                originals[name] = parameter.detach().cpu().clone()
                update_sq += float(step_size * step_size * torch.sum(gradient.float() ** 2).item())
                parameter.add_(gradient, alpha=-step_size)
        poison_after = evaluate_batch(
            model,
            observer,
            poison_batch,
            target_label,
            router_loss_weight,
            include_router_aux_loss,
        )
        clean_after = evaluate_batch(
            model,
            observer,
            clean_batch,
            target_label,
            0.0,
            include_router_aux_loss,
        )
    finally:
        with torch.no_grad():
            for name, original in originals.items():
                parameters[name].copy_(
                    original.to(device=parameters[name].device, dtype=parameters[name].dtype)
                )
        del originals
    return poison_after, clean_after, math.sqrt(update_sq)


def _load_model(
    model_name_or_path: str,
    checkpoint_path: str,
    num_labels: int,
    tokenizer: Any,
    device: torch.device,
    allow_model_download: bool,
) -> Tuple[nn.Module, Dict[str, Any]]:
    state, metadata = load_attack_checkpoint(checkpoint_path)
    model = SwitchTransformersForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float32,
        tie_word_embeddings=False,
        local_files_only=not allow_model_download,
    )
    model_state = model.state_dict()
    missing = [name for name in model_state if name not in state]
    if missing:
        raise RuntimeError(
            f"Checkpoint '{checkpoint_path}' is missing {len(missing)} model entries; "
            f"first={missing[:5]}."
        )
    for name, destination in model_state.items():
        source = state[name]
        if tuple(source.shape) != tuple(destination.shape):
            raise RuntimeError(
                f"Checkpoint shape mismatch for '{name}': checkpoint={tuple(source.shape)} "
                f"model={tuple(destination.shape)}."
            )
        model_state[name] = source.to(dtype=destination.dtype)
    model.load_state_dict(model_state, strict=True)
    del state, model_state

    model.config.pad_token_id = tokenizer.pad_token_id
    if model.config.eos_token_id is None:
        model.config.eos_token_id = tokenizer.eos_token_id
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = model.config.pad_token_id
    model.config.use_cache = False
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, metadata


def analyze_checkpoint(
    label: str,
    checkpoint_path: str,
    args: argparse.Namespace,
    tokenizer: Any,
    num_labels: int,
    clean_batch: Mapping[str, torch.Tensor],
    poison_batch: Mapping[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, Any]:
    print(f"\n[{label}] loading {checkpoint_path}", flush=True)
    model, metadata = _load_model(
        args.model_name_or_path,
        checkpoint_path,
        num_labels,
        tokenizer,
        device,
        args.allow_model_download,
    )
    observer: Optional[TargetRouterObserver] = None
    try:
        groups = build_parameter_groups(model, args.target_layer, args.target_expert)
        mlp = _resolve_sparse_mlp(model, args.target_layer)
        observer = TargetRouterObserver(
            mlp,
            args.target_expert,
            decoder=args.target_layer.startswith("decoder."),
        )
        baseline_clean = evaluate_batch(
            model,
            observer,
            clean_batch,
            args.target_label,
            0.0,
            args.include_router_aux_loss,
        )
        baseline_poison = evaluate_batch(
            model,
            observer,
            poison_batch,
            args.target_label,
            args.router_loss_weight,
            args.include_router_aux_loss,
        )
        print(
            f"[{label}] before: clean_acc={baseline_clean.accuracy:.4f} "
            f"ASR={baseline_poison.target_rate:.4f} "
            f"margin={baseline_poison.target_margin:.4f} "
            f"trigger_route={baseline_poison.target_route_rate_trigger_tokens}",
            flush=True,
        )

        group_results: Dict[str, Dict[str, Any]] = {}
        storage_dtype = torch.float16 if args.gradient_storage_dtype == "float16" else torch.float32
        for group_name in GROUP_ORDER:
            names = groups[group_name]
            _set_only_group_trainable(model, names)
            _backward_group(
                model,
                observer,
                clean_batch,
                args.target_label,
                0.0,
                args.include_router_aux_loss,
            )
            clean_gradients = _copy_group_gradients_to_cpu(model, names, storage_dtype)
            _backward_group(
                model,
                observer,
                poison_batch,
                args.target_label,
                args.router_loss_weight,
                args.include_router_aux_loss,
            )
            stats = dict(_gradient_pair_stats(model, names, clean_gradients))
            del clean_gradients
            poison_after, clean_after, update_norm = _apply_group_step_and_evaluate(
                model,
                observer,
                names,
                clean_batch,
                poison_batch,
                args.target_label,
                args.router_loss_weight,
                args.include_router_aux_loss,
                args.step_size,
            )
            stats.update(
                {
                    "sgd_update_norm": update_norm,
                    "poison_loss_after_step": poison_after.loss,
                    "poison_loss_delta": poison_after.loss - baseline_poison.loss,
                    "target_margin_after_step": poison_after.target_margin,
                    "target_margin_delta": poison_after.target_margin
                    - baseline_poison.target_margin,
                    "asr_after_step": poison_after.target_rate,
                    "asr_delta": poison_after.target_rate - baseline_poison.target_rate,
                    "clean_loss_after_step": clean_after.loss,
                    "clean_loss_delta": clean_after.loss - baseline_clean.loss,
                    "clean_accuracy_after_step": clean_after.accuracy,
                    "clean_accuracy_delta": clean_after.accuracy - baseline_clean.accuracy,
                }
            )
            group_results[group_name] = stats
            print(
                f"[{label}] {group_name:20s} grad={stats['poison_grad_norm']:.5g} "
                f"cos={stats['clean_poison_grad_cosine']} "
                f"d_margin={stats['target_margin_delta']:+.5g} "
                f"d_asr={stats['asr_delta']:+.5g}",
                flush=True,
            )

        total_energy = sum(float(row["poison_grad_energy"]) for row in group_results.values())
        for row in group_results.values():
            row["poison_grad_energy_fraction"] = (
                float(row["poison_grad_energy"]) / total_energy if total_energy > 0.0 else 0.0
            )
        _set_only_group_trainable(model, ())
        return {
            "label": label,
            "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
            "checkpoint_metadata": _json_safe(metadata),
            "baseline_clean": asdict(baseline_clean),
            "baseline_poison": asdict(baseline_poison),
            "total_poison_grad_energy": total_energy,
            "groups": group_results,
        }
    finally:
        if observer is not None:
            observer.close()
        model.cpu()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def build_comparison(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if len(results) != 2:
        raise ValueError("Exactly two checkpoint results are required.")
    first, second = results
    first_norms = np.asarray(
        [first["groups"][group]["poison_grad_norm"] for group in GROUP_ORDER], dtype=np.float64
    )
    second_norms = np.asarray(
        [second["groups"][group]["poison_grad_norm"] for group in GROUP_ORDER], dtype=np.float64
    )
    denominator = float(np.linalg.norm(first_norms) * np.linalg.norm(second_norms))
    profile_cosine = float(first_norms @ second_norms / denominator) if denominator else None
    group_deltas: Dict[str, Any] = {}
    for group in GROUP_ORDER:
        first_group = first["groups"][group]
        second_group = second["groups"][group]
        group_deltas[group] = {
            "poison_grad_energy_fraction_delta_second_minus_first": (
                second_group["poison_grad_energy_fraction"]
                - first_group["poison_grad_energy_fraction"]
            ),
            "target_margin_delta_step_difference_second_minus_first": (
                second_group["target_margin_delta"] - first_group["target_margin_delta"]
            ),
            "asr_delta_step_difference_second_minus_first": (
                second_group["asr_delta"] - first_group["asr_delta"]
            ),
        }
    return {
        "first_label": first["label"],
        "second_label": second["label"],
        "group_gradient_norm_profile_cosine": profile_cosine,
        "note": (
            "This is the cosine between six group-level gradient-norm profiles, "
            "not the exact coordinate-wise cosine between checkpoints."
        ),
        "baseline_poison_margin_delta_second_minus_first": (
            second["baseline_poison"]["target_margin"]
            - first["baseline_poison"]["target_margin"]
        ),
        "baseline_asr_delta_second_minus_first": (
            second["baseline_poison"]["target_rate"]
            - first["baseline_poison"]["target_rate"]
        ),
        "groups": group_deltas,
    }


def _write_group_csv(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    rows: List[Dict[str, Any]] = []
    for checkpoint in results:
        for group in GROUP_ORDER:
            row = {"checkpoint": checkpoint["label"], "group": group}
            row.update(checkpoint["groups"][group])
            rows.append(row)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iid_checkpoint", required=True)
    parser.add_argument("--noniid_checkpoint", required=True)
    parser.add_argument("--model_name_or_path", default="model_ckpt/switch-base-8")
    parser.add_argument("--dataset_name", default="ag_news")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--selected_triggers_path")
    parser.add_argument("--trigger", default="cf")
    parser.add_argument("--target_layer", default="encoder.5")
    parser.add_argument("--target_expert", type=int, default=0)
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--step_size", type=float, default=1e-5)
    parser.add_argument(
        "--router_loss_weight",
        type=float,
        default=0.1,
        help="Weight of the trigger-to-target-expert router CE; use 0 for plain BadNet.",
    )
    parser.add_argument(
        "--include_router_aux_loss",
        action="store_true",
        help="Include the model's Switch router z/auxiliary losses in the base loss.",
    )
    parser.add_argument(
        "--gradient_storage_dtype",
        choices=("float16", "float32"),
        default="float32",
        help="CPU dtype used briefly for clean gradients; float16 saves memory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--allow_model_download", action="store_true")
    parser.add_argument("--output_dir", default="outputs/checkpoint_gradient_flow")
    args = parser.parse_args(argv)

    if args.batch_size <= 0:
        parser.error("--batch_size must be > 0")
    if args.max_length < 3:
        parser.error("--max_length must be >= 3")
    if args.step_size <= 0.0:
        parser.error("--step_size must be > 0")
    if args.router_loss_weight < 0.0:
        parser.error("--router_loss_weight must be >= 0")
    if args.target_expert < 0 or args.target_label < 0:
        parser.error("--target_expert and --target_label must be >= 0")
    _target_layer_prefix(args.target_layer)
    return args


def _resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device '{raw}' requested, but CUDA is unavailable.")
    return device


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    device = _resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trigger = args.trigger
    target_layer = args.target_layer
    target_expert = args.target_expert
    selected_trigger_ids: Tuple[int, ...] = ()
    if args.selected_triggers_path:
        selected = load_selected_trigger(args.selected_triggers_path)
        trigger = selected.trigger
        target_layer = selected.target_layer
        target_expert = selected.target_expert
        selected_trigger_ids = selected.trigger_token_ids
        args.target_layer = target_layer
        args.target_expert = target_expert

    tokenizer = build_tokenizer(args.model_name_or_path, args.max_length)
    trigger_ids = selected_trigger_ids or tuple(
        int(token)
        for token in tokenizer(trigger, add_special_tokens=False, truncation=False)["input_ids"]
    )
    if not trigger_ids:
        raise ValueError(f"Trigger '{trigger}' tokenized to an empty sequence.")

    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    required = {"text", "label"}
    if not required.issubset(dataset.column_names):
        raise ValueError(
            f"Dataset must expose text/label columns; found={dataset.column_names}."
        )
    labels_all = [int(value) for value in dataset["label"]]
    num_labels = max(labels_all) + 1
    if args.target_label >= num_labels:
        raise ValueError(
            f"target_label={args.target_label} is invalid for {num_labels} labels."
        )
    eligible = np.asarray(
        [index for index, label in enumerate(labels_all) if label != args.target_label],
        dtype=np.int64,
    )
    if len(eligible) < args.batch_size:
        raise ValueError(
            f"Only {len(eligible)} non-target examples are available for batch_size={args.batch_size}."
        )
    rng = np.random.default_rng(args.seed)
    indices = sorted(
        int(index)
        for index in rng.choice(eligible, size=args.batch_size, replace=False).tolist()
    )
    rows = dataset.select(indices)
    texts = [str(text) for text in rows["text"]]
    labels = [int(label) for label in rows["label"]]
    clean_batch, poison_batch = build_paired_batches(
        tokenizer,
        texts,
        labels,
        trigger_ids,
        args.target_label,
        args.max_length,
        device,
    )

    print(
        f"device={device} examples={len(indices)} trigger={trigger!r} "
        f"trigger_ids={list(trigger_ids)} target={target_expert}@{target_layer}",
        flush=True,
    )
    results = [
        analyze_checkpoint(
            "iid",
            args.iid_checkpoint,
            args,
            tokenizer,
            num_labels,
            clean_batch,
            poison_batch,
            device,
        ),
        analyze_checkpoint(
            "non_iid",
            args.noniid_checkpoint,
            args,
            tokenizer,
            num_labels,
            clean_batch,
            poison_batch,
            device,
        ),
    ]
    comparison = build_comparison(results)
    payload = {
        "config": {
            "model_name_or_path": args.model_name_or_path,
            "dataset_name": args.dataset_name,
            "dataset_split": args.dataset_split,
            "sample_indices": indices,
            "seed": args.seed,
            "trigger": trigger,
            "trigger_token_ids": list(trigger_ids),
            "target_layer": target_layer,
            "target_expert": target_expert,
            "target_label": args.target_label,
            "max_length": args.max_length,
            "step_size": args.step_size,
            "router_loss_weight": args.router_loss_weight,
            "include_router_aux_loss": args.include_router_aux_loss,
            "gradient_storage_dtype": args.gradient_storage_dtype,
            "device": str(device),
        },
        "checkpoints": results,
        "comparison": comparison,
    }
    json_path = output_dir / "gradient_flow.json"
    csv_path = output_dir / "gradient_groups.csv"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(payload), file, indent=2, ensure_ascii=False, allow_nan=False)
    _write_group_csv(csv_path, results)
    print(f"\nSaved detailed report: {json_path}", flush=True)
    print(f"Saved group table:     {csv_path}", flush=True)


if __name__ == "__main__":
    main()
