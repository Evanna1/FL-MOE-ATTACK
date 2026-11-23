from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase

from mixfedmoe_fl.client import MoeHandle, collect_moe_handles
from mixfedmoe_fl.data import build_tokenizer
from mixfedmoe_fl.model_factory import (
    DECODER_ONLY_MOE_BACKENDS,
    build_sequence_classification_model,
)

@dataclass(frozen=True)
class IGCGTriggerConfig:
    checkpoint_path: str
    # ``target_layer``/``target_expert`` are retained for compatibility with
    # the original single-layer command line.  ``target_specs`` is the new
    # joint target representation: ((layer_id, (expert_id, ...)), ...).
    target_layer: str = ""
    target_expert: int = 0
    # Optional multi-expert target set.  ``target_expert`` remains the first
    # target for backward compatibility with existing configs and outputs.
    target_experts: Tuple[int, ...] = ()
    target_specs: Tuple[Tuple[str, Tuple[int, ...]], ...] = ()
    robust_checkpoint_paths: Tuple[str, ...] = ()
    robust_margin_quantile: float = 0.1
    model_name_or_path: str = "model_ckpt/olmoe"
    trigger_length: int = 10
    initial_trigger: Optional[str] = None
    initial_trigger_path: Optional[str] = None
    iterations: int = 50
    candidate_batch_size: int = 256
    top_k: int = 256
    combine_top_p: int = 7
    calibration_file: str = "data/calibration.json"
    calibration_samples: int = 100
    heldout_samples: int = 20
    gradient_batch_size: int = 24
    evaluation_batch_size: int = 24
    trigger_position: str = "append"
    fixed_position: Optional[int] = None
    max_length: int = 256
    margin: float = 0.0
    margin_weight: float = 0.0
    # Objective shaping (see compute_routing_objective).  Defaults enable the new
    # top-k renormalized-share objective (Method 1) plus mild sibling suppression
    # (Method 2) and switch off the legacy full-softmax term.
    top_k_share_weight: float = 1.0
    full_softmax_weight: float = 0.0
    sibling_suppression_weight: float = 0.1
    target_probability_threshold: float = 0.97
    target_hit_rate_threshold: float = 0.95
    output_dir: str = "outputs/igcg_trigger_optimization"
    device: str = "auto"
    seed: int = 42
    eps: float = 1e-8
    # Log a per-(layer, expert) metric breakdown each iteration for multi-target
    # runs.  Adds one evaluation forward pass per iteration.
    log_per_target_breakdown: bool = True


@dataclass(frozen=True)
class PreparedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    trigger_mask: torch.Tensor


@dataclass(frozen=True)
class TriggerMetrics:
    loss: float
    target_probability: float
    target_hit_rate: float
    target_margin: float
    # Target probability after renormalizing over the selected top-k experts.
    # The default keeps this value backward-compatible for callers that build
    # TriggerMetrics directly (for example, older tests or analysis scripts).
    target_top_k_probability: float = 0.0


@dataclass(frozen=True)
class RobustTriggerMetrics:
    """Aggregate one trigger's metrics over a checkpoint ensemble."""

    mean: TriggerMetrics
    worst_target_probability: float
    worst_target_hit_rate: float
    low_quantile_margin: float
    per_checkpoint: Tuple[TriggerMetrics, ...]
    worst_target_top_k_probability: float = 0.0


@dataclass(frozen=True)
class TriggerRecord:
    iteration: int
    trigger_ids: List[int]
    trigger_text: str
    routing_loss: float
    target_expert_probability: float
    target_expert_hit_rate: float
    target_expert_margin: float
    accepted_mutations: int
    target_expert_top_k_probability: float = 0.0
    robust_worst_target_probability: Optional[float] = None
    robust_worst_target_hit_rate: Optional[float] = None
    robust_worst_target_top_k_probability: Optional[float] = None
    robust_low_quantile_margin: Optional[float] = None
    heldout_routing_loss: Optional[float] = None
    heldout_target_probability: Optional[float] = None
    heldout_target_hit_rate: Optional[float] = None
    heldout_target_margin: Optional[float] = None
    heldout_target_top_k_probability: Optional[float] = None
    heldout_robust_worst_target_probability: Optional[float] = None
    heldout_robust_worst_target_hit_rate: Optional[float] = None
    heldout_robust_worst_target_top_k_probability: Optional[float] = None
    heldout_robust_low_quantile_margin: Optional[float] = None


@dataclass(frozen=True)
class TopKExpertProfile:
    """Average post-top-k routing share for one trigger evaluation."""

    # (expert id, mean gate share over trigger tokens, selection rate)
    experts: Tuple[Tuple[int, float, float], ...]
    # Mean post-top-k mass assigned to all target experts.
    target_share: float = 0.0
    # Mass belonging to experts outside the displayed aggregate top-k list.
    omitted_share: float = 0.0


def heldout_selection_key(metrics: TriggerMetrics) -> Tuple[float, float, float, float]:
    """Rank validation results by actual Top-1 routing, then robustness margin."""
    return (
        metrics.target_hit_rate,
        metrics.target_margin,
        metrics.target_probability,
        -metrics.loss,
    )


def aggregate_robust_metrics(
    metrics: Sequence[TriggerMetrics], margin_quantile: float
) -> RobustTriggerMetrics:
    """Aggregate checkpoint metrics without hiding the weakest router state."""
    if not metrics:
        raise ValueError("At least one checkpoint metric is required.")
    if not 0.0 <= margin_quantile <= 1.0:
        raise ValueError("margin_quantile must be in [0, 1].")

    count = float(len(metrics))
    mean = TriggerMetrics(
        sum(item.loss for item in metrics) / count,
        sum(item.target_probability for item in metrics) / count,
        sum(item.target_hit_rate for item in metrics) / count,
        sum(item.target_margin for item in metrics) / count,
        sum(item.target_top_k_probability for item in metrics) / count,
    )
    margins = torch.tensor([item.target_margin for item in metrics], dtype=torch.float64)
    return RobustTriggerMetrics(
        mean=mean,
        worst_target_probability=min(item.target_probability for item in metrics),
        worst_target_hit_rate=min(item.target_hit_rate for item in metrics),
        low_quantile_margin=float(torch.quantile(margins, margin_quantile)),
        per_checkpoint=tuple(metrics),
        worst_target_top_k_probability=min(item.target_top_k_probability for item in metrics),
    )


def robust_selection_key(
    metrics: RobustTriggerMetrics,
) -> Tuple[float, float, float, float, float, float]:
    """Prefer worst-checkpoint hit rate, then low-quantile routing margin."""
    return (
        metrics.worst_target_hit_rate,
        metrics.low_quantile_margin,
        metrics.mean.target_hit_rate,
        metrics.mean.target_margin,
        metrics.mean.target_probability,
        -metrics.mean.loss,
    )

def normalize_target_experts(
    target_expert: Any,
    num_experts: int,
) -> Tuple[int, ...]:
    """Normalize legacy scalar and new multi-expert target specifications."""
    if isinstance(target_expert, bool):
        raise ValueError("target_expert must contain integer expert IDs.")
    if isinstance(target_expert, int):
        raw_targets = (target_expert,)
    else:
        try:
            raw_targets = tuple(target_expert)
        except TypeError as exc:
            raise ValueError("target_expert must be an integer or a sequence of integers.") from exc
    if not raw_targets:
        raise ValueError("At least one target expert is required.")
    normalized = []
    for expert in raw_targets:
        if isinstance(expert, bool) or not isinstance(expert, int):
            raise ValueError("target_expert must contain integer expert IDs.")
        if not 0 <= expert < num_experts:
            raise ValueError("target_expert is outside the router output range.")
        if expert not in normalized:
            normalized.append(int(expert))
    return tuple(normalized)


# Return loss, target-set probability, Top-K hit rate, and target/boundary margin.
def compute_routing_objective(
    router_logits: torch.Tensor,
    trigger_mask: torch.Tensor,
    target_expert: Any,
    *,
    routing_top_k: int = 1,
    margin: float = 0.0,
    margin_weight: float = 0.0,
    top_k_share_weight: float = 0.0,
    full_softmax_weight: float = 1.0,
    sibling_suppression_weight: float = 0.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if router_logits.ndim == 2:
        if router_logits.shape[0] != trigger_mask.numel():
            raise ValueError("Flattened router logits and trigger mask have different token counts.")
        router_logits = router_logits.reshape(*trigger_mask.shape, router_logits.shape[-1])
    if router_logits.ndim != 3 or router_logits.shape[:2] != trigger_mask.shape:
        raise ValueError(
            f"Router logits {tuple(router_logits.shape)} do not match mask {tuple(trigger_mask.shape)}."
        )
    target_experts = normalize_target_experts(target_expert, router_logits.shape[-1])
    if routing_top_k <= 0:
        raise ValueError("routing_top_k must be > 0.")
    if len(target_experts) > min(int(routing_top_k), router_logits.shape[-1] - 1):
        raise ValueError(
            "The target expert set cannot fit inside the configured routing top-k."
        )
    selected = trigger_mask.bool()
    if not bool(selected.any()):
        raise ValueError("The trigger mask has no active token.")

    logits = router_logits.float()[selected]
    probabilities = torch.softmax(logits, dim=-1)
    target_indices = torch.tensor(target_experts, device=probabilities.device)
    target_probabilities = probabilities.index_select(-1, target_indices).sum(dim=-1)

    # Legacy objective: the target's share of the full softmax over *all* experts.
    # This spends gradient suppressing the many experts outside the top-k, so it
    # caps the achievable gate share.  Kept as an optional blend term.
    full_softmax_loss = -torch.log(target_probabilities + eps).mean()

    top_k = min(int(routing_top_k), logits.shape[-1])
    topk_vals, topk_idx = torch.topk(logits, k=top_k, dim=-1)
    target_slot = topk_idx.unsqueeze(-1).eq(target_indices.view(1, 1, -1)).any(dim=-1)

    # Method 1 --- differentiable top-k renormalized share.  Softmax is taken over
    # *only* the selected top-k logits, so the gradient competes against the
    # (top_k - 1) co-selected siblings that actually share the gate mass rather
    # than all experts.  The hard index choice is detached; a target outside the
    # top-k contributes no share gradient here and is pulled in by the margin
    # term below instead.
    topk_probs = torch.softmax(topk_vals, dim=-1)
    target_share = (topk_probs * target_slot.to(topk_probs.dtype)).sum(dim=-1)
    all_targets_selected = (
        topk_idx.unsqueeze(-1).eq(target_indices.view(1, 1, -1)).any(dim=1).all(dim=-1)
    )
    if bool(all_targets_selected.any()):
        share_loss = -torch.log(target_share[all_targets_selected].clamp_min(eps)).mean()
    else:
        share_loss = logits.new_zeros(())

    # Method 2 --- suppress the co-selected siblings.  Pushing down the non-target
    # logits inside the top-k shrinks the renormalization denominator, so the
    # target's share rises faster than by raising its own logit alone.
    sibling_logits = topk_vals.masked_fill(target_slot, torch.finfo(topk_vals.dtype).min)
    sibling_penalty = torch.logsumexp(sibling_logits, dim=-1).mean()

    loss = (
        full_softmax_weight * full_softmax_loss
        + top_k_share_weight * share_loss
        + sibling_suppression_weight * sibling_penalty
    )

    other_logits = logits.clone()
    other_logits[:, target_indices] = -torch.inf
    # For a target set, all target experts are inside top-k only if each target
    # clears the (top_k - number_of_targets)-th strongest non-target logit.
    boundary_k = min(
        max(1, int(routing_top_k) - len(target_experts)),
        logits.shape[-1] - len(target_experts),
    )
    routing_boundary = torch.topk(
        other_logits,
        k=boundary_k,
        dim=-1,
    ).values[:, -1]
    target_logits = logits.index_select(-1, target_indices)
    logit_margin = (target_logits - routing_boundary.unsqueeze(-1)).amin(dim=-1)
    if margin_weight:
        loss = loss + margin_weight * torch.relu(margin - logit_margin).mean()

    target_selected = topk_idx.unsqueeze(-1).eq(target_indices.view(1, 1, -1)).any(dim=1)
    hit_rate = target_selected.all(dim=-1).float().mean()
    return loss, target_probabilities.mean(), hit_rate, logit_margin.mean()


def compute_target_top_k_probability(
    router_logits: torch.Tensor,
    trigger_mask: torch.Tensor,
    target_expert: Any,
    *,
    routing_top_k: int = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return the target share after renormalizing over the selected top-k.

    Router logits are first converted to probabilities over all experts.  The
    probabilities of the actual top-k experts are then renormalized to sum to
    one.  A target expert outside that top-k set has share zero, matching the
    expert selection semantics of top-k MoE routing.
    """
    if router_logits.ndim == 2:
        if router_logits.shape[0] != trigger_mask.numel():
            raise ValueError("Flattened router logits and trigger mask have different token counts.")
        router_logits = router_logits.reshape(*trigger_mask.shape, router_logits.shape[-1])
    if router_logits.ndim != 3 or router_logits.shape[:2] != trigger_mask.shape:
        raise ValueError(
            f"Router logits {tuple(router_logits.shape)} do not match mask {tuple(trigger_mask.shape)}."
        )
    target_experts = normalize_target_experts(target_expert, router_logits.shape[-1])
    if routing_top_k <= 0:
        raise ValueError("routing_top_k must be > 0.")
    if len(target_experts) > min(int(routing_top_k), router_logits.shape[-1] - 1):
        raise ValueError(
            "The target expert set cannot fit inside the configured routing top-k."
        )
    selected = trigger_mask.bool()
    if not bool(selected.any()):
        raise ValueError("The trigger mask has no active token.")

    probabilities = torch.softmax(router_logits.float()[selected], dim=-1)
    top_k = min(int(routing_top_k), probabilities.shape[-1])
    top_k_indices = torch.topk(probabilities, k=top_k, dim=-1).indices
    top_k_probabilities = probabilities.gather(-1, top_k_indices)
    top_k_total = top_k_probabilities.sum(dim=-1).clamp_min(eps)
    target_indices = torch.tensor(target_experts, device=probabilities.device)
    target_selected_per_expert = (
        top_k_indices.unsqueeze(-1)
        .eq(target_indices.view(1, 1, -1))
        .any(dim=-2)
    )
    target_probability = (
        probabilities.index_select(-1, target_indices) * target_selected_per_expert
    ).sum(dim=-1)
    target_selected = target_selected_per_expert.any(dim=-1)
    target_share = torch.where(
        target_selected,
        target_probability / top_k_total,
        torch.zeros_like(target_probability),
    )
    return target_share.mean()

# Rank replacement tokens by first-order loss change for I-GCG proposals.
def top_k_tokens_from_gradient(
    current_ids: Sequence[int],
    position: int,
    gradient: torch.Tensor,
    embedding_matrix: torch.Tensor,
    top_k: int,
    excluded_ids: Iterable[int] = (),
) -> List[int]:
    if not 0 <= position < len(current_ids):
        raise IndexError("Trigger position is out of range.")
    row_gradient = gradient[position] if gradient.ndim == 2 else gradient
    if row_gradient.ndim != 1 or embedding_matrix.ndim != 2:
        raise ValueError("Expected a vector gradient and a [vocab, hidden] embedding matrix.")
    scores = embedding_matrix.float().mv(row_gradient.float())
    forbidden = {int(token_id) for token_id in excluded_ids}
    forbidden.add(int(current_ids[position]))
    valid_forbidden = [idx for idx in forbidden if 0 <= idx < scores.numel()]
    if valid_forbidden:
        scores[torch.tensor(valid_forbidden, device=scores.device)] = torch.inf
    count = min(int(top_k), int(torch.isfinite(scores).sum().item()))
    if count <= 0:
        raise ValueError(f"No valid replacement token remains at trigger position {position}.")
    return torch.topk(scores, k=count, largest=False).indices.cpu().tolist()

# Sample single-coordinate proposals for I-GCG cumulative merging.
def sample_single_coordinate_candidates(
    current_ids: Sequence[int],
    candidates_by_position: Sequence[Sequence[int]],
    batch_size: int,
    rng: random.Random,
) -> List[List[int]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    if len(candidates_by_position) != len(current_ids):
        raise ValueError("A candidate list is required for every trigger position.")
    if any(not choices for choices in candidates_by_position):
        raise ValueError("Every trigger position must have at least one replacement token.")
    generated: List[List[int]] = []
    trigger_length = len(current_ids)
    for candidate_index in range(batch_size):
        # Integer equivalent of arange(0, length, length / batch_size).
        position = min((candidate_index * trigger_length) // batch_size, trigger_length - 1)
        choices = candidates_by_position[position]
        candidate = list(current_ids)
        candidate[position] = int(rng.choice(choices))
        generated.append(candidate)
    return generated

# Canonicalize candidates through text and reject unstable tokenizations.
def filter_roundtrip_candidates(
    tokenizer: PreTrainedTokenizerBase,
    current_ids: Sequence[int],
    candidates: Sequence[Sequence[int]],
) -> List[List[int]]:
    expected_length = len(current_ids)
    filtered: List[List[int]] = []
    seen = set()
    for candidate in candidates:
        text = tokenizer.decode(
            list(candidate),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        encoded = list(tokenizer(text, add_special_tokens=False)["input_ids"])
        if len(encoded) != expected_length or encoded == list(current_ids):
            continue
        key = tuple(int(token_id) for token_id in encoded)
        if key not in seen:
            seen.add(key)
            filtered.append(list(key))
    return filtered

# I-GCG cumulative merge of the best round-trip-stable candidates.
def combine_top_single_coordinate_candidates(
    current_ids: Sequence[int],
    candidates: Sequence[Sequence[int]],
    losses: Sequence[float],
    top_p: int,
) -> List[List[int]]:
    if len(candidates) != len(losses):
        raise ValueError("candidates and losses must have equal length.")
    if top_p <= 0:
        raise ValueError("top_p must be > 0.")
    ranked = sorted(range(len(candidates)), key=lambda idx: losses[idx])[:top_p]
    merged = list(current_ids)
    output: List[List[int]] = []
    seen = set()
    for candidate_index in ranked:
        candidate = list(candidates[candidate_index])
        changed = [idx for idx, (old, new) in enumerate(zip(current_ids, candidate)) if old != new]
        if not changed:
            continue
        # A raw proposal changes one token, but decode/encode can alter several
        # token IDs at a word boundary. The original I-GCG merges every changed
        # coordinate after this round trip, rather than rejecting the proposal.
        for position in changed:
            merged[position] = candidate[position]
        key = tuple(merged)
        if key != tuple(current_ids) and key not in seen:
            seen.add(key)
            output.append(list(merged))
    return output

# Select the first ranked low-frequency expert from an LFE profile.
def select_low_frequency_target(profile_path: str, layer_id: str) -> int:
    payload = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    try:
        experts = payload["layers"][layer_id]["top_low_frequency_experts"]
        expert = experts[0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            f"Profile '{profile_path}' has no low-frequency ranking for layer '{layer_id}'."
        ) from exc
    if isinstance(expert, bool) or not isinstance(expert, int) or expert < 0:
        raise ValueError("The selected low-frequency expert ID is invalid.")
    return expert


def _load_checkpoint(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file() and not checkpoint_path.is_absolute():
        repository_relative = Path(__file__).resolve().parent.parent / checkpoint_path
        if repository_relative.is_file():
            checkpoint_path = repository_relative
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: '{path}'.")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        return dict(payload["state_dict"]), {k: v for k, v in payload.items() if k != "state_dict"}
    if isinstance(payload, dict) and payload and all(torch.is_tensor(v) for v in payload.values()):
        return dict(payload), {}
    raise ValueError("Checkpoint must be a state_dict or contain a 'state_dict' object.")


def _load_state_into_model(model: nn.Module, checkpoint_path: str) -> Dict[str, Any]:
    state, metadata = _load_checkpoint(checkpoint_path)
    incompatible = model.load_state_dict(state, strict=False)
    missing = [name for name in incompatible.missing_keys if not name.endswith("position_ids")]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint '{checkpoint_path}' mismatch: missing={missing[:10]}, "
            f"unexpected={incompatible.unexpected_keys[:10]}."
        )
    return metadata


def resolve_checkpoint_ensemble(
    primary_path: str, robust_paths: Sequence[str]
) -> Tuple[str, ...]:
    """Return primary + robust checkpoints in order, with duplicates removed."""
    result: List[str] = []
    seen: set[str] = set()
    for raw_path in (primary_path, *robust_paths):
        checkpoint_path = Path(raw_path).expanduser()
        if not checkpoint_path.is_file() and not checkpoint_path.is_absolute():
            repository_relative = Path(__file__).resolve().parent.parent / checkpoint_path
            if repository_relative.is_file():
                checkpoint_path = repository_relative
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: '{raw_path}'.")
        resolved = str(checkpoint_path.resolve())
        key = resolved.casefold()
        if key not in seen:
            seen.add(key)
            result.append(resolved)
    return tuple(result)


def load_victim_model(
    model_name_or_path: str,
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[nn.Module, PreTrainedTokenizerBase, Dict[str, Any]]:
    state, metadata = _load_checkpoint(checkpoint_path)
    classifier = state.get("classification_head.out_proj.weight")
    num_labels = int(classifier.shape[0]) if classifier is not None else 2
    tokenizer = build_tokenizer(model_name_or_path, max_length=1_000_000)
    model = build_sequence_classification_model(
        model_name_or_path,
        num_labels=num_labels,
        id2label={index: str(index) for index in range(num_labels)},
        label2id={str(index): index for index in range(num_labels)},
        tokenizer=tokenizer,
        model_backend="auto",
        model_dtype="auto",
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    incompatible = model.load_state_dict(state, strict=False)
    missing = [name for name in incompatible.missing_keys if not name.endswith("position_ids")]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing[:10]}, unexpected={incompatible.unexpected_keys[:10]}."
        )
    
    model.config.use_cache = False
    model.eval().requires_grad_(False).to(device)
    return model, tokenizer, metadata


def resolve_moe_layer(model: nn.Module, layer_id: str) -> MoeHandle:
    matches = [handle for handle in collect_moe_handles(model) if handle.layer_id == layer_id]
    if len(matches) == 1:
        return matches[0]
    available = ", ".join(handle.layer_id for handle in collect_moe_handles(model))
    raise ValueError(f"Unknown MoE layer '{layer_id}'. Available layers: {available or '<none>'}.")


def get_router(handle: MoeHandle) -> nn.Module:
    if handle.backend in DECODER_ONLY_MOE_BACKENDS and isinstance(handle.router_module, nn.Module):
        return handle.router_module
    router = getattr(getattr(handle.mlp_module, "router", None), "classifier", None)
    if not isinstance(router, nn.Linear):
        raise RuntimeError(f"Layer '{handle.layer_id}' does not expose a linear router classifier.")
    return router


def build_triggered_batch(
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    trigger_ids: Sequence[int],
    max_length: int,
    trigger_position: str,
    fixed_position: Optional[int],
) -> PreparedBatch:
    eos_id, pad_id = tokenizer.eos_token_id, tokenizer.pad_token_id
    if eos_id is None or pad_id is None:
        raise ValueError("Tokenizer must define EOS and padding token IDs.")
    if max_length <= len(trigger_ids) + 1:
        raise ValueError("max_length leaves no room for prompt tokens and EOS.")
    rows, attentions, trigger_masks = [], [], []
    budget = max_length - len(trigger_ids) - 1
    for prompt in prompts:
        prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        prompt_ids = [token for token in prompt_ids if token != eos_id][:budget]
        if trigger_position == "append":
            insertion = len(prompt_ids)
        elif trigger_position == "prepend":
            insertion = 0
        elif trigger_position == "fixed_position" and fixed_position is not None and fixed_position >= 0:
            insertion = min(fixed_position, len(prompt_ids))
        else:
            raise ValueError("fixed_position >= 0 is required when using fixed_position mode.")
        ids = prompt_ids[:insertion] + list(trigger_ids) + prompt_ids[insertion:] + [eos_id]
        mask = [False] * insertion + [True] * len(trigger_ids)
        mask.extend([False] * (len(ids) - len(mask)))
        rows.append(ids)
        attentions.append([1] * len(ids))
        trigger_masks.append(mask)
    width = max(map(len, rows))
    for ids, attention, mask in zip(rows, attentions, trigger_masks):
        padding = width - len(ids)
        ids.extend([pad_id] * padding)
        attention.extend([0] * padding)
        mask.extend([False] * padding)
    return PreparedBatch(
        torch.tensor(rows, dtype=torch.long),
        torch.tensor(attentions, dtype=torch.long),
        torch.tensor(trigger_masks, dtype=torch.bool),
    )


def _stack_trigger_mask(
    mask: torch.Tensor,
    stack_name: str,
    backend: str = "switch",
) -> torch.Tensor:
    if backend in DECODER_ONLY_MOE_BACKENDS:
        return mask
    if stack_name == "encoder":
        return mask
    if stack_name == "decoder":
        shifted = torch.zeros_like(mask)
        shifted[:, 1:] = mask[:, :-1]
        return shifted
    raise ValueError(f"Unsupported stack '{stack_name}'.")


def load_calibration_texts(path: str, limit: int) -> List[str]:
    calibration_path = Path(path)
    if not calibration_path.is_file() and not calibration_path.is_absolute():
        repository_relative = Path(__file__).resolve().parent.parent / calibration_path
        if repository_relative.is_file():
            calibration_path = repository_relative
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Calibration file does not exist: '{path}'.")
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Calibration file must contain a JSON list.")
    texts = [row.get("text") for row in payload if isinstance(row, dict)]
    if not texts or not all(isinstance(text, str) and text.strip() for text in texts):
        raise ValueError("Calibration rows must contain non-empty 'text' strings.")
    return [str(text) for text in texts[:limit]]

# Deterministically split calibration contexts into search and held-out sets.
def split_calibration_texts(
    texts: Sequence[str], heldout_samples: int, seed: int
) -> Tuple[List[str], List[str]]:
    if heldout_samples <= 0:
        raise ValueError("heldout_samples must be > 0.")
    if len(texts) <= heldout_samples:
        raise ValueError(
            f"Calibration data has {len(texts)} samples, but heldout_samples={heldout_samples} "
            "must leave at least one optimization sample."
        )
    indices = list(range(len(texts)))
    random.Random(seed).shuffle(indices)
    heldout_indices = set(indices[:heldout_samples])
    optimization = [text for index, text in enumerate(texts) if index not in heldout_indices]
    heldout = [text for index, text in enumerate(texts) if index in heldout_indices]
    return optimization, heldout


class IGCGTriggerOptimizer:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        config: IGCGTriggerConfig,
        calibration_texts: Sequence[str],
        heldout_texts: Sequence[str],
        device: torch.device,
    ) -> None:
        self.model, self.tokenizer, self.config = model, tokenizer, config
        self.texts = list(calibration_texts)
        self.heldout_texts = list(heldout_texts)
        self.device = device
        if not self.texts or not self.heldout_texts:
            raise ValueError("Both optimization and held-out calibration sets must be non-empty.")
        raw_specs = config.target_specs
        if raw_specs:
            specs = tuple((str(layer), tuple(int(expert) for expert in experts)) for layer, experts in raw_specs)
        else:
            if not config.target_layer:
                raise ValueError("target_layer is required when target_specs is empty.")
            specs = ((config.target_layer, config.target_experts or (config.target_expert,)),)
        if not specs:
            raise ValueError("At least one layer/expert target is required.")

        self.handles: Dict[str, MoeHandle] = {}
        self.routers: Dict[str, nn.Module] = {}
        normalized_specs: List[Tuple[str, Tuple[int, ...]]] = []
        for layer_id, configured_targets in specs:
            if layer_id in self.handles:
                raise ValueError(f"Duplicate iGCG target layer '{layer_id}'.")
            handle = resolve_moe_layer(model, layer_id)
            router = get_router(handle)
            router_weight = getattr(router, "weight", None)
            if not isinstance(router_weight, torch.Tensor) or router_weight.ndim != 2:
                raise RuntimeError(f"Layer '{layer_id}' router has no matrix weight.")
            targets = normalize_target_experts(configured_targets, int(router_weight.shape[0]))
            if len(targets) > int(handle.router_top_k):
                raise ValueError(
                    f"Layer '{layer_id}' has {handle.router_top_k}-way routing, "
                    f"but {len(targets)} target experts were requested."
                )
            self.handles[layer_id] = handle
            self.routers[layer_id] = router
            normalized_specs.append((layer_id, targets))

        # Joint optimization is intentionally restricted to one stack.  This
        # keeps the trigger mask/decoder shift unambiguous while supporting the
        # intended cross-decoder-layer use case.
        stacks = {handle.stack_name for handle in self.handles.values()}
        if len(stacks) != 1:
            raise ValueError(
                "Cross-layer iGCG targets must belong to the same model stack "
                "(for example, decoder.2 and decoder.7)."
            )
        self.target_specs = tuple(normalized_specs)
        self.handle = self.handles[self.target_specs[0][0]]
        self.router = self.routers[self.target_specs[0][0]]
        self.target_experts = self.target_specs[0][1]
        self.num_router_experts = int(getattr(self.router, "weight").shape[0])
        # Keep legacy scalar fields aligned with the first target while storing
        # the complete cross-layer target set in the config and result files.
        self.config = replace(
            config,
            target_layer=self.target_specs[0][0],
            target_expert=self.target_specs[0][1][0],
            target_experts=self.target_specs[0][1],
            target_specs=self.target_specs,
        )
        if self.handle.backend in DECODER_ONLY_MOE_BACKENDS:
            embedding = model.get_input_embeddings()
        else:
            # Switch/T5 may tie this weight to ``shared`` while calling the
            # stack-specific embedding module in forward. Hook the module that
            # actually executes so trigger gradients remain connected.
            embedding = getattr(model, self.handle.stack_name).embed_tokens
        if not isinstance(embedding, nn.Embedding):
            raise RuntimeError("iGCG requires an nn.Embedding input module.")
        self.embedding = embedding
        self.rng = random.Random(config.seed)
        self.records: List[TriggerRecord] = []
        self.heldout_evaluations: List[Dict[str, Any]] = []
        self.checkpoint_paths = resolve_checkpoint_ensemble(
            config.checkpoint_path, config.robust_checkpoint_paths
        )
        self.primary_checkpoint_path = self.checkpoint_paths[0]
        self.robust_screening_enabled = len(self.checkpoint_paths) > 1

    def _activate_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        metadata = _load_state_into_model(self.model, checkpoint_path)
        self.model.eval().requires_grad_(False)
        return metadata

    def _prepare(self, prompts: Sequence[str], trigger_ids: Sequence[int]) -> PreparedBatch:
        batch = build_triggered_batch(
            self.tokenizer,
            prompts,
            trigger_ids,
            self.config.max_length,
            self.config.trigger_position,
            self.config.fixed_position,
        )
        return PreparedBatch(
            batch.input_ids.to(self.device),
            batch.attention_mask.to(self.device),
            # Keep the unshifted mask in PreparedBatch.  Each target layer
            # applies its own stack/backend alignment below.
            batch.trigger_mask.to(self.device),
        )

    def _forward_router(
        self, batch: PreparedBatch, capture_embedding: bool
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        router_outputs: Dict[str, List[torch.Tensor]] = {
            layer_id: [] for layer_id in self.routers
        }
        embedding_outputs: List[torch.Tensor] = []

        def make_router_hook(layer_id: str):
            handle = self.handles[layer_id]

            def router_hook(_module: nn.Module, args: Any, output: Any) -> None:
                if handle.backend == "olmoe":
                    if not args or not isinstance(args[0], torch.Tensor):
                        raise RuntimeError(
                            f"OLMoE iGCG router hook at '{layer_id}' received no hidden states."
                        )
                    router_outputs[layer_id].append(
                        nn.functional.linear(
                            args[0].float(),
                            _module.weight.float(),
                        )
                    )
                    return
                if not isinstance(output, torch.Tensor):
                    raise RuntimeError(
                        f"Router at '{layer_id}' did not return logits."
                    )
                router_outputs[layer_id].append(output)

            return router_hook

        def embedding_hook(_module: nn.Module, _args: Any, output: torch.Tensor) -> torch.Tensor:
            leaf = output.detach().requires_grad_(True)
            leaf.retain_grad()
            embedding_outputs.append(leaf)
            return leaf

        hooks = [
            router.register_forward_hook(make_router_hook(layer_id))
            for layer_id, router in self.routers.items()
        ]
        if capture_embedding:
            hooks.append(self.embedding.register_forward_hook(embedding_hook))
        try:
            self.model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                output_router_logits=False,
                use_cache=False,
                return_dict=True,
            )
        finally:
            for hook in hooks:
                hook.remove()
        missing = [layer_id for layer_id, outputs in router_outputs.items() if len(outputs) != 1]
        if missing:
            details = ", ".join(
                f"{layer_id}={len(router_outputs[layer_id])}" for layer_id in missing
            )
            raise RuntimeError(f"Expected one router call per target layer, observed {details}.")
        selected_embedding = None
        if capture_embedding:
            # Switch/T5 commonly shares the embedding module. Its first call
            # is encoder input and its second call is shifted decoder input.
            if not embedding_outputs:
                raise RuntimeError("The embedding hook did not run.")
            index = 0 if self.handle.stack_name == "encoder" else -1
            selected_embedding = embedding_outputs[index]
        return {
            layer_id: outputs[0] for layer_id, outputs in router_outputs.items()
        }, selected_embedding

    def _objective(self, layer_id: str, logits: torch.Tensor, mask: torch.Tensor):
        handle = self.handles[layer_id]
        aligned_mask = _stack_trigger_mask(
            mask,
            handle.stack_name,
            handle.backend,
        ).to(device=logits.device)
        targets = dict(self.target_specs)[layer_id]
        return compute_routing_objective(
            logits,
            aligned_mask,
            targets,
            routing_top_k=int(handle.router_top_k),
            margin=self.config.margin,
            margin_weight=self.config.margin_weight,
            top_k_share_weight=self.config.top_k_share_weight,
            full_softmax_weight=self.config.full_softmax_weight,
            sibling_suppression_weight=self.config.sibling_suppression_weight,
            eps=self.config.eps,
        )

    def _joint_objective(
        self,
        logits_by_layer: Dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aggregate objectives conservatively across all target layers.

        Loss is averaged for gradient stability.  Metrics use the minimum
        across layers, so a candidate cannot look successful while one target
        layer remains unreachable.
        """
        values = [
            self._objective(layer_id, logits_by_layer[layer_id], mask)
            for layer_id, _targets in self.target_specs
        ]
        losses, probabilities, hits, margins = zip(*values)
        loss = torch.stack(tuple(losses)).mean()
        probability = torch.stack(tuple(probabilities)).amin()
        hit_rate = torch.stack(tuple(hits)).amin()
        target_margin = torch.stack(tuple(margins)).amin()
        top_k_values = [
            compute_target_top_k_probability(
                logits_by_layer[layer_id],
                _stack_trigger_mask(
                    mask,
                    self.handles[layer_id].stack_name,
                    self.handles[layer_id].backend,
                ).to(device=logits_by_layer[layer_id].device),
                targets,
                routing_top_k=int(self.handles[layer_id].router_top_k),
                eps=self.config.eps,
            )
            for layer_id, targets in self.target_specs
        ]
        target_top_k_probability = torch.stack(tuple(top_k_values)).amin()
        return loss, probability, hit_rate, target_margin, target_top_k_probability

    def evaluate(
        self, trigger_ids: Sequence[int], texts: Optional[Sequence[str]] = None
    ) -> TriggerMetrics:
        evaluation_texts = list(self.texts if texts is None else texts)
        if not evaluation_texts:
            raise ValueError("Evaluation texts must not be empty.")
        loss_sum = probability_sum = hit_sum = top_k_probability_sum = 0.0
        margin_sum = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, len(evaluation_texts), self.config.evaluation_batch_size):
                prompts = evaluation_texts[start : start + self.config.evaluation_batch_size]
                batch = self._prepare(prompts, trigger_ids)
                logits_by_layer, _ = self._forward_router(batch, False)
                loss, probability, hit_rate, target_margin, top_k_probability = (
                    self._joint_objective(logits_by_layer, batch.trigger_mask)
                )
                size = len(prompts)
                loss_sum += float(loss) * size
                probability_sum += float(probability) * size
                hit_sum += float(hit_rate) * size
                margin_sum += float(target_margin) * size
                top_k_probability_sum += float(top_k_probability) * size
                count += size
        return TriggerMetrics(
            loss_sum / count,
            probability_sum / count,
            hit_sum / count,
            margin_sum / count,
            top_k_probability_sum / count,
        )

    def evaluate_breakdown(
        self, trigger_ids: Sequence[int], texts: Optional[Sequence[str]] = None
    ) -> List[Dict[str, Any]]:
        """Evaluate every target expert at every layer *separately*.

        Each ``(layer_id, expert)`` is scored on its own as a single-element
        target set, so the metrics are not merged across experts or layers.
        """
        evaluation_texts = list(self.texts if texts is None else texts)
        if not evaluation_texts:
            raise ValueError("Evaluation texts must not be empty.")
        targets = [
            (layer_id, int(expert))
            for layer_id, experts in self.target_specs
            for expert in experts
        ]
        sums = {
            (layer_id, expert): {
                "loss": 0.0, "prob": 0.0, "hit": 0.0, "margin": 0.0, "top_k": 0.0
            }
            for layer_id, expert in targets
        }
        count = 0
        with torch.no_grad():
            for start in range(0, len(evaluation_texts), self.config.evaluation_batch_size):
                prompts = evaluation_texts[start : start + self.config.evaluation_batch_size]
                batch = self._prepare(prompts, trigger_ids)
                logits_by_layer, _ = self._forward_router(batch, False)
                size = len(prompts)
                for layer_id, expert in targets:
                    handle = self.handles[layer_id]
                    aligned_mask = _stack_trigger_mask(
                        batch.trigger_mask, handle.stack_name, handle.backend
                    ).to(device=logits_by_layer[layer_id].device)
                    loss, probability, hit_rate, target_margin = compute_routing_objective(
                        logits_by_layer[layer_id],
                        aligned_mask,
                        (expert,),
                        routing_top_k=int(handle.router_top_k),
                        margin=self.config.margin,
                        margin_weight=self.config.margin_weight,
                        eps=self.config.eps,
                    )
                    top_k_probability = compute_target_top_k_probability(
                        logits_by_layer[layer_id],
                        aligned_mask,
                        (expert,),
                        routing_top_k=int(handle.router_top_k),
                        eps=self.config.eps,
                    )
                    bucket = sums[(layer_id, expert)]
                    bucket["loss"] += float(loss) * size
                    bucket["prob"] += float(probability) * size
                    bucket["hit"] += float(hit_rate) * size
                    bucket["margin"] += float(target_margin) * size
                    bucket["top_k"] += float(top_k_probability) * size
                count += size
        return [
            {
                "layer_id": layer_id,
                "expert": expert,
                "routing_loss": sums[(layer_id, expert)]["loss"] / count,
                "target_probability": sums[(layer_id, expert)]["prob"] / count,
                "target_top_k_probability": sums[(layer_id, expert)]["top_k"] / count,
                "target_hit_rate": sums[(layer_id, expert)]["hit"] / count,
                "target_margin": sums[(layer_id, expert)]["margin"] / count,
            }
            for layer_id, expert in targets
        ]

    def gradient(self, trigger_ids: Sequence[int]) -> torch.Tensor:
        total = len(self.texts)
        result = torch.zeros(
            len(trigger_ids), self.embedding.embedding_dim, device=self.device, dtype=torch.float32
        )
        for start in range(0, total, self.config.gradient_batch_size):
            prompts = self.texts[start : start + self.config.gradient_batch_size]
            batch = self._prepare(prompts, trigger_ids)
            logits_by_layer, embedding_output = self._forward_router(batch, True)
            if embedding_output is None:
                raise RuntimeError("No differentiable embedding output was captured.")
            loss, _, _, _, _ = self._joint_objective(logits_by_layer, batch.trigger_mask)
            (loss * (len(prompts) / total)).backward()
            if embedding_output.grad is None:
                raise RuntimeError("Routing objective did not reach the input embeddings.")
            for row in range(len(prompts)):
                row_gradient = embedding_output.grad[row][batch.trigger_mask[row]]
                if row_gradient.shape != result.shape:
                    raise RuntimeError(
                        f"Trigger gradient shape {tuple(row_gradient.shape)} != {tuple(result.shape)}."
                    )
                result += row_gradient
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("Non-finite trigger gradient.")
        return result

    def evaluate_many(
        self,
        candidates: Sequence[Sequence[int]],
        texts: Optional[Sequence[str]] = None,
    ) -> List[TriggerMetrics]:
        if not candidates:
            raise ValueError("At least one candidate trigger is required.")
        evaluation_texts = list(self.texts if texts is None else texts)
        if not evaluation_texts:
            raise ValueError("Evaluation texts must not be empty.")
        loss_sums = [0.0] * len(candidates)
        probability_sums = [0.0] * len(candidates)
        hit_sums = [0.0] * len(candidates)
        margin_sums = [0.0] * len(candidates)
        top_k_probability_sums = [0.0] * len(candidates)
        counts = [0] * len(candidates)
        pairs = [(index, text) for index in range(len(candidates)) for text in evaluation_texts]
        with torch.no_grad():
            for start in range(0, len(pairs), self.config.evaluation_batch_size):
                chunk = pairs[start : start + self.config.evaluation_batch_size]
                prepared = [self._prepare([text], candidates[index]) for index, text in chunk]
                width = max(item.input_ids.shape[1] for item in prepared)
                pad_id = int(self.tokenizer.pad_token_id)
                ids = torch.full(
                    (len(prepared), width), pad_id, dtype=torch.long, device=self.device
                )
                attention = torch.zeros(
                    (len(prepared), width), dtype=torch.long, device=self.device
                )
                masks = torch.zeros(
                    (len(prepared), width), dtype=torch.bool, device=self.device
                )
                for row, item in enumerate(prepared):
                    length = item.input_ids.shape[1]
                    ids[row, :length] = item.input_ids[0]
                    attention[row, :length] = item.attention_mask[0]
                    masks[row, :length] = item.trigger_mask[0]
                logits_by_layer, _ = self._forward_router(
                    PreparedBatch(ids, attention, masks), False
                )
                logits_by_layer = {
                    layer_id: logits.reshape(len(chunk), width, logits.shape[-1])
                    if logits.ndim == 2 else logits
                    for layer_id, logits in logits_by_layer.items()
                }
                for row, (candidate_index, _text) in enumerate(chunk):
                    row_logits = {
                        layer_id: logits[row : row + 1]
                        for layer_id, logits in logits_by_layer.items()
                    }
                    loss, probability, hit_rate, target_margin, top_k_probability = (
                        self._joint_objective(row_logits, masks[row : row + 1])
                    )
                    loss_sums[candidate_index] += float(loss)
                    probability_sums[candidate_index] += float(probability)
                    hit_sums[candidate_index] += float(hit_rate)
                    margin_sums[candidate_index] += float(target_margin)
                    top_k_probability_sums[candidate_index] += float(top_k_probability)
                    counts[candidate_index] += 1
        return [
            TriggerMetrics(
                loss_sums[index] / counts[index],
                probability_sums[index] / counts[index],
                hit_sums[index] / counts[index],
                margin_sums[index] / counts[index],
                top_k_probability_sums[index] / counts[index],
            )
            for index in range(len(candidates))
        ]

    def evaluate_top_k_profile(
        self,
        trigger_ids: Sequence[int],
        texts: Optional[Sequence[str]] = None,
    ) -> TopKExpertProfile:
        """Summarize actual top-k gate shares for the trigger tokens.

        The reported probability is the post-top-k share (the selected expert
        probabilities renormalized to sum to one), averaged over trigger tokens
        for which the target expert is also present in the selected top-k set.
        Selection rate is the fraction of those target-selected trigger tokens
        on which an expert appeared in the selected top-k set.
        """
        evaluation_texts = list(self.texts if texts is None else texts)
        if not evaluation_texts:
            raise ValueError("Evaluation texts must not be empty.")

        profile_layer_id, profile_targets = self.target_specs[0]
        profile_handle = self.handles[profile_layer_id]
        profile_num_experts = int(getattr(self.routers[profile_layer_id], "weight").shape[0])
        share_sum = torch.zeros(profile_num_experts, dtype=torch.float64)
        selected_count = torch.zeros(profile_num_experts, dtype=torch.float64)
        token_count = 0
        routing_top_k = min(int(profile_handle.router_top_k), profile_num_experts)

        with torch.no_grad():
            for start in range(0, len(evaluation_texts), self.config.evaluation_batch_size):
                prompts = evaluation_texts[start : start + self.config.evaluation_batch_size]
                batch = self._prepare(prompts, trigger_ids)
                logits_by_layer, _ = self._forward_router(batch, False)
                logits = logits_by_layer[profile_layer_id]
                # OLMoE may return router logits flattened as [batch * seq, experts],
                # while the trigger mask remains [batch, seq]. Restore the token
                # dimensions before applying boolean indexing below.
                if logits.ndim == 2:
                    if logits.shape[0] != batch.trigger_mask.numel():
                        raise ValueError(
                            "Flattened router logits and trigger mask have different token counts."
                        )
                    logits = logits.reshape(
                        *batch.trigger_mask.shape,
                        logits.shape[-1],
                    )
                if logits.ndim != 3 or logits.shape[:2] != batch.trigger_mask.shape:
                    raise ValueError(
                        f"Router logits {tuple(logits.shape)} do not match trigger mask "
                        f"{tuple(batch.trigger_mask.shape)}."
                    )
                probabilities = torch.softmax(logits.float(), dim=-1)
                top_k_probabilities, top_k_indices = torch.topk(
                    probabilities,
                    k=routing_top_k,
                    dim=-1,
                )
                top_k_total = top_k_probabilities.sum(dim=-1, keepdim=True).clamp_min(
                    self.config.eps
                )
                top_k_shares = top_k_probabilities / top_k_total

                active = _stack_trigger_mask(
                    batch.trigger_mask,
                    profile_handle.stack_name,
                    profile_handle.backend,
                ).to(device=logits.device).bool()
                active_indices = top_k_indices[active]
                active_shares = top_k_shares[active]
                if active_indices.numel() == 0:
                    continue

                # Restrict the profile to the actual top-k set that contains all
                # target experts. This makes the reported competitors the other
                # co-selected experts, rather than experts from tokens where a
                # target was not selected.
                target_indices = torch.tensor(
                    profile_targets,
                    device=active_indices.device,
                )
                target_selected = (
                    active_indices.unsqueeze(-1)
                    .eq(target_indices.view(1, 1, -1))
                    # Reduce over the K selected experts first, leaving one
                    # boolean per target expert and token.
                    .any(dim=-2)
                    .all(dim=-1)
                )
                if not bool(target_selected.any()):
                    continue
                active_indices = active_indices[target_selected]
                active_shares = active_shares[target_selected]
                token_count += int(active_indices.shape[0])
                flat_indices = active_indices.reshape(-1).cpu()
                flat_shares = active_shares.reshape(-1).double().cpu()
                share_sum.scatter_add_(0, flat_indices, flat_shares)
                selected_count.scatter_add_(
                    0,
                    flat_indices,
                    torch.ones_like(flat_shares),
                )

        if token_count <= 0:
            # The target expert did not appear in top-k for any trigger token.
            # Keep logging usable during early search iterations.
            return TopKExpertProfile(experts=())
        mean_share = share_sum / float(token_count)
        selection_rate = selected_count / float(token_count)
        ranked = torch.argsort(mean_share, descending=True)
        # Always retain every target expert in the displayed aggregate set,
        # then fill the remaining slots with the strongest non-target experts.
        # This keeps the displayed target + competitors mass accounting exact.
        target_id_set = set(profile_targets)
        target_ids_sorted = sorted(
            target_id_set,
            key=lambda expert_id: float(mean_share[expert_id]),
            reverse=True,
        )
        non_target_ids = [
            int(expert_id)
            for expert_id in ranked.tolist()
            if int(expert_id) not in target_id_set
        ]
        displayed_ids = torch.tensor(
            (target_ids_sorted + non_target_ids)[:routing_top_k],
            dtype=torch.long,
        )
        target_indices_cpu = torch.tensor(profile_targets, dtype=torch.long)
        target_share = float(mean_share[target_indices_cpu].sum())
        displayed_share = float(mean_share[displayed_ids].sum())
        omitted_share = max(0.0, 1.0 - displayed_share)
        experts = tuple(
            (
                int(expert_id),
                float(mean_share[expert_id]),
                float(selection_rate[expert_id]),
            )
            for expert_id in displayed_ids.tolist()
        )
        return TopKExpertProfile(
            experts=experts,
            target_share=target_share,
            omitted_share=omitted_share,
        )

    def evaluate_many_robust(
        self,
        candidates: Sequence[Sequence[int]],
        texts: Optional[Sequence[str]] = None,
    ) -> List[RobustTriggerMetrics]:
        """Evaluate every candidate on every configured benign checkpoint."""
        per_candidate: List[List[TriggerMetrics]] = [[] for _ in candidates]
        if not per_candidate:
            raise ValueError("At least one candidate trigger is required.")
        try:
            for checkpoint_path in self.checkpoint_paths:
                self._activate_checkpoint(checkpoint_path)
                checkpoint_metrics = self.evaluate_many(candidates, texts=texts)
                for index, metrics in enumerate(checkpoint_metrics):
                    per_candidate[index].append(metrics)
        finally:
            # Gradients and token proposals always continue from the designated
            # primary checkpoint, regardless of screening order or exceptions.
            self._activate_checkpoint(self.primary_checkpoint_path)
        return [
            aggregate_robust_metrics(metrics, self.config.robust_margin_quantile)
            for metrics in per_candidate
        ]

    def _stable_text_ids(self, text: str) -> List[int]:
        """Return the fixed point of encode(decode(ids)) for a trigger string."""
        ids = list(self.tokenizer(text.strip(), add_special_tokens=False)["input_ids"])
        if not ids:
            raise ValueError("Initial trigger tokenized to an empty sequence.")
        for _ in range(8):
            decoded = self.tokenizer.decode(
                ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            encoded = list(self.tokenizer(decoded, add_special_tokens=False)["input_ids"])
            if encoded == ids:
                return [int(token_id) for token_id in ids]
            if not encoded:
                raise ValueError("Initial trigger became empty after tokenizer round trip.")
            ids = encoded
        raise RuntimeError("Initial trigger tokenization did not reach a stable round-trip encoding.")

    def _initial_ids(self) -> List[int]:
        initial = self.config.initial_trigger
        if self.config.initial_trigger_path:
            payload = json.loads(Path(self.config.initial_trigger_path).read_text(encoding="utf-8"))
            initial = payload.get("best_trigger", {}).get("trigger")
            if not isinstance(initial, str) or not initial.strip():
                raise ValueError("initial_trigger_path has no best_trigger.trigger string.")
        if initial:
            ids = self._stable_text_ids(initial)
            if len(ids) != self.config.trigger_length:
                print(
                    f"Initial trigger has {len(ids)} stable tokens; configured trigger_length="
                    f"{self.config.trigger_length}. Preserving the supplied trigger's stable length.",
                    flush=True,
                )
            return ids

        # Match the original I-GCG initialization: spaced exclamation marks as
        # text. Repeating the token ID for a bare '!' decodes to '!!!!', which
        # SentencePiece can re-encode into a shorter, unstable sequence.
        exact: Optional[List[int]] = None
        closest: Optional[List[int]] = None
        for count in range(1, max(4 * self.config.trigger_length, 64) + 1):
            candidate = self._stable_text_ids(" ".join(["!"] * count))
            if closest is None or abs(len(candidate) - self.config.trigger_length) < abs(
                len(closest) - self.config.trigger_length
            ):
                closest = candidate
            if len(candidate) == self.config.trigger_length:
                exact = candidate
                break
        ids = exact if exact is not None else closest
        if ids is None:
            raise RuntimeError("Could not construct a stable exclamation-mark trigger.")
        if len(ids) != self.config.trigger_length:
            print(
                f"Tokenizer cannot represent a stable spaced-'!' trigger with exactly "
                f"{self.config.trigger_length} tokens; using {len(ids)} tokens.",
                flush=True,
            )
        return ids

    def _record(
        self,
        iteration: int,
        ids: Sequence[int],
        metrics: TriggerMetrics,
        mutations: int,
        robust: Optional[RobustTriggerMetrics] = None,
        top_k_profile: Optional[TopKExpertProfile] = None,
    ) -> TriggerRecord:
        record = TriggerRecord(
            iteration,
            list(ids),
            self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip(),
            metrics.loss,
            metrics.target_probability,
            metrics.target_hit_rate,
            metrics.target_margin,
            mutations,
            target_expert_top_k_probability=metrics.target_top_k_probability,
            robust_worst_target_probability=(
                robust.worst_target_probability if robust is not None else None
            ),
            robust_worst_target_hit_rate=(
                robust.worst_target_hit_rate if robust is not None else None
            ),
            robust_worst_target_top_k_probability=(
                robust.worst_target_top_k_probability if robust is not None else None
            ),
            robust_low_quantile_margin=(
                robust.low_quantile_margin if robust is not None else None
            ),
        )
        self.records.append(record)
        robust_summary = ""
        if robust is not None:
            robust_summary = (
                f" worst_p={robust.worst_target_probability:.4f} "
                f"worst_topk_p={robust.worst_target_top_k_probability:.4f} "
                f"worst_hit={robust.worst_target_hit_rate:.4f} "
                f"q{self.config.robust_margin_quantile:g}_margin="
                f"{robust.low_quantile_margin:.4f}"
            )
        profile_summary = ""
        if top_k_profile is not None:
            if top_k_profile.experts:
                other_experts = [
                    item
                    for item in top_k_profile.experts
                    if item[0] not in self.target_experts
                ]
                profile_summary = " top8_others=" + ",".join(
                    f"expert{expert_id}:{share:.4f}"
                    for expert_id, share, _selection_rate in other_experts
                )
                listed_other_share = sum(share for _expert_id, share, _rate in other_experts)
                profile_summary += (
                    f" target_top8_mean={top_k_profile.target_share:.4f}"
                    f" listed_sum={top_k_profile.target_share + listed_other_share:.4f}"
                    f" omitted_top8_mass={top_k_profile.omitted_share:.4f}"
                )
            else:
                profile_summary = " top8_others=<target_not_selected>"
        print(
            f"[I-GCG {iteration:03d}] loss={metrics.loss:.6f} "
            f"p(target)={metrics.target_probability:.4f} hit_rate={metrics.target_hit_rate:.4f} "
            f"p(target|top-k)={metrics.target_top_k_probability:.4f} "
            f"margin={metrics.target_margin:.4f}{robust_summary} "
            f"mutations={mutations} trigger={record.trigger_text!r}{profile_summary}",
            flush=True,
        )
        total_targets = sum(len(experts) for _layer_id, experts in self.target_specs)
        if self.config.log_per_target_breakdown and total_targets > 1:
            for row in self.evaluate_breakdown(ids):
                print(
                    f"[I-GCG {iteration:03d}]   {row['layer_id']}:{row['expert']} "
                    f"p(target)={row['target_probability']:.4f} "
                    f"hit_rate={row['target_hit_rate']:.4f} "
                    f"p(target|top-k)={row['target_top_k_probability']:.4f} "
                    f"margin={row['target_margin']:.4f}",
                    flush=True,
                )
        return record

    def optimize(self) -> TriggerRecord:
        current = self._initial_ids()
        current_robust: Optional[RobustTriggerMetrics] = None
        if self.robust_screening_enabled:
            current_robust = self.evaluate_many_robust([current])[0]
            current_metrics = current_robust.mean
        else:
            current_metrics = self.evaluate(current)
        best = self._record(
            0,
            current,
            current_metrics,
            0,
            current_robust,
            self.evaluate_top_k_profile(current),
        )
        best_robust = current_robust
        excluded = set(self.tokenizer.all_special_ids)
        for iteration in range(1, self.config.iterations + 1):
            gradient = self.gradient(current)
            per_position = [
                top_k_tokens_from_gradient(
                    current, position, gradient, self.embedding.weight.detach(), self.config.top_k, excluded
                )
                for position in range(len(current))
            ]
            sampled_singles = sample_single_coordinate_candidates(
                current, per_position, self.config.candidate_batch_size, self.rng
            )
            singles = filter_roundtrip_candidates(
                self.tokenizer,
                current,
                sampled_singles,
            )
            if not singles:
                raise RuntimeError(
                    "All sampled candidates changed token length after decode/encode. "
                    "Increase candidate_batch_size or top_k."
                )
            if self.robust_screening_enabled:
                single_robust = self.evaluate_many_robust(singles)
                ranked_indices = sorted(
                    range(len(singles)),
                    key=lambda idx: robust_selection_key(single_robust[idx]),
                    reverse=True,
                )
                single_ranks = [0.0] * len(singles)
                for rank, index in enumerate(ranked_indices):
                    single_ranks[index] = float(rank)
            else:
                single_metrics = self.evaluate_many(singles)
                single_ranks = [metrics.loss for metrics in single_metrics]
            combined = combine_top_single_coordinate_candidates(
                current,
                singles,
                single_ranks,
                self.config.combine_top_p,
            )
            combined = filter_roundtrip_candidates(
                self.tokenizer,
                current,
                combined,
            )
            if not combined:
                raise RuntimeError(
                    "All combined I-GCG candidates changed token length after decode/encode."
                )
            candidate_robust: Optional[RobustTriggerMetrics] = None
            if self.robust_screening_enabled:
                combined_robust = self.evaluate_many_robust(combined)
                selected_index = max(
                    range(len(combined)),
                    key=lambda idx: robust_selection_key(combined_robust[idx]),
                )
                candidate_robust = combined_robust[selected_index]
                candidate_metrics = candidate_robust.mean
            else:
                combined_metrics = self.evaluate_many(combined)
                selected_index = min(
                    range(len(combined)), key=lambda idx: combined_metrics[idx].loss
                )
                candidate_metrics = combined_metrics[selected_index]
            candidate = combined[selected_index]
            mutations = sum(left != right for left, right in zip(current, candidate))
            current, current_metrics = list(candidate), candidate_metrics
            current_robust = candidate_robust
            record = self._record(
                iteration,
                current,
                current_metrics,
                mutations,
                current_robust,
                self.evaluate_top_k_profile(current),
            )
            if current_robust is not None:
                if best_robust is None or robust_selection_key(current_robust) > robust_selection_key(
                    best_robust
                ):
                    best, best_robust = record, current_robust
                search_probability = current_robust.worst_target_probability
                search_hit_rate = current_robust.worst_target_hit_rate
            else:
                if record.routing_loss < best.routing_loss:
                    best = record
                search_probability = current_metrics.target_probability
                search_hit_rate = current_metrics.target_hit_rate
            if (
                search_probability >= self.config.target_probability_threshold
                and search_hit_rate >= self.config.target_hit_rate_threshold
            ):
                if self.robust_screening_enabled:
                    heldout_robust = self.evaluate_many_robust(
                        [record.trigger_ids], texts=self.heldout_texts
                    )[0]
                    heldout_probability = heldout_robust.worst_target_probability
                    heldout_hit_rate = heldout_robust.worst_target_hit_rate
                else:
                    heldout_metrics = self.evaluate(record.trigger_ids, self.heldout_texts)
                    heldout_probability = heldout_metrics.target_probability
                    heldout_hit_rate = heldout_metrics.target_hit_rate
                if (
                    heldout_probability >= self.config.target_probability_threshold
                    and heldout_hit_rate >= self.config.target_hit_rate_threshold
                ):
                    break
        unique_records: Dict[Tuple[int, ...], TriggerRecord] = {}
        for candidate_record in self.records:
            key = tuple(candidate_record.trigger_ids)
            previous = unique_records.get(key)
            if previous is None or candidate_record.routing_loss < previous.routing_loss:
                unique_records[key] = candidate_record

        candidate_records = list(unique_records.values())
        if self.robust_screening_enabled:
            robust_heldout = self.evaluate_many_robust(
                [record.trigger_ids for record in candidate_records], texts=self.heldout_texts
            )
            robust_evaluated = list(zip(candidate_records, robust_heldout))
            for candidate_record, heldout in robust_evaluated:
                self.heldout_evaluations.append(
                    {
                        "iteration": candidate_record.iteration,
                        "trigger_ids": candidate_record.trigger_ids,
                        "trigger_text": candidate_record.trigger_text,
                        "mean": asdict(heldout.mean),
                        "worst_target_probability": heldout.worst_target_probability,
                        "worst_target_hit_rate": heldout.worst_target_hit_rate,
                        "worst_target_top_k_probability": heldout.worst_target_top_k_probability,
                        "low_quantile_margin": heldout.low_quantile_margin,
                        "per_checkpoint": [
                            {"checkpoint_path": path, **asdict(metrics)}
                            for path, metrics in zip(
                                self.checkpoint_paths, heldout.per_checkpoint
                            )
                        ],
                    }
                )
            best_record, robust_heldout_metrics = max(
                robust_evaluated,
                key=lambda item: robust_selection_key(item[1]),
            )
            heldout = robust_heldout_metrics.mean
            selected = replace(
                best_record,
                heldout_routing_loss=heldout.loss,
                heldout_target_probability=heldout.target_probability,
                heldout_target_hit_rate=heldout.target_hit_rate,
                heldout_target_margin=heldout.target_margin,
                heldout_target_top_k_probability=heldout.target_top_k_probability,
                heldout_robust_worst_target_probability=(
                    robust_heldout_metrics.worst_target_probability
                ),
                heldout_robust_worst_target_hit_rate=(
                    robust_heldout_metrics.worst_target_hit_rate
                ),
                heldout_robust_worst_target_top_k_probability=(
                    robust_heldout_metrics.worst_target_top_k_probability
                ),
                heldout_robust_low_quantile_margin=(
                    robust_heldout_metrics.low_quantile_margin
                ),
            )
            print(
                "[I-GCG robust held-out selection] "
                f"worst_hit={robust_heldout_metrics.worst_target_hit_rate:.4f} "
                f"q{self.config.robust_margin_quantile:g}_margin="
                f"{robust_heldout_metrics.low_quantile_margin:.4f} "
                f"mean_hit={heldout.target_hit_rate:.4f} "
                f"mean_p={heldout.target_probability:.4f} "
                f"mean_topk_p={heldout.target_top_k_probability:.4f} "
                f"trigger={selected.trigger_text!r}",
                flush=True,
            )
            return selected

        evaluated: List[Tuple[TriggerRecord, TriggerMetrics]] = []
        for candidate_record in candidate_records:
            heldout_metrics = self.evaluate(candidate_record.trigger_ids, self.heldout_texts)
            evaluated.append((candidate_record, heldout_metrics))
            self.heldout_evaluations.append(
                {
                    "iteration": candidate_record.iteration,
                    "trigger_ids": candidate_record.trigger_ids,
                    "trigger_text": candidate_record.trigger_text,
                    "routing_loss": heldout_metrics.loss,
                    "target_probability": heldout_metrics.target_probability,
                    "target_top_k_probability": heldout_metrics.target_top_k_probability,
                    "target_hit_rate": heldout_metrics.target_hit_rate,
                    "target_margin": heldout_metrics.target_margin,
                }
            )
        best_record, heldout = max(
            evaluated,
            key=lambda item: heldout_selection_key(item[1]),
        )
        selected = replace(
            best_record,
            heldout_routing_loss=heldout.loss,
            heldout_target_probability=heldout.target_probability,
            heldout_target_hit_rate=heldout.target_hit_rate,
            heldout_target_margin=heldout.target_margin,
            heldout_target_top_k_probability=heldout.target_top_k_probability,
        )
        print(
            "[I-GCG held-out selection] "
            f"hit_rate={heldout.target_hit_rate:.4f} margin={heldout.target_margin:.4f} "
            f"p(target)={heldout.target_probability:.4f} loss={heldout.loss:.6f} "
            f"p(target|top-k)={heldout.target_top_k_probability:.4f} "
            f"trigger={selected.trigger_text!r}",
            flush=True,
        )
        return selected


def save_results(
    path: Path,
    config: IGCGTriggerConfig,
    best: TriggerRecord,
    history: Sequence[TriggerRecord],
    heldout_evaluations: Sequence[Dict[str, Any]],
):
    path.mkdir(parents=True, exist_ok=True)
    result = {
        "method": "routing_i_gcg",
        "target": {
            "layer_id": config.target_layer,
            # Keep expert_id for backward-compatible consumers and add the
            # complete target set for multi-expert runs.
            "expert_id": config.target_expert,
            "expert_ids": list(config.target_experts or (config.target_expert,)),
            "targets": [
                {"layer_id": layer_id, "expert_ids": list(experts)}
                for layer_id, experts in (
                    config.target_specs
                    or ((config.target_layer, config.target_experts or (config.target_expert,)),)
                )
            ],
        },
        "best_trigger": {
            "trigger": best.trigger_text,
            "trigger_token_ids": best.trigger_ids,
            "routing_loss": best.routing_loss,
            "mean_target_probability": best.target_expert_probability,
            "mean_target_probability_in_top_k": best.target_expert_top_k_probability,
            "target_hit_rate": best.target_expert_hit_rate,
            "mean_target_margin": best.target_expert_margin,
            "heldout_routing_loss": best.heldout_routing_loss,
            "heldout_mean_target_probability": best.heldout_target_probability,
            "heldout_mean_target_probability_in_top_k": best.heldout_target_top_k_probability,
            "heldout_target_hit_rate": best.heldout_target_hit_rate,
            "heldout_mean_target_margin": best.heldout_target_margin,
            "robust_worst_target_probability": best.robust_worst_target_probability,
            "robust_worst_target_probability_in_top_k": (
                best.robust_worst_target_top_k_probability
            ),
            "robust_worst_target_hit_rate": best.robust_worst_target_hit_rate,
            "robust_low_quantile_margin": best.robust_low_quantile_margin,
            "heldout_robust_worst_target_probability": (
                best.heldout_robust_worst_target_probability
            ),
            "heldout_robust_worst_target_probability_in_top_k": (
                best.heldout_robust_worst_target_top_k_probability
            ),
            "heldout_robust_worst_target_hit_rate": best.heldout_robust_worst_target_hit_rate,
            "heldout_robust_low_quantile_margin": best.heldout_robust_low_quantile_margin,
        },
        "config": asdict(config),
    }
    (path / "selected_triggers.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (path / "optimization_history.json").write_text(
        json.dumps([asdict(record) for record in history], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (path / "heldout_evaluations.json").write_text(
        json.dumps(list(heldout_evaluations), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _device(value: str) -> torch.device:
    resolved = torch.device("cuda" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return resolved


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="I-GCG-style MoE routing trigger optimization.")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument(
        "--robust_checkpoint_paths",
        nargs="+",
        default=[],
        help=(
            "Additional benign checkpoints used only for robust candidate screening. "
            "The primary --checkpoint_path remains the gradient/search model."
        ),
    )
    parser.add_argument(
        "--robust_margin_quantile",
        type=float,
        default=0.1,
        help="Checkpoint-level margin quantile used after worst-checkpoint hit rate.",
    )
    parser.add_argument("--model_name_or_path", default="model_ckpt/olmoe")
    parser.add_argument(
        "--target_layer",
        help="Legacy single-layer target. Use --target_spec for cross-layer optimization.",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--target_expert",
        type=int,
        nargs="+",
        help="One or more target expert IDs. A single ID preserves legacy behavior.",
    )
    target.add_argument("--lfe_profile")
    target.add_argument(
        "--target_spec",
        nargs="+",
        help=(
            "Cross-layer targets in LAYER:EXPERT[,EXPERT...] form; for example "
            "decoder.2:3,7 decoder.6:1. All target layers must use the same stack."
        ),
    )
    parser.add_argument("--trigger_length", type=int, default=10)
    parser.add_argument("--initial_trigger")
    parser.add_argument("--initial_trigger_path")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--candidate_batch_size", type=int, default=256)
    parser.add_argument("--top_k", type=int, default=256)
    parser.add_argument("--combine_top_p", type=int, default=7)
    parser.add_argument("--calibration_file", default="data/calibration.json")
    parser.add_argument("--calibration_samples", type=int, default=100)
    parser.add_argument("--heldout_samples", type=int, default=20)
    parser.add_argument("--gradient_batch_size", type=int, default=24)
    parser.add_argument("--evaluation_batch_size", type=int, default=24)
    parser.add_argument("--trigger_position", choices=["append", "prepend", "fixed_position"], default="append")
    parser.add_argument("--fixed_position", type=int)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--margin_weight", type=float, default=0.0)
    parser.add_argument(
        "--top_k_share_weight",
        type=float,
        default=1.0,
        help="Weight on the top-k renormalized target-share objective (Method 1).",
    )
    parser.add_argument(
        "--full_softmax_weight",
        type=float,
        default=0.0,
        help="Weight on the legacy full-softmax target-probability objective.",
    )
    parser.add_argument(
        "--sibling_suppression_weight",
        type=float,
        default=0.1,
        help="Weight on suppressing co-selected sibling logits inside top-k (Method 2).",
    )
    parser.add_argument("--target_probability_threshold", type=float, default=0.97)
    parser.add_argument("--target_hit_rate_threshold", type=float, default=0.95)
    parser.add_argument("--output_dir", default="outputs/igcg_trigger_optimization")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log_per_target_breakdown",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log per-(layer, expert) metrics each iteration for multi-target runs.",
    )
    return parser


def parse_config(argv: Optional[Sequence[str]] = None) -> IGCGTriggerConfig:
    args = _parser().parse_args(argv)
    values = vars(args)
    profile = values.pop("lfe_profile")
    raw_target_specs = values.pop("target_spec")
    values["robust_checkpoint_paths"] = tuple(values["robust_checkpoint_paths"])
    target_specs: Tuple[Tuple[str, Tuple[int, ...]], ...] = ()
    if raw_target_specs:
        if values["target_layer"]:
            raise ValueError("--target_layer cannot be combined with --target_spec.")
        parsed: List[Tuple[str, Tuple[int, ...]]] = []
        for raw in raw_target_specs:
            if ":" not in raw:
                raise ValueError(
                    f"Invalid --target_spec '{raw}'; expected LAYER:EXPERT[,EXPERT...]."
                )
            layer_id, raw_experts = raw.split(":", 1)
            layer_id = layer_id.strip()
            if not layer_id or not raw_experts.strip():
                raise ValueError(
                    f"Invalid --target_spec '{raw}'; layer and expert IDs are required."
                )
            try:
                experts = tuple(int(item.strip()) for item in raw_experts.split(","))
            except ValueError as exc:
                raise ValueError(f"Invalid expert IDs in --target_spec '{raw}'.") from exc
            if not experts or any(expert < 0 for expert in experts):
                raise ValueError(f"Expert IDs in --target_spec '{raw}' must be >= 0.")
            parsed.append((layer_id, experts))
        target_specs = tuple(parsed)
        values["target_layer"] = target_specs[0][0]
        values["target_expert"] = int(target_specs[0][1][0])
        values["target_experts"] = target_specs[0][1]
        values["target_specs"] = target_specs
        target_experts = target_specs[0][1]
    elif values["target_expert"] is None:
        selected = select_low_frequency_target(profile, values["target_layer"])
        target_experts = (selected,)
    else:
        target_experts = tuple(values["target_expert"])
    if not target_experts:
        raise ValueError("At least one target expert is required.")
    values["target_expert"] = int(target_experts[0])
    values["target_experts"] = target_experts
    if not target_specs:
        if not values["target_layer"]:
            raise ValueError("--target_layer is required unless --target_spec is used.")
        values["target_specs"] = ((values["target_layer"], target_experts),)
    config = IGCGTriggerConfig(**values)
    positive = (
        config.trigger_length,
        config.iterations,
        config.candidate_batch_size,
        config.top_k,
        config.combine_top_p,
        config.calibration_samples,
        config.heldout_samples,
        config.gradient_batch_size,
        config.evaluation_batch_size,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("Trigger/search/batch parameters must all be > 0.")
    if config.combine_top_p > config.candidate_batch_size:
        raise ValueError("combine_top_p cannot exceed candidate_batch_size.")
    if config.margin_weight < 0:
        raise ValueError("margin_weight must be >= 0.")
    if not 0 <= config.robust_margin_quantile <= 1:
        raise ValueError("robust_margin_quantile must be in [0, 1].")
    if config.heldout_samples >= config.calibration_samples:
        raise ValueError("heldout_samples must be smaller than calibration_samples.")
    for value in (config.target_probability_threshold, config.target_hit_rate_threshold):
        if not 0 <= value <= 1:
            raise ValueError("Stopping thresholds must be in [0, 1].")
    return config


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_config(argv)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = _device(config.device)
    calibration_texts = load_calibration_texts(
        config.calibration_file, config.calibration_samples
    )
    optimization_texts, heldout_texts = split_calibration_texts(
        calibration_texts, config.heldout_samples, config.seed
    )
    model, tokenizer, metadata = load_victim_model(
        config.model_name_or_path, config.checkpoint_path, device
    )
    print(f"Loaded checkpoint metadata: {metadata}", flush=True)
    print(
        f"Calibration split: optimization={len(optimization_texts)}, "
        f"held_out={len(heldout_texts)}",
        flush=True,
    )
    optimizer = IGCGTriggerOptimizer(
        model, tokenizer, config, optimization_texts, heldout_texts, device
    )
    if optimizer.robust_screening_enabled:
        print(
            "Robust checkpoint screening enabled: "
            f"checkpoints={len(optimizer.checkpoint_paths)}, "
            f"margin_quantile={config.robust_margin_quantile:g}",
            flush=True,
        )
        for checkpoint_path in optimizer.checkpoint_paths:
            print(f"  - {checkpoint_path}", flush=True)
    best = optimizer.optimize()
    output = Path(config.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(output, config, best, optimizer.records, optimizer.heldout_evaluations)

    # Per-(layer, expert) breakdown for the selected trigger, so cross-layer /
    # multi-expert runs can be read one target at a time instead of merged.
    breakdown = {
        "trigger": best.trigger_text,
        "trigger_token_ids": best.trigger_ids,
        "optimization": optimizer.evaluate_breakdown(best.trigger_ids, optimization_texts),
        "heldout": optimizer.evaluate_breakdown(best.trigger_ids, heldout_texts),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "per_target_breakdown.json").write_text(
        json.dumps(breakdown, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("[I-GCG per-target breakdown] held-out:", flush=True)
    for row in breakdown["heldout"]:
        print(
            f"  {row['layer_id']}:{row['expert']}  "
            f"hit_rate={row['target_hit_rate']:.4f} "
            f"margin={row['target_margin']:.4f} "
            f"p(target)={row['target_probability']:.4f} "
            f"p(target|top-k)={row['target_top_k_probability']:.4f}",
            flush=True,
        )
    print(f"Best routing trigger saved to {output / 'selected_triggers.json'}", flush=True)


if __name__ == "__main__":
    main()
