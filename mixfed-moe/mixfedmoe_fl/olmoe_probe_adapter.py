from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common import NDArrays, RecordDict

from mixfedmoe_fl.expert_probe import (
    choose_lazy_probe_expert,
    inject_paired_probe,
    load_probe_state,
    observe_global_and_update_history,
    rank_lazy_candidates_across_layers,
    rank_lazy_expert_candidates,
    save_probe_state,
)
from mixfedmoe_fl.olmoe_adapter import (
    OLMOE_PACKED_EXPERT_PATTERN,
    make_olmoe_expert_slice_name,
    parse_olmoe_expert_slice_name,
)


OLMOE_PROBE_STATE_KEY = "olmoe_expert_probe.state_json"


def load_olmoe_probe_state(client_state: RecordDict) -> Dict[str, Any]:
    return load_probe_state(client_state, state_key=OLMOE_PROBE_STATE_KEY)


def save_olmoe_probe_state(
    client_state: RecordDict,
    state: Mapping[str, Any],
) -> None:
    save_probe_state(
        client_state,
        state,
        state_key=OLMOE_PROBE_STATE_KEY,
    )


def parse_olmoe_probe_parameter(name: str) -> Optional[Tuple[str, int]]:
    """Parse only the virtual expert-slice namespace used by OLMoE probes."""
    parsed = parse_olmoe_expert_slice_name(name)
    if parsed is None:
        return None
    layer_id, expert_id, _kind = parsed
    return layer_id, expert_id


def project_packed_olmoe_experts(
    parameter_names: Sequence[str],
    parameters: Sequence[np.ndarray],
) -> Tuple[List[str], NDArrays]:
    """Expose packed ``[experts, ...]`` tensors as zero-copy virtual slices.

    The virtual names are identical to OLMoE's sparse-upload names.  They are
    deliberately outside the Switch parameter namespace, so a persisted
    tracker from one backend cannot be interpreted by the other backend.
    """
    if len(parameter_names) != len(parameters):
        raise ValueError("OLMoE probe parameter names and arrays have different lengths.")

    projected_names: List[str] = []
    projected_parameters: NDArrays = []
    seen: set[str] = set()
    for name, raw_array in zip(parameter_names, parameters):
        packed_match = OLMOE_PACKED_EXPERT_PATTERN.match(str(name))
        if packed_match is None:
            # Also accept an already projected list in focused unit tests and
            # adapter-to-adapter calls, without accepting Switch names.
            parsed_slice = parse_olmoe_expert_slice_name(str(name))
            if parsed_slice is None:
                continue
            if str(name) in seen:
                raise ValueError(f"Duplicate OLMoE probe parameter name: {name!r}.")
            projected_names.append(str(name))
            projected_parameters.append(np.asarray(raw_array))
            seen.add(str(name))
            continue

        layer_idx = int(packed_match.group(1))
        kind = packed_match.group(2)
        packed = np.asarray(raw_array)
        if packed.ndim < 2 or packed.shape[0] < 1:
            raise ValueError(
                f"Packed OLMoE expert tensor {name!r} must have shape [experts, ...], "
                f"got {packed.shape}."
            )
        layer_id = f"decoder.{layer_idx}"
        for expert_id in range(int(packed.shape[0])):
            virtual_name = make_olmoe_expert_slice_name(
                layer_id,
                expert_id,
                kind,
            )
            if virtual_name in seen:
                raise ValueError(f"Duplicate OLMoE probe parameter name: {virtual_name!r}.")
            # NumPy indexing returns a view.  History tracking is read-only, so
            # this does not duplicate OLMoE's full expert bank in host memory.
            projected_names.append(virtual_name)
            projected_parameters.append(packed[expert_id])
            seen.add(virtual_name)
    return projected_names, projected_parameters


def observe_olmoe_global_and_update_history(
    *,
    state: MutableMapping[str, Any],
    parameter_names: Sequence[str],
    parameters: Sequence[np.ndarray],
    server_round: int,
    target_layer: Optional[str],
    candidate_pool: int,
    repetitions: int,
    drift_window: int = 5,
    max_consistency_error: float = float("inf"),
) -> Dict[str, Any]:
    names, slices = project_packed_olmoe_experts(parameter_names, parameters)
    return observe_global_and_update_history(
        state=state,
        parameter_names=names,
        parameters=slices,
        server_round=server_round,
        target_layer=target_layer,
        candidate_pool=candidate_pool,
        repetitions=repetitions,
        drift_window=drift_window,
        max_consistency_error=max_consistency_error,
        parameter_parser=parse_olmoe_probe_parameter,
        # Track one tensor per expert during model-wide discovery.  The fused
        # gate/up projection is consistently present in every OLMoE layer.
        automatic_tensor_suffix=".gate_up_proj",
    )


def inject_olmoe_paired_probe(
    *,
    state: MutableMapping[str, Any],
    incoming_names: Sequence[str],
    incoming_parameters: Sequence[np.ndarray],
    outgoing_global_names: Sequence[str],
    outgoing_parameters: NDArrays,
    server_round: int,
    target_layer: str,
    target_experts: Sequence[int],
    warmup_observations: int,
    pairs: int,
    relative_scale: float,
    max_relative_scale: float,
    noise_multiplier: float,
    seed: int,
    client_id: int,
    append_pending: bool = False,
) -> List[Dict[str, Any]]:
    names, slices = project_packed_olmoe_experts(incoming_names, incoming_parameters)
    return inject_paired_probe(
        state=state,
        incoming_names=names,
        incoming_parameters=slices,
        outgoing_global_names=outgoing_global_names,
        outgoing_parameters=outgoing_parameters,
        server_round=server_round,
        target_layer=target_layer,
        target_experts=target_experts,
        warmup_observations=warmup_observations,
        pairs=pairs,
        relative_scale=relative_scale,
        max_relative_scale=max_relative_scale,
        noise_multiplier=noise_multiplier,
        seed=seed,
        client_id=client_id,
        append_pending=append_pending,
        parameter_parser=parse_olmoe_probe_parameter,
    )


def rank_olmoe_lazy_expert_candidates(**kwargs: Any) -> List[Dict[str, Any]]:
    return rank_lazy_expert_candidates(
        **kwargs,
        parameter_parser=parse_olmoe_probe_parameter,
    )


def rank_olmoe_lazy_candidates_across_layers(
    **kwargs: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return rank_lazy_candidates_across_layers(
        **kwargs,
        parameter_parser=parse_olmoe_probe_parameter,
    )


def choose_olmoe_lazy_probe_expert(**kwargs: Any) -> Optional[Dict[str, Any]]:
    return choose_lazy_probe_expert(
        **kwargs,
        parameter_parser=parse_olmoe_probe_parameter,
    )


def capture_olmoe_router_logits(
    router: nn.Module,
    inputs: Tuple[Any, ...],
) -> torch.Tensor:
    """Recover raw logits from an OLMoE router that returns a Top-K tuple."""
    if not inputs or not isinstance(inputs[0], torch.Tensor):
        raise RuntimeError("OLMoE probe router did not receive tensor hidden states.")
    weight = getattr(router, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise RuntimeError("OLMoE probe router does not expose a matrix weight.")
    bias = getattr(router, "bias", None)
    if isinstance(bias, torch.Tensor):
        bias = bias.float()
    return F.linear(
        inputs[0].float(),
        weight.float(),
        bias,
    )


def _topk_boundary_metrics(
    router_logits: np.ndarray,
    router_weight: np.ndarray,
    *,
    routing_top_k: int,
    margin: float,
    quantile: float,
    eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    logits = np.asarray(router_logits, dtype=np.float64)
    weight = np.asarray(router_weight, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError("OLMoE trigger logits must be non-empty [tokens, experts].")
    if weight.ndim != 2 or weight.shape[0] != logits.shape[1]:
        raise ValueError("OLMoE router logits and weight have incompatible shapes.")
    if not 1 <= int(routing_top_k) < int(logits.shape[1]):
        raise ValueError("OLMoE probe routing_top_k must be in [1, num_experts - 1].")

    tokens, experts = logits.shape
    distances = np.zeros((tokens, experts), dtype=np.float64)
    margins = np.zeros((tokens, experts), dtype=np.float64)
    hits = np.zeros((tokens, experts), dtype=np.float64)
    all_indices = np.arange(experts)
    for expert_id in range(experts):
        competitors = all_indices[all_indices != expert_id]
        competitor_logits = logits[:, competitors]
        # The K-th strongest non-target expert is the admission boundary for
        # a Top-K router.  This is intentionally different from Switch's
        # strongest-competitor (Top-1) boundary.
        order = np.argsort(-competitor_logits, axis=1, kind="stable")
        boundary_positions = order[:, int(routing_top_k) - 1]
        boundary_experts = competitors[boundary_positions]
        rows = np.arange(tokens)
        boundary_logits = logits[rows, boundary_experts]
        target_margin = logits[:, expert_id] - boundary_logits
        row_delta = weight[expert_id] - weight[boundary_experts]
        denominator = np.maximum(np.linalg.norm(row_delta, axis=-1), float(eps))
        distances[:, expert_id] = np.maximum(float(margin) - target_margin, 0.0) / denominator
        margins[:, expert_id] = target_margin
        hits[:, expert_id] = target_margin >= 0.0

    return {
        "q_distance": np.quantile(distances, float(quantile), axis=0),
        "median_distance": np.median(distances, axis=0),
        "q10_margin": np.quantile(margins, 0.1, axis=0),
        "hit_rate": np.mean(hits, axis=0),
    }


def rank_olmoe_candidates_by_boundary_distance(
    *,
    candidates: Sequence[Mapping[str, Any]],
    trigger_router_logits: Sequence[np.ndarray],
    router_weight: np.ndarray,
    routing_top_k: int,
    margin: float = 0.0,
    quantile: float = 0.8,
) -> List[Dict[str, Any]]:
    """Rank OLMoE candidates against the Top-K admission boundary."""
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("quantile must be in [0, 1].")
    if not trigger_router_logits:
        raise ValueError("At least one OLMoE trigger-logit array is required.")
    if margin < 0.0:
        raise ValueError("margin must be >= 0.")

    weight = np.asarray(router_weight)
    per_trigger = [
        _topk_boundary_metrics(
            raw_logits,
            weight,
            routing_top_k=int(routing_top_k),
            margin=float(margin),
            quantile=float(quantile),
        )
        for raw_logits in trigger_router_logits
    ]
    ranked: List[Dict[str, Any]] = []
    for raw_candidate in candidates:
        expert_id = int(raw_candidate["expert_id"])
        if not 0 <= expert_id < int(weight.shape[0]):
            raise ValueError(f"Candidate expert {expert_id} is outside the router range.")
        best_trigger = min(
            range(len(per_trigger)),
            key=lambda index: (
                float(per_trigger[index]["q_distance"][expert_id]),
                -float(per_trigger[index]["q10_margin"][expert_id]),
                -float(per_trigger[index]["hit_rate"][expert_id]),
                index,
            ),
        )
        metrics = per_trigger[best_trigger]
        item = dict(raw_candidate)
        topk_hit_rate = float(metrics["hit_rate"][expert_id])
        item.update(
            {
                "boundary_best_trigger_index": int(best_trigger),
                "boundary_distance_quantile": float(metrics["q_distance"][expert_id]),
                "boundary_distance_median": float(metrics["median_distance"][expert_id]),
                "boundary_margin_q10": float(metrics["q10_margin"][expert_id]),
                "boundary_topk_hit_rate": topk_hit_rate,
                "boundary_router_top_k": int(routing_top_k),
                # The common scheduler sorts this legacy field.  For OLMoE it
                # explicitly carries Top-K admission rate, not Top-1 hit rate.
                "boundary_top1_hit_rate": topk_hit_rate,
            }
        )
        ranked.append(item)

    ranked.sort(
        key=lambda item: (
            float(item["boundary_distance_quantile"]),
            -float(item["boundary_margin_q10"]),
            -float(item["boundary_topk_hit_rate"]),
            float(item.get("normalized_activity_drift", float("inf"))),
            int(item["expert_id"]),
        )
    )
    return ranked

