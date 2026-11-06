from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from flwr.common import ArrayRecord, NDArrays, RecordDict


EXPERT_PARAM_PATTERN = re.compile(
    r"^(encoder|decoder)\.block\.(\d+)\.layer\.\d+\.mlp\.experts\.expert_(\d+)\.(.+)$"
)
PROBE_STATE_KEY = "expert_probe.state_json"
PROBE_STATE_VERSION = 1


def parse_expert_parameter(name: str) -> Optional[Tuple[str, int]]:
    match = EXPERT_PARAM_PATTERN.match(name)
    if match is None:
        return None
    return f"{match.group(1)}.{int(match.group(2))}", int(match.group(3))


def map_outgoing_parameter_names(
    return_names: Sequence[str],
    mode: str,
    assignments: Mapping[str, Sequence[int]],
) -> List[str]:
    """Map a drop client's local expert indices back to global parameter names."""
    if mode != "drop":
        return list(return_names)
    mapped: List[str] = []
    for name in return_names:
        match = EXPERT_PARAM_PATTERN.match(name)
        if match is None:
            mapped.append(name)
            continue
        layer_id = f"{match.group(1)}.{int(match.group(2))}"
        local_expert = int(match.group(3))
        layer_assignment = assignments.get(layer_id)
        if layer_assignment is None or local_expert >= len(layer_assignment):
            raise ValueError(
                f"Cannot map probe parameter '{name}': missing/out-of-range assignment for {layer_id}."
            )
        global_expert = int(layer_assignment[local_expert])
        mapped.append(
            name.replace(
                f".experts.expert_{local_expert}.",
                f".experts.expert_{global_expert}.",
                1,
            )
        )
    return mapped


def _empty_state() -> Dict[str, Any]:
    return {
        "version": PROBE_STATE_VERSION,
        "trackers": {},
        "pending": [],
        "results": {},
        "attempts": {},
        "scheduled_expert": None,
        "scheduled_target": None,
        "phase": "scan",
        "confirmation_expert": None,
        "confirmation_expert_id": None,
        "confirmation_target": None,
        "confirmation_successes": 0,
        "candidate_queue": [],
        "rejected_experts": [],
        "rejected_candidates": [],
        "found_low_frequency_expert": None,
        "scheduled_targets": [],
        "batch_candidates": {},
        "boundary_screen": None,
    }


def load_probe_state(client_state: RecordDict) -> Dict[str, Any]:
    record = client_state.array_records.get(PROBE_STATE_KEY)
    if record is None:
        return _empty_state()
    arrays = record.to_numpy_ndarrays()
    if len(arrays) != 1:
        raise ValueError("Expert probe state must contain exactly one encoded JSON array.")
    payload = bytes(np.asarray(arrays[0], dtype=np.uint8).tolist()).decode("utf-8")
    state = json.loads(payload)
    if not isinstance(state, dict) or int(state.get("version", -1)) != PROBE_STATE_VERSION:
        raise ValueError("Unsupported or malformed expert probe state.")
    state.setdefault("attempts", {})
    # Migrate probe state written by the earlier server-side assignment-request
    # implementation without discarding an in-flight target.
    state.setdefault("scheduled_expert", state.pop("requested_expert", None))
    state.setdefault("scheduled_target", None)
    state.setdefault("phase", "scan")
    state.setdefault("confirmation_expert", None)
    state.setdefault("confirmation_expert_id", None)
    state.setdefault("confirmation_target", None)
    state.setdefault("confirmation_successes", 0)
    state.setdefault("candidate_queue", [])
    state.setdefault("rejected_experts", [])
    state.setdefault("rejected_candidates", [])
    state.setdefault("found_low_frequency_expert", None)
    state.setdefault("scheduled_targets", [])
    state.setdefault("batch_candidates", {})
    state.setdefault("boundary_screen", None)
    if not state["scheduled_targets"]:
        legacy_target = state.get("scheduled_target")
        if isinstance(legacy_target, dict):
            state["scheduled_targets"] = [legacy_target]
    return state


def boundary_distance_lower_bounds(
    router_logits: np.ndarray,
    router_weight: np.ndarray,
    *,
    margin: float = 0.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return the pairwise-boundary distance lower bound for every token/expert.

    For target expert ``e`` and competitor ``j``, the distance to the
    corresponding linear-router half-space is

        relu(margin - z_e + z_j) / ||w_e - w_j||.

    Any hidden-state displacement that makes ``e`` top-1 must cross every
    violated half-space, so the maximum pairwise distance is a cheap lower
    bound on the exact distance to the intersection of all target half-spaces.
    """
    logits = np.asarray(router_logits, dtype=np.float64)
    weight = np.asarray(router_weight, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError("router_logits must have shape [tokens, experts].")
    if weight.ndim != 2:
        raise ValueError("router_weight must have shape [experts, hidden_size].")
    if logits.shape[1] != weight.shape[0]:
        raise ValueError("Router logits and weight disagree on the number of experts.")
    if margin < 0.0:
        raise ValueError("margin must be >= 0.")
    if eps <= 0.0:
        raise ValueError("eps must be > 0.")

    # [tokens, target, competitor]
    deficits = np.maximum(
        float(margin) - logits[:, :, None] + logits[:, None, :],
        0.0,
    )
    row_differences = weight[:, None, :] - weight[None, :, :]
    row_distances = np.linalg.norm(row_differences, axis=-1)
    denominator = np.maximum(row_distances, float(eps))
    pairwise = deficits / denominator[None, :, :]
    diagonal = np.arange(weight.shape[0])
    pairwise[:, diagonal, diagonal] = 0.0
    return np.max(pairwise, axis=-1).astype(np.float32, copy=False)


def rank_candidates_by_boundary_distance(
    *,
    candidates: Sequence[Mapping[str, Any]],
    trigger_router_logits: Sequence[np.ndarray],
    router_weight: np.ndarray,
    margin: float = 0.0,
    quantile: float = 0.8,
) -> List[Dict[str, Any]]:
    """Rank passive fixed-layer candidates with a shared trigger bank.

    Each item in ``trigger_router_logits`` contains the target-router logits on
    the exact trigger-token positions for one shared-bank trigger.  For each
    expert we first aggregate across contexts/tokens, then choose the best
    *single* bank trigger.  This preserves the universal-trigger ordering
    (best trigger outside the context aggregation) and compares every expert
    under identical trigger and compute budgets.
    """
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("quantile must be in [0, 1].")
    if not trigger_router_logits:
        raise ValueError("At least one shared-bank router-logit array is required.")

    weight = np.asarray(router_weight)
    num_experts = int(weight.shape[0])
    per_trigger: List[Dict[str, np.ndarray]] = []
    for raw_logits in trigger_router_logits:
        logits = np.asarray(raw_logits, dtype=np.float64)
        if logits.ndim != 2 or logits.shape[0] == 0:
            raise ValueError("Each trigger must provide non-empty [tokens, experts] logits.")
        distances = boundary_distance_lower_bounds(
            logits,
            weight,
            margin=margin,
        ).astype(np.float64, copy=False)
        other_logits = np.broadcast_to(logits[:, None, :], (logits.shape[0], num_experts, num_experts)).copy()
        diagonal = np.arange(num_experts)
        other_logits[:, diagonal, diagonal] = -np.inf
        target_margins = logits - np.max(other_logits, axis=-1)
        top1 = np.argmax(logits, axis=-1)
        per_trigger.append(
            {
                "q_distance": np.quantile(distances, float(quantile), axis=0),
                "median_distance": np.median(distances, axis=0),
                "q10_margin": np.quantile(target_margins, 0.1, axis=0),
                "hit_rate": np.asarray(
                    [np.mean(top1 == expert_id) for expert_id in range(num_experts)],
                    dtype=np.float64,
                ),
            }
        )

    ranked: List[Dict[str, Any]] = []
    for candidate in candidates:
        expert_id = int(candidate["expert_id"])
        if not 0 <= expert_id < num_experts:
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
        item = dict(candidate)
        item.update(
            {
                "boundary_best_trigger_index": int(best_trigger),
                "boundary_distance_quantile": float(metrics["q_distance"][expert_id]),
                "boundary_distance_median": float(metrics["median_distance"][expert_id]),
                "boundary_margin_q10": float(metrics["q10_margin"][expert_id]),
                "boundary_top1_hit_rate": float(metrics["hit_rate"][expert_id]),
            }
        )
        ranked.append(item)

    ranked.sort(
        key=lambda item: (
            float(item["boundary_distance_quantile"]),
            -float(item["boundary_margin_q10"]),
            -float(item["boundary_top1_hit_rate"]),
            float(item.get("normalized_activity_drift", float("inf"))),
            int(item["expert_id"]),
        )
    )
    return ranked


def save_probe_state(client_state: RecordDict, state: Mapping[str, Any]) -> None:
    payload = json.dumps(state, separators=(",", ":"), sort_keys=True).encode("utf-8")
    client_state.array_records[PROBE_STATE_KEY] = ArrayRecord(
        numpy_ndarrays=[np.frombuffer(payload, dtype=np.uint8).copy()]
    )


def _candidate_indices(size: int, pool_size: int) -> List[int]:
    count = min(max(int(pool_size), 1), int(size))
    if count == size:
        return list(range(size))
    return [int(value) for value in np.linspace(0, size - 1, num=count, dtype=np.int64)]


def _probe_sign(seed: int, client_id: int, server_round: int, expert_key: str) -> int:
    digest = hashlib.sha256(
        f"{seed}:{client_id}:{server_round}:{expert_key}".encode("utf-8")
    ).digest()
    return 1 if digest[0] & 1 else -1


def _rankings(state: Mapping[str, Any], repetitions: int) -> List[Dict[str, Any]]:
    rankings: List[Dict[str, Any]] = []
    for expert_key, raw_values in state.get("results", {}).items():
        values = np.asarray(raw_values[-max(int(repetitions), 1) :], dtype=np.float64)
        if values.size == 0:
            continue
        robust_values = np.clip(values, -0.5, 1.5)
        median = float(np.median(robust_values))
        low_quantile = float(np.quantile(robust_values, 0.2))
        mad = float(np.median(np.abs(robust_values - median)))
        rankings.append(
            {
                "expert": str(expert_key),
                "observations": int(values.size),
                "median_retention": median,
                "q20_retention": low_quantile,
                "retention_mad": mad,
                "score": float(low_quantile - mad),
            }
        )
    rankings.sort(key=lambda item: (-float(item["score"]), str(item["expert"])))
    return rankings


def observe_global_and_update_history(
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
    """Decode probes and track low-drift coordinates in one or every MoE layer.

    ``target_layer=None`` enables automatic layer discovery.  Drift statistics
    use a bounded recent window so an old transient update cannot permanently
    disqualify a coordinate that later becomes dormant.
    """
    if len(parameter_names) != len(parameters):
        raise ValueError("Expert probe parameter names and arrays have different lengths.")
    by_name = dict(zip(parameter_names, parameters))
    decoded: List[Dict[str, Any]] = []
    excluded: Dict[str, set[int]] = {}

    pending = list(state.get("pending", []))
    for item in pending:
        name = str(item["parameter_name"])
        excluded.setdefault(name, set()).update(int(x) for x in item["positive_indices"])
        excluded[name].update(int(x) for x in item["negative_indices"])
        round_gap = int(server_round) - int(item["round"])
        report: Dict[str, Any] = {
            "expert": str(item["expert"]),
            "parameter_name": name,
            "injected_round": int(item["round"]),
            "observed_round": int(server_round),
            "round_gap": round_gap,
            "valid": False,
        }
        if round_gap == 1 and name in by_name:
            flat = np.asarray(by_name[name]).reshape(-1).astype(np.float64, copy=False)
            pos = np.asarray(item["positive_indices"], dtype=np.int64)
            neg = np.asarray(item["negative_indices"], dtype=np.int64)
            base_pos = np.asarray(item["baseline_positive"], dtype=np.float64)
            base_neg = np.asarray(item["baseline_negative"], dtype=np.float64)
            delta = float(item["delta"])
            sign = int(item["sign"])
            pos_gain = float(sign * np.mean(flat[pos] - base_pos) / delta)
            neg_gain = float(-sign * np.mean(flat[neg] - base_neg) / delta)
            retention = float(0.5 * (pos_gain + neg_gain))
            consistency_error = float(abs(pos_gain - neg_gain))
            measurement_valid = bool(
                np.isfinite(retention)
                and np.isfinite(consistency_error)
                and consistency_error <= float(max_consistency_error)
            )
            report.update(
                {
                    "valid": True,
                    "measurement_valid": measurement_valid,
                    "quality": "reliable" if measurement_valid else "distorted",
                    "retention": retention,
                    "positive_retention": pos_gain,
                    "negative_retention": neg_gain,
                    "consistency_error": consistency_error,
                    "delta": delta,
                }
            )
            # Temporal validity alone is not enough: a large positive/negative
            # disagreement means the background update overwhelmed the probe.
            # Do not let such samples pollute the expert ranking.
            if measurement_valid:
                values = state.setdefault("results", {}).setdefault(str(item["expert"]), [])
                values.append(retention)
                del values[: -max(int(repetitions) * 4, 8)]
        decoded.append(report)
    state["pending"] = []

    trackers = state.setdefault("trackers", {})
    for name, array in zip(parameter_names, parameters):
        parsed = parse_expert_parameter(name)
        if parsed is None or (target_layer is not None and parsed[0] != target_layer):
            continue
        # Automatic discovery spans many layers.  Tracking one stable,
        # consistently available tensor per expert bounds the persisted JSON
        # state without weakening cross-layer screening.
        if target_layer is None and not name.endswith(".wi.weight"):
            continue
        values = np.asarray(array)
        if values.size < 2 or not np.issubdtype(values.dtype, np.floating):
            continue
        flat = values.reshape(-1).astype(np.float64, copy=False)
        tracker = trackers.get(name)
        if tracker is None or int(tracker.get("size", -1)) != int(flat.size):
            indices = _candidate_indices(flat.size, candidate_pool)
            trackers[name] = {
                "size": int(flat.size),
                "indices": indices,
                "last": [float(flat[index]) for index in indices],
                "max_drift": [0.0] * len(indices),
                "mean_drift": [0.0] * len(indices),
                "drift_history": [],
                "scale_rms": float(
                    np.sqrt(np.mean(np.square(flat, dtype=np.float64)))
                ),
                "observations": 0,
            }
            continue

        indices = [int(index) for index in tracker["indices"]]
        current = np.asarray([flat[index] for index in indices], dtype=np.float64)
        last = np.asarray(tracker["last"], dtype=np.float64)
        drift = np.abs(current - last)
        skip = excluded.get(name, set())
        mask = np.asarray([index not in skip for index in indices], dtype=bool)
        old_count = int(tracker["observations"])
        previous_mean = np.asarray(tracker["mean_drift"], dtype=np.float64)
        safe_drift = drift.copy()
        safe_drift[~mask] = previous_mean[~mask]
        history = list(tracker.get("drift_history", []))
        history.append([float(value) for value in safe_drift])
        del history[: -max(int(drift_window), 1)]
        history_array = np.asarray(history, dtype=np.float64)
        max_drift = np.max(history_array, axis=0)
        mean_drift = np.mean(history_array, axis=0)
        # Pending probe coordinates are not valid background-drift samples.
        # Preserve their previous estimates until the injected signal leaves
        # the sliding window naturally.
        previous_max = np.asarray(tracker["max_drift"], dtype=np.float64)
        max_drift[~mask] = previous_max[~mask]
        mean_drift[~mask] = previous_mean[~mask]
        tracker["last"] = [float(value) for value in current]
        tracker["max_drift"] = [float(value) for value in max_drift]
        tracker["mean_drift"] = [float(value) for value in mean_drift]
        tracker["drift_history"] = history
        tracker["scale_rms"] = float(
            np.sqrt(np.mean(np.square(flat, dtype=np.float64)))
        )
        tracker["observations"] = old_count + 1

    return {"decoded": decoded, "rankings": _rankings(state, repetitions)}


def _expert_id_from_key(expert_key: str) -> int:
    match = re.search(r"/expert_(\d+)$", str(expert_key))
    if match is None:
        raise ValueError(f"Malformed expert probe key '{expert_key}'.")
    return int(match.group(1))


def _layer_from_expert_key(expert_key: str) -> str:
    if "/expert_" not in str(expert_key):
        raise ValueError(f"Malformed expert probe key '{expert_key}'.")
    return str(expert_key).rsplit("/expert_", 1)[0]


def _target_from_expert_key(expert_key: str) -> Dict[str, Any]:
    return {
        "layer": _layer_from_expert_key(expert_key),
        "expert_id": _expert_id_from_key(expert_key),
    }


def _keep_completed_probe_stopped(state: MutableMapping[str, Any]) -> bool:
    """Make a persisted discovery terminal across all later client rounds."""
    if state.get("found_low_frequency_expert") is None:
        return False
    state["phase"] = "complete"
    state["scheduled_targets"] = []
    state["scheduled_expert"] = None
    state["scheduled_target"] = None
    state["pending"] = []
    return True


def advance_probe_confirmation(
    *,
    state: MutableMapping[str, Any],
    decoded: Sequence[MutableMapping[str, Any]],
    confirmation_rounds: int,
    single_retention_min: float,
    single_retention_max: float,
    max_pilot_attempts: int = 2,
) -> List[Dict[str, Any]]:
    """Advance scan/confirmation state from newly decoded probe observations.

    A reliable near-one observation locks the first candidate.  The discovery
    sample is not counted toward confirmation: the same expert must then pass
    ``confirmation_rounds`` additional probes.  A statistically distorted or
    reliably non-single observation rejects a locked candidate and requests a
    fallback from the persisted candidate queue.
    """
    if _keep_completed_probe_stopped(state):
        return []

    required = int(confirmation_rounds)
    if required < 1:
        raise ValueError("confirmation_rounds must be >= 1.")
    retention_min = float(single_retention_min)
    retention_max = float(single_retention_max)
    if retention_min > retention_max:
        raise ValueError("single_retention_min must be <= single_retention_max.")

    events: List[Dict[str, Any]] = []
    for report in decoded:
        if not bool(report.get("valid", False)):
            report["classification"] = "temporal_invalid"
            continue

        expert_key = str(report["expert"])
        expert_id = _expert_id_from_key(expert_key)
        retention = float(report["retention"])
        measurement_valid = bool(report.get("measurement_valid", True))
        if not measurement_valid:
            classification = "distorted"
        elif retention_min <= retention <= retention_max:
            classification = "near_single_contributor"
        else:
            classification = "not_single_contributor"
        report["classification"] = classification

        phase = str(state.get("phase", "scan"))
        if phase == "complete":
            continue

        if phase == "scan":
            if classification != "near_single_contributor":
                attempts = int(state.get("attempts", {}).get(expert_key, 0))
                should_reject = bool(
                    classification == "not_single_contributor"
                    or attempts >= int(max_pilot_attempts)
                )
                if should_reject:
                    rejected_candidates = list(
                        state.get("rejected_candidates", [])
                    )
                    if expert_key not in rejected_candidates:
                        rejected_candidates.append(expert_key)
                    state["rejected_candidates"] = rejected_candidates
                    events.append(
                        {
                            "event": "pilot_candidate_rejected",
                            "expert": expert_key,
                            "round": int(report["observed_round"]),
                            "reason": classification,
                            "attempts": attempts,
                        }
                    )
                else:
                    events.append(
                        {
                            "event": "pilot_retry",
                            "expert": expert_key,
                            "round": int(report["observed_round"]),
                            "reason": classification,
                            "attempts": attempts,
                            "maximum_attempts": int(max_pilot_attempts),
                        }
                    )
                continue
            state["phase"] = "confirm"
            state["confirmation_expert"] = expert_key
            state["confirmation_expert_id"] = expert_id
            state["confirmation_target"] = _target_from_expert_key(expert_key)
            state["confirmation_successes"] = 0
            state["scheduled_expert"] = expert_id
            state["scheduled_target"] = _target_from_expert_key(expert_key)
            state["confirmation_discovery"] = {
                "round": int(report["observed_round"]),
                "retention": retention,
                "consistency_error": float(report["consistency_error"]),
            }
            events.append(
                {
                    "event": "candidate_locked",
                    "expert": expert_key,
                    "round": int(report["observed_round"]),
                    "required_confirmations": required,
                }
            )
            continue

        if phase != "confirm" or expert_key != state.get("confirmation_expert"):
            continue

        if classification == "near_single_contributor":
            successes = int(state.get("confirmation_successes", 0)) + 1
            state["confirmation_successes"] = successes
            events.append(
                {
                    "event": "confirmation_passed",
                    "expert": expert_key,
                    "round": int(report["observed_round"]),
                    "confirmation": successes,
                    "required_confirmations": required,
                }
            )
            if successes >= required:
                found = {
                    "expert": expert_key,
                    "layer": _layer_from_expert_key(expert_key),
                    "expert_id": expert_id,
                    "found_round": int(report["observed_round"]),
                    "confirmation_rounds": successes,
                    "discovery": state.get("confirmation_discovery"),
                    "final_retention": retention,
                    "final_consistency_error": float(report["consistency_error"]),
                }
                state["phase"] = "complete"
                state["scheduled_expert"] = None
                state["scheduled_target"] = None
                state["found_low_frequency_expert"] = found
                events.append({"event": "low_frequency_expert_found", **found})
            continue

        rejected_candidates = list(state.get("rejected_candidates", []))
        if expert_key not in rejected_candidates:
            rejected_candidates.append(expert_key)
        state["rejected_candidates"] = rejected_candidates
        rejected = list(state.get("rejected_experts", []))
        if expert_id not in rejected:
            rejected.append(expert_id)
        state["rejected_experts"] = rejected
        state["phase"] = "fallback"
        state["confirmation_expert"] = None
        state["confirmation_expert_id"] = None
        state["confirmation_target"] = None
        state["confirmation_successes"] = 0
        state["scheduled_expert"] = None
        state["scheduled_target"] = None
        events.append(
            {
                "event": "candidate_rejected",
                "expert": expert_key,
                "round": int(report["observed_round"]),
                "reason": classification,
                "retention": retention,
                "consistency_error": float(report["consistency_error"]),
            }
        )

    return events


def advance_batch_probe_confirmation(
    *,
    state: MutableMapping[str, Any],
    decoded: Sequence[MutableMapping[str, Any]],
    confirmation_rounds: int,
    single_retention_min: float,
    single_retention_max: float,
    max_pilot_attempts: int = 2,
) -> List[Dict[str, Any]]:
    """Advance independent pilot/confirmation state for a model-wide batch.

    Unlike :func:`advance_probe_confirmation`, one failed candidate never
    blocks or resets another candidate.  Discovery observations lock each
    candidate independently and are not counted as confirmation observations.
    If multiple candidates complete together, the geometry-ranked candidate
    (smaller boundary-distance quantile) is selected deterministically.
    """
    if _keep_completed_probe_stopped(state):
        return []

    required = int(confirmation_rounds)
    if required < 1:
        raise ValueError("confirmation_rounds must be >= 1.")
    retention_min = float(single_retention_min)
    retention_max = float(single_retention_max)
    if retention_min > retention_max:
        raise ValueError("single_retention_min must be <= single_retention_max.")

    candidate_states = state.setdefault("batch_candidates", {})
    rejected_candidates = list(state.get("rejected_candidates", []))
    events: List[Dict[str, Any]] = []
    newly_confirmed: List[Tuple[str, MutableMapping[str, Any], MutableMapping[str, Any]]] = []

    for report in decoded:
        if not bool(report.get("valid", False)):
            report["classification"] = "temporal_invalid"
            continue

        expert_key = str(report["expert"])
        expert_id = _expert_id_from_key(expert_key)
        retention = float(report["retention"])
        measurement_valid = bool(report.get("measurement_valid", True))
        if not measurement_valid:
            classification = "distorted"
        elif retention_min <= retention <= retention_max:
            classification = "near_single_contributor"
        else:
            classification = "not_single_contributor"
        report["classification"] = classification

        entry = candidate_states.setdefault(
            expert_key,
            {
                "status": "pilot",
                "confirmation_successes": 0,
                "retentions": [],
            },
        )
        entry.setdefault("retentions", []).append(retention)
        entry["last_consistency_error"] = float(report["consistency_error"])
        entry["last_observed_round"] = int(report["observed_round"])
        status = str(entry.get("status", "pilot"))
        if status in {"rejected", "confirmed"}:
            continue

        if status == "pilot":
            if classification == "near_single_contributor":
                entry["status"] = "confirming"
                entry["confirmation_successes"] = 0
                entry["discovery"] = {
                    "round": int(report["observed_round"]),
                    "retention": retention,
                    "consistency_error": float(report["consistency_error"]),
                }
                events.append(
                    {
                        "event": "candidate_locked",
                        "expert": expert_key,
                        "round": int(report["observed_round"]),
                        "required_confirmations": required,
                    }
                )
                continue

            attempts = int(state.get("attempts", {}).get(expert_key, 0))
            should_reject = bool(
                classification == "not_single_contributor"
                or attempts >= int(max_pilot_attempts)
            )
            if should_reject:
                entry["status"] = "rejected"
                entry["rejection_reason"] = classification
                if expert_key not in rejected_candidates:
                    rejected_candidates.append(expert_key)
                events.append(
                    {
                        "event": "pilot_candidate_rejected",
                        "expert": expert_key,
                        "round": int(report["observed_round"]),
                        "reason": classification,
                        "attempts": attempts,
                    }
                )
            else:
                events.append(
                    {
                        "event": "pilot_retry",
                        "expert": expert_key,
                        "round": int(report["observed_round"]),
                        "reason": classification,
                        "attempts": attempts,
                        "maximum_attempts": int(max_pilot_attempts),
                    }
                )
            continue

        if classification != "near_single_contributor":
            entry["status"] = "rejected"
            entry["rejection_reason"] = classification
            if expert_key not in rejected_candidates:
                rejected_candidates.append(expert_key)
            events.append(
                {
                    "event": "candidate_rejected",
                    "expert": expert_key,
                    "round": int(report["observed_round"]),
                    "reason": classification,
                    "retention": retention,
                    "consistency_error": float(report["consistency_error"]),
                }
            )
            continue

        successes = int(entry.get("confirmation_successes", 0)) + 1
        entry["confirmation_successes"] = successes
        events.append(
            {
                "event": "confirmation_passed",
                "expert": expert_key,
                "round": int(report["observed_round"]),
                "confirmation": successes,
                "required_confirmations": required,
            }
        )
        if successes >= required:
            entry["status"] = "confirmed"
            newly_confirmed.append((expert_key, entry, report))

    state["rejected_candidates"] = rejected_candidates
    if newly_confirmed:
        expert_key, entry, report = min(
            newly_confirmed,
            key=lambda item: (
                float(item[1].get("boundary_distance_quantile", float("inf"))),
                int(_expert_id_from_key(item[0])),
            ),
        )
        found = {
            "expert": expert_key,
            "layer": _layer_from_expert_key(expert_key),
            "expert_id": _expert_id_from_key(expert_key),
            "found_round": int(report["observed_round"]),
            "confirmation_rounds": int(entry["confirmation_successes"]),
            "discovery": entry.get("discovery"),
            "final_retention": float(report["retention"]),
            "final_consistency_error": float(report["consistency_error"]),
            "boundary_distance_quantile": entry.get("boundary_distance_quantile"),
        }
        state["phase"] = "complete"
        state["scheduled_targets"] = []
        state["scheduled_expert"] = None
        state["scheduled_target"] = None
        state["found_low_frequency_expert"] = found
        events.append({"event": "low_frequency_expert_found", **found})
    elif candidate_states:
        active = {
            str(item.get("status", "pilot"))
            for item in candidate_states.values()
            if str(item.get("status", "pilot")) not in {"rejected", "confirmed"}
        }
        state["phase"] = "batch_confirm" if "confirming" in active else "batch_scan"

    return events


def choose_batch_probe_candidates(
    *,
    state: MutableMapping[str, Any],
    ranked_candidates: Sequence[Mapping[str, Any]],
    batch_size: int,
    currently_assigned: Sequence[int] | Mapping[str, Sequence[int]],
    max_per_layer: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Choose a model-wide batch while enforcing an optional per-layer cap."""
    limit = int(batch_size)
    if limit < 1:
        raise ValueError("batch_size must be >= 1.")
    if max_per_layer is not None and int(max_per_layer) < 1:
        raise ValueError("max_per_layer must be >= 1 when provided.")
    if str(state.get("phase")) == "complete":
        return []

    assigned_by_layer: Optional[Dict[str, set[int]]]
    assigned_flat: set[int]
    if isinstance(currently_assigned, Mapping):
        assigned_by_layer = {
            str(layer): {int(value) for value in experts}
            for layer, experts in currently_assigned.items()
        }
        assigned_flat = set()
    else:
        assigned_by_layer = None
        assigned_flat = {int(value) for value in currently_assigned}

    def candidate_layer(item: Mapping[str, Any]) -> str:
        return str(item.get("target_layer") or _layer_from_expert_key(str(item["expert"])))

    def is_currently_assigned(item: Mapping[str, Any]) -> bool:
        expert_id = int(item["expert_id"])
        if assigned_by_layer is None:
            return expert_id in assigned_flat
        return expert_id in assigned_by_layer.get(candidate_layer(item), set())

    rejected = {str(value) for value in state.get("rejected_candidates", [])}
    candidate_states = state.setdefault("batch_candidates", {})
    available: List[Dict[str, Any]] = []
    for raw_candidate in ranked_candidates:
        candidate = dict(raw_candidate)
        expert_key = str(candidate["expert"])
        if expert_key in rejected:
            continue
        entry = candidate_states.setdefault(
            expert_key,
            {
                "status": "pilot",
                "confirmation_successes": 0,
                "retentions": [],
            },
        )
        for key in (
            "boundary_best_trigger_index",
            "boundary_distance_quantile",
            "boundary_distance_median",
            "boundary_margin_q10",
            "boundary_top1_hit_rate",
        ):
            if key in candidate:
                entry[key] = candidate[key]
        status = str(entry.get("status", "pilot"))
        if status in {"rejected", "confirmed"}:
            continue
        candidate["probe_status"] = status
        available.append(candidate)

    available.sort(
        key=lambda item: (
            0 if item["probe_status"] == "confirming" else 1,
            0 if is_currently_assigned(item) else 1,
            float(item.get("boundary_distance_quantile", float("inf"))),
            float(item.get("normalized_activity_drift", float("inf"))),
            candidate_layer(item),
            int(item["expert_id"]),
        )
    )
    selected: List[Dict[str, Any]] = []
    per_layer_counts: Dict[str, int] = {}
    for candidate in available:
        layer = candidate_layer(candidate)
        if (
            max_per_layer is not None
            and per_layer_counts.get(layer, 0) >= int(max_per_layer)
        ):
            continue
        selected.append(candidate)
        per_layer_counts[layer] = per_layer_counts.get(layer, 0) + 1
        if len(selected) >= limit:
            break
    return selected


def inject_paired_probe(
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
    """Overwrite uploaded coordinates with exact incoming-value +/- probes.

    ``append_pending`` is used by model-wide probing to combine independently
    injected batches from several MoE layers into one next-round decode set.
    """
    incoming = dict(zip(incoming_names, incoming_parameters))
    outgoing_lookup = {name: index for index, name in enumerate(outgoing_global_names)}
    trackers: Mapping[str, Any] = state.get("trackers", {})
    injected: List[Dict[str, Any]] = []
    new_pending: List[Dict[str, Any]] = []

    for expert_id in sorted(set(int(value) for value in target_experts)):
        expert_key = f"{target_layer}/expert_{expert_id}"
        candidates: List[Tuple[float, str]] = []
        for name, tracker in trackers.items():
            if parse_expert_parameter(name) != (target_layer, expert_id):
                continue
            if name not in incoming or name not in outgoing_lookup:
                continue
            if int(tracker.get("observations", 0)) < int(warmup_observations):
                continue
            if len(tracker.get("indices", [])) < 2 * int(pairs):
                continue
            scale = max(
                float(tracker.get("scale_rms", 1.0)),
                float(np.finfo(np.float32).eps),
            )
            candidates.append(
                (float(np.median(tracker["max_drift"])) / scale, name)
            )
        if not candidates:
            continue

        _, name = min(candidates, key=lambda item: (item[0], item[1]))
        tracker = trackers[name]
        order = np.argsort(np.asarray(tracker["max_drift"], dtype=np.float64), kind="stable")
        chosen_positions = order[: 2 * int(pairs)]
        candidate_indices = np.asarray(tracker["indices"], dtype=np.int64)
        positive = candidate_indices[chosen_positions[:pairs]]
        negative = candidate_indices[chosen_positions[pairs:]]

        baseline_array = np.asarray(incoming[name])
        baseline = baseline_array.reshape(-1).astype(np.float64, copy=False)
        rms = float(np.sqrt(np.mean(np.square(baseline, dtype=np.float64))))
        scale_reference = max(rms, float(np.finfo(np.float32).eps))
        selected_noise = np.asarray(tracker["mean_drift"], dtype=np.float64)[chosen_positions]
        noise = float(np.median(selected_noise))
        delta = max(float(relative_scale) * scale_reference, float(noise_multiplier) * noise)
        delta = min(delta, float(max_relative_scale) * scale_reference)
        delta = max(delta, float(np.finfo(np.float32).eps) * scale_reference)
        sign = _probe_sign(seed, client_id, server_round, expert_key)

        outbound_index = outgoing_lookup[name]
        outbound_flat = outgoing_parameters[outbound_index].reshape(-1)
        outbound_flat[positive] = (
            baseline[positive] + sign * delta
        ).astype(outbound_flat.dtype, copy=False)
        outbound_flat[negative] = (
            baseline[negative] - sign * delta
        ).astype(outbound_flat.dtype, copy=False)
        pending = {
            "expert": expert_key,
            "parameter_name": name,
            "round": int(server_round),
            "sign": int(sign),
            "delta": float(delta),
            "positive_indices": [int(value) for value in positive],
            "negative_indices": [int(value) for value in negative],
            "baseline_positive": [float(baseline[index]) for index in positive],
            "baseline_negative": [float(baseline[index]) for index in negative],
        }
        new_pending.append(pending)
        attempts = state.setdefault("attempts", {})
        attempts[expert_key] = int(attempts.get(expert_key, 0)) + 1
        injected.append(
            {
                "expert": expert_key,
                "parameter_name": name,
                "round": int(server_round),
                "sign": int(sign),
                "delta": float(delta),
                "pairs": int(pairs),
                "tensor_rms": rms,
                "estimated_coordinate_noise": noise,
            }
        )

    if append_pending:
        state["pending"] = list(state.get("pending", [])) + new_pending
    else:
        state["pending"] = new_pending
    return injected


def rank_lazy_expert_candidates(
    *,
    state: Mapping[str, Any],
    target_layer: str,
    num_experts: int,
    warmup_observations: int,
    pairs: int,
) -> List[Dict[str, Any]]:
    """Rank experts by whole-sample activity while retaining safe probe coordinates.

    Expert/layer selection uses the median drift across every sampled
    coordinate.  The lowest ``2 * pairs`` coordinates are reported separately
    and are used only to place a low-noise paired probe.  Keeping these two
    scores separate prevents a generally active expert from looking dormant
    merely because it has a handful of unchanged coordinates.
    """
    trackers: Mapping[str, Any] = state.get("trackers", {})
    attempts: Mapping[str, Any] = state.get("attempts", {})
    candidates: List[Dict[str, Any]] = []
    for expert_id in range(int(num_experts)):
        tensor_candidates: List[Tuple[float, float, float, float, str, int, float]] = []
        for name, tracker in trackers.items():
            if parse_expert_parameter(name) != (target_layer, expert_id):
                continue
            observations = int(tracker.get("observations", 0))
            if observations < int(warmup_observations):
                continue
            if len(tracker.get("indices", [])) < 2 * int(pairs):
                continue
            max_drift = np.asarray(tracker["max_drift"], dtype=np.float64)
            coordinate_count = min(2 * int(pairs), max_drift.size)
            lowest = np.partition(max_drift, coordinate_count - 1)[:coordinate_count]
            probe_drift = float(np.median(lowest))
            activity_drift = float(np.median(max_drift))
            scale = max(
                float(tracker.get("scale_rms", 1.0)),
                float(np.finfo(np.float32).eps),
            )
            tensor_candidates.append(
                (
                    activity_drift / scale,
                    activity_drift,
                    probe_drift / scale,
                    probe_drift,
                    str(name),
                    observations,
                    float(np.mean(max_drift == 0.0)),
                )
            )
        if not tensor_candidates:
            continue
        (
            normalized_activity,
            activity_drift,
            normalized_probe_drift,
            probe_drift,
            parameter_name,
            observations,
            stable_coordinate_fraction,
        ) = min(
            tensor_candidates, key=lambda item: (item[0], item[1])
        )
        expert_key = f"{target_layer}/expert_{expert_id}"
        candidates.append(
            {
                "expert": expert_key,
                "target_layer": str(target_layer),
                "expert_id": int(expert_id),
                "lazy_parameter_name": parameter_name,
                # Legacy aliases retain the actual probe-coordinate statistic.
                "lazy_parameter_drift": float(probe_drift),
                "normalized_lazy_drift": float(normalized_probe_drift),
                "expert_activity_drift": float(activity_drift),
                "normalized_activity_drift": float(normalized_activity),
                "probe_coordinate_drift": float(probe_drift),
                "normalized_probe_coordinate_drift": float(normalized_probe_drift),
                "stable_coordinate_fraction": float(stable_coordinate_fraction),
                "history_observations": int(observations),
                "probe_attempts": int(attempts.get(expert_key, 0)),
            }
        )
    candidates.sort(
        key=lambda item: (
            int(item["probe_attempts"]),
            float(item["normalized_activity_drift"]),
            int(item["expert_id"]),
        )
    )
    return candidates


def rank_lazy_candidates_across_layers(
    *,
    state: Mapping[str, Any],
    layer_num_experts: Mapping[str, int],
    warmup_observations: int,
    pairs: int,
    top_experts_per_layer: int = 2,
    max_layers: int = 0,
    min_layer_contrast: float = 1.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build a scale-normalized, balanced shortlist across all MoE layers.

    Passive drift is only a priority signal.  Each retained layer contributes
    at most ``top_experts_per_layer`` candidates, preventing a globally
    converged layer from monopolizing every active pilot probe.
    """
    per_layer: Dict[str, List[Dict[str, Any]]] = {}
    layer_rankings: List[Dict[str, Any]] = []
    for layer_id in sorted(layer_num_experts):
        candidates = rank_lazy_expert_candidates(
            state=state,
            target_layer=layer_id,
            num_experts=int(layer_num_experts[layer_id]),
            warmup_observations=warmup_observations,
            pairs=pairs,
        )
        if not candidates:
            continue
        by_drift = sorted(
            candidates,
            key=lambda item: (
                float(item["normalized_activity_drift"]),
                int(item["expert_id"]),
            ),
        )
        normalized = np.asarray(
            [float(item["normalized_activity_drift"]) for item in by_drift],
            dtype=np.float64,
        )
        best = float(normalized[0])
        median = float(np.median(normalized))
        epsilon = float(np.finfo(np.float64).eps)
        contrast = float((median + epsilon) / (best + epsilon))
        # Contrast is the primary layer signal: an entirely converged layer
        # should not outrank a layer with one unusually dormant expert merely
        # because every parameter in the converged layer has near-zero drift.
        priority = contrast
        layer_rankings.append(
            {
                "layer": layer_id,
                "best_expert": int(by_drift[0]["expert_id"]),
                "best_normalized_drift": best,
                "median_normalized_drift": median,
                "best_normalized_activity_drift": best,
                "median_normalized_activity_drift": median,
                "layer_contrast": contrast,
                "priority_score": priority,
                "eligible_experts": len(by_drift),
            }
        )
        for candidate in by_drift:
            candidate["layer_contrast"] = contrast
            candidate["layer_priority_score"] = priority
        per_layer[layer_id] = by_drift

    layer_rankings.sort(
        key=lambda item: (
            -float(item["priority_score"]),
            float(item["best_normalized_drift"]),
            str(item["layer"]),
        )
    )
    eligible_layers = [
        item
        for item in layer_rankings
        if float(item["layer_contrast"]) >= float(min_layer_contrast)
    ]
    candidate_states = state.get("batch_candidates", {})
    confirming = {
        str(expert_key)
        for expert_key, entry in candidate_states.items()
        if str(entry.get("status", "pilot")) == "confirming"
    }
    scheduled = {
        f"{target['layer']}/expert_{int(target['expert_id'])}"
        for target in state.get("scheduled_targets", [])
        if isinstance(target, Mapping)
        and "layer" in target
        and "expert_id" in target
    }
    active_layers = {
        _layer_from_expert_key(expert_key)
        for expert_key in confirming | scheduled
    }
    if int(max_layers) > 0:
        eligible_layers = eligible_layers[: int(max_layers)]
    selected_layer_ids = {str(item["layer"]) for item in eligible_layers}
    for layer in layer_rankings:
        layer_id = str(layer["layer"])
        if layer_id in active_layers and layer_id not in selected_layer_ids:
            eligible_layers.append(layer)
            selected_layer_ids.add(layer_id)

    shortlist: List[Dict[str, Any]] = []
    per_layer_limit = max(int(top_experts_per_layer), 1)
    retained_by_layer: Dict[str, List[Dict[str, Any]]] = {}
    for layer in eligible_layers:
        layer_id = str(layer["layer"])
        retained_by_layer[layer_id] = sorted(
            per_layer[layer_id],
            key=lambda item: (
                0
                if str(item["expert"]) in confirming
                else 1
                if str(item["expert"]) in scheduled
                else 2,
                float(item["normalized_activity_drift"]),
                int(item["expert_id"]),
            ),
        )[:per_layer_limit]
    # Interleave ranks across layers: every plausible layer gets a pilot before
    # a second expert from an already tested layer.
    for expert_rank in range(per_layer_limit):
        for layer in eligible_layers:
            candidates = retained_by_layer[str(layer["layer"])]
            if expert_rank < len(candidates):
                shortlist.append(candidates[expert_rank])
    shortlist = [
        item
        for _index, item in sorted(
            enumerate(shortlist),
            key=lambda pair: (int(pair[1]["probe_attempts"]), pair[0]),
        )
    ]
    return shortlist, layer_rankings


def build_probe_candidate_queue(
    candidates: Sequence[Mapping[str, Any]],
    *,
    current_expert: Optional[str] = None,
    rejected_candidates: Sequence[str] = (),
) -> List[str]:
    """Preserve the cross-layer ranking order in a bounded fallback queue."""
    rejected = {str(value) for value in rejected_candidates}
    current = None if current_expert is None else str(current_expert)
    queue: List[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        expert_key = str(candidate["expert"])
        if expert_key == current or expert_key in rejected or expert_key in seen:
            continue
        queue.append(expert_key)
        seen.add(expert_key)
    return queue


def choose_lazy_probe_expert(
    *,
    state: Mapping[str, Any],
    target_layer: str,
    num_experts: int,
    warmup_observations: int,
    pairs: int,
    currently_assigned: Sequence[int],
    excluded_experts: Sequence[str] = (),
) -> Optional[Dict[str, Any]]:
    """Choose the least-probed lazy expert, preferring one not already assigned."""
    candidates = rank_lazy_expert_candidates(
        state=state,
        target_layer=target_layer,
        num_experts=num_experts,
        warmup_observations=warmup_observations,
        pairs=pairs,
    )
    assigned = {int(value) for value in currently_assigned}
    excluded = {str(value) for value in excluded_experts}
    candidates = [
        candidate
        for candidate in candidates
        if str(candidate["expert"]) not in excluded
    ]
    candidates.sort(
        key=lambda item: (
            int(item["probe_attempts"]),
            int(item["expert_id"]) in assigned,
            float(item["normalized_activity_drift"]),
            int(item["expert_id"]),
        )
    )
    return candidates[0] if candidates else None


def promote_probe_expert_in_activation_profile(
    *,
    activation_profile: Mapping[str, Sequence[int]],
    target_layer: str,
    target_expert: int,
    k: int,
) -> Tuple[Dict[str, List[int]], Dict[str, Any]]:
    """Promote a probe target through the server's ordinary hot top-k policy.

    The malicious client swaps the target expert's reported activation count
    with the current k-th ranked expert.  The server remains authoritative over
    assignments and only sees a normal ``expert_activation_map_json`` payload.
    A one-count tie-break is used only when an exact swap would still leave the
    target outside top-k because the server resolves equal counts by expert ID.
    """
    reported = {
        str(layer_id): [int(value) for value in counts]
        for layer_id, counts in activation_profile.items()
    }
    if target_layer not in reported:
        raise KeyError(f"Activation profile has no probe target layer '{target_layer}'.")

    counts = reported[target_layer]
    num_experts = len(counts)
    if num_experts == 0:
        raise ValueError(f"Activation profile for '{target_layer}' is empty.")
    if not 0 <= int(target_expert) < num_experts:
        raise ValueError(
            f"Probe target expert {target_expert} is outside [0, {num_experts}) "
            f"for layer '{target_layer}'."
        )
    effective_k = int(k)
    if not 1 <= effective_k <= num_experts:
        raise ValueError(f"Probe activation promotion requires 1 <= k <= {num_experts}, got {k}.")

    def preference_order(values: Sequence[int]) -> List[int]:
        return sorted(
            range(num_experts),
            key=lambda expert_id: (-int(values[expert_id]), expert_id),
        )

    target_expert = int(target_expert)
    ranking_before = preference_order(counts)
    report: Dict[str, Any] = {
        "target_layer": target_layer,
        "target_expert": target_expert,
        "k": effective_k,
        "ranking_before": ranking_before,
        "target_count_before": int(counts[target_expert]),
        "displaced_expert": None,
        "displaced_count_before": None,
        "tie_break_increment": 0,
        "applied": False,
        "reason": "already_top_k",
    }

    if target_expert not in ranking_before[:effective_k]:
        displaced_expert = int(ranking_before[effective_k - 1])
        target_count = int(counts[target_expert])
        displaced_count = int(counts[displaced_expert])
        counts[target_expert], counts[displaced_expert] = displaced_count, target_count

        # Equal activation counts are ordered by expert ID on the server.  In
        # that degenerate case, an exact swap can fail to cross the top-k
        # boundary; one extra synthetic activation is the smallest correction.
        ranking_after_swap = preference_order(counts)
        tie_break_increment = 0
        if target_expert not in ranking_after_swap[:effective_k]:
            counts[target_expert] += 1
            tie_break_increment = 1

        report.update(
            {
                "applied": True,
                "reason": "swapped_with_top_k_boundary",
                "displaced_expert": displaced_expert,
                "displaced_count_before": displaced_count,
                "tie_break_increment": tie_break_increment,
            }
        )

    ranking_after = preference_order(counts)
    report.update(
        {
            "target_count_after": int(counts[target_expert]),
            "displaced_count_after": (
                None
                if report["displaced_expert"] is None
                else int(counts[int(report["displaced_expert"])])
            ),
            "ranking_after": ranking_after,
            "target_in_top_k": target_expert in ranking_after[:effective_k],
        }
    )
    return reported, report


def promote_probe_experts_in_activation_profile(
    *,
    activation_profile: Mapping[str, Sequence[int]],
    target_layer: str,
    target_experts: Sequence[int],
    k: int,
) -> Tuple[Dict[str, List[int]], Dict[str, Any]]:
    """Jointly promote up to ``k`` fixed-layer probe targets into hot top-k.

    Promotion is computed as one set operation so a later target cannot evict
    an earlier target, which can happen when the single-target swap is applied
    repeatedly.  Existing top-k non-targets are retained in preference order
    for the remaining slots, and activation-count swaps preserve the original
    count multiset except for a minimal tie-break increment when required.
    """
    reported = {
        str(layer_id): [int(value) for value in counts]
        for layer_id, counts in activation_profile.items()
    }
    if target_layer not in reported:
        raise KeyError(f"Activation profile has no probe target layer '{target_layer}'.")
    counts = reported[target_layer]
    num_experts = len(counts)
    effective_k = int(k)
    if not 1 <= effective_k <= num_experts:
        raise ValueError(f"Probe activation promotion requires 1 <= k <= {num_experts}, got {k}.")
    targets = sorted(set(int(value) for value in target_experts))
    if not targets:
        return reported, {
            "target_layer": target_layer,
            "target_experts": [],
            "k": effective_k,
            "applied": False,
            "reason": "no_targets",
        }
    if len(targets) > effective_k:
        raise ValueError(
            f"Cannot promote {len(targets)} probe targets into top-{effective_k}."
        )
    if any(target < 0 or target >= num_experts for target in targets):
        raise ValueError(
            f"Probe target experts {targets} must lie inside [0, {num_experts})."
        )

    def preference_order(values: Sequence[int]) -> List[int]:
        return sorted(
            range(num_experts),
            key=lambda expert_id: (-int(values[expert_id]), expert_id),
        )

    ranking_before = preference_order(counts)
    target_set = set(targets)
    keep_count = effective_k - len(targets)
    keepers = [expert for expert in ranking_before if expert not in target_set][:keep_count]
    desired = set(keepers) | target_set
    displaced = [expert for expert in ranking_before[:effective_k] if expert not in desired]
    entrants = [expert for expert in targets if expert not in ranking_before[:effective_k]]
    swaps: List[Dict[str, int]] = []
    for target, old_top in zip(entrants, displaced):
        target_before = int(counts[target])
        displaced_before = int(counts[old_top])
        counts[target], counts[old_top] = displaced_before, target_before
        swaps.append(
            {
                "target_expert": target,
                "displaced_expert": old_top,
                "target_count_before": target_before,
                "displaced_count_before": displaced_before,
            }
        )

    tie_break_increments: Dict[str, int] = {}
    # Resolve count ties deterministically until every desired target occupies
    # top-k.  At most one increment above the current boundary is necessary.
    ranking_after = preference_order(counts)
    for target in targets:
        if target in ranking_after[:effective_k]:
            continue
        boundary = int(counts[ranking_after[effective_k - 1]])
        increment = max(0, boundary + 1 - int(counts[target]))
        counts[target] += increment
        tie_break_increments[str(target)] = increment
        ranking_after = preference_order(counts)

    success = all(target in ranking_after[:effective_k] for target in targets)
    return reported, {
        "target_layer": target_layer,
        "target_experts": targets,
        "k": effective_k,
        "ranking_before": ranking_before,
        "ranking_after": ranking_after,
        "retained_non_targets": keepers,
        "swaps": swaps,
        "tie_break_increments": tie_break_increments,
        "applied": bool(swaps or tie_break_increments),
        "reason": "joint_top_k_promotion" if swaps or tie_break_increments else "already_top_k",
        "targets_in_top_k": success,
    }

