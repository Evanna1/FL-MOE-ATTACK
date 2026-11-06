from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from mixfedmoe_fl import defenses
from mixfedmoe_fl.config import MixFedMoEConfig
from mixfedmoe_fl.logging_utils import (
    append_benign_lfe_activation_rows,
    append_expert_assignment_log,
    append_expert_probe_reports,
    append_poison_gradient_flow_reports,
    write_expert_probe_discovery,
)

LAYER_ID_PATTERN = re.compile(r"^(encoder|decoder)\.(\d+)$")
EXPERT_PARAM_PATTERN = re.compile(
    r"^(encoder|decoder)\.block\.(\d+)\.layer\.\d+\.mlp\.experts\.expert_(\d+)\.(.+)$"
)
ROUTER_WEIGHT_PATTERN = re.compile(
    r"^(encoder|decoder)\.block\.(\d+)\.layer\.\d+\.mlp\.router\.classifier\.weight$"
)
ROUTER_BIAS_PATTERN = re.compile(
    r"^(encoder|decoder)\.block\.(\d+)\.layer\.\d+\.mlp\.router\.classifier\.bias$"
)


def _layer_sort_key(layer_id: str) -> Tuple[int, int]:
    match = LAYER_ID_PATTERN.match(layer_id)
    if match is None:
        raise ValueError(f"Invalid layer id '{layer_id}'. Expected 'encoder.N' or 'decoder.N'.")
    stack = 0 if match.group(1) == "encoder" else 1
    return stack, int(match.group(2))


def _cid_sort_key(cid: str) -> Tuple[int, str]:
    try:
        return 0, f"{int(cid):09d}"
    except ValueError:
        return 1, cid


def _as_str(value: Scalar) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _as_float(value: Scalar) -> float:
    if isinstance(value, bytes):
        return float(value.decode("utf-8"))
    return float(value)


def _parse_json_dict_of_int_lists(raw: str) -> Dict[str, List[int]]:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object.")
    parsed: Dict[str, List[int]] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise ValueError(f"Expected string layer id key, got {type(key)}.")
        if not isinstance(value, list):
            raise ValueError(f"Expected list[int] for layer '{key}'.")
        parsed[key] = [int(x) for x in value]
    return parsed


def _parse_json_list_of_str(raw: str) -> List[str]:
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError("Expected JSON list.")
    return [str(x) for x in payload]


def _aggregate_local_asr_reports(
    reports: Sequence[Tuple[str, Dict[str, Scalar]]],
) -> Dict[str, Scalar]:
    """Combine malicious-client ASR probes using their evaluation-set sizes."""

    if not reports:
        return {}

    normalized: List[Dict[str, Any]] = []
    for cid, metrics in reports:
        examples_value = _as_float(metrics["local_asr_eval_examples"])
        examples = int(examples_value)
        if examples <= 0 or examples_value != float(examples):
            raise ValueError(
                f"Client {cid} reported invalid local ASR example count={examples_value}."
            )
        pre = _as_float(metrics["local_asr_pre"])
        post = _as_float(metrics["local_asr_post"])
        reported_gain = _as_float(metrics["local_asr_gain"])
        if not 0.0 <= pre <= 1.0 or not 0.0 <= post <= 1.0:
            raise ValueError(
                f"Client {cid} reported local ASR outside [0, 1]: pre={pre}, post={post}."
            )
        if not math.isclose(reported_gain, post - pre, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError(
                f"Client {cid} reported inconsistent local ASR gain={reported_gain}; "
                f"expected {post - pre}."
            )
        normalized.append(
            {
                "client_id": str(cid),
                "pre": pre,
                "post": post,
                "gain": post - pre,
                "examples": examples,
            }
        )

    total_examples = sum(int(item["examples"]) for item in normalized)

    def weighted(key: str) -> float:
        return float(
            sum(float(item[key]) * int(item["examples"]) for item in normalized)
            / total_examples
        )

    local_pre = weighted("pre")
    local_post = weighted("post")
    return {
        "local_asr_pre": local_pre,
        "local_asr_post": local_post,
        "local_asr_gain": float(local_post - local_pre),
        "local_asr_eval_examples": int(total_examples),
        "local_asr_num_clients": int(len(normalized)),
        "local_asr_clients_json": json.dumps(normalized, sort_keys=True),
    }


def _array_update_l2(
    before: Sequence[np.ndarray],
    after: Sequence[np.ndarray],
    indices: Sequence[int],
) -> Tuple[float, float]:
    """Return absolute and baseline-relative L2 update for selected tensors."""

    if len(before) != len(after):
        raise ValueError("before/after array collections must have the same length.")
    update_squared = 0.0
    baseline_squared = 0.0
    for raw_index in indices:
        index = int(raw_index)
        if not 0 <= index < len(before):
            raise IndexError(f"Parameter index {index} is outside [0, {len(before) - 1}].")
        before_array = np.asarray(before[index], dtype=np.float64)
        after_array = np.asarray(after[index], dtype=np.float64)
        if before_array.shape != after_array.shape:
            raise ValueError(
                f"Parameter index {index} shape mismatch: "
                f"before={before_array.shape}, after={after_array.shape}."
            )
        delta = after_array - before_array
        update_squared += float(np.dot(delta.reshape(-1), delta.reshape(-1)))
        baseline_squared += float(
            np.dot(before_array.reshape(-1), before_array.reshape(-1))
        )
    absolute = math.sqrt(update_squared)
    baseline_norm = math.sqrt(baseline_squared)
    relative = absolute / baseline_norm if baseline_norm > 0.0 else (0.0 if absolute == 0.0 else float("inf"))
    return float(absolute), float(relative)


def _selected_update_statistics(
    *,
    before: Sequence[np.ndarray],
    local_after: Mapping[int, np.ndarray],
    global_after: Sequence[np.ndarray],
    selections: Mapping[int, Optional[int]],
) -> Dict[str, float]:
    """Compare one local update with the global update on identical coordinates.

    A selection value of ``None`` uses the complete tensor. An integer selects
    one router row, allowing target-router retention to be measured without
    unrelated expert rows inflating its norm.
    """

    local_squared = 0.0
    global_squared = 0.0
    baseline_squared = 0.0
    dot_product = 0.0
    used_coordinates = 0
    for index, row in selections.items():
        if index not in local_after:
            continue
        base = np.asarray(before[index], dtype=np.float64)
        local = np.asarray(local_after[index], dtype=np.float64)
        global_value = np.asarray(global_after[index], dtype=np.float64)
        if base.shape != local.shape or base.shape != global_value.shape:
            raise ValueError(
                f"Diagnostic parameter index {index} shape mismatch: "
                f"baseline={base.shape}, local={local.shape}, global={global_value.shape}."
            )
        if row is not None:
            if base.ndim < 1 or not 0 <= int(row) < base.shape[0]:
                raise ValueError(
                    f"Diagnostic router row {row} is invalid for parameter index "
                    f"{index} with shape={base.shape}."
                )
            base = base[int(row)]
            local = local[int(row)]
            global_value = global_value[int(row)]
        local_delta = (local - base).reshape(-1)
        global_delta = (global_value - base).reshape(-1)
        local_squared += float(np.dot(local_delta, local_delta))
        global_squared += float(np.dot(global_delta, global_delta))
        baseline_vector = base.reshape(-1)
        baseline_squared += float(np.dot(baseline_vector, baseline_vector))
        dot_product += float(np.dot(local_delta, global_delta))
        used_coordinates += int(local_delta.size)

    if used_coordinates == 0:
        return {}
    local_l2 = math.sqrt(local_squared)
    global_l2 = math.sqrt(global_squared)
    baseline_l2 = math.sqrt(baseline_squared)
    return {
        "local_update_l2": float(local_l2),
        "local_relative_update_l2": float(
            local_l2 / baseline_l2
            if baseline_l2 > 0.0
            else (0.0 if local_l2 == 0.0 else float("inf"))
        ),
        "projection_retention": float(
            dot_product / local_squared if local_squared > 0.0 else 0.0
        ),
        "update_cosine": float(
            dot_product / (local_l2 * global_l2)
            if local_l2 > 0.0 and global_l2 > 0.0
            else 0.0
        ),
        "coordinates": float(used_coordinates),
    }


def _selected_global_update_l2(
    *,
    before: Sequence[np.ndarray],
    after: Sequence[np.ndarray],
    selections: Mapping[int, Optional[int]],
) -> Tuple[float, float]:
    update_squared = 0.0
    baseline_squared = 0.0
    for index, row in selections.items():
        base = np.asarray(before[index], dtype=np.float64)
        value = np.asarray(after[index], dtype=np.float64)
        if base.shape != value.shape:
            raise ValueError(
                f"Global diagnostic parameter index {index} shape mismatch: "
                f"baseline={base.shape}, aggregated={value.shape}."
            )
        if row is not None:
            if base.ndim < 1 or not 0 <= int(row) < base.shape[0]:
                raise ValueError(
                    f"Diagnostic router row {row} is invalid for parameter index "
                    f"{index} with shape={base.shape}."
                )
            base = base[int(row)]
            value = value[int(row)]
        delta = (value - base).reshape(-1)
        base_vector = base.reshape(-1)
        update_squared += float(np.dot(delta, delta))
        baseline_squared += float(np.dot(base_vector, base_vector))
    update_l2 = math.sqrt(update_squared)
    baseline_l2 = math.sqrt(baseline_squared)
    relative = (
        update_l2 / baseline_l2
        if baseline_l2 > 0.0
        else (0.0 if update_l2 == 0.0 else float("inf"))
    )
    return float(update_l2), float(relative)


def _aggregate_update_group_diagnostics(
    *,
    prefix: str,
    before: Sequence[np.ndarray],
    aggregated: Sequence[np.ndarray],
    selections: Mapping[int, Optional[int]],
    malicious_updates: Mapping[
        str, Tuple[Mapping[int, np.ndarray], float]
    ],
    contributors: int,
) -> Tuple[Dict[str, Scalar], List[Dict[str, Any]]]:
    """Summarize local-to-global retention for one parameter group."""

    if not selections:
        return {}, []
    global_l2, global_relative = _selected_global_update_l2(
        before=before,
        after=aggregated,
        selections=selections,
    )
    reports: List[Dict[str, Any]] = []
    for cid, (local_after, weight) in malicious_updates.items():
        stats = _selected_update_statistics(
            before=before,
            local_after=local_after,
            global_after=aggregated,
            selections=selections,
        )
        if not stats:
            continue
        reports.append(
            {
                "client_id": str(cid),
                "aggregation_weight": float(weight),
                **stats,
            }
        )

    metrics: Dict[str, Scalar] = {
        f"{prefix}_global_update_l2": float(global_l2),
        f"{prefix}_global_relative_update_l2": float(global_relative),
        f"{prefix}_contributors": int(contributors),
    }
    malicious_count_key = (
        "num_malicious_target_updates"
        if prefix == "target_expert"
        else f"{prefix}_num_malicious_updates"
    )
    metrics[malicious_count_key] = int(len(reports))
    if not reports:
        return metrics, reports

    total_weight = sum(float(report["aggregation_weight"]) for report in reports)
    if total_weight <= 0.0:
        raise ValueError(f"{prefix} diagnostics contain no positive malicious weight.")

    def weighted(key: str) -> float:
        return float(
            sum(
                float(report[key]) * float(report["aggregation_weight"])
                for report in reports
            )
            / total_weight
        )

    local_l2 = weighted("local_update_l2")
    projection_retention = weighted("projection_retention")
    metrics.update(
        {
            f"{prefix}_local_update_l2": local_l2,
            f"{prefix}_local_relative_update_l2": weighted(
                "local_relative_update_l2"
            ),
            # Norm retention says how large the aggregated movement is; signed
            # projection retention says how much movement remains specifically
            # in the malicious local-update direction.
            f"{prefix}_update_norm_retention": float(
                global_l2 / local_l2 if local_l2 > 0.0 else 0.0
            ),
            f"{prefix}_aggregation_retention": projection_retention,
            f"{prefix}_update_cosine": weighted("update_cosine"),
        }
    )
    return metrics, reports


@dataclass(frozen=True)
class ExpertParamMeta:
    layer_id: str
    expert_idx: int
    suffix: str


@dataclass(frozen=True)
class RouterParamMeta:
    layer_id: str
    kind: str  # weight | bias


CentralEvaluateFn = Callable[
    [int, NDArrays, Dict[str, Scalar]],
    tuple[float, Dict[str, Scalar]] | None,
]


class MixFedMoEStrategy(FedAvg):
    def __init__(
        self,
        runtime_config: MixFedMoEConfig,
        parameter_names: Sequence[str],
        initial_parameters: Optional[Parameters] = None,
        evaluate_fn: Optional[CentralEvaluateFn] = None,
    ) -> None:
        self.runtime_config = runtime_config
        self.parameter_names: List[str] = [str(x) for x in parameter_names]
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("parameter_names contains duplicates.")

        self.param_name_to_idx: Dict[str, int] = {name: i for i, name in enumerate(self.parameter_names)}
        self.expert_param_meta: Dict[int, ExpertParamMeta] = {}
        self.router_param_meta: Dict[int, RouterParamMeta] = {}
        self.expert_name_lookup: Dict[Tuple[str, int, str], str] = {}
        self.layer_num_experts: Dict[str, int] = self._infer_layer_num_experts()
        self.layer_ids: List[str] = sorted(self.layer_num_experts.keys(), key=_layer_sort_key)

        if self.runtime_config.mode == "flex" and self.runtime_config.assignment_policy != "hot":
            raise ValueError("--mode=flex requires assignment_policy='hot'.")
        if self.runtime_config.mode == "flex" and self.runtime_config.k != 1:
            raise ValueError("--mode=flex requires k=1.")

        min_fit_clients = 1
        if self.runtime_config.mode in {"mix", "drop", "flex"} and self.layer_ids:
            effective_k = self._effective_k()
            for layer_id in self.layer_ids:
                num_experts = self.layer_num_experts[layer_id]
                if effective_k > num_experts:
                    raise ValueError(
                        f"--k={effective_k} exceeds experts={num_experts} at layer={layer_id}. "
                        "Exact-K assignment is infeasible."
                    )
            if self._requires_global_coverage():
                layer_min_clients = [math.ceil(self.layer_num_experts[layer_id] / effective_k) for layer_id in self.layer_ids]
                min_fit_clients = max(layer_min_clients)

        super().__init__(
            fraction_fit=self.runtime_config.fraction_fit,
            fraction_evaluate=0.0,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=0,
            min_available_clients=max(self.runtime_config.num_clients, min_fit_clients),
            evaluate_fn=evaluate_fn,
            accept_failures=False,
            initial_parameters=initial_parameters,
        )

        self._current_global_arrays: Optional[List[np.ndarray]] = None
        self._last_round_assignments: Dict[str, Dict[str, List[int]]] = {}
        self._last_activation_profiles: Dict[str, Dict[str, List[int]]] = {}
        # Genuine profiles reported by malicious clients in the most recently
        # aggregated round.  These let centralized evaluation undo a spoof
        # from that same round when it discovers that ASR has reached 1.
        self._latest_unspoofed_activation_profiles: Dict[
            str, Dict[str, List[int]]
        ] = {}
        self.cumulative_round_time: float = 0.0
        self._latest_fit_metrics: Dict[str, Scalar] = {}
        self._last_defense_metrics: Dict[str, Scalar] = {}
        self._a3fl_eval_trigger_token_ids: Optional[List[int]] = None
        self._a3fl_eval_trigger_text: Optional[str] = None
        # Once a live attack reaches perfect centralized ASR, subsequent rounds
        # become benign persistence rounds.  This is strategy state rather than a
        # mutation of the frozen runtime config so it is also visible to newly
        # constructed simulation clients through each round's FitIns config.
        self._attack_stopped_on_asr: bool = False
        self._attack_stop_round: Optional[int] = None
        self._expert_probe_stop_requested: bool = False
        self._found_low_frequency_expert: Optional[Dict[str, Any]] = None

        self.defense = str(getattr(self.runtime_config, "defense", "none"))
        self.defense_trim_ratio = float(getattr(self.runtime_config, "defense_trim_ratio", 0.2))
        configured_malicious = int(getattr(self.runtime_config, "defense_num_malicious", 0))
        self.defense_num_malicious = (
            configured_malicious if configured_malicious > 0 else len(self.runtime_config.malicious_clients)
        )
        configured_m = int(getattr(self.runtime_config, "defense_krum_m", 0))
        # Keep zero as an automatic sentinel. Multi-Krum is applied separately to
        # the shared group and every expert group, whose contributor counts differ.
        self.defense_krum_m = configured_m
        if self.defense != "none" and self.runtime_config.mode != "mix":
            raise ValueError("Robust-aggregation defenses are implemented for mix mode only.")

    def _effective_k(self) -> int:
        if self.runtime_config.mode == "flex":
            return 1
        return int(self.runtime_config.k)

    @property
    def expert_probe_stop_requested(self) -> bool:
        """Return whether the server should stop after the current fit round."""
        return self._expert_probe_stop_requested

    @property
    def found_low_frequency_expert(self) -> Optional[Dict[str, Any]]:
        """Return a copy of the terminal expert discovery, when available."""
        if self._found_low_frequency_expert is None:
            return None
        return dict(self._found_low_frequency_expert)

    def _record_expert_probe_discovery(
        self,
        *,
        server_round: int,
        reports: Sequence[Dict[str, Any]],
        arrays: Sequence[np.ndarray],
    ) -> None:
        """Validate, persist, and expose a newly reported terminal discovery."""
        if self._expert_probe_stop_requested:
            return

        discoveries: List[Dict[str, Any]] = []
        for report in reports:
            events = report.get("state_events", [])
            if not isinstance(events, list):
                raise ValueError("Expert probe state_events must be a list.")
            found_events = [
                event
                for event in events
                if isinstance(event, dict)
                and event.get("event") == "low_frequency_expert_found"
            ]
            if not found_events:
                continue
            found = report.get("found_low_frequency_expert")
            if not isinstance(found, dict):
                raise ValueError(
                    "Expert probe reported a discovery event without "
                    "found_low_frequency_expert metadata."
                )
            for key in ("layer", "expert_id", "expert", "found_round"):
                if key not in found:
                    raise ValueError(
                        f"Expert probe discovery is missing required field '{key}'."
                    )
            event = found_events[-1]
            if (
                str(event.get("layer")) != str(found["layer"])
                or int(event.get("expert_id", -1)) != int(found["expert_id"])
            ):
                raise ValueError(
                    "Expert probe discovery event disagrees with terminal metadata."
                )
            if int(found["found_round"]) != int(server_round):
                raise ValueError(
                    "Expert probe discovery round does not match the aggregation round."
                )
            discoveries.append(
                {
                    **found,
                    "client_id": int(report["client_id"]),
                    "reported_round": int(server_round),
                }
            )

        if not discoveries:
            return
        identities = {
            (str(item["layer"]), int(item["expert_id"]))
            for item in discoveries
        }
        if len(identities) != 1:
            raise ValueError(
                f"Clients reported conflicting low-frequency experts: {sorted(identities)}."
            )

        discovery = discoveries[0]
        self._found_low_frequency_expert = discovery
        self._expert_probe_stop_requested = True
        discovery_path = write_expert_probe_discovery(
            self.runtime_config.output_dir,
            discovery,
        )
        self._save_checkpoint(
            server_round=server_round,
            arrays=arrays,
            force=True,
            metadata={
                "checkpoint_reason": "low_frequency_expert_found",
                "found_low_frequency_expert": discovery,
            },
        )
        retention = discovery.get("final_retention")
        retention_text = (
            "unknown" if retention is None else f"{float(retention):.6f}"
        )
        print(
            "[MixFedMoE][EXPERT PROBE COMPLETE] "
            f"Found low-frequency expert {discovery['expert']} "
            f"(layer={discovery['layer']}, expert_id={int(discovery['expert_id'])}, "
            f"round={server_round}, retention={retention_text}). "
            f"Saved discovery to {discovery_path}. Stopping simulation.",
            flush=True,
        )

    def _assignment_rng(self, server_round: int, layer_id: str) -> np.random.Generator:
        stack_idx, layer_idx = _layer_sort_key(layer_id)
        seed = (
            int(self.runtime_config.seed) * 1_000_003
            + int(server_round) * 10_007
            + int(stack_idx) * 313
            + int(layer_idx)
        )
        return np.random.default_rng(seed & 0xFFFFFFFF)

    def _requires_global_coverage(self) -> bool:
        if self.runtime_config.mode not in {"mix", "drop"}:
            return True
        policy = str(self.runtime_config.assignment_policy).lower()
        if policy == "random":
            return False
        if policy == "hot":
            return self.runtime_config.coverage_guarantee
        raise ValueError(f"Unsupported assignment policy '{self.runtime_config.assignment_policy}'.")

    def _infer_layer_num_experts(self) -> Dict[str, int]:
        layer_max_expert: Dict[str, int] = {}
        for idx, name in enumerate(self.parameter_names):
            expert_match = EXPERT_PARAM_PATTERN.match(name)
            if expert_match is not None:
                stack = expert_match.group(1)
                block_idx = int(expert_match.group(2))
                expert_idx = int(expert_match.group(3))
                suffix = expert_match.group(4)
                layer_id = f"{stack}.{block_idx}"
                self.expert_param_meta[idx] = ExpertParamMeta(
                    layer_id=layer_id, expert_idx=expert_idx, suffix=suffix
                )
                self.expert_name_lookup[(layer_id, expert_idx, suffix)] = name
                layer_max_expert[layer_id] = max(layer_max_expert.get(layer_id, -1), expert_idx)
                continue

            router_weight_match = ROUTER_WEIGHT_PATTERN.match(name)
            if router_weight_match is not None:
                layer_id = f"{router_weight_match.group(1)}.{int(router_weight_match.group(2))}"
                self.router_param_meta[idx] = RouterParamMeta(layer_id=layer_id, kind="weight")
                continue

            router_bias_match = ROUTER_BIAS_PATTERN.match(name)
            if router_bias_match is not None:
                layer_id = f"{router_bias_match.group(1)}.{int(router_bias_match.group(2))}"
                self.router_param_meta[idx] = RouterParamMeta(layer_id=layer_id, kind="bias")

        return {layer_id: max_expert + 1 for layer_id, max_expert in layer_max_expert.items()}

    def _parse_assignments_metric(
        self,
        metrics: Dict[str, Scalar],
        client_cid: str,
    ) -> Dict[str, List[int]]:
        if client_cid not in self._last_round_assignments:
            raise KeyError(
                f"Server has no authoritative assignment for client {client_cid}."
            )

        authoritative = {
            layer_id: sorted(set(vals))
            for layer_id, vals in self._last_round_assignments[client_cid].items()
        }
        if "assigned_experts_json" in metrics:
            reported = _parse_json_dict_of_int_lists(
                _as_str(metrics["assigned_experts_json"])
            )
            normalized_reported = {
                layer_id: sorted(set(vals)) for layer_id, vals in reported.items()
            }
            if normalized_reported != authoritative:
                raise ValueError(
                    "Client-reported expert assignment does not match the "
                    f"server-issued assignment for client {client_cid}: "
                    f"reported={normalized_reported}, authoritative={authoritative}."
                )
        return authoritative

    def _preference_order(self, cid: str, layer_id: str, num_experts: int) -> List[int]:
        profile = self._last_activation_profiles.get(cid)
        if profile is None or layer_id not in profile:
            return list(range(num_experts))
        counts = profile[layer_id]
        if len(counts) != num_experts:
            return list(range(num_experts))
        return sorted(range(num_experts), key=lambda expert_idx: (-int(counts[expert_idx]), expert_idx))

    def _assign_round1(self, sorted_cids: List[str], num_experts: int, k: int) -> Dict[str, List[int]]:
        if len(sorted_cids) < math.ceil(num_experts / k):
            raise ValueError(
                f"Coverage infeasible: selected_clients={len(sorted_cids)} < ceil({num_experts}/{k})."
            )
        assignments: Dict[str, List[int]] = {cid: [] for cid in sorted_cids}

        slot_cycle: List[str] = []
        for _ in range(k):
            slot_cycle.extend(sorted_cids)

        for expert_idx in range(num_experts):
            assignments[slot_cycle[expert_idx]].append(expert_idx)

        for cid_idx, cid in enumerate(sorted_cids):
            candidate = cid_idx % num_experts
            while len(assignments[cid]) < k:
                if candidate not in assignments[cid]:
                    assignments[cid].append(candidate)
                candidate = (candidate + 1) % num_experts
            assignments[cid].sort()
        return assignments

    def _assign_by_preferences_with_coverage(
        self,
        sorted_cids: List[str],
        num_experts: int,
        k: int,
        preferences: Dict[str, List[int]],
    ) -> Dict[str, List[int]]:
        assignments: Dict[str, List[int]] = {cid: [] for cid in sorted_cids}
        remaining_slots: Dict[str, int] = {cid: k for cid in sorted_cids}

        rank_lookup: Dict[str, Dict[int, int]] = {}
        for cid in sorted_cids:
            rank_lookup[cid] = {expert_idx: rank for rank, expert_idx in enumerate(preferences[cid])}

        for expert_idx in range(num_experts):
            candidates = [cid for cid in sorted_cids if remaining_slots[cid] > 0]
            if not candidates:
                raise RuntimeError("No remaining client capacity while enforcing expert coverage.")
            chosen_cid = sorted(
                candidates,
                key=lambda cid: (
                    rank_lookup[cid].get(expert_idx, num_experts + 1),
                    _cid_sort_key(cid),
                ),
            )[0]
            assignments[chosen_cid].append(expert_idx)
            remaining_slots[chosen_cid] -= 1

        for cid in sorted_cids:
            for expert_idx in preferences[cid]:
                if len(assignments[cid]) >= k:
                    break
                if expert_idx not in assignments[cid]:
                    assignments[cid].append(expert_idx)
            if len(assignments[cid]) != k:
                raise RuntimeError(
                    f"Failed to fill exact-K assignment for client={cid}: got {len(assignments[cid])}, want {k}."
                )
            assignments[cid].sort()
        return assignments

    def _assign_roundn(self, sorted_cids: List[str], layer_id: str, num_experts: int, k: int) -> Dict[str, List[int]]:
        if len(sorted_cids) < math.ceil(num_experts / k):
            raise ValueError(
                f"Coverage infeasible at layer={layer_id}: "
                f"selected_clients={len(sorted_cids)} < ceil({num_experts}/{k})."
            )
        preferences = {cid: self._preference_order(cid, layer_id, num_experts) for cid in sorted_cids}
        rank_lookup = {
            cid: {expert_idx: rank for rank, expert_idx in enumerate(preferences[cid])} for cid in sorted_cids
        }

        assignments: Dict[str, List[int]] = {}
        for cid in sorted_cids:
            chosen: List[int] = []
            for expert_idx in preferences[cid]:
                if expert_idx not in chosen:
                    chosen.append(expert_idx)
                if len(chosen) == k:
                    break
            assignments[cid] = sorted(chosen)

        coverage_count = [0 for _ in range(num_experts)]
        for cid in sorted_cids:
            for expert_idx in assignments[cid]:
                coverage_count[expert_idx] += 1

        while True:
            missing = [expert_idx for expert_idx in range(num_experts) if coverage_count[expert_idx] == 0]
            if not missing:
                break

            progressed = False
            for missing_expert in missing:
                for cid in sorted_cids:
                    if missing_expert in assignments[cid]:
                        continue
                    removable = [x for x in assignments[cid] if coverage_count[x] > 1]
                    if not removable:
                        continue
                    removable.sort(
                        key=lambda expert_idx: (
                            -rank_lookup[cid].get(expert_idx, num_experts + 1),
                            expert_idx,
                        )
                    )
                    to_remove = removable[0]
                    assignments[cid].remove(to_remove)
                    assignments[cid].append(missing_expert)
                    assignments[cid].sort()
                    coverage_count[to_remove] -= 1
                    coverage_count[missing_expert] += 1
                    progressed = True
                    break
                if progressed:
                    break

            if not progressed:
                return self._assign_by_preferences_with_coverage(
                    sorted_cids=sorted_cids,
                    num_experts=num_experts,
                    k=k,
                    preferences=preferences,
                )

        return assignments

    def _assign_roundn_no_coverage(
        self,
        sorted_cids: List[str],
        layer_id: str,
        num_experts: int,
        k: int,
    ) -> Dict[str, List[int]]:
        """Hot policy assignment without orphan expert repair (no coverage guarantee)."""
        preferences = {cid: self._preference_order(cid, layer_id, num_experts) for cid in sorted_cids}
        assignments: Dict[str, List[int]] = {}
        for cid in sorted_cids:
            chosen: List[int] = []
            for expert_idx in preferences[cid]:
                if expert_idx not in chosen:
                    chosen.append(expert_idx)
                if len(chosen) == k:
                    break
            assignments[cid] = sorted(chosen)
        return assignments

    def _assign_random_no_coverage(
        self,
        sorted_cids: List[str],
        num_experts: int,
        k: int,
        rng: np.random.Generator,
    ) -> Dict[str, List[int]]:
        assignments: Dict[str, List[int]] = {cid: [] for cid in sorted_cids}
        for cid in sorted_cids:
            sampled = rng.choice(np.arange(num_experts, dtype=np.int64), size=k, replace=False).tolist()
            assignments[cid] = sorted([int(x) for x in sampled])

        return assignments

    def _build_assignments_for_clients(
        self,
        server_round: int,
        cids: Sequence[str],
    ) -> Dict[str, Dict[str, List[int]]]:
        sorted_cids = sorted([str(cid) for cid in cids], key=_cid_sort_key)
        per_client: Dict[str, Dict[str, List[int]]] = {cid: {} for cid in sorted_cids}
        policy = str(self.runtime_config.assignment_policy).lower()
        if policy not in {"hot", "random"}:
            raise ValueError(f"Unsupported assignment policy '{self.runtime_config.assignment_policy}'.")
        if self.runtime_config.mode == "flex" and policy != "hot":
            raise ValueError("Flex mode requires assignment_policy='hot'.")

        for layer_id in self.layer_ids:
            num_experts = self.layer_num_experts[layer_id]
            if self.runtime_config.mode == "full":
                layer_assignment = {cid: list(range(num_experts)) for cid in sorted_cids}
            else:
                k = self._effective_k()
                if policy == "random":
                    layer_assignment = self._assign_random_no_coverage(
                        sorted_cids=sorted_cids,
                        num_experts=num_experts,
                        k=k,
                        rng=self._assignment_rng(server_round=server_round, layer_id=layer_id),
                    )
                elif server_round == 1:
                    layer_assignment = self._assign_round1(sorted_cids=sorted_cids, num_experts=num_experts, k=k)
                else:
                    if policy == "hot" and not self.runtime_config.coverage_guarantee:
                        layer_assignment = self._assign_roundn_no_coverage(
                            sorted_cids=sorted_cids,
                            layer_id=layer_id,
                            num_experts=num_experts,
                            k=k,
                        )
                    else:
                        layer_assignment = self._assign_roundn(
                            sorted_cids=sorted_cids,
                            layer_id=layer_id,
                            num_experts=num_experts,
                            k=k,
                        )
            for cid in sorted_cids:
                per_client[cid][layer_id] = layer_assignment[cid]

        return per_client

    def _validate_round_coverage(
        self,
        round_assignments: Dict[str, Dict[str, List[int]]],
        enforce_global_coverage: bool,
    ) -> None:
        for layer_id in self.layer_ids:
            num_experts = self.layer_num_experts[layer_id]
            covered = [False for _ in range(num_experts)]
            for client_assignment in round_assignments.values():
                experts = client_assignment.get(layer_id, [])
                for expert_idx in experts:
                    if 0 <= expert_idx < num_experts:
                        covered[expert_idx] = True
                    else:
                        raise ValueError(
                            f"Invalid assigned expert index={expert_idx} for layer={layer_id} "
                            f"(num_experts={num_experts})."
                        )
                if self.runtime_config.mode in {"mix", "drop", "flex"} and len(experts) != self._effective_k():
                    raise ValueError(
                        f"Exact-K violation at layer={layer_id}: got {len(experts)} "
                        f"experts for mode={self.runtime_config.mode}, expected {self._effective_k()}."
                    )
            if enforce_global_coverage:
                missing = [i for i, is_covered in enumerate(covered) if not is_covered]
                if missing:
                    raise RuntimeError(f"Coverage violation at layer={layer_id}: missing experts={missing}.")

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        self._current_global_arrays = parameters_to_ndarrays(parameters)

        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        sampled_clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        sampled_cids = [client.cid for client in sampled_clients]

        per_client_assignments = self._build_assignments_for_clients(server_round=server_round, cids=sampled_cids)
        self._validate_round_coverage(
            per_client_assignments,
            enforce_global_coverage=self._requires_global_coverage(),
        )
        self._last_round_assignments = per_client_assignments
        append_expert_assignment_log(
            self.runtime_config.output_dir,
            server_round,
            per_client_assignments,
            {layer_id: self.layer_num_experts[layer_id] for layer_id in self.layer_ids},
        )

        incoming_names_json = json.dumps(self.parameter_names)
        fit_instructions: List[Tuple[ClientProxy, FitIns]] = []
        for client in sampled_clients:
            config: Dict[str, Scalar] = {
                "mode": self.runtime_config.mode,
                "server_round": int(server_round),
                "assignment_policy": self.runtime_config.assignment_policy,
                "k": int(self._effective_k()),
                "local_epochs": float(self.runtime_config.local_epochs),
                "learning_rate": float(self.runtime_config.learning_rate),
                "weight_decay": float(self.runtime_config.weight_decay),
                "train_batch_size": int(self.runtime_config.train_batch_size),
                "eval_batch_size": int(self.runtime_config.eval_batch_size),
                # This server-side switch can disable every malicious client from
                # the round after perfect ASR is observed, even when Ray recreates
                # the client object with the original runtime configuration.
                "attack_enabled": bool(
                    self.runtime_config.attack_enabled
                    and not self._attack_stopped_on_asr
                ),
                "monitor_asr": bool(
                    self.runtime_config.monitor_asr
                    or self._attack_stopped_on_asr
                ),
                "incoming_parameter_names_json": incoming_names_json,
                "assigned_experts_json": json.dumps(per_client_assignments[client.cid], sort_keys=True),
            }
            fit_instructions.append((client, FitIns(parameters=parameters, config=config)))
        return fit_instructions

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        return []

    def _accumulate_dense(
        self,
        dense_sums: Dict[int, np.ndarray],
        dense_weights: Dict[int, float],
        param_idx: int,
        array: np.ndarray,
        weight: float,
    ) -> None:
        weighted = array.astype(np.float32, copy=False) * np.float32(weight)
        if param_idx not in dense_sums:
            dense_sums[param_idx] = weighted.astype(np.float32, copy=True)
            dense_weights[param_idx] = weight
            return
        dense_sums[param_idx] += weighted
        dense_weights[param_idx] += weight

    def _accumulate_router_rows(
        self,
        router_row_sums: Dict[int, np.ndarray],
        router_row_weights: Dict[int, np.ndarray],
        baseline: Sequence[np.ndarray],
        param_idx: int,
        array: np.ndarray,
        weight: float,
        assignments: Dict[str, List[int]],
        mode: str,
    ) -> None:
        meta = self.router_param_meta[param_idx]
        target_shape = baseline[param_idx].shape
        if param_idx not in router_row_sums:
            router_row_sums[param_idx] = np.zeros(target_shape, dtype=np.float32)
            router_row_weights[param_idx] = np.zeros(target_shape[0], dtype=np.float32)

        if array.shape == target_shape:
            router_row_sums[param_idx] += array.astype(np.float32, copy=False) * np.float32(weight)
            router_row_weights[param_idx] += np.float32(weight)
            return

        if mode != "drop":
            raise ValueError(
                f"Router parameter shape mismatch for non-drop mode: idx={param_idx}, "
                f"got={array.shape}, expected={target_shape}."
            )

        if meta.layer_id not in assignments:
            raise KeyError(f"Missing assignments for layer={meta.layer_id} while aggregating drop router rows.")
        assigned_experts = assignments[meta.layer_id]
        if len(assigned_experts) != array.shape[0]:
            raise ValueError(
                f"Drop router rows mismatch for layer={meta.layer_id}: "
                f"len(assigned_experts)={len(assigned_experts)} vs array.shape[0]={array.shape[0]}."
            )

        if len(target_shape) == 2:
            if array.shape[1] != target_shape[1]:
                raise ValueError(
                    f"Drop router weight width mismatch: got={array.shape[1]}, expected={target_shape[1]}."
                )
            for local_idx, global_expert_idx in enumerate(assigned_experts):
                router_row_sums[param_idx][global_expert_idx] += (
                    array[local_idx].astype(np.float32, copy=False) * np.float32(weight)
                )
                router_row_weights[param_idx][global_expert_idx] += np.float32(weight)
        elif len(target_shape) == 1:
            for local_idx, global_expert_idx in enumerate(assigned_experts):
                router_row_sums[param_idx][global_expert_idx] += (
                    np.float32(array[local_idx]) * np.float32(weight)
                )
                router_row_weights[param_idx][global_expert_idx] += np.float32(weight)
        else:
            raise ValueError(
                f"Unsupported router tensor ndim={len(target_shape)} for idx={param_idx}; expected 1 or 2."
            )

    def _defense_aggregate(
        self,
        baseline: Sequence[np.ndarray],
        param_contribs: Dict[int, List[Tuple[str, np.ndarray, float]]],
    ) -> List[np.ndarray]:
        """Aggregate collected per-parameter contributions under a robust defense.

        Norm clipping and Multi-Krum treat all non-expert parameters as one shared
        group and each (layer, expert) as its own group. Multi-Krum selects clients
        independently inside each group; coordinate defenses (coord_median,
        trimmed_mean) replace the per-parameter weighted mean. A cold expert with
        one contributor cannot be identified as an outlier by contributor-comparison
        defenses.
        """
        client_param_delta: Dict[str, Dict[int, np.ndarray]] = {}
        for param_idx, contribs in param_contribs.items():
            base = baseline[param_idx].astype(np.float64)
            for cid, value, _weight in contribs:
                client_param_delta.setdefault(cid, {})[param_idx] = value.astype(np.float64) - base

        self._last_defense_metrics = {}
        shared_group = ("shared", "", -1)
        scale: Dict[Tuple[Tuple[str, str, int], str], float] = {}
        if self.defense == "norm_clip":
            group_client_squared: Dict[Tuple[str, str, int], Dict[str, float]] = {}
            for cid, deltas in client_param_delta.items():
                for param_idx, delta in deltas.items():
                    meta = self.expert_param_meta.get(param_idx)
                    group = (
                        ("expert", meta.layer_id, meta.expert_idx)
                        if meta is not None
                        else shared_group
                    )
                    client_squared = group_client_squared.setdefault(group, {})
                    client_squared[cid] = client_squared.get(cid, 0.0) + float(
                        np.sum(delta * delta)
                    )

            group_clip_norms: Dict[Tuple[str, str, int], float] = {}
            all_update_norms: List[float] = []
            for group, client_squared in group_client_squared.items():
                update_norms = {
                    cid: math.sqrt(squared) for cid, squared in client_squared.items()
                }
                clip_k = 3.0 if group == shared_group else 2.0
                clip_norm = defenses.median_mad_clip_norm(update_norms.values(), k=clip_k)
                group_clip_norms[group] = clip_norm
                all_update_norms.extend(update_norms.values())
                for cid, update_norm in update_norms.items():
                    scale[(group, cid)] = defenses.norm_clip_scale(update_norm, clip_norm)

            num_clipped = sum(value < 1.0 for value in scale.values())
            min_update_norm = min(all_update_norms)
            max_update_norm = max(all_update_norms)
            min_clip_norm = min(group_clip_norms.values())
            max_clip_norm = max(group_clip_norms.values())
            self._last_defense_metrics = {
                "defense_num_clipped": int(num_clipped),
                "defense_num_clip_groups": int(len(group_clip_norms)),
                "defense_min_update_norm": float(min_update_norm),
                "defense_max_update_norm": float(max_update_norm),
                "defense_min_group_clip_norm": float(min_clip_norm),
                "defense_max_group_clip_norm": float(max_clip_norm),
            }
            if shared_group in group_clip_norms:
                shared_clip_norm = float(group_clip_norms[shared_group])
                self._last_defense_metrics["defense_clip_norm"] = shared_clip_norm
                self._last_defense_metrics["defense_shared_clip_norm"] = shared_clip_norm
            else:
                shared_clip_norm = float("nan")
            print(
                "[MixFedMoE][norm_clip] "
                f"groups={len(group_clip_norms)} shared_k=3 expert_k=2 "
                f"shared_clip_norm={shared_clip_norm:.6f} "
                f"group_clip_norm_range=[{min_clip_norm:.6f}, {max_clip_norm:.6f}] "
                f"update_norm_range=[{min_update_norm:.6f}, {max_update_norm:.6f}] "
                f"clipped_group_updates={num_clipped}/{len(scale)}",
                flush=True,
            )

        selected_by_group: Dict[Tuple[str, str, int], set[str]] = {}
        if self.defense == "multi_krum":
            group_client_deltas: Dict[
                Tuple[str, str, int], Dict[str, Dict[int, np.ndarray]]
            ] = {}
            for cid, deltas in client_param_delta.items():
                for param_idx, delta in deltas.items():
                    meta = self.expert_param_meta.get(param_idx)
                    group = (
                        ("expert", meta.layer_id, meta.expert_idx)
                        if meta is not None
                        else shared_group
                    )
                    group_client_deltas.setdefault(group, {}).setdefault(cid, {})[
                        param_idx
                    ] = delta

            contributor_counts: List[int] = []
            selected_counts: List[int] = []
            for group, client_deltas in group_client_deltas.items():
                group_param_indices = sorted(
                    {idx for deltas in client_deltas.values() for idx in deltas}
                )
                vectors: Dict[str, np.ndarray] = {}
                for cid, deltas in client_deltas.items():
                    parts = [
                        deltas[idx].ravel()
                        if idx in deltas
                        else np.zeros(baseline[idx].size, dtype=np.float64)
                        for idx in group_param_indices
                    ]
                    vectors[cid] = (
                        np.concatenate(parts) if parts else np.zeros(1, dtype=np.float64)
                    )

                num_contributors = len(vectors)
                num_select = (
                    self.defense_krum_m
                    if self.defense_krum_m > 0
                    else max(1, num_contributors - self.defense_num_malicious - 2)
                )
                selected_by_group[group] = set(
                    defenses.multi_krum_select(
                        vectors,
                        self.defense_num_malicious,
                        num_select,
                    )
                )
                contributor_counts.append(num_contributors)
                selected_counts.append(len(selected_by_group[group]))

            self._last_defense_metrics = {
                "defense_num_krum_groups": int(len(selected_by_group)),
                "defense_krum_total_contributors": int(sum(contributor_counts)),
                "defense_krum_total_selected": int(sum(selected_counts)),
                "defense_krum_min_group_contributors": int(min(contributor_counts)),
                "defense_krum_max_group_contributors": int(max(contributor_counts)),
            }
            print(
                "[MixFedMoE][multi_krum] "
                f"groups={len(selected_by_group)} f={self.defense_num_malicious} "
                f"configured_m={self.defense_krum_m} "
                f"contributors_per_group=[{min(contributor_counts)}, {max(contributor_counts)}] "
                f"selected_per_group=[{min(selected_counts)}, {max(selected_counts)}] "
                f"selected_group_updates={sum(selected_counts)}/{sum(contributor_counts)}",
                flush=True,
            )

        aggregated = [arr.copy() for arr in baseline]
        for param_idx, contribs in param_contribs.items():
            base = baseline[param_idx].astype(np.float64)
            values: List[np.ndarray] = []
            weights: List[float] = []
            meta = self.expert_param_meta.get(param_idx)
            group = (
                ("expert", meta.layer_id, meta.expert_idx)
                if meta is not None
                else shared_group
            )
            selected = selected_by_group.get(group)
            for cid, value, weight in contribs:
                if selected is not None and cid not in selected:
                    continue
                current = value.astype(np.float64)
                if self.defense == "norm_clip":
                    current = base + scale.get((group, cid), 1.0) * (current - base)
                values.append(current)
                weights.append(float(weight))
            if not values:
                continue
            if self.defense == "coord_median":
                agg = defenses.coordinate_median(values)
            elif self.defense == "trimmed_mean":
                agg = defenses.trimmed_mean(values, self.defense_trim_ratio)
            else:
                agg = defenses.weighted_mean(values, weights)
            aggregated[param_idx] = agg.astype(baseline[param_idx].dtype, copy=False)
        return aggregated

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes] | BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        if failures and not self.accept_failures:
            return None, {}
        if self._current_global_arrays is None:
            raise RuntimeError("Missing baseline global arrays before aggregate_fit.")
        if len(self._current_global_arrays) != len(self.parameter_names):
            raise RuntimeError(
                "Baseline global array length mismatch. "
                f"len(arrays)={len(self._current_global_arrays)} len(parameter_names)={len(self.parameter_names)}."
            )

        baseline = self._current_global_arrays
        dense_sums: Dict[int, np.ndarray] = {}
        dense_weights: Dict[int, float] = {}
        router_row_sums: Dict[int, np.ndarray] = {}
        router_row_weights: Dict[int, np.ndarray] = {}
        # For robust-aggregation defenses (mix mode): collect per-parameter contributor
        # (cid, value, weight) tuples so a robust rule can replace the weighted mean.
        param_contribs: Dict[int, List[Tuple[str, np.ndarray, float]]] = {}

        round_times: List[float] = []
        weighted_train_loss = 0.0
        weighted_train_examples = 0.0
        round_assignments: Dict[str, Dict[str, List[int]]] = {}
        next_profiles = dict(self._last_activation_profiles)
        latest_unspoofed_profiles: Dict[str, Dict[str, List[int]]] = {}
        benign_target_activations: List[Dict[str, object]] = []
        a3fl_eval_reports: List[Tuple[str, Dict[str, Scalar]]] = []
        expert_probe_reports: List[Dict[str, Any]] = []
        local_asr_reports: List[Tuple[str, Dict[str, Scalar]]] = []
        poison_gradient_reports: List[Dict[str, Any]] = []
        update_group_selections: Dict[str, Dict[int, Optional[int]]] = {}
        target_layer = self.runtime_config.lfe_target_layer
        target_expert = self.runtime_config.lfe_target_expert
        if (
            self.runtime_config.measure_local_asr
            and self.runtime_config.mode == "mix"
            and target_layer is not None
            and target_expert is not None
        ):
            target_expert_selection = {
                index: None
                for index, meta in self.expert_param_meta.items()
                if meta.layer_id == target_layer
                and meta.expert_idx == int(target_expert)
            }
            target_router_selection = {
                index: int(target_expert)
                for index, meta in self.router_param_meta.items()
                if meta.layer_id == target_layer
            }
            shared_selection = {
                index: None
                for index in range(len(self.parameter_names))
                if index not in self.expert_param_meta
                and index not in self.router_param_meta
            }
            if not target_expert_selection:
                raise ValueError(
                    f"No parameters found for target expert "
                    f"{target_expert}@{target_layer}."
                )
            if not target_router_selection:
                raise ValueError(
                    f"No router parameters found for target layer '{target_layer}'."
                )
            update_group_selections = {
                "target_expert": target_expert_selection,
                "target_router": target_router_selection,
                "shared": shared_selection,
            }
        diagnostic_groups_by_index: Dict[int, List[str]] = {}
        for group_name, selections in update_group_selections.items():
            for parameter_index in selections:
                diagnostic_groups_by_index.setdefault(parameter_index, []).append(
                    group_name
                )
        diagnostic_contributors: Dict[str, set[str]] = {
            group_name: set() for group_name in update_group_selections
        }
        malicious_local_updates: Dict[
            str, Tuple[Dict[int, np.ndarray], float]
        ] = {}

        for client, fit_res in results:
            client_cid = client.cid
            num_examples = float(fit_res.num_examples)
            if num_examples <= 0:
                raise ValueError(f"Client {client_cid} reported non-positive num_examples={fit_res.num_examples}.")

            metrics = fit_res.metrics
            mode = _as_str(metrics.get("mode", self.runtime_config.mode)).lower()
            if mode not in {"full", "mix", "drop", "flex"}:
                raise ValueError(f"Client {client_cid} reported invalid mode='{mode}'.")
            assignments = self._parse_assignments_metric(metrics=metrics, client_cid=client_cid)
            round_assignments[client_cid] = assignments

            if "local_training_time" in metrics:
                round_times.append(_as_float(metrics["local_training_time"]))
            if "train_loss" in metrics:
                weighted_train_loss += _as_float(metrics["train_loss"]) * num_examples
                weighted_train_examples += num_examples

            if "expert_activation_map_json" in metrics:
                profile = _parse_json_dict_of_int_lists(_as_str(metrics["expert_activation_map_json"]))
                next_profiles[client_cid] = profile

            if "unspoofed_expert_activation_map_json" in metrics:
                latest_unspoofed_profiles[client_cid] = _parse_json_dict_of_int_lists(
                    _as_str(metrics["unspoofed_expert_activation_map_json"])
                )

            if "target_expert_activation_json" in metrics and not bool(metrics.get("is_malicious", False)):
                target_activation = json.loads(_as_str(metrics["target_expert_activation_json"]))
                if target_activation:
                    required = {"layer_id", "expert_id", "count", "valid_tokens", "activation_rate"}
                    missing = required.difference(target_activation)
                    if missing:
                        raise ValueError(
                            f"Client {client_cid} target activation metric is missing {sorted(missing)}."
                        )
                    benign_target_activations.append(
                        {
                            "client_id": int(metrics.get("client_id", client_cid)),
                            "layer_id": str(target_activation["layer_id"]),
                            "expert_id": int(target_activation["expert_id"]),
                            "count": int(target_activation["count"]),
                            "valid_tokens": int(target_activation["valid_tokens"]),
                            "activation_rate": float(target_activation["activation_rate"]),
                        }
                    )

            if "a3fl_eval_trigger_token_ids_json" in metrics:
                raw_trigger_ids = json.loads(
                    _as_str(metrics["a3fl_eval_trigger_token_ids_json"])
                )
                if not isinstance(raw_trigger_ids, list) or not raw_trigger_ids:
                    raise ValueError(
                        f"Client {client_cid} reported an invalid A3FL evaluation trigger."
                    )
                if any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in raw_trigger_ids
                ):
                    raise ValueError(
                        f"Client {client_cid} reported invalid A3FL evaluation token IDs."
                    )
                a3fl_eval_reports.append((client_cid, metrics))

            if "local_asr_pre" in metrics or "local_asr_post" in metrics:
                required_local_asr = {
                    "local_asr_pre",
                    "local_asr_post",
                    "local_asr_gain",
                    "local_asr_eval_examples",
                }
                missing_local_asr = required_local_asr.difference(metrics)
                if missing_local_asr:
                    raise ValueError(
                        f"Client {client_cid} local ASR report is missing "
                        f"{sorted(missing_local_asr)}."
                    )
                if mode != "mix":
                    raise ValueError(
                        f"Client {client_cid} reported local ASR in unsupported mode='{mode}'."
                    )
                local_asr_reports.append((client_cid, metrics))

            if "expert_probe_report_json" in metrics:
                probe_payload = json.loads(_as_str(metrics["expert_probe_report_json"]))
                if not isinstance(probe_payload, dict):
                    raise ValueError(
                        f"Client {client_cid} reported a non-object expert probe payload."
                    )
                expert_probe_reports.append(
                    {
                        "round": int(server_round),
                        "client_id": int(metrics.get("client_id", client_cid)),
                        **probe_payload,
                    }
                )

            if "poison_gradient_flow_json" in metrics:
                gradient_payload = json.loads(
                    _as_str(metrics["poison_gradient_flow_json"])
                )
                if not isinstance(gradient_payload, dict):
                    raise ValueError(
                        f"Client {client_cid} poison-gradient report is not an object."
                    )
                groups = gradient_payload.get("groups")
                if not isinstance(groups, dict) or not groups:
                    raise ValueError(
                        f"Client {client_cid} poison-gradient report has no groups."
                    )
                poison_gradient_reports.append(
                    {
                        "round": int(server_round),
                        "client_id": int(metrics.get("client_id", client_cid)),
                        **gradient_payload,
                    }
                )

            if "returned_parameter_names_json" not in metrics:
                raise KeyError(
                    f"Client {client_cid} did not report returned_parameter_names_json, which is required."
                )
            returned_names = _parse_json_list_of_str(_as_str(metrics["returned_parameter_names_json"]))
            returned_arrays = parameters_to_ndarrays(fit_res.parameters)
            if len(returned_names) != len(returned_arrays):
                raise ValueError(
                    f"Client {client_cid} returned parameter metadata mismatch: "
                    f"len(names)={len(returned_names)} len(arrays)={len(returned_arrays)}."
                )

            client_diagnostic_arrays: Dict[int, np.ndarray] = {}
            client_diagnostic_groups: set[str] = set()
            captures_malicious_update = "local_asr_pre" in metrics

            def record_diagnostic_parameter(
                parameter_index: int,
                value: np.ndarray,
            ) -> None:
                groups = diagnostic_groups_by_index.get(parameter_index, [])
                if not groups:
                    return
                client_diagnostic_groups.update(groups)
                if captures_malicious_update:
                    client_diagnostic_arrays[parameter_index] = np.asarray(
                        value
                    ).copy()

            for local_name, array in zip(returned_names, returned_arrays):
                mapped_name = local_name
                expert_match = EXPERT_PARAM_PATTERN.match(local_name)
                if mode == "drop" and expert_match is not None:
                    layer_id = f"{expert_match.group(1)}.{int(expert_match.group(2))}"
                    local_expert_idx = int(expert_match.group(3))
                    suffix = expert_match.group(4)
                    if layer_id not in assignments:
                        raise KeyError(
                            f"Client {client_cid} missing assignment for drop layer={layer_id} while remapping."
                        )
                    layer_assignment = assignments[layer_id]
                    if local_expert_idx >= len(layer_assignment):
                        raise ValueError(
                            f"Client {client_cid} local expert idx {local_expert_idx} out of bounds for layer "
                            f"{layer_id} assignment size {len(layer_assignment)}."
                        )
                    global_expert_idx = int(layer_assignment[local_expert_idx])
                    lookup_key = (layer_id, global_expert_idx, suffix)
                    if lookup_key not in self.expert_name_lookup:
                        raise KeyError(
                            f"Client {client_cid} remap lookup failed for layer={layer_id}, "
                            f"expert={global_expert_idx}, suffix='{suffix}'."
                        )
                    mapped_name = self.expert_name_lookup[lookup_key]

                if mapped_name not in self.param_name_to_idx:
                    raise KeyError(f"Client {client_cid} returned unknown parameter name '{mapped_name}'.")
                param_idx = self.param_name_to_idx[mapped_name]

                if self.defense == "none" and param_idx in self.router_param_meta:
                    if mode == "mix" and array.shape != baseline[param_idx].shape:
                        raise ValueError(
                            f"Mix-mode router diagnostic shape mismatch for "
                            f"'{mapped_name}': got={array.shape}, "
                            f"expected={baseline[param_idx].shape}."
                        )
                    record_diagnostic_parameter(param_idx, array)
                    self._accumulate_router_rows(
                        router_row_sums=router_row_sums,
                        router_row_weights=router_row_weights,
                        baseline=baseline,
                        param_idx=param_idx,
                        array=array,
                        weight=num_examples,
                        assignments=assignments,
                        mode=mode,
                    )
                    continue

                expected_shape = baseline[param_idx].shape
                if array.shape != expected_shape:
                    raise ValueError(
                        f"Shape mismatch for '{mapped_name}' from client={client_cid}: "
                        f"got={array.shape} expected={expected_shape}."
                    )

                if param_idx in self.expert_param_meta:
                    meta = self.expert_param_meta[param_idx]
                    if meta.layer_id not in assignments:
                        raise KeyError(
                            f"Client {client_cid} missing layer assignment for expert key '{mapped_name}'."
                        )
                    if meta.expert_idx not in assignments[meta.layer_id]:
                        continue

                record_diagnostic_parameter(param_idx, array)

                if self.defense == "none":
                    self._accumulate_dense(
                        dense_sums=dense_sums,
                        dense_weights=dense_weights,
                        param_idx=param_idx,
                        array=array,
                        weight=num_examples,
                    )
                else:
                    param_contribs.setdefault(param_idx, []).append(
                        (client_cid, array.astype(np.float32, copy=False), num_examples)
                    )

            for group_name in client_diagnostic_groups:
                diagnostic_contributors[group_name].add(client_cid)
            if captures_malicious_update and client_diagnostic_arrays:
                malicious_local_updates[client_cid] = (
                    client_diagnostic_arrays,
                    num_examples,
                )

        enforce_global_coverage = self._requires_global_coverage()
        self._validate_round_coverage(round_assignments, enforce_global_coverage=enforce_global_coverage)

        # Per-expert contributor counts at the target layer: direct evidence that a
        # cold expert has ~1 (only the malicious) contributor and is thus applied
        # verbatim without FedAvg dilution, while hot experts are averaged over many.
        contributor_layer = self.runtime_config.lfe_target_layer or (self.layer_ids[0] if self.layer_ids else None)
        if contributor_layer and contributor_layer in self.layer_num_experts:
            contrib = [0 for _ in range(self.layer_num_experts[contributor_layer])]
            for client_assignment in round_assignments.values():
                for expert_idx in client_assignment.get(contributor_layer, []):
                    if 0 <= expert_idx < len(contrib):
                        contrib[expert_idx] += 1
            print(f"[MixFedMoE][round={server_round}] contributors@{contributor_layer}: {contrib}", flush=True)

        if enforce_global_coverage and self.defense != "none":
            for param_idx, meta in self.expert_param_meta.items():
                if not param_contribs.get(param_idx):
                    raise RuntimeError(
                        f"Sparse aggregation failure (defense={self.defense}): no contributor for expert "
                        f"parameter '{self.parameter_names[param_idx]}' in layer={meta.layer_id}, "
                        f"expert={meta.expert_idx}."
                    )
        if enforce_global_coverage and self.defense == "none":
            for param_idx, meta in self.expert_param_meta.items():
                if dense_weights.get(param_idx, 0.0) <= 0.0:
                    raise RuntimeError(
                        f"Sparse aggregation failure: no contributor for expert parameter "
                        f"'{self.parameter_names[param_idx]}' in layer={meta.layer_id}, expert={meta.expert_idx}."
                    )
            for param_idx, meta in self.router_param_meta.items():
                if param_idx not in router_row_sums:
                    raise RuntimeError(
                        f"Router aggregation failure: no contributors for '{self.parameter_names[param_idx]}' "
                        f"(layer={meta.layer_id})."
                    )
                missing_rows = np.where(router_row_weights[param_idx] <= 0)[0].tolist()
                if missing_rows:
                    raise RuntimeError(
                        f"Router aggregation failure for '{self.parameter_names[param_idx]}': "
                        f"missing expert rows={missing_rows}."
                    )

        if self.defense == "none":
            aggregated = [arr.copy() for arr in baseline]
            for param_idx, weighted_sum in dense_sums.items():
                weight = dense_weights[param_idx]
                if weight <= 0.0:
                    raise RuntimeError(f"Zero aggregation weight for parameter idx={param_idx}.")
                aggregated[param_idx] = (weighted_sum / np.float32(weight)).astype(
                    baseline[param_idx].dtype, copy=False
                )

            for param_idx, weighted_rows in router_row_sums.items():
                row_weights = router_row_weights[param_idx]
                averaged = baseline[param_idx].astype(np.float32, copy=True)
                valid_rows = row_weights > 0
                if weighted_rows.ndim == 2:
                    averaged[valid_rows] = weighted_rows[valid_rows] / row_weights[valid_rows, None]
                elif weighted_rows.ndim == 1:
                    averaged[valid_rows] = weighted_rows[valid_rows] / row_weights[valid_rows]
                else:
                    raise RuntimeError(
                        f"Unsupported router tensor ndim={weighted_rows.ndim} for idx={param_idx}."
                    )
                aggregated[param_idx] = averaged.astype(baseline[param_idx].dtype, copy=False)
        else:
            aggregated = self._defense_aggregate(baseline=baseline, param_contribs=param_contribs)

        round_time = max(round_times) if round_times else 0.0
        self.cumulative_round_time += round_time
        self._last_activation_profiles = next_profiles
        self._latest_unspoofed_activation_profiles = latest_unspoofed_profiles
        self._current_global_arrays = aggregated
        self._save_checkpoint(server_round=server_round, arrays=aggregated)
        if expert_probe_reports:
            append_expert_probe_reports(
                self.runtime_config.output_dir,
                expert_probe_reports,
            )
            self._record_expert_probe_discovery(
                server_round=server_round,
                reports=expert_probe_reports,
                arrays=aggregated,
            )
        poison_gradient_jsonl_path = ""
        poison_gradient_csv_path = ""
        if poison_gradient_reports:
            (
                poison_gradient_jsonl_path,
                poison_gradient_csv_path,
            ) = append_poison_gradient_flow_reports(
                self.runtime_config.output_dir,
                poison_gradient_reports,
            )

        metrics_aggregated: Dict[str, Scalar] = {
            "num_clients_sampled": int(len(results)),
            "round_time": float(round_time),
            "cumulative_time": float(self.cumulative_round_time),
            "assignment_policy": self.runtime_config.assignment_policy,
        }
        if poison_gradient_reports:
            metrics_aggregated.update(
                {
                    "poison_gradient_flow_num_clients": int(
                        len(poison_gradient_reports)
                    ),
                    "poison_gradient_flow_jsonl_path": poison_gradient_jsonl_path,
                    "poison_gradient_flow_csv_path": poison_gradient_csv_path,
                }
            )
        if self._expert_probe_stop_requested:
            found = self._found_low_frequency_expert or {}
            metrics_aggregated.update(
                {
                    "expert_probe_stop_requested": True,
                    "found_low_frequency_expert": str(found.get("expert", "")),
                    "found_low_frequency_expert_layer": str(found.get("layer", "")),
                    "found_low_frequency_expert_id": int(found.get("expert_id", -1)),
                }
            )
        metrics_aggregated.update(self._last_defense_metrics)
        if weighted_train_examples > 0:
            metrics_aggregated["train_loss"] = float(weighted_train_loss / weighted_train_examples)
        metrics_aggregated.update(_aggregate_local_asr_reports(local_asr_reports))
        if local_asr_reports and update_group_selections:
            diagnostic_payload: Dict[str, Any] = {}
            for group_name, selections in update_group_selections.items():
                group_metrics, client_reports = _aggregate_update_group_diagnostics(
                    prefix=group_name,
                    before=baseline,
                    aggregated=aggregated,
                    selections=selections,
                    malicious_updates=malicious_local_updates,
                    contributors=len(diagnostic_contributors[group_name]),
                )
                metrics_aggregated.update(group_metrics)
                diagnostic_payload[group_name] = {
                    "contributors": len(diagnostic_contributors[group_name]),
                    "clients": client_reports,
                }
            metrics_aggregated["parameter_update_diagnostics_json"] = json.dumps(
                diagnostic_payload,
                sort_keys=True,
            )
            print(
                f"[MixFedMoE][round={server_round}] update retention: "
                + " ".join(
                    f"{group_name}(local_l2="
                    f"{float(metrics_aggregated.get(f'{group_name}_local_update_l2', 0.0)):.6g},"
                    f"global_l2="
                    f"{float(metrics_aggregated.get(f'{group_name}_global_update_l2', 0.0)):.6g},"
                    f"projection="
                    f"{float(metrics_aggregated.get(f'{group_name}_aggregation_retention', 0.0)):.6g},"
                    f"contributors="
                    f"{int(metrics_aggregated.get(f'{group_name}_contributors', 0))})"
                    for group_name in update_group_selections
                ),
                flush=True,
            )
        if a3fl_eval_reports:
            reported_eval_triggers = {
                tuple(
                    int(value)
                    for value in json.loads(
                        _as_str(metrics["a3fl_eval_trigger_token_ids_json"])
                    )
                )
                for _cid, metrics in a3fl_eval_reports
            }
            if len(reported_eval_triggers) != 1:
                raise ValueError(
                    "Malicious clients used different A3FL triggers in the same round."
                )
            _leader_cid, leader_metrics = min(
                a3fl_eval_reports, key=lambda item: _cid_sort_key(item[0])
            )
            self._a3fl_eval_trigger_token_ids = [
                int(value)
                for value in json.loads(
                    _as_str(leader_metrics["a3fl_eval_trigger_token_ids_json"])
                )
            ]
            self._a3fl_eval_trigger_text = _as_str(
                leader_metrics.get("a3fl_eval_trigger_text", "")
            )
        if self._a3fl_eval_trigger_token_ids is not None:
            # Retain the last attack-window trigger to measure post-attack ASR
            # and lifespan, matching the BadNet centralized evaluation flow.
            metrics_aggregated["a3fl_eval_trigger_token_ids_json"] = json.dumps(
                self._a3fl_eval_trigger_token_ids
            )
            metrics_aggregated["a3fl_eval_trigger_text"] = (
                self._a3fl_eval_trigger_text or ""
            )
        if benign_target_activations:
            target_keys = {
                (str(item["layer_id"]), int(item["expert_id"]))
                for item in benign_target_activations
            }
            if len(target_keys) != 1:
                raise ValueError(f"Benign clients reported inconsistent target experts: {sorted(target_keys)}.")
            layer_id, expert_id = next(iter(target_keys))
            total_count = sum(int(item["count"]) for item in benign_target_activations)
            total_tokens = sum(int(item["valid_tokens"]) for item in benign_target_activations)
            aggregate_rate = float(total_count / total_tokens) if total_tokens else 0.0
            csv_rows: List[Dict[str, object]] = [
                {
                    "round": int(server_round),
                    "client_id": int(item["client_id"]),
                    "layer_id": layer_id,
                    "expert_id": expert_id,
                    "activation_count": int(item["count"]),
                    "valid_tokens": int(item["valid_tokens"]),
                    "activation_rate": float(item["activation_rate"]),
                }
                for item in sorted(benign_target_activations, key=lambda value: int(value["client_id"]))
            ]
            csv_rows.append(
                {
                    "round": int(server_round),
                    "client_id": "ALL_BENIGN",
                    "layer_id": layer_id,
                    "expert_id": expert_id,
                    "activation_count": total_count,
                    "valid_tokens": total_tokens,
                    "activation_rate": aggregate_rate,
                }
            )
            activation_path = append_benign_lfe_activation_rows(
                self.runtime_config.output_dir, csv_rows
            )
            metrics_aggregated["benign_lfe_activation_rate"] = aggregate_rate
            metrics_aggregated["benign_lfe_activation_count"] = int(total_count)
            metrics_aggregated["benign_lfe_valid_tokens"] = int(total_tokens)
            metrics_aggregated["num_benign_clients_profiled"] = int(len(benign_target_activations))
            print(
                f"[MixFedMoE][round={server_round}] benign expert {expert_id}@{layer_id} "
                f"activation={total_count}/{total_tokens} ({aggregate_rate:.6f}) across "
                f"{len(benign_target_activations)} clients; csv={activation_path}",
                flush=True,
            )
        self._latest_fit_metrics = dict(metrics_aggregated)

        return ndarrays_to_parameters(aggregated), metrics_aggregated

    def _save_checkpoint(
        self,
        server_round: int,
        arrays: Sequence[np.ndarray],
        *,
        force: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        checkpoint_interval = int(
            getattr(self.runtime_config, "checkpoint_interval", 5)
        )
        is_periodic_round = (
            server_round > 0
            and checkpoint_interval > 0
            and server_round % checkpoint_interval == 0
        )
        is_configured_round = server_round in self.runtime_config.checkpoint_rounds
        if not force and not is_periodic_round and not is_configured_round:
            return

        reasons: List[str] = []
        if is_periodic_round:
            reasons.append("periodic_interval")
        if is_configured_round:
            reasons.append("configured_round")
        event_reason = metadata.get("checkpoint_reason") if metadata else None
        if event_reason is not None and str(event_reason) not in reasons:
            reasons.append(str(event_reason))
        if force and not reasons:
            reasons.append("forced")

        checkpoint_dir = os.path.join(self.runtime_config.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"round_{server_round:04d}.pt")
        state_dict = {
            name: torch.from_numpy(array.copy())
            for name, array in zip(self.parameter_names, arrays)
        }
        payload: Dict[str, Any] = {
            "round": int(server_round),
            "parameter_names": list(self.parameter_names),
            "state_dict": state_dict,
            "checkpoint_reason": reasons[-1],
            "checkpoint_reasons": reasons,
        }
        if metadata:
            payload.update(metadata)
        torch.save(payload, checkpoint_path)
        print(f"[MixFedMoE] Saved checkpoint: {checkpoint_path}", flush=True)

    def _is_live_attack_round(self, server_round: int) -> bool:
        """Return whether this round contains malicious local optimization."""
        if not self.runtime_config.attack_enabled or self._attack_stopped_on_asr:
            return False
        # Client round one is always a routing-profile warmup without updates.
        if server_round < max(2, int(self.runtime_config.attack_start_round)):
            return False
        end_round = int(self.runtime_config.attack_end_round)
        return end_round <= 0 or server_round <= end_round

    def _is_persistence_monitor_round(self, server_round: int) -> bool:
        if self.runtime_config.monitor_asr:
            return True
        if not self.runtime_config.attack_enabled:
            return False
        if self._attack_stopped_on_asr:
            return bool(
                self._attack_stop_round is not None
                and server_round > self._attack_stop_round
            )
        end_round = int(self.runtime_config.attack_end_round)
        return end_round > 0 and server_round > end_round

    def evaluate(self, server_round: int, parameters: Parameters) -> tuple[float, Dict[str, Scalar]] | None:
        """Centralized server-side evaluation callback wrapper."""
        if self.evaluate_fn is None:
            return None

        eval_config: Dict[str, Scalar] = {
            "round_time": float(self._latest_fit_metrics.get("round_time", 0.0)),
            "cumulative_time": float(self.cumulative_round_time),
            "num_clients_sampled": int(self._latest_fit_metrics.get("num_clients_sampled", 0)),
            "mode": self.runtime_config.mode,
            "assignment_policy": self.runtime_config.assignment_policy,
            "test_samples": int(self.runtime_config.test_samples),
            "attack_enabled": self._is_live_attack_round(server_round),
            "monitor_asr": self._is_persistence_monitor_round(server_round),
        }
        if "train_loss" in self._latest_fit_metrics:
            eval_config["train_loss"] = _as_float(self._latest_fit_metrics["train_loss"])
        for key in (
            "benign_lfe_activation_rate",
            "benign_lfe_activation_count",
            "benign_lfe_valid_tokens",
            "num_benign_clients_profiled",
            "a3fl_eval_trigger_token_ids_json",
            "a3fl_eval_trigger_text",
            "local_asr_pre",
            "local_asr_post",
            "local_asr_gain",
            "local_asr_eval_examples",
            "local_asr_num_clients",
            "local_asr_clients_json",
        ):
            if key in self._latest_fit_metrics:
                eval_config[key] = self._latest_fit_metrics[key]
        for key, value in self._latest_fit_metrics.items():
            if key.startswith(("target_expert_", "target_router_", "shared_")) or key in {
                "num_malicious_target_updates",
                "parameter_update_diagnostics_json",
            }:
                eval_config[key] = value

        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, eval_config)
        if eval_res is None:
            return None
        loss, metrics = eval_res
        merged_metrics: Dict[str, Scalar] = dict(metrics)
        for key, value in eval_config.items():
            if key not in merged_metrics:
                merged_metrics[key] = value

        asr_value = merged_metrics.get("asr")
        if (
            not self._attack_stopped_on_asr
            and self._is_live_attack_round(server_round)
            and asr_value is not None
            and _as_float(asr_value) >= 1.0
        ):
            perfect_asr = _as_float(asr_value)
            restored_profiles: List[str] = []
            removed_profiles: List[str] = []
            for malicious_client_id in self.runtime_config.malicious_clients:
                client_cid = str(malicious_client_id)
                genuine_profile = self._latest_unspoofed_activation_profiles.get(
                    client_cid
                )
                if genuine_profile is not None:
                    self._last_activation_profiles[client_cid] = genuine_profile
                    restored_profiles.append(client_cid)
                elif client_cid in self._last_activation_profiles:
                    # Be conservative with clients that do not provide the new
                    # genuine-profile field: dropping their cached profile is
                    # safer than carrying a potentially spoofed one into the
                    # first persistence-monitoring round.
                    del self._last_activation_profiles[client_cid]
                    removed_profiles.append(client_cid)
            self._save_checkpoint(
                server_round=server_round,
                arrays=parameters_ndarrays,
                force=True,
                metadata={
                    "asr": perfect_asr,
                    "attack_stopped_on_asr": True,
                    "attack_stop_round": int(server_round),
                    "checkpoint_reason": "asr_reached_1",
                },
            )
            self._attack_stopped_on_asr = True
            self._attack_stop_round = int(server_round)
            merged_metrics["attack_stopped_on_asr"] = True
            merged_metrics["attack_stop_round"] = int(server_round)
            print(
                f"[MixFedMoE][round={server_round}] ASR={perfect_asr:.6f} reached 1.0; "
                "checkpoint saved, current-round malicious activation-profile "
                f"spoofs were discarded (restored={restored_profiles}, "
                f"removed={removed_profiles}), and attacks are disabled from the next round. "
                "Continuing with monitor_asr persistence evaluation.",
                flush=True,
            )
        return float(loss), merged_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> Tuple[float | None, Dict[str, Scalar]]:
        return None, {}
