from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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
from mixfedmoe_fl.logging_utils import append_benign_lfe_activation_rows

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
        self.cumulative_round_time: float = 0.0
        self._latest_fit_metrics: Dict[str, Scalar] = {}
        # Centralized ASR is only known on the server after a round has been
        # aggregated. Keep the transition here (rather than in a simulation
        # client, which may be recreated) and broadcast it on subsequent fits.
        self._inject_benign_training_active: bool = False
        self._inject_benign_training_trigger_round: Optional[int] = None

        self.defense = str(getattr(self.runtime_config, "defense", "none"))
        self.defense_clip_norm = float(getattr(self.runtime_config, "defense_clip_norm", 5.0))
        self.defense_trim_ratio = float(getattr(self.runtime_config, "defense_trim_ratio", 0.2))
        configured_malicious = int(getattr(self.runtime_config, "defense_num_malicious", 0))
        self.defense_num_malicious = (
            configured_malicious if configured_malicious > 0 else len(self.runtime_config.malicious_clients)
        )
        configured_m = int(getattr(self.runtime_config, "defense_krum_m", 0))
        self.defense_krum_m = (
            configured_m
            if configured_m > 0
            else max(1, int(self.runtime_config.num_clients) - self.defense_num_malicious)
        )
        if self.defense != "none" and self.runtime_config.mode != "mix":
            raise ValueError("Robust-aggregation defenses are implemented for mix mode only.")

    def _effective_k(self) -> int:
        if self.runtime_config.mode == "flex":
            return 1
        return int(self.runtime_config.k)

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
        if "assigned_experts_json" in metrics:
            parsed = _parse_json_dict_of_int_lists(_as_str(metrics["assigned_experts_json"]))
            return {layer_id: sorted(set(vals)) for layer_id, vals in parsed.items()}
        if client_cid in self._last_round_assignments:
            return self._last_round_assignments[client_cid]
        raise KeyError(f"Client {client_cid} did not report assigned_experts_json.")

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
                "incoming_parameter_names_json": incoming_names_json,
                "assigned_experts_json": json.dumps(per_client_assignments[client.cid], sort_keys=True),
                "inject_benign_training_active": bool(self._inject_benign_training_active),
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

        Client-level defenses (norm_clip, multi_krum) reweight/select whole clients
        via their flat update vectors; coordinate defenses (coord_median,
        trimmed_mean) replace the per-parameter weighted mean. A cold expert has a
        single contributor, so median/trimmed/krum cannot filter it -- the defense
        degenerates and the poisoned update passes through unchanged.
        """
        client_param_delta: Dict[str, Dict[int, np.ndarray]] = {}
        for param_idx, contribs in param_contribs.items():
            base = baseline[param_idx].astype(np.float64)
            for cid, value, _weight in contribs:
                client_param_delta.setdefault(cid, {})[param_idx] = value.astype(np.float64) - base

        scale: Dict[str, float] = {}
        if self.defense == "norm_clip":
            for cid, deltas in client_param_delta.items():
                squared = sum(float(np.sum(delta * delta)) for delta in deltas.values())
                scale[cid] = defenses.norm_clip_scale(math.sqrt(squared), self.defense_clip_norm)

        selected: Optional[set] = None
        if self.defense == "multi_krum":
            all_idx = sorted({idx for deltas in client_param_delta.values() for idx in deltas})
            vectors: Dict[str, np.ndarray] = {}
            for cid, deltas in client_param_delta.items():
                parts = [
                    deltas[idx].ravel()
                    if idx in deltas
                    else np.zeros(baseline[idx].size, dtype=np.float64)
                    for idx in all_idx
                ]
                vectors[cid] = np.concatenate(parts) if parts else np.zeros(1, dtype=np.float64)
            selected = set(
                defenses.multi_krum_select(vectors, self.defense_num_malicious, self.defense_krum_m)
            )

        aggregated = [arr.copy() for arr in baseline]
        for param_idx, contribs in param_contribs.items():
            base = baseline[param_idx].astype(np.float64)
            values: List[np.ndarray] = []
            weights: List[float] = []
            for cid, value, weight in contribs:
                if selected is not None and cid not in selected:
                    continue
                current = value.astype(np.float64)
                if self.defense == "norm_clip":
                    current = base + scale.get(cid, 1.0) * (current - base)
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
        benign_target_activations: List[Dict[str, object]] = []

        for client, fit_res in results:
            client_cid = client.cid
            num_examples = float(fit_res.num_examples)
            if num_examples <= 0:
                raise ValueError(f"Client {client_cid} reported non-positive num_examples={fit_res.num_examples}.")

            metrics = fit_res.metrics
            mode = _as_str(metrics.get("mode", self.runtime_config.mode)).lower()
            if mode not in {"full", "mix", "drop", "flex"}:
                raise ValueError(f"Client {client_cid} reported invalid mode='{mode}'.")
            reported_benign_training = bool(metrics.get("inject_benign_training_active", False))
            reported_client_id = int(metrics.get("client_id", client_cid))
            attack_end = int(self.runtime_config.attack_end_round)
            attack_window_active = server_round >= int(self.runtime_config.attack_start_round) and (
                attack_end <= 0 or server_round <= attack_end
            )
            inject_benign_training_active = bool(
                self._inject_benign_training_active
                and self.runtime_config.attack_enabled
                and self.runtime_config.inject_target_expert
                and reported_client_id in self.runtime_config.malicious_clients
                and attack_window_active
            )
            if reported_benign_training != inject_benign_training_active:
                raise ValueError(
                    f"Client {client_cid} reported inject_benign_training_active="
                    f"{reported_benign_training}, expected {inject_benign_training_active}."
                )

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

        enforce_global_coverage = self._requires_global_coverage()
        self._validate_round_coverage(round_assignments, enforce_global_coverage=enforce_global_coverage)

        # Per-expert contributor counts at the target layer: direct evidence that a
        # cold expert has ~1 (only the malicious) contributor and is thus applied
        # verbatim without FedAvg dilution, while hot experts are averaged over many.
        target_layer = self.runtime_config.lfe_target_layer or (self.layer_ids[0] if self.layer_ids else None)
        if target_layer and target_layer in self.layer_num_experts:
            contrib = [0 for _ in range(self.layer_num_experts[target_layer])]
            for client_assignment in round_assignments.values():
                for expert_idx in client_assignment.get(target_layer, []):
                    if 0 <= expert_idx < len(contrib):
                        contrib[expert_idx] += 1
            print(f"[MixFedMoE][round={server_round}] contributors@{target_layer}: {contrib}", flush=True)

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
        self._current_global_arrays = aggregated
        self._save_checkpoint(server_round=server_round, arrays=aggregated)

        metrics_aggregated: Dict[str, Scalar] = {
            "num_clients_sampled": int(len(results)),
            "round_time": float(round_time),
            "cumulative_time": float(self.cumulative_round_time),
            "assignment_policy": self.runtime_config.assignment_policy,
        }
        if weighted_train_examples > 0:
            metrics_aggregated["train_loss"] = float(weighted_train_loss / weighted_train_examples)
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

    def _save_checkpoint(self, server_round: int, arrays: Sequence[np.ndarray]) -> None:
        if server_round not in self.runtime_config.checkpoint_rounds:
            return
        checkpoint_dir = os.path.join(self.runtime_config.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"round_{server_round:04d}.pt")
        state_dict = {
            name: torch.from_numpy(array.copy())
            for name, array in zip(self.parameter_names, arrays)
        }
        torch.save(
            {
                "round": int(server_round),
                "parameter_names": list(self.parameter_names),
                "state_dict": state_dict,
            },
            checkpoint_path,
        )
        print(f"[MixFedMoE] Saved checkpoint: {checkpoint_path}", flush=True)

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
        }
        if "train_loss" in self._latest_fit_metrics:
            eval_config["train_loss"] = _as_float(self._latest_fit_metrics["train_loss"])
        for key in (
            "benign_lfe_activation_rate",
            "benign_lfe_activation_count",
            "benign_lfe_valid_tokens",
            "num_benign_clients_profiled",
        ):
            if key in self._latest_fit_metrics:
                eval_config[key] = self._latest_fit_metrics[key]

        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, eval_config)
        if eval_res is None:
            return None
        loss, metrics = eval_res
        merged_metrics: Dict[str, Scalar] = dict(metrics)
        if (
            not self._inject_benign_training_active
            and self.runtime_config.attack_enabled
            and self.runtime_config.inject_target_expert
            and "asr" in metrics
            and _as_float(metrics["asr"])
            >= float(self.runtime_config.inject_benign_train_asr_threshold)
        ):
            self._inject_benign_training_active = True
            self._inject_benign_training_trigger_round = int(server_round)
            print(
                f"[MixFedMoE][round={server_round}] ASR={_as_float(metrics['asr']):.6f} reached "
                f"{self.runtime_config.inject_benign_train_asr_threshold:.6f}; malicious clients "
                "will resume benign-style mode trainability with poisoned data from the next "
                "attack round onward.",
                flush=True,
            )
        merged_metrics["inject_benign_training_active"] = bool(self._inject_benign_training_active)
        if self._inject_benign_training_trigger_round is not None:
            merged_metrics["inject_benign_training_trigger_round"] = int(
                self._inject_benign_training_trigger_round
            )
        for key, value in eval_config.items():
            if key not in merged_metrics:
                merged_metrics[key] = value
        return float(loss), merged_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> Tuple[float | None, Dict[str, Scalar]]:
        return None, {}
