from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pytest
import torch
from flwr.common import Code, FitRes, Parameters, Status, ndarrays_to_parameters, parameters_to_ndarrays

from mixfedmoe_fl.config import MixFedMoEConfig
from mixfedmoe_fl.strategy import MixFedMoEStrategy


@dataclass(frozen=True)
class DummyClientProxy:
    cid: str


class DummyClientManager:
    def __init__(self, clients: Sequence[DummyClientProxy]) -> None:
        self._clients = list(clients)

    def num_available(self) -> int:
        return len(self._clients)

    def sample(self, num_clients: int, min_num_clients: int) -> List[DummyClientProxy]:
        if len(self._clients) < min_num_clients:
            raise RuntimeError(
                f"Not enough dummy clients: have={len(self._clients)} required={min_num_clients}."
            )
        if num_clients > len(self._clients):
            raise RuntimeError(
                f"Requested num_clients={num_clients}, available={len(self._clients)}."
            )
        return self._clients[:num_clients]


def _build_config(
    *,
    mode: str = "mix",
    num_clients: int = 2,
    k: int = 1,
    fraction_fit: float = 1.0,
    local_epochs: float = 1.0,
    assignment_policy: str = "hot",
) -> MixFedMoEConfig:
    return MixFedMoEConfig(
        mode=mode,  # type: ignore[arg-type]
        model_name_or_path="model_ckpt/switch-base-8",
        dataset_name="ag_news",
        num_clients=num_clients,
        num_rounds=2,
        k=k,
        alpha=0.5,
        fraction_fit=fraction_fit,
        fraction_evaluate=0.0,
        local_epochs=local_epochs,
        learning_rate=2e-5,
        weight_decay=0.0,
        train_batch_size=2,
        eval_batch_size=2,
        max_length=32,
        client_eval_ratio=0.2,
        seed=42,
        num_cpus_per_client=1,
        num_gpus_per_client=0.0,
        output_dir="outputs/mixfedmoe",
        test_samples=16,
        assignment_policy=assignment_policy,  # type: ignore[arg-type]
    )


def _parameter_names() -> List[str]:
    return [
        "shared.weight",
        "encoder.block.0.layer.1.mlp.experts.expert_0.wi.weight",
        "encoder.block.0.layer.1.mlp.experts.expert_1.wi.weight",
        "encoder.block.0.layer.1.mlp.router.classifier.weight",
        "encoder.block.0.layer.1.mlp.router.classifier.bias",
    ]


def _initial_arrays() -> List[np.ndarray]:
    return [
        np.array([0.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.zeros((2, 1), dtype=np.float32),
        np.zeros((2,), dtype=np.float32),
    ]


def _build_strategy(
    *,
    mode: str = "mix",
    num_clients: int = 2,
    k: int = 1,
    fraction_fit: float = 1.0,
    local_epochs: float = 1.0,
    assignment_policy: str = "hot",
) -> Tuple[MixFedMoEStrategy, Parameters]:
    names = _parameter_names()
    initial_parameters = ndarrays_to_parameters(_initial_arrays())
    strategy = MixFedMoEStrategy(
        runtime_config=_build_config(
            mode=mode,
            num_clients=num_clients,
            k=k,
            fraction_fit=fraction_fit,
            local_epochs=local_epochs,
            assignment_policy=assignment_policy,
        ),
        parameter_names=names,
        initial_parameters=initial_parameters,
    )
    return strategy, initial_parameters


def _build_fit_res(
    *,
    arrays: Sequence[np.ndarray],
    num_examples: int,
    metrics: Dict[str, object],
) -> FitRes:
    return FitRes(
        status=Status(code=Code.OK, message="ok"),
        parameters=ndarrays_to_parameters(list(arrays)),
        num_examples=num_examples,
        metrics=metrics,
    )


def test_configure_fit_assigns_exact_k_with_full_coverage() -> None:
    strategy, initial_parameters = _build_strategy(mode="mix", num_clients=2, k=1)
    manager = DummyClientManager([DummyClientProxy(cid="0"), DummyClientProxy(cid="1")])

    fit_instructions = strategy.configure_fit(
        server_round=1,
        parameters=initial_parameters,
        client_manager=manager,  # type: ignore[arg-type]
    )
    assert len(fit_instructions) == 2

    covered = set()
    for _client, fit_ins in fit_instructions:
        assigned = json.loads(str(fit_ins.config["assigned_experts_json"]))
        assert len(assigned["encoder.0"]) == 1
        covered.update(int(x) for x in assigned["encoder.0"])
        incoming_names = json.loads(str(fit_ins.config["incoming_parameter_names_json"]))
        assert len(incoming_names) == len(_parameter_names())
        assert fit_ins.config["local_epochs"] == 1.0
    assert covered == {0, 1}


def test_configure_fit_flex_assigns_one_expert_with_full_coverage() -> None:
    strategy, initial_parameters = _build_strategy(mode="flex", num_clients=2, k=1, assignment_policy="hot")
    manager = DummyClientManager([DummyClientProxy(cid="0"), DummyClientProxy(cid="1")])

    fit_instructions = strategy.configure_fit(
        server_round=1,
        parameters=initial_parameters,
        client_manager=manager,  # type: ignore[arg-type]
    )
    assert len(fit_instructions) == 2

    covered = set()
    for _client, fit_ins in fit_instructions:
        assigned = json.loads(str(fit_ins.config["assigned_experts_json"]))
        assert len(assigned["encoder.0"]) == 1
        covered.update(int(x) for x in assigned["encoder.0"])
        assert fit_ins.config["mode"] == "flex"
        assert int(fit_ins.config["k"]) == 1
    assert covered == {0, 1}


def test_configure_fit_preserves_fractional_local_epochs() -> None:
    strategy, initial_parameters = _build_strategy(mode="mix", num_clients=2, k=1, local_epochs=0.5)
    manager = DummyClientManager([DummyClientProxy(cid="0"), DummyClientProxy(cid="1")])

    fit_instructions = strategy.configure_fit(
        server_round=1,
        parameters=initial_parameters,
        client_manager=manager,  # type: ignore[arg-type]
    )
    assert len(fit_instructions) == 2
    for _client, fit_ins in fit_instructions:
        assert fit_ins.config["local_epochs"] == 0.5
        assert fit_ins.config["server_round"] == 1


def test_configure_fit_fails_when_selected_clients_cannot_cover_experts() -> None:
    strategy, initial_parameters = _build_strategy(mode="mix", num_clients=1, k=1)
    manager = DummyClientManager([DummyClientProxy(cid="0")])

    with pytest.raises(RuntimeError, match="Not enough dummy clients"):
        strategy.configure_fit(
            server_round=1,
            parameters=initial_parameters,
            client_manager=manager,  # type: ignore[arg-type]
        )


def test_configure_fit_random_policy_assigns_exact_k_across_rounds() -> None:
    strategy, initial_parameters = _build_strategy(
        mode="mix",
        num_clients=2,
        k=1,
        assignment_policy="random",
    )
    manager = DummyClientManager([DummyClientProxy(cid="0"), DummyClientProxy(cid="1")])

    fit_round1 = strategy.configure_fit(
        server_round=1,
        parameters=initial_parameters,
        client_manager=manager,  # type: ignore[arg-type]
    )
    assert len(fit_round1) == 2
    for _client, fit_ins in fit_round1:
        assigned = json.loads(str(fit_ins.config["assigned_experts_json"]))
        assert len(assigned["encoder.0"]) == 1

    fit_round2 = strategy.configure_fit(
        server_round=2,
        parameters=initial_parameters,
        client_manager=manager,  # type: ignore[arg-type]
    )
    assert len(fit_round2) == 2
    for _client, fit_ins in fit_round2:
        assigned = json.loads(str(fit_ins.config["assigned_experts_json"]))
        assert len(assigned["encoder.0"]) == 1


def test_flex_rejects_random_assignment_policy() -> None:
    with pytest.raises(ValueError, match="Flex mode requires assignment_policy='hot'"):
        _build_strategy(mode="flex", num_clients=2, k=1, assignment_policy="random")


def test_aggregate_fit_random_drop_allows_missing_expert_contributors() -> None:
    strategy, initial_parameters = _build_strategy(
        mode="drop",
        num_clients=2,
        k=1,
        assignment_policy="random",
    )
    manager = DummyClientManager([DummyClientProxy(cid="0"), DummyClientProxy(cid="1")])
    strategy.configure_fit(
        server_round=1,
        parameters=initial_parameters,
        client_manager=manager,  # type: ignore[arg-type]
    )

    names = _parameter_names()
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    expert0_name = names[name_to_idx["encoder.block.0.layer.1.mlp.experts.expert_0.wi.weight"]]
    router_w_name = names[name_to_idx["encoder.block.0.layer.1.mlp.router.classifier.weight"]]
    router_b_name = names[name_to_idx["encoder.block.0.layer.1.mlp.router.classifier.bias"]]

    results: List[Tuple[DummyClientProxy, FitRes]] = []
    for client_idx in [0, 1]:
        returned_names = [expert0_name, router_w_name, router_b_name]
        returned_arrays = [
            np.array([5.0 + 2.0 * client_idx], dtype=np.float32),
            np.array([[10.0 + 4.0 * client_idx]], dtype=np.float32),
            np.array([20.0 + 4.0 * client_idx], dtype=np.float32),
        ]
        fit_res = _build_fit_res(
            arrays=returned_arrays,
            num_examples=1,
            metrics={
                "mode": "drop",
                "local_training_time": 0.2,
                "train_loss": 0.1,
                "assigned_experts_json": json.dumps({"encoder.0": [0]}),
                "returned_parameter_names_json": json.dumps(returned_names),
            },
        )
        results.append((DummyClientProxy(cid=str(client_idx)), fit_res))

    aggregated_parameters, _ = strategy.aggregate_fit(
        server_round=1,
        results=results,  # type: ignore[arg-type]
        failures=[],
    )
    assert aggregated_parameters is not None

    aggregated_arrays = parameters_to_ndarrays(aggregated_parameters)
    expert0_idx = name_to_idx["encoder.block.0.layer.1.mlp.experts.expert_0.wi.weight"]
    expert1_idx = name_to_idx["encoder.block.0.layer.1.mlp.experts.expert_1.wi.weight"]
    router_w_idx = name_to_idx["encoder.block.0.layer.1.mlp.router.classifier.weight"]
    router_b_idx = name_to_idx["encoder.block.0.layer.1.mlp.router.classifier.bias"]

    assert aggregated_arrays[expert0_idx][0] == pytest.approx(6.0)
    assert aggregated_arrays[expert1_idx][0] == pytest.approx(0.0)
    assert aggregated_arrays[router_w_idx][0, 0] == pytest.approx(12.0)
    assert aggregated_arrays[router_w_idx][1, 0] == pytest.approx(0.0)
    assert aggregated_arrays[router_b_idx][0] == pytest.approx(22.0)
    assert aggregated_arrays[router_b_idx][1] == pytest.approx(0.0)


@pytest.mark.parametrize("mode", ["mix", "flex"])
def test_aggregate_fit_sparse_experts_use_only_assigned_contributors(mode: str) -> None:
    strategy, initial_parameters = _build_strategy(mode=mode, num_clients=2, k=1)
    manager = DummyClientManager([DummyClientProxy(cid="0"), DummyClientProxy(cid="1")])
    fit_instructions = strategy.configure_fit(
        server_round=1,
        parameters=initial_parameters,
        client_manager=manager,  # type: ignore[arg-type]
    )

    names = _parameter_names()
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    expert0_idx = name_to_idx["encoder.block.0.layer.1.mlp.experts.expert_0.wi.weight"]
    expert1_idx = name_to_idx["encoder.block.0.layer.1.mlp.experts.expert_1.wi.weight"]
    shared_idx = name_to_idx["shared.weight"]
    router_w_idx = name_to_idx["encoder.block.0.layer.1.mlp.router.classifier.weight"]
    router_b_idx = name_to_idx["encoder.block.0.layer.1.mlp.router.classifier.bias"]

    expected_expert_values: Dict[int, float] = {}
    results: List[Tuple[DummyClientProxy, FitRes]] = []
    for client_idx, (client, fit_ins) in enumerate(fit_instructions):
        assigned = json.loads(str(fit_ins.config["assigned_experts_json"]))
        layer_assignment = [int(x) for x in assigned["encoder.0"]]
        assert len(layer_assignment) == 1
        assigned_expert = layer_assignment[0]

        arrays = _initial_arrays()
        arrays[shared_idx] = np.array([2.0 + 2.0 * client_idx], dtype=np.float32)
        arrays[expert0_idx] = np.array([1000.0 + client_idx], dtype=np.float32)
        arrays[expert1_idx] = np.array([2000.0 + client_idx], dtype=np.float32)
        assigned_value = 10.0 + client_idx
        arrays[expert0_idx if assigned_expert == 0 else expert1_idx] = np.array(
            [assigned_value], dtype=np.float32
        )
        expected_expert_values[assigned_expert] = assigned_value
        arrays[router_w_idx] = np.array([[1.0 + client_idx], [2.0 + client_idx]], dtype=np.float32)
        arrays[router_b_idx] = np.array([3.0 + client_idx, 4.0 + client_idx], dtype=np.float32)

        metrics = {
            "mode": mode,
            "local_training_time": float(0.7 + 0.4 * client_idx),
            "train_loss": float(0.1 + 0.1 * client_idx),
            "assigned_experts_json": fit_ins.config["assigned_experts_json"],
            "returned_parameter_names_json": json.dumps(names),
        }
        fit_res = _build_fit_res(arrays=arrays, num_examples=1, metrics=metrics)
        results.append((client, fit_res))

    aggregated_parameters, agg_metrics = strategy.aggregate_fit(
        server_round=1,
        results=results,  # type: ignore[arg-type]
        failures=[],
    )
    assert aggregated_parameters is not None

    aggregated_arrays = parameters_to_ndarrays(aggregated_parameters)
    assert aggregated_arrays[shared_idx][0] == pytest.approx(3.0)
    assert aggregated_arrays[expert0_idx][0] == pytest.approx(expected_expert_values[0])
    assert aggregated_arrays[expert1_idx][0] == pytest.approx(expected_expert_values[1])
    assert float(agg_metrics["round_time"]) == pytest.approx(1.1, abs=1e-8)
    assert float(agg_metrics["cumulative_time"]) == pytest.approx(1.1, abs=1e-8)


def test_aggregate_fit_round_time_uses_max_and_accumulates() -> None:
    strategy, initial_parameters = _build_strategy(mode="mix", num_clients=2, k=1)
    manager = DummyClientManager([DummyClientProxy(cid="0"), DummyClientProxy(cid="1")])
    names = _parameter_names()

    def make_round_results(
        server_round: int,
        parameters: Parameters,
        times: Sequence[float],
    ) -> Tuple[List[Tuple[DummyClientProxy, FitRes]], List[Tuple[DummyClientProxy, object]]]:
        fit_instructions = strategy.configure_fit(
            server_round=server_round,
            parameters=parameters,
            client_manager=manager,  # type: ignore[arg-type]
        )
        results: List[Tuple[DummyClientProxy, FitRes]] = []
        for client_idx, (client, fit_ins) in enumerate(fit_instructions):
            arrays = _initial_arrays()
            metrics = {
                "mode": "mix",
                "local_training_time": float(times[client_idx]),
                "train_loss": float(0.2 + 0.1 * client_idx),
                "assigned_experts_json": fit_ins.config["assigned_experts_json"],
                "returned_parameter_names_json": json.dumps(names),
            }
            fit_res = _build_fit_res(arrays=arrays, num_examples=1, metrics=metrics)
            results.append((client, fit_res))
        return results, fit_instructions

    round1_results, _ = make_round_results(1, initial_parameters, [0.5, 1.25])
    agg_parameters_1, agg_metrics_1 = strategy.aggregate_fit(
        server_round=1,
        results=round1_results,  # type: ignore[arg-type]
        failures=[],
    )
    assert agg_parameters_1 is not None
    assert float(agg_metrics_1["round_time"]) == pytest.approx(1.25, abs=1e-8)
    assert float(agg_metrics_1["cumulative_time"]) == pytest.approx(1.25, abs=1e-8)

    round2_results, _ = make_round_results(2, agg_parameters_1, [0.2, 0.8])
    _agg_parameters_2, agg_metrics_2 = strategy.aggregate_fit(
        server_round=2,
        results=round2_results,  # type: ignore[arg-type]
        failures=[],
    )
    assert float(agg_metrics_2["round_time"]) == pytest.approx(0.8, abs=1e-8)
    assert float(agg_metrics_2["cumulative_time"]) == pytest.approx(2.05, abs=1e-8)


def test_round_checkpoint_can_be_saved(tmp_path) -> None:
    names = _parameter_names()
    cfg = replace(
        _build_config(mode="full"),
        output_dir=str(tmp_path),
        checkpoint_rounds=(2,),
    )
    strategy = MixFedMoEStrategy(
        runtime_config=cfg,
        parameter_names=names,
        initial_parameters=ndarrays_to_parameters(_initial_arrays()),
    )

    strategy._save_checkpoint(server_round=1, arrays=_initial_arrays())
    assert not (tmp_path / "checkpoints" / "round_0001.pt").exists()

    strategy._save_checkpoint(server_round=2, arrays=_initial_arrays())
    checkpoint_path = tmp_path / "checkpoints" / "round_0002.pt"
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["round"] == 2
    assert checkpoint["parameter_names"] == names
    assert set(checkpoint["state_dict"]) == set(names)
