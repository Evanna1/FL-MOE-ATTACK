from __future__ import annotations

from dataclasses import dataclass
import json
from typing import List, Sequence

import numpy as np
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays

from mixfedmoe_fl.config import MixFedMoEConfig, parse_config
from mixfedmoe_fl.strategy import MixFedMoEStrategy


@dataclass(frozen=True)
class _Client:
    cid: str


class _ClientManager:
    def __init__(self, clients: Sequence[_Client]) -> None:
        self.clients = list(clients)

    def num_available(self) -> int:
        return len(self.clients)

    def sample(self, num_clients: int, min_num_clients: int) -> List[_Client]:
        assert len(self.clients) >= min_num_clients
        return self.clients[:num_clients]


def _config() -> MixFedMoEConfig:
    return MixFedMoEConfig(
        mode="full",
        model_name_or_path="unused",
        dataset_name="ag_news",
        num_clients=1,
        num_rounds=3,
        k=1,
        alpha=0.5,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        local_epochs=1.0,
        learning_rate=1e-4,
        weight_decay=0.0,
        train_batch_size=2,
        eval_batch_size=2,
        max_length=16,
        client_eval_ratio=0.2,
        seed=1,
        num_cpus_per_client=1,
        num_gpus_per_client=0.0,
        output_dir="unused",
        test_samples=4,
        attack_enabled=True,
        malicious_clients=(0,),
        inject_target_expert=True,
        inject_benign_train_asr_threshold=0.95,
        lfe_target_layer="encoder.0",
        lfe_target_expert=0,
    )


def test_strategy_latches_benign_style_training_after_asr_reaches_threshold() -> None:
    observed_asr = {1: 0.949, 2: 0.95, 3: 0.10}

    def evaluate_fn(server_round, parameters, config):
        _ = parameters, config
        return 0.0, {"asr": observed_asr[server_round]}

    parameters = ndarrays_to_parameters([np.zeros((1,), dtype=np.float32)])
    strategy = MixFedMoEStrategy(
        runtime_config=_config(),
        parameter_names=["weight"],
        initial_parameters=parameters,
        evaluate_fn=evaluate_fn,
    )
    manager = _ClientManager([_Client("0")])

    before = strategy.configure_fit(1, parameters, manager)[0][1].config
    assert before["inject_benign_training_active"] is False

    strategy.evaluate(1, parameters)
    still_before = strategy.configure_fit(2, parameters, manager)[0][1].config
    assert still_before["inject_benign_training_active"] is False

    _, transition_metrics = strategy.evaluate(2, parameters)  # type: ignore[misc]
    assert transition_metrics["inject_benign_training_active"] is True
    assert transition_metrics["inject_benign_training_trigger_round"] == 2
    after = strategy.configure_fit(3, parameters, manager)[0][1].config
    assert after["inject_benign_training_active"] is True

    # The transition is permanent even if a later ASR falls below the threshold.
    strategy.evaluate(3, parameters)
    assert strategy.configure_fit(3, parameters, manager)[0][1].config[
        "inject_benign_training_active"
    ] is True


def test_benign_training_asr_threshold_defaults_to_point_95_and_keeps_old_alias() -> None:
    default = parse_config(["--mode", "full"])
    custom = parse_config(
        ["--mode", "full", "--inject_benign_train_asr_threshold", "0.9"]
    )
    legacy_alias = parse_config(
        ["--mode", "full", "--inject_full_train_asr_threshold", "0.8"]
    )

    assert default.inject_benign_train_asr_threshold == 0.95
    assert custom.inject_benign_train_asr_threshold == 0.9
    assert legacy_alias.inject_benign_train_asr_threshold == 0.8


def test_adaptive_benign_style_training_keeps_sparse_assignment_boundary() -> None:
    cfg = _config()
    cfg = MixFedMoEConfig(
        **{
            **cfg.to_dict(),
            "mode": "mix",
            "coverage_guarantee": False,
        }
    )
    names = [
        "encoder.block.0.layer.1.mlp.experts.expert_0.wi.weight",
        "encoder.block.0.layer.1.mlp.experts.expert_1.wi.weight",
    ]
    initial = ndarrays_to_parameters(
        [np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32)]
    )
    strategy = MixFedMoEStrategy(cfg, names, initial_parameters=initial)
    strategy._inject_benign_training_active = True
    client = _Client("0")
    manager = _ClientManager([client])
    fit_config = strategy.configure_fit(2, initial, manager)[0][1].config
    assignment = json.loads(str(fit_config["assigned_experts_json"]))
    assigned = assignment["encoder.0"]
    assert len(assigned) == 1
    unassigned = 1 - assigned[0]

    updates = [np.array([10.0], dtype=np.float32), np.array([20.0], dtype=np.float32)]
    fit_res = FitRes(
        status=Status(code=Code.OK, message="ok"),
        parameters=ndarrays_to_parameters(updates),
        num_examples=1,
        metrics={
            "client_id": 0,
            "mode": "mix",
            "is_malicious": True,
            "inject_benign_training_active": True,
            "assigned_experts_json": json.dumps(assignment),
            "returned_parameter_names_json": json.dumps(names),
        },
    )

    aggregated, _ = strategy.aggregate_fit(2, [(client, fit_res)], [])  # type: ignore[arg-type]
    assert aggregated is not None
    arrays = parameters_to_ndarrays(aggregated)
    assert arrays[assigned[0]].item() == updates[assigned[0]].item()
    assert arrays[unassigned].item() == 0.0
