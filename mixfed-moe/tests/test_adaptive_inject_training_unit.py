from __future__ import annotations

from dataclasses import dataclass
import json
from typing import List, Sequence

import numpy as np
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays

from mixfedmoe_fl.config import MixFedMoEConfig
from mixfedmoe_fl.expert_probe import promote_probe_expert_in_activation_profile
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
        lfe_target_layer="encoder.0",
        lfe_target_expert=1,
    )


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
            "assigned_experts_json": json.dumps(assignment),
            "returned_parameter_names_json": json.dumps(names),
        },
    )

    aggregated, _ = strategy.aggregate_fit(2, [(client, fit_res)], [])  # type: ignore[arg-type]
    assert aggregated is not None
    arrays = parameters_to_ndarrays(aggregated)
    assert arrays[assigned[0]].item() == updates[assigned[0]].item()
    assert arrays[unassigned].item() == 0.0


def test_warmup_profile_promotes_target_for_first_formal_training_round() -> None:
    cfg = MixFedMoEConfig(
        **{
            **_config().to_dict(),
            "mode": "mix",
            "num_clients": 2,
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
    clients = [_Client("0"), _Client("1")]
    manager = _ClientManager(clients)

    round1_fits = strategy.configure_fit(1, initial, manager)
    round1_by_cid = {client.cid: fit_ins for client, fit_ins in round1_fits}
    round1_assignment = json.loads(
        str(round1_by_cid["0"].config["assigned_experts_json"])
    )
    assert round1_assignment == {"encoder.0": [0]}

    reported_profile, spoof = promote_probe_expert_in_activation_profile(
        activation_profile={"encoder.0": [20, 1]},
        target_layer="encoder.0",
        target_expert=1,
        k=1,
    )
    assert reported_profile == {"encoder.0": [1, 20]}
    assert spoof["displaced_expert"] == 0

    results = []
    for client in clients:
        fit_ins = round1_by_cid[client.cid]
        profile = reported_profile if client.cid == "0" else {"encoder.0": [1, 20]}
        fit_res = FitRes(
            status=Status(code=Code.OK, message="ok"),
            parameters=initial,
            num_examples=1,
            metrics={
                "client_id": int(client.cid),
                "mode": "mix",
                "is_malicious": client.cid == "0",
                # Each client reports the untouched server-issued assignment.
                "assigned_experts_json": fit_ins.config["assigned_experts_json"],
                "expert_activation_map_json": json.dumps(profile),
                "returned_parameter_names_json": json.dumps(names),
            },
        )
        results.append((client, fit_res))
    aggregated, _ = strategy.aggregate_fit(1, results, [])  # type: ignore[arg-type]
    assert aggregated is not None

    round2_fits = strategy.configure_fit(2, aggregated, manager)
    round2_by_cid = {client.cid: fit_ins for client, fit_ins in round2_fits}
    round2_assignment = json.loads(
        str(round2_by_cid["0"].config["assigned_experts_json"])
    )
    assert round2_assignment == {"encoder.0": [1]}
