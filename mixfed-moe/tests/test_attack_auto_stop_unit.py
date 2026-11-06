from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pytest
import torch
from flwr.common import ndarrays_to_parameters

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
        assert len(self._clients) >= min_num_clients
        return self._clients[:num_clients]


def _attack_config(output_dir: Path) -> MixFedMoEConfig:
    return MixFedMoEConfig(
        mode="full",
        model_name_or_path="unused",
        dataset_name="ag_news",
        num_clients=2,
        num_rounds=4,
        k=1,
        alpha=0.5,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        local_epochs=1.0,
        learning_rate=1e-4,
        weight_decay=0.0,
        train_batch_size=1,
        eval_batch_size=1,
        max_length=8,
        client_eval_ratio=0.0,
        seed=1,
        num_cpus_per_client=1,
        num_gpus_per_client=0.0,
        output_dir=str(output_dir),
        test_samples=1,
        attack_enabled=True,
        attack_start_round=2,
        checkpoint_rounds=(),
    )


def test_perfect_asr_saves_checkpoint_and_switches_to_persistence_monitoring(
    tmp_path: Path,
) -> None:
    names = ["shared.weight"]
    arrays = [np.array([3.0], dtype=np.float32)]
    parameters = ndarrays_to_parameters(arrays)
    eval_configs: List[Dict[str, object]] = []

    def evaluate_fn(server_round, received_arrays, config):
        _ = received_arrays
        eval_configs.append(dict(config))
        return 0.25, {"asr": 1.0, "round": server_round}

    strategy = MixFedMoEStrategy(
        runtime_config=_attack_config(tmp_path),
        parameter_names=names,
        initial_parameters=parameters,
        evaluate_fn=evaluate_fn,
    )
    strategy._last_activation_profiles = {
        "0": {"encoder.0": [1, 99]},
        "1": {"encoder.0": [8, 2]},
    }
    strategy._latest_unspoofed_activation_profiles = {
        "0": {"encoder.0": [10, 1]},
    }

    result = strategy.evaluate(server_round=2, parameters=parameters)
    assert result is not None
    _loss, metrics = result

    checkpoint_path = tmp_path / "checkpoints" / "round_0002.pt"
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["asr"] == pytest.approx(1.0)
    assert checkpoint["attack_stopped_on_asr"] is True
    assert checkpoint["attack_stop_round"] == 2
    assert checkpoint["checkpoint_reason"] == "asr_reached_1"
    assert metrics["attack_stopped_on_asr"] is True
    assert strategy._last_activation_profiles == {
        "0": {"encoder.0": [10, 1]},
        "1": {"encoder.0": [8, 2]},
    }
    assert eval_configs[0]["attack_enabled"] is True
    assert eval_configs[0]["monitor_asr"] is False

    manager = DummyClientManager(
        [DummyClientProxy(cid="0"), DummyClientProxy(cid="1")]
    )
    fit_instructions = strategy.configure_fit(
        server_round=3,
        parameters=parameters,
        client_manager=manager,  # type: ignore[arg-type]
    )
    assert all(ins.config["attack_enabled"] is False for _, ins in fit_instructions)
    assert all(ins.config["monitor_asr"] is True for _, ins in fit_instructions)

    strategy.evaluate(server_round=3, parameters=parameters)
    assert eval_configs[1]["attack_enabled"] is False
    assert eval_configs[1]["monitor_asr"] is True


def test_perfect_asr_drops_malicious_profile_when_genuine_profile_is_missing(
    tmp_path: Path,
) -> None:
    arrays = [np.array([3.0], dtype=np.float32)]
    parameters = ndarrays_to_parameters(arrays)
    strategy = MixFedMoEStrategy(
        runtime_config=_attack_config(tmp_path),
        parameter_names=["shared.weight"],
        initial_parameters=parameters,
        evaluate_fn=lambda server_round, received_arrays, config: (
            0.25,
            {"asr": 1.0, "round": server_round},
        ),
    )
    strategy._last_activation_profiles = {
        "0": {"encoder.0": [1, 99]},
        "1": {"encoder.0": [8, 2]},
    }

    strategy.evaluate(server_round=2, parameters=parameters)

    assert strategy._last_activation_profiles == {
        "1": {"encoder.0": [8, 2]},
    }


def test_periodic_checkpoint_and_perfect_asr_share_one_round_artifact(
    tmp_path: Path,
) -> None:
    names = ["shared.weight"]
    arrays = [np.array([5.0], dtype=np.float32)]
    parameters = ndarrays_to_parameters(arrays)
    strategy = MixFedMoEStrategy(
        runtime_config=_attack_config(tmp_path),
        parameter_names=names,
        initial_parameters=parameters,
        evaluate_fn=lambda server_round, received_arrays, eval_config: (
            0.1,
            {"asr": 1.0, "round": server_round},
        ),
    )

    strategy._save_checkpoint(server_round=5, arrays=arrays)
    strategy.evaluate(server_round=5, parameters=parameters)

    checkpoint_path = tmp_path / "checkpoints" / "round_0005.pt"
    assert checkpoint_path.exists()
    assert list((tmp_path / "checkpoints").glob("round_0005*.pt")) == [
        checkpoint_path
    ]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["checkpoint_reason"] == "asr_reached_1"
    assert checkpoint["checkpoint_reasons"] == [
        "periodic_interval",
        "asr_reached_1",
    ]
    assert checkpoint["attack_stopped_on_asr"] is True


def test_periodic_checkpoint_continues_beyond_legacy_round_twenty(
    tmp_path: Path,
) -> None:
    names = ["shared.weight"]
    arrays = [np.array([7.0], dtype=np.float32)]
    strategy = MixFedMoEStrategy(
        runtime_config=_attack_config(tmp_path),
        parameter_names=names,
        initial_parameters=ndarrays_to_parameters(arrays),
    )

    strategy._save_checkpoint(server_round=24, arrays=arrays)
    strategy._save_checkpoint(server_round=25, arrays=arrays)

    assert not (tmp_path / "checkpoints" / "round_0024.pt").exists()
    checkpoint_path = tmp_path / "checkpoints" / "round_0025.pt"
    assert checkpoint_path.exists()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["checkpoint_reason"] == "periodic_interval"
    assert checkpoint["checkpoint_reasons"] == ["periodic_interval"]
