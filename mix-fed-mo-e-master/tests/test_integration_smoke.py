from __future__ import annotations

import json
import os
from typing import Dict, List

import pytest

from mixfedmoe_fl.client import MixFedMoEClient
from mixfedmoe_fl.config import MixFedMoEConfig
from mixfedmoe_fl.data import ClientDatasetBundle, MixFedMoEDataManager


pytestmark = pytest.mark.skipif(
    os.environ.get("MIXFEDMOE_RUN_INTEGRATION") != "1",
    reason="Set MIXFEDMOE_RUN_INTEGRATION=1 to run model/dataset integration smoke tests.",
)


def _build_cfg(mode: str) -> MixFedMoEConfig:
    k = 1 if mode == "flex" else 2
    return MixFedMoEConfig(
        mode=mode,  # type: ignore[arg-type]
        model_name_or_path="model_ckpt/switch-base-8",
        dataset_name="ag_news",
        num_clients=2,
        num_rounds=1,
        k=k,
        alpha=0.5,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        local_epochs=1,
        learning_rate=2e-5,
        weight_decay=0.0,
        train_batch_size=2,
        eval_batch_size=2,
        max_length=64,
        client_eval_ratio=0.2,
        seed=42,
        num_cpus_per_client=1,
        num_gpus_per_client=0.0,
        output_dir="outputs/mixfedmoe",
        test_samples=16,
        assignment_policy="hot",
    )


def _tiny_bundle(
    dm: MixFedMoEDataManager,
    client_id: int,
    train_n: int = 4,
    eval_n: int = 4,
) -> ClientDatasetBundle:
    full = dm.load_client_dataset(partition_id=client_id)
    train_count = min(train_n, len(full.train_dataset))
    eval_count = min(eval_n, len(full.eval_dataset))
    return ClientDatasetBundle(
        train_dataset=full.train_dataset.select(range(train_count)),
        eval_dataset=full.eval_dataset.select(range(eval_count)),
        num_train_examples=train_count,
        num_eval_examples=eval_count,
        label_info=full.label_info,
    )


@pytest.mark.integration
def test_client_fit_smoke_across_modes() -> None:
    required_metrics: List[str] = [
        "train_loss",
        "local_training_time",
        "assigned_experts_json",
        "expert_activation_map_json",
        "returned_parameter_names_json",
    ]

    for mode in ["full", "mix", "drop", "flex"]:
        cfg = _build_cfg(mode)
        dm = MixFedMoEDataManager.from_config(cfg)
        client = MixFedMoEClient(client_id=0, runtime_config=cfg, data_manager=dm)
        client._dataset_bundle = _tiny_bundle(dm=dm, client_id=0)

        inbound = client.get_parameters({})
        fit_cfg: Dict[str, object] = {
            "mode": mode,
            "local_epochs": 1,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "train_batch_size": cfg.train_batch_size,
            "eval_batch_size": cfg.eval_batch_size,
            "k": cfg.k,
            "calib_samples": 4,
        }
        outbound, num_examples, metrics = client.fit(inbound, fit_cfg)

        assert len(outbound) > 0
        assert num_examples > 0
        for key in required_metrics:
            assert key in metrics
        json.loads(str(metrics["assigned_experts_json"]))
        json.loads(str(metrics["expert_activation_map_json"]))
        returned_names = json.loads(str(metrics["returned_parameter_names_json"]))
        assert isinstance(returned_names, list)
        assert len(returned_names) == len(outbound)
