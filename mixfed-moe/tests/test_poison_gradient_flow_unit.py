from __future__ import annotations

import csv

import pytest
import torch
from datasets import Dataset

from mixfedmoe_fl.client import (
    _select_poison_gradient_probe_datasets,
    _summarize_clean_poison_gradient_flow,
)
from mixfedmoe_fl.logging_utils import append_poison_gradient_flow_reports


def test_select_poison_gradient_probe_datasets_balances_groups() -> None:
    dataset = Dataset.from_dict(
        {
            "value": list(range(10)),
            "is_poisoned": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        }
    )

    clean, poison = _select_poison_gradient_probe_datasets(
        dataset=dataset,
        max_samples=3,
        seed=7,
    )

    assert len(clean) == len(poison) == 3
    assert set(clean["is_poisoned"]) == {0}
    assert set(poison["is_poisoned"]) == {1}


def test_gradient_flow_summary_partitions_target_router_row() -> None:
    poison = {
        "encoder.block.0.layer.1.mlp.experts.expert_1.wi.weight": torch.tensor(
            [3.0, 4.0]
        ),
        "encoder.block.0.layer.1.mlp.experts.expert_0.wi.weight": torch.tensor(
            [1.0]
        ),
        "encoder.block.0.layer.1.mlp.router.classifier.weight": torch.tensor(
            [[1.0, 0.0], [0.0, 2.0]]
        ),
        "classification_head.weight": torch.tensor([2.0]),
    }
    clean = {name: value.clone() for name, value in poison.items()}

    summary = _summarize_clean_poison_gradient_flow(
        clean_gradients=clean,
        poison_gradients=poison,
        target_layer="encoder.0",
        target_expert=1,
    )

    # Total poison energy: 25 target expert + 1 other expert +
    # 4 target-router row + 1 other-router row + 4 shared = 35.
    assert summary["target_expert"]["poison_grad_l2"] == pytest.approx(5.0)
    assert summary["target_expert"]["poison_energy_ratio"] == pytest.approx(25 / 35)
    assert summary["target_router"]["poison_grad_l2"] == pytest.approx(2.0)
    assert summary["other_routers"]["poison_grad_l2"] == pytest.approx(1.0)
    assert summary["shared"]["poison_grad_l2"] == pytest.approx(2.0)
    assert summary["shared"]["clean_poison_cosine"] == pytest.approx(1.0)


def test_poison_gradient_flow_reports_write_long_form_csv(tmp_path) -> None:
    report = {
        "round": 2,
        "client_id": 0,
        "target_layer": "encoder.0",
        "target_expert": 1,
        "clean_examples": 8,
        "poison_examples": 8,
        "clean_ce_loss": 0.4,
        "poison_ce_loss": 1.2,
        "groups": {
            "target_expert": {
                "poison_grad_l2": 3.0,
                "clean_grad_l2": 1.0,
                "poison_energy_ratio": 0.75,
                "clean_energy_ratio": 0.25,
                "clean_poison_cosine": -0.2,
                "poison_to_clean_norm_ratio": 3.0,
                "coordinates": 10,
            }
        },
    }

    jsonl_path, csv_path = append_poison_gradient_flow_reports(
        str(tmp_path), [report]
    )

    assert jsonl_path.endswith("poison_gradient_flow.jsonl")
    with open(csv_path, encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["round"] == "2"
    assert row["group"] == "target_expert"
    assert row["poison_energy_ratio"] == "0.75"
