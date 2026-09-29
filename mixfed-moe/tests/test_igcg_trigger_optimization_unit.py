from __future__ import annotations

import json

import pytest
import torch

from mixfedmoe_fl.igcg_trigger_optimization import (
    compute_routing_objective,
    heldout_selection_key,
    load_calibration_texts,
    parse_config,
    split_calibration_texts,
    TriggerMetrics,
)


def test_calibration_split_is_deterministic_and_disjoint() -> None:
    texts = [f"sample-{index}" for index in range(10)]
    first_train, first_heldout = split_calibration_texts(texts, heldout_samples=3, seed=17)
    second_train, second_heldout = split_calibration_texts(texts, heldout_samples=3, seed=17)

    assert first_train == second_train
    assert first_heldout == second_heldout
    assert len(first_train) == 7
    assert len(first_heldout) == 3
    assert set(first_train).isdisjoint(first_heldout)


def test_default_config_always_uses_calibration_file() -> None:
    config = parse_config(
        ["--checkpoint_path", "model.pt", "--target_layer", "encoder.5", "--target_expert", "2"]
    )

    assert config.calibration_file == "data/calibration.json"
    assert config.target_hit_rate_threshold == 0.95
    assert not hasattr(config, "optimization_context")


def test_hit_rate_stopping_threshold_can_be_overridden() -> None:
    config = parse_config(
        [
            "--checkpoint_path",
            "model.pt",
            "--target_layer",
            "encoder.5",
            "--target_expert",
            "2",
            "--target_hit_rate_threshold",
            "0.97",
        ]
    )

    assert config.target_hit_rate_threshold == 0.97


def test_load_calibration_texts_reads_ag_news_rows(tmp_path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps([{"text": "news one"}, {"text": "news two"}]), encoding="utf-8")

    assert load_calibration_texts(str(path), 1) == ["news one"]


def test_routing_objective_reports_target_logit_margin() -> None:
    logits = torch.tensor([[[1.0, 3.0, 2.0], [4.0, 0.0, 1.0]]])
    mask = torch.tensor([[True, False]])

    _loss, _probability, hit_rate, margin = compute_routing_objective(
        logits, mask, target_expert=1
    )

    assert hit_rate.item() == 1.0
    assert margin.item() == pytest.approx(1.0)


def test_heldout_selection_prioritizes_hit_rate_then_margin() -> None:
    lower_hit = TriggerMetrics(0.01, 0.99, 0.8, 10.0)
    higher_hit = TriggerMetrics(2.0, 0.20, 0.9, -0.5)
    same_hit_larger_margin = TriggerMetrics(3.0, 0.10, 0.9, 0.7)

    assert heldout_selection_key(higher_hit) > heldout_selection_key(lower_hit)
    assert heldout_selection_key(same_hit_larger_margin) > heldout_selection_key(higher_hit)
