from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from mixfedmoe_fl.igcg_trigger_optimization import (
    aggregate_robust_metrics,
    compute_routing_objective,
    heldout_selection_key,
    IGCGTriggerOptimizer,
    load_calibration_texts,
    parse_config,
    robust_selection_key,
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


def test_config_accepts_robust_checkpoint_screening() -> None:
    config = parse_config(
        [
            "--checkpoint_path",
            "checkpoints/round01.pt",
            "--robust_checkpoint_paths",
            "checkpoints/round05.pt",
            "checkpoints/round10.pt",
            "--robust_margin_quantile",
            "0.2",
            "--target_layer",
            "encoder.5",
            "--target_expert",
            "2",
        ]
    )

    assert config.robust_checkpoint_paths == (
        "checkpoints/round05.pt",
        "checkpoints/round10.pt",
    )
    assert config.robust_margin_quantile == pytest.approx(0.2)


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


def test_robust_selection_prioritizes_worst_hit_then_margin_quantile() -> None:
    high_average_fragile = aggregate_robust_metrics(
        [
            TriggerMetrics(0.1, 0.99, 0.99, 4.0),
            TriggerMetrics(2.0, 0.40, 0.60, -1.0),
        ],
        margin_quantile=0.1,
    )
    stable = aggregate_robust_metrics(
        [
            TriggerMetrics(0.5, 0.85, 0.90, 1.0),
            TriggerMetrics(0.6, 0.80, 0.88, 0.8),
        ],
        margin_quantile=0.1,
    )
    same_worst_hit_better_tail_margin = aggregate_robust_metrics(
        [
            TriggerMetrics(0.7, 0.75, 0.90, 1.2),
            TriggerMetrics(0.8, 0.70, 0.88, 1.0),
        ],
        margin_quantile=0.1,
    )

    assert robust_selection_key(stable) > robust_selection_key(high_average_fragile)
    assert robust_selection_key(same_worst_hit_better_tail_margin) > robust_selection_key(stable)
    assert stable.low_quantile_margin == pytest.approx(0.82)


def test_robust_evaluation_visits_all_checkpoints_and_restores_primary() -> None:
    optimizer = object.__new__(IGCGTriggerOptimizer)
    optimizer.checkpoint_paths = ("round01.pt", "round05.pt", "round10.pt")
    optimizer.primary_checkpoint_path = "round01.pt"
    optimizer.config = SimpleNamespace(robust_margin_quantile=0.1)
    active = {"path": ""}
    visits = []

    def activate(path: str) -> None:
        active["path"] = path
        visits.append(path)

    checkpoint_values = {
        "round01.pt": (0.95, 1.0),
        "round05.pt": (0.85, 0.4),
        "round10.pt": (0.90, 0.8),
    }

    def evaluate_many(candidates, texts=None):
        hit_rate, margin = checkpoint_values[active["path"]]
        return [TriggerMetrics(0.5, 0.8, hit_rate, margin) for _ in candidates]

    optimizer._activate_checkpoint = activate
    optimizer.evaluate_many = evaluate_many

    result = optimizer.evaluate_many_robust([[1, 2]], texts=["held-out"])[0]

    assert visits == ["round01.pt", "round05.pt", "round10.pt", "round01.pt"]
    assert result.worst_target_hit_rate == pytest.approx(0.85)
    assert result.low_quantile_margin == pytest.approx(0.48)
