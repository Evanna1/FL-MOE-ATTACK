from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from mixfedmoe_fl.logging_utils import RoundMetricsLogger
from mixfedmoe_fl.strategy import (
    _aggregate_local_asr_reports,
    _aggregate_update_group_diagnostics,
    _selected_update_statistics,
)


def test_aggregate_local_asr_reports_weights_clients_by_eval_examples() -> None:
    result = _aggregate_local_asr_reports(
        [
            (
                "2",
                {
                    "local_asr_pre": 0.1,
                    "local_asr_post": 0.7,
                    "local_asr_gain": 0.6,
                    "local_asr_eval_examples": 100,
                },
            ),
            (
                "5",
                {
                    "local_asr_pre": 0.2,
                    "local_asr_post": 0.4,
                    "local_asr_gain": 0.2,
                    "local_asr_eval_examples": 300,
                },
            ),
        ]
    )

    assert result["local_asr_pre"] == pytest.approx(0.175)
    assert result["local_asr_post"] == pytest.approx(0.475)
    assert result["local_asr_gain"] == pytest.approx(0.3)
    assert result["local_asr_eval_examples"] == 400
    assert result["local_asr_num_clients"] == 2
    reports = json.loads(str(result["local_asr_clients_json"]))
    assert reports[0]["client_id"] == "2"
    assert reports[1]["examples"] == 300


def test_aggregate_local_asr_reports_rejects_inconsistent_gain() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        _aggregate_local_asr_reports(
            [
                (
                    "2",
                    {
                        "local_asr_pre": 0.1,
                        "local_asr_post": 0.7,
                        "local_asr_gain": 0.1,
                        "local_asr_eval_examples": 100,
                    },
                )
            ]
        )


def test_round_logger_persists_local_and_global_asr_diagnostics(tmp_path) -> None:
    logger = RoundMetricsLogger(str(tmp_path))
    logger.log_round(
        {
            "round": 2,
            "asr": 0.5,
            "local_asr_pre": 0.1,
            "local_asr_post": 0.8,
            "local_asr_gain": 0.7,
            "local_global_asr_gap": 0.3,
            "local_asr_aggregation_retention": 4 / 7,
            "local_asr_eval_examples": 100,
            "local_asr_num_clients": 1,
        }
    )

    with open(logger.csv_path, encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["local_asr_pre"] == "0.1"
    assert row["local_asr_post"] == "0.8"
    assert row["local_global_asr_gap"] == "0.3"
    assert row["local_asr_num_clients"] == "1"


def test_selected_update_statistics_uses_only_target_router_row() -> None:
    before = [
        np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    ]
    local = {
        0: np.array([[101.0, 101.0], [3.0, 2.0]], dtype=np.float32)
    }
    aggregated = [
        np.array([[51.0, 51.0], [2.25, 2.0]], dtype=np.float32)
    ]

    stats = _selected_update_statistics(
        before=before,
        local_after=local,
        global_after=aggregated,
        selections={0: 1},
    )

    assert stats["local_update_l2"] == pytest.approx(1.0)
    assert stats["projection_retention"] == pytest.approx(0.25)
    assert stats["update_cosine"] == pytest.approx(1.0)


def test_update_group_diagnostics_reports_l2_retention_and_contributors() -> None:
    before = [np.array([1.0, 2.0], dtype=np.float32)]
    local = {0: np.array([2.0, 2.0], dtype=np.float32)}
    aggregated = [np.array([1.25, 2.0], dtype=np.float32)]

    metrics, reports = _aggregate_update_group_diagnostics(
        prefix="target_expert",
        before=before,
        aggregated=aggregated,
        selections={0: None},
        malicious_updates={"0": (local, 10.0)},
        contributors=1,
    )

    assert metrics["target_expert_local_update_l2"] == pytest.approx(1.0)
    assert metrics["target_expert_global_update_l2"] == pytest.approx(0.25)
    assert metrics["target_expert_update_norm_retention"] == pytest.approx(0.25)
    assert metrics["target_expert_aggregation_retention"] == pytest.approx(0.25)
    assert metrics["target_expert_update_cosine"] == pytest.approx(1.0)
    assert metrics["target_expert_contributors"] == 1
    assert metrics["num_malicious_target_updates"] == 1
    assert reports[0]["client_id"] == "0"
