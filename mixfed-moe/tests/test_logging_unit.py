from __future__ import annotations

import csv

from mixfedmoe_fl.logging_utils import (
    RoundMetricsLogger,
    append_benign_lfe_activation_rows,
    build_timestamped_output_dir,
)


def test_build_timestamped_output_dir() -> None:
    path = build_timestamped_output_dir("outputs/experiment", timestamp="20260911_143025_123456")
    assert path.endswith("outputs/experiment/20260911_143025_123456") or path.endswith(
        "outputs\\experiment\\20260911_143025_123456"
    )


def test_attack_results_csv_is_created_and_logged(tmp_path) -> None:
    logger = RoundMetricsLogger(str(tmp_path))
    logger.log_round({"round": 1, "clean_accuracy": 0.75, "asr": 0.5})

    with open(logger.attack_csv_path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "round": "1",
            "clean_accuracy": "0.75",
            "asr": "0.5",
            "trigger_target_probability": "",
            "trigger_target_hit_rate": "",
            "trigger_target_execution_rate": "",
            "trigger_routing_tokens": "",
        }
    ]


def test_benign_lfe_activation_csv_contains_clients_and_aggregate(tmp_path) -> None:
    path = append_benign_lfe_activation_rows(
        str(tmp_path),
        [
            {
                "round": 1,
                "client_id": 1,
                "layer_id": "encoder.5",
                "expert_id": 0,
                "activation_count": 2,
                "valid_tokens": 10,
                "activation_rate": 0.2,
            },
            {
                "round": 1,
                "client_id": "ALL_BENIGN",
                "layer_id": "encoder.5",
                "expert_id": 0,
                "activation_count": 2,
                "valid_tokens": 10,
                "activation_rate": 0.2,
            },
        ],
    )

    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["client_id"] for row in rows] == ["1", "ALL_BENIGN"]
    assert rows[-1]["activation_rate"] == "0.2"
