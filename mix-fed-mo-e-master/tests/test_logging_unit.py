from __future__ import annotations

import csv

from mixfedmoe_fl.logging_utils import RoundMetricsLogger, build_timestamped_output_dir


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
    assert rows == [{"round": "1", "clean_accuracy": "0.75", "asr": "0.5"}]
