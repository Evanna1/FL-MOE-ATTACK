from __future__ import annotations

import csv
import logging
import sys

from mixfedmoe_fl.logging_utils import (
    RoundMetricsLogger,
    append_benign_lfe_activation_rows,
    build_timestamped_output_dir,
    tee_output_to_log,
)


def test_build_timestamped_output_dir() -> None:
    path = build_timestamped_output_dir("outputs/experiment", timestamp="20260911_143025_123456")
    assert path.endswith("outputs/experiment/20260911_143025_123456") or path.endswith(
        "outputs\\experiment\\20260911_143025_123456"
    )


def test_tee_output_to_log_mirrors_stdout_stderr_and_logging(tmp_path) -> None:
    handler = logging.StreamHandler()
    logger = logging.getLogger("mixfedmoe.test.run_log")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        with tee_output_to_log(str(tmp_path)) as log_path:
            print("stdout message", flush=True)
            with open(log_path, encoding="utf-8") as handle:
                assert "stdout message" in handle.read()
            print("stderr message", file=sys.stderr, flush=True)
            logger.info("logging message")
    finally:
        logger.removeHandler(handler)
        handler.close()

    with open(log_path, encoding="utf-8") as handle:
        contents = handle.read()
    assert "stdout message" in contents
    assert "stderr message" in contents
    assert "logging message" in contents


def test_attack_results_csv_is_created_and_logged(tmp_path) -> None:
    logger = RoundMetricsLogger(str(tmp_path))
    logger.log_round(
        {
            "round": 1,
            "clean_accuracy": 0.75,
            "asr": 0.5,
            "target_expert_local_update_l2": 0.4,
            "target_expert_global_update_l2": 0.1,
            "target_expert_contributors": 5,
        }
    )

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
            "target_expert_local_update_l2": "0.4",
            "target_expert_local_relative_update_l2": "",
            "target_expert_global_update_l2": "0.1",
            "target_expert_global_relative_update_l2": "",
            "target_expert_contributors": "5",
            "num_malicious_target_updates": "",
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
