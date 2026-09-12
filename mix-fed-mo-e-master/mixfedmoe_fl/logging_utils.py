from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict

from mixfedmoe_fl.config import MixFedMoEConfig


ROUND_CSV_COLUMNS = [
    "round",
    "eval_loss",
    "eval_accuracy",
    "clean_accuracy",
    "asr",
    "train_loss",
    "round_time",
    "cumulative_time",
    "effective_test_samples",
    "num_clients_sampled",
    "mode",
    "attack_enabled",
]

ATTACK_CSV_COLUMNS = ["round", "clean_accuracy", "asr"]


def build_timestamped_output_dir(base_output_dir: str, timestamp: str | None = None) -> str:
    """Return a unique per-run directory below the configured output directory."""
    if not base_output_dir.strip():
        raise ValueError("base_output_dir must not be empty.")
    run_timestamp = timestamp or datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    return os.path.join(base_output_dir, run_timestamp)


class RoundMetricsLogger:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.output_dir, "round_metrics.jsonl")
        self.csv_path = os.path.join(self.output_dir, "round_metrics.csv")
        self.attack_csv_path = os.path.join(self.output_dir, "attack_results.csv")
        self._ensure_csv_header()

    def _ensure_csv_header(self) -> None:
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=ROUND_CSV_COLUMNS)
                writer.writeheader()
        if not os.path.exists(self.attack_csv_path):
            with open(self.attack_csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=ATTACK_CSV_COLUMNS)
                writer.writeheader()

    def log_round(self, row: Dict[str, Any]) -> None:
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

        csv_row = {key: row.get(key, "") for key in ROUND_CSV_COLUMNS}
        with open(self.csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ROUND_CSV_COLUMNS)
            writer.writerow(csv_row)

        attack_row = {key: row.get(key, "") for key in ATTACK_CSV_COLUMNS}
        with open(self.attack_csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ATTACK_CSV_COLUMNS)
            writer.writerow(attack_row)


def write_run_config(config: MixFedMoEConfig, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "run_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, sort_keys=True)
    return path
