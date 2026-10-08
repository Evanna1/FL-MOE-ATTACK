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
    "trigger_target_probability",
    "trigger_target_hit_rate",
    "trigger_target_execution_rate",
    "trigger_routing_tokens",
    "train_loss",
    "round_time",
    "cumulative_time",
    "effective_test_samples",
    "num_clients_sampled",
    "benign_lfe_activation_rate",
    "benign_lfe_activation_count",
    "benign_lfe_valid_tokens",
    "num_benign_clients_profiled",
    "malicious_router_trigger_tokens",
    "malicious_router_margin",
    "malicious_router_target_probability",
    "malicious_router_target_hit_rate",
    "malicious_router_margin_loss",
    "malicious_router_clean_kl",
    "malicious_router_update_norm",
    "mode",
    "attack_enabled",
    "monitor_asr",
]

ATTACK_CSV_COLUMNS = [
    "round",
    "clean_accuracy",
    "asr",
    "trigger_target_probability",
    "trigger_target_hit_rate",
    "trigger_target_execution_rate",
    "trigger_routing_tokens",
]

BENIGN_LFE_ACTIVATION_COLUMNS = [
    "round",
    "client_id",
    "layer_id",
    "expert_id",
    "activation_count",
    "valid_tokens",
    "activation_rate",
]


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


def append_benign_lfe_activation_rows(output_dir: str, rows: list[Dict[str, Any]]) -> str:
    """Append one round of per-client and aggregate benign target-expert usage."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "benign_lfe_activation.csv")
    needs_header = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BENIGN_LFE_ACTIVATION_COLUMNS)
        if needs_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in BENIGN_LFE_ACTIVATION_COLUMNS})
    return path


def write_run_config(config: MixFedMoEConfig, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "run_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, sort_keys=True)
    return path
