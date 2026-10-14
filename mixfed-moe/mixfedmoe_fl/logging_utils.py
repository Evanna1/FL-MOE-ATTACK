from __future__ import annotations

import csv
import json
import logging
import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Iterator, TextIO

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


class _SharedLogSink:
    """Thread-safe sink shared by the stdout and stderr tee streams."""

    def __init__(self, path: str) -> None:
        self.handle = open(path, "a", encoding="utf-8", buffering=1)
        self.lock = threading.Lock()

    def write(self, text: str) -> None:
        with self.lock:
            self.handle.write(text)
            self.handle.flush()

    def flush(self) -> None:
        with self.lock:
            self.handle.flush()

    def close(self) -> None:
        with self.lock:
            self.handle.close()


class _TeeStream:
    """Mirror a text stream to a shared run log without hiding console output."""

    def __init__(self, console: TextIO, sink: _SharedLogSink) -> None:
        self.console = console
        self.sink = sink

    def write(self, text: str) -> int:
        written = self.console.write(text)
        self.sink.write(text)
        return len(text) if written is None else written

    def flush(self) -> None:
        self.console.flush()
        self.sink.flush()

    def isatty(self) -> bool:
        return self.console.isatty()

    def fileno(self) -> int:
        return self.console.fileno()

    @property
    def encoding(self) -> str | None:
        return self.console.encoding

    def __getattr__(self, name: str) -> Any:
        return getattr(self.console, name)


def _redirect_existing_console_handlers(
    stdout: TextIO,
    stderr: TextIO,
    tee_stdout: _TeeStream,
    tee_stderr: _TeeStream,
) -> list[tuple[logging.StreamHandler[Any], TextIO]]:
    """Point already-created console handlers at the corresponding tee."""
    loggers = [logging.getLogger()]
    loggers.extend(
        logger
        for logger in logging.root.manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
    )
    redirected: list[tuple[logging.StreamHandler[Any], TextIO]] = []
    seen: set[int] = set()
    for logger in loggers:
        for handler in logger.handlers:
            if id(handler) in seen or not isinstance(handler, logging.StreamHandler):
                continue
            seen.add(id(handler))
            if handler.stream is stdout:
                redirected.append((handler, stdout))
                handler.setStream(tee_stdout)
            elif handler.stream is stderr:
                redirected.append((handler, stderr))
                handler.setStream(tee_stderr)
    return redirected


@contextmanager
def tee_output_to_log(
    output_dir: str,
    filename: str = "run.log",
) -> Iterator[str]:
    """Mirror stdout and stderr to a live log file under ``output_dir``.

    Keeping stdout/stderr visible is useful for interactive runs, while line
    buffering and an explicit flush make the same output available in the log
    file while a long-running simulation is still in progress.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, filename)
    sink = _SharedLogSink(log_path)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    tee_stdout = _TeeStream(original_stdout, sink)
    tee_stderr = _TeeStream(original_stderr, sink)
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr
    redirected_handlers = _redirect_existing_console_handlers(
        original_stdout,
        original_stderr,
        tee_stdout,
        tee_stderr,
    )
    try:
        yield log_path
    finally:
        for handler, original_stream in redirected_handlers:
            handler.setStream(original_stream)
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        sink.close()


def build_timestamped_output_dir(base_output_dir: str, timestamp: str | None = None) -> str:
    """Return a unique per-run directory below the configured output directory."""
    if not base_output_dir.strip():
        raise ValueError("base_output_dir must not be empty.")
    run_timestamp = timestamp or datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    return os.path.join(os.path.normpath(base_output_dir), run_timestamp)


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


def append_expert_assignment_log(
    output_dir: str,
    server_round: int,
    round_assignments: Dict[str, Dict[str, list[int]]],
    layer_num_experts: Dict[str, int],
) -> str:
    """Append one round of per-layer expert assignments to expert_assignments.jsonl.

    Per layer: `per_client` maps client id -> assigned experts, and `coverage` is a
    list indexed by expert id giving how many sampled clients were assigned it
    (e.g. encoder.11: [4, 4, 4, 4, 4, 4, 4, 4]).
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "expert_assignments.jsonl")
    layers: Dict[str, Any] = {}
    for layer_id, num_experts in layer_num_experts.items():
        coverage = [0] * num_experts
        per_client: Dict[str, list[int]] = {}
        for cid, client_assignment in round_assignments.items():
            experts = [int(x) for x in client_assignment.get(layer_id, [])]
            per_client[cid] = experts
            for expert_idx in experts:
                coverage[expert_idx] += 1
        layers[layer_id] = {"coverage": coverage, "per_client": per_client}
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"round": int(server_round), "layers": layers}, sort_keys=True) + "\n")
    return path


def write_run_config(config: MixFedMoEConfig, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "run_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, sort_keys=True)
    return path
