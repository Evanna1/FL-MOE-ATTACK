from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
from datasets import Dataset


@dataclass(frozen=True)
class SelectedTrigger:
    """Trigger and target metadata produced by trigger optimization."""

    trigger: str
    target_layer: str
    target_expert: int


def load_selected_trigger(path: str) -> SelectedTrigger:
    """Load the best low-frequency-expert trigger from selected_triggers.json."""
    trigger_path = Path(path)
    if not trigger_path.is_file():
        raise FileNotFoundError(f"selected_triggers.json not found: '{path}'.")
    with trigger_path.open("r", encoding="utf-8") as file:
        payload: Any = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("selected_triggers.json must contain a JSON object.")

    best_trigger = payload.get("best_trigger")
    target = payload.get("target")
    if not isinstance(best_trigger, dict) or not isinstance(target, dict):
        raise ValueError(
            "selected_triggers.json must contain object fields 'best_trigger' and 'target'."
        )

    trigger = best_trigger.get("trigger")
    target_layer = target.get("layer_id")
    target_expert = target.get("expert_id")
    if not isinstance(trigger, str) or not trigger.strip():
        raise ValueError("selected_triggers.json field 'best_trigger.trigger' must be a non-empty string.")
    if not isinstance(target_layer, str) or not target_layer.strip():
        raise ValueError("selected_triggers.json field 'target.layer_id' must be a non-empty string.")
    if isinstance(target_expert, bool) or not isinstance(target_expert, int) or target_expert < 0:
        raise ValueError("selected_triggers.json field 'target.expert_id' must be an integer >= 0.")

    return SelectedTrigger(
        trigger=trigger.strip(),
        target_layer=target_layer.strip(),
        target_expert=target_expert,
    )


def add_trigger(text: str, trigger: str) -> str:
    """Append a fixed text trigger while preserving the original text."""
    clean_trigger = trigger.strip()
    if not clean_trigger:
        raise ValueError("BadNet trigger must not be empty.")
    return f"{text.rstrip()} {clean_trigger}"


def select_poison_indices(num_examples: int, poison_rate: float, seed: int) -> List[int]:
    """Select a deterministic subset of local example indices for poisoning."""
    if num_examples < 0:
        raise ValueError("num_examples must be >= 0.")
    if not 0.0 <= poison_rate <= 1.0:
        raise ValueError("poison_rate must be in [0, 1].")

    num_poisoned = int(num_examples * poison_rate)
    if num_poisoned == 0:
        return []
    rng = np.random.default_rng(seed)
    selected = rng.choice(num_examples, size=num_poisoned, replace=False)
    return sorted(int(idx) for idx in selected.tolist())


def poison_text_classification_dataset(
    dataset: Dataset,
    poison_rate: float,
    target_label: int,
    trigger: str,
    seed: int,
) -> Tuple[Dataset, int]:
    """Apply standard BadNet poisoning to a text/label HuggingFace Dataset."""
    required_columns = {"text", "label"}
    missing = sorted(required_columns.difference(dataset.column_names))
    if missing:
        raise ValueError(f"BadNet poisoning requires text/label columns; missing={missing}.")
    if target_label < 0:
        raise ValueError("target_label must be >= 0.")

    poison_indices = select_poison_indices(len(dataset), poison_rate, seed)
    poison_set = set(poison_indices)

    def poison_example(example, index: int):
        if index not in poison_set:
            return example
        poisoned = dict(example)
        poisoned["text"] = add_trigger(str(example["text"]), trigger)
        poisoned["label"] = int(target_label)
        return poisoned

    poisoned_dataset = dataset.map(poison_example, with_indices=True)
    return poisoned_dataset, len(poison_indices)


def build_triggered_test_dataset(dataset: Dataset, trigger: str) -> Dataset:
    """Add the training trigger to every test input without changing labels."""
    if "text" not in dataset.column_names:
        raise ValueError("Triggered evaluation requires a text column.")

    def trigger_example(example):
        triggered = dict(example)
        triggered["text"] = add_trigger(str(example["text"]), trigger)
        return triggered

    return dataset.map(trigger_example)
