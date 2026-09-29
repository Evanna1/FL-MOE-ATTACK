from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from datasets import Dataset


@dataclass(frozen=True)
class SelectedTrigger:
    """Trigger and target metadata produced by trigger optimization."""

    trigger: str
    target_layer: str
    target_expert: int
    trigger_token_ids: Tuple[int, ...] = ()


def load_attack_checkpoint(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Load model weights used as the starting point for poisoning training.

    Flower round checkpoints produced by this project wrap the weights in a
    ``state_dict`` field.  ``model_state_dict`` and an unwrapped PyTorch state
    dict are accepted as well so centrally trained checkpoints can be reused.
    Only tensor-like state-dict values are accepted.
    """
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: '{checkpoint_path}'.")

    try:
        payload: Any = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise ValueError(f"Unable to load checkpoint '{checkpoint_path}': {exc}") from exc

    metadata: Dict[str, Any] = {}
    raw_state: Any = payload
    if isinstance(payload, Mapping) and "state_dict" in payload:
        raw_state = payload["state_dict"]
        metadata = {str(key): value for key, value in payload.items() if key != "state_dict"}
    elif isinstance(payload, Mapping) and "model_state_dict" in payload:
        raw_state = payload["model_state_dict"]
        metadata = {
            str(key): value for key, value in payload.items() if key != "model_state_dict"
        }

    if not isinstance(raw_state, Mapping) or not raw_state:
        raise ValueError(
            "Unsupported checkpoint format. Expected a non-empty state_dict, "
            "{'state_dict': ...}, or {'model_state_dict': ...}."
        )

    state_dict: Dict[str, torch.Tensor] = {}
    for name, value in raw_state.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Checkpoint state_dict keys must be non-empty strings.")
        if not isinstance(value, (torch.Tensor, np.ndarray)):
            raise ValueError(
                f"Checkpoint parameter '{name}' must be a tensor or NumPy array, "
                f"got {type(value).__name__}."
            )
        state_dict[name] = torch.as_tensor(value).detach().cpu()

    return state_dict, metadata


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
    raw_trigger_ids = best_trigger.get("trigger_token_ids", [])
    if not isinstance(trigger, str) or not trigger.strip():
        raise ValueError("selected_triggers.json field 'best_trigger.trigger' must be a non-empty string.")
    if not isinstance(target_layer, str) or not target_layer.strip():
        raise ValueError("selected_triggers.json field 'target.layer_id' must be a non-empty string.")
    if isinstance(target_expert, bool) or not isinstance(target_expert, int) or target_expert < 0:
        raise ValueError("selected_triggers.json field 'target.expert_id' must be an integer >= 0.")
    if not isinstance(raw_trigger_ids, list) or any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in raw_trigger_ids
    ):
        raise ValueError(
            "selected_triggers.json field 'best_trigger.trigger_token_ids' must be a list of integers >= 0."
        )

    return SelectedTrigger(
        trigger=trigger.strip(),
        target_layer=target_layer.strip(),
        target_expert=target_expert,
        trigger_token_ids=tuple(raw_trigger_ids),
    )

# Append a fixed text trigger while preserving the original text.
def add_trigger(text: str, trigger: str) -> str:
    clean_trigger = trigger.strip()
    if not clean_trigger:
        raise ValueError("BadNet trigger must not be empty.")
    return f"{text.rstrip()} {clean_trigger}"

# Select a deterministic subset of local example indices for poisoning.
def select_poison_indices(
    num_examples: int,
    poison_rate: float,
    seed: int,
    candidate_indices: List[int] | None = None,
) -> List[int]:
    if num_examples < 0:
        raise ValueError("num_examples must be >= 0.")
    if not 0.0 <= poison_rate <= 1.0:
        raise ValueError("poison_rate must be in [0, 1].")

    candidates = list(range(num_examples)) if candidate_indices is None else candidate_indices
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate_indices must not contain duplicates.")
    if any(isinstance(idx, bool) or not isinstance(idx, int) for idx in candidates):
        raise ValueError("candidate_indices must contain only integers.")
    if any(idx < 0 or idx >= num_examples for idx in candidates):
        raise ValueError("candidate_indices must be within the dataset bounds.")

    num_poisoned = int(num_examples * poison_rate)
    if len(candidates) < num_poisoned:
        raise ValueError(
            "Not enough candidate_indices for poisoning: "
            f"required={num_poisoned}, available={len(candidates)}."
        )
    if num_poisoned == 0:
        return []
    rng = np.random.default_rng(seed)
    selected = rng.choice(candidates, size=num_poisoned, replace=False)
    return sorted(int(idx) for idx in selected.tolist())

# Apply standard BadNet poisoning to a text/label HuggingFace Dataset.
def poison_text_classification_dataset(
    dataset: Dataset,
    poison_rate: float,
    target_label: int,
    trigger: str,
    seed: int,
    trigger_token_ids: Sequence[int] = (),
) -> Tuple[Dataset, int]:
    required_columns = {"text", "label"}
    missing = sorted(required_columns.difference(dataset.column_names))
    if missing:
        raise ValueError(f"BadNet poisoning requires text/label columns; missing={missing}.")
    if target_label < 0:
        raise ValueError("target_label must be >= 0.")

    eligible_indices = [
        index
        for index, label in enumerate(dataset["label"])
        if int(label) != target_label
    ]
    poison_indices = select_poison_indices(
        len(dataset),
        poison_rate,
        seed,
        candidate_indices=eligible_indices,
    )
    poison_set = set(poison_indices)
    exact_trigger_ids = [int(token_id) for token_id in trigger_token_ids]

    def poison_example(example, index: int):
        if index not in poison_set:
            if exact_trigger_ids:
                clean = dict(example)
                clean["_mixfedmoe_trigger_ids"] = []
                return clean
            return example
        poisoned = dict(example)
        if exact_trigger_ids:
            poisoned["_mixfedmoe_trigger_ids"] = exact_trigger_ids
        else:
            poisoned["text"] = add_trigger(str(example["text"]), trigger)
        poisoned["label"] = int(target_label)
        return poisoned

    poisoned_dataset = dataset.map(poison_example, with_indices=True)
    return poisoned_dataset, len(poison_indices)

# Add the training trigger to non-target-class test inputs without changing labels.
def build_triggered_test_dataset(
    dataset: Dataset,
    trigger: str,
    target_label: int,
    trigger_token_ids: Sequence[int] = (),
) -> Dataset:
    required_columns = {"text", "label"}
    missing = sorted(required_columns.difference(dataset.column_names))
    if missing:
        raise ValueError(f"Triggered evaluation requires text/label columns; missing={missing}.")
    if target_label < 0:
        raise ValueError("target_label must be >= 0.")

    # Standard targeted-attack ASR excludes examples whose ground-truth label is
    # already the attack target; otherwise ordinary correct predictions inflate ASR.
    eligible_dataset = dataset.filter(
        lambda example: int(example["label"]) != target_label
    )
    if len(eligible_dataset) == 0:
        raise ValueError(
            "Triggered evaluation has no samples whose label differs from target_label."
        )

    exact_trigger_ids = [int(token_id) for token_id in trigger_token_ids]

    def trigger_example(example):
        triggered = dict(example)
        if exact_trigger_ids:
            triggered["_mixfedmoe_trigger_ids"] = exact_trigger_ids
        else:
            triggered["text"] = add_trigger(str(example["text"]), trigger)
        return triggered

    return eligible_dataset.map(trigger_example)
