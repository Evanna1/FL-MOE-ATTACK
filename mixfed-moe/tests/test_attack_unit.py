from __future__ import annotations

import pytest
import torch
from datasets import Dataset

from mixfedmoe_fl.attack import (
    SelectedTrigger,
    add_trigger,
    build_triggered_test_dataset,
    load_attack_checkpoint,
    load_selected_trigger,
    poison_text_classification_dataset,
    select_poison_indices,
)


def _dataset(size: int = 10) -> Dataset:
    return Dataset.from_dict(
        {
            "text": [f"sample-{idx}" for idx in range(size)],
            "label": [idx % 3 for idx in range(size)],
        }
    )


def test_add_trigger_appends_fixed_trigger() -> None:
    assert add_trigger("hello world.", "cf") == "hello world. cf"


def test_poison_selection_is_exact_and_deterministic() -> None:
    first = select_poison_indices(num_examples=10, poison_rate=0.2, seed=42)
    second = select_poison_indices(num_examples=10, poison_rate=0.2, seed=42)
    assert first == second
    assert len(first) == 2


def test_poison_dataset_changes_only_selected_examples() -> None:
    clean = _dataset()
    poisoned, count = poison_text_classification_dataset(
        clean,
        poison_rate=0.2,
        target_label=0,
        trigger="cf",
        seed=7,
    )

    assert count == 2
    selected = {
        idx
        for idx in range(len(clean))
        if poisoned[idx]["text"] != clean[idx]["text"]
    }
    assert len(selected) == count
    for idx in range(len(clean)):
        if idx in selected:
            assert clean[idx]["label"] != 0
            assert poisoned[idx]["text"] == f"sample-{idx} cf"
            assert poisoned[idx]["label"] == 0
            assert poisoned[idx]["_mixfedmoe_is_poisoned"] is True
        else:
            assert poisoned[idx]["text"] == clean[idx]["text"]
            assert poisoned[idx]["label"] == clean[idx]["label"]
            assert poisoned[idx]["_mixfedmoe_is_poisoned"] is False


def test_poison_dataset_never_selects_examples_already_in_target_class() -> None:
    clean = _dataset(size=6)
    poisoned, count = poison_text_classification_dataset(
        clean,
        poison_rate=0.5,
        target_label=0,
        trigger="cf",
        seed=3,
    )

    assert count == 3
    for idx in range(len(clean)):
        if clean[idx]["label"] == 0:
            assert poisoned[idx]["text"] == clean[idx]["text"]
            assert poisoned[idx]["label"] == clean[idx]["label"]
            assert poisoned[idx]["_mixfedmoe_is_poisoned"] is False
        elif poisoned[idx]["text"] != clean[idx]["text"]:
            assert poisoned[idx]["text"] == f"sample-{idx} cf"
            assert poisoned[idx]["label"] == 0


def test_poison_dataset_raises_when_eligible_examples_are_insufficient() -> None:
    clean = _dataset(size=6)

    with pytest.raises(
        ValueError,
        match=r"required=6, available=4",
    ):
        poison_text_classification_dataset(
            clean,
            poison_rate=1.0,
            target_label=0,
            trigger="cf",
            seed=3,
        )


def test_triggered_test_dataset_excludes_target_class_and_preserves_other_labels() -> None:
    clean = _dataset(size=6)
    triggered = build_triggered_test_dataset(clean, trigger="cf", target_label=0)

    assert triggered["label"] == [1, 2, 1, 2]
    assert triggered["text"] == ["sample-1 cf", "sample-2 cf", "sample-4 cf", "sample-5 cf"]
    assert all(label != 0 for label in triggered["label"])


def test_triggered_test_dataset_rejects_test_set_with_only_target_class() -> None:
    clean = Dataset.from_dict({"text": ["a", "b"], "label": [0, 0]})

    with pytest.raises(ValueError, match="no samples"):
        build_triggered_test_dataset(clean, trigger="cf", target_label=0)


def test_load_selected_trigger_reads_optimizer_output(tmp_path) -> None:
    path = tmp_path / "selected_triggers.json"
    path.write_text(
        '{"target":{"layer_id":"encoder.1","expert_id":7},'
        '"best_trigger":{"trigger":" rare phrase ","combined_score":0.8}}',
        encoding="utf-8",
    )

    assert load_selected_trigger(str(path)) == SelectedTrigger(
        trigger="rare phrase",
        target_layer="encoder.1",
        target_expert=7,
    )


def test_load_selected_trigger_preserves_exact_token_ids(tmp_path) -> None:
    path = tmp_path / "selected_triggers.json"
    path.write_text(
        '{"target":{"layer_id":"encoder.5","expert_id":3},'
        '"best_trigger":{"trigger":"stable trigger","trigger_token_ids":[11,22,33]}}',
        encoding="utf-8",
    )

    selected = load_selected_trigger(str(path))

    assert selected.trigger_token_ids == (11, 22, 33)


def test_token_id_poisoning_marks_only_selected_examples_without_editing_text() -> None:
    clean = _dataset()
    poisoned, count = poison_text_classification_dataset(
        clean,
        poison_rate=0.2,
        target_label=0,
        trigger="display text",
        seed=7,
        trigger_token_ids=[91, 92],
    )

    assert count == 2
    marked = [index for index, ids in enumerate(poisoned["_mixfedmoe_trigger_ids"]) if ids]
    assert len(marked) == count
    for index in range(len(clean)):
        assert poisoned[index]["text"] == clean[index]["text"]
        if index in marked:
            assert poisoned[index]["_mixfedmoe_trigger_ids"] == [91, 92]
            assert poisoned[index]["label"] == 0
            assert poisoned[index]["_mixfedmoe_is_poisoned"] is True
        else:
            assert poisoned[index]["_mixfedmoe_trigger_ids"] == []
            assert poisoned[index]["_mixfedmoe_is_poisoned"] is False


def test_load_attack_checkpoint_reads_flower_round_checkpoint(tmp_path) -> None:
    path = tmp_path / "round_0004.pt"
    torch.save(
        {"round": 4, "state_dict": {"weight": torch.tensor([1.0, 2.0])}},
        path,
    )

    state_dict, metadata = load_attack_checkpoint(str(path))

    assert torch.equal(state_dict["weight"], torch.tensor([1.0, 2.0]))
    assert metadata["round"] == 4


def test_load_attack_checkpoint_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Checkpoint does not exist"):
        load_attack_checkpoint(str(tmp_path / "missing.pt"))
