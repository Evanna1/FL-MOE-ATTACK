from __future__ import annotations

from datasets import Dataset

from mixfedmoe_fl.attack import (
    SelectedTrigger,
    add_trigger,
    build_triggered_test_dataset,
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
    selected = set(select_poison_indices(10, poison_rate=0.2, seed=7))
    poisoned, count = poison_text_classification_dataset(
        clean,
        poison_rate=0.2,
        target_label=0,
        trigger="cf",
        seed=7,
    )

    assert count == 2
    for idx in range(len(clean)):
        if idx in selected:
            assert poisoned[idx]["text"] == f"sample-{idx} cf"
            assert poisoned[idx]["label"] == 0
        else:
            assert poisoned[idx] == clean[idx]


def test_triggered_test_dataset_preserves_labels() -> None:
    clean = _dataset(size=4)
    triggered = build_triggered_test_dataset(clean, trigger="cf")
    assert triggered["label"] == clean["label"]
    assert triggered["text"] == [f"sample-{idx} cf" for idx in range(4)]


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
