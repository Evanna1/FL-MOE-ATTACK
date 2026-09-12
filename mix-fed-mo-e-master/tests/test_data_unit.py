from __future__ import annotations

from typing import List

from datasets import Dataset

import mixfedmoe_fl.data as data_module
from mixfedmoe_fl.config import MixFedMoEConfig
from mixfedmoe_fl.data import MixFedMoEDataManager, _build_label_info


def _build_config(test_samples: int) -> MixFedMoEConfig:
    return MixFedMoEConfig(
        mode="full",
        model_name_or_path="model_ckpt/switch-base-8",
        dataset_name="ag_news",
        num_clients=2,
        num_rounds=1,
        k=1,
        alpha=0.5,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        local_epochs=1,
        learning_rate=2e-5,
        weight_decay=0.0,
        train_batch_size=2,
        eval_batch_size=2,
        max_length=32,
        client_eval_ratio=0.2,
        seed=7,
        num_cpus_per_client=1,
        num_gpus_per_client=0.0,
        output_dir="outputs/mixfedmoe",
        test_samples=test_samples,
        assignment_policy="hot",
    )


def _build_data_manager(test_samples: int) -> MixFedMoEDataManager:
    cfg = _build_config(test_samples=test_samples)
    manager = MixFedMoEDataManager(config=cfg, tokenizer=object())  # type: ignore[arg-type]

    train_split = Dataset.from_dict(
        {
            "id": [0, 1, 2, 3],
            "text": ["a", "b", "c", "d"],
            "label": [0, 1, 0, 1],
        }
    )
    test_split = Dataset.from_dict(
        {
            "id": list(range(10)),
            "text": [f"row-{i}" for i in range(10)],
            "label": [i % 2 for i in range(10)],
        }
    )
    manager._full_train_split = train_split
    manager._full_test_split = test_split
    manager._label_info = _build_label_info(train_split)
    return manager


def test_load_server_test_dataset_caps_test_samples(monkeypatch) -> None:
    monkeypatch.setattr(
        data_module,
        "_tokenize_split",
        lambda split, tokenizer, max_length: split,
    )
    manager = _build_data_manager(test_samples=3)
    bundle = manager.load_server_test_dataset()

    assert bundle.effective_test_samples == 3
    assert len(bundle.test_dataset) == 3
    ids: List[int] = bundle.test_dataset["id"]  # type: ignore[assignment]
    assert len(ids) == 3


def test_load_server_test_dataset_uses_full_split_when_test_samples_non_positive(monkeypatch) -> None:
    monkeypatch.setattr(
        data_module,
        "_tokenize_split",
        lambda split, tokenizer, max_length: split,
    )
    manager = _build_data_manager(test_samples=0)
    bundle = manager.load_server_test_dataset()

    assert bundle.effective_test_samples == 10
    assert len(bundle.test_dataset) == 10


def test_load_server_test_dataset_is_deterministic_for_same_seed(monkeypatch) -> None:
    monkeypatch.setattr(
        data_module,
        "_tokenize_split",
        lambda split, tokenizer, max_length: split,
    )
    manager = _build_data_manager(test_samples=4)

    first = manager.load_server_test_dataset()
    second = manager.load_server_test_dataset()

    first_ids: List[int] = first.test_dataset["id"]  # type: ignore[assignment]
    second_ids: List[int] = second.test_dataset["id"]  # type: ignore[assignment]
    assert first_ids == second_ids
