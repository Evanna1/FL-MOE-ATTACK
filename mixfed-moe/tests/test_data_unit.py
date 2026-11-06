from __future__ import annotations

from dataclasses import replace
from typing import List

from datasets import Dataset

import mixfedmoe_fl.data as data_module
from mixfedmoe_fl.config import MixFedMoEConfig
from mixfedmoe_fl.data import MixFedMoEDataManager, _build_label_info, _tokenize_split


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


def test_federated_dataset_uses_dirichlet_partitioner(monkeypatch) -> None:
    captured = {}

    class FakeDirichletPartitioner:
        def __init__(self, **kwargs) -> None:
            captured["partitioner_kwargs"] = kwargs

    class FakeFederatedDataset:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(data_module, "DirichletPartitioner", FakeDirichletPartitioner)
    monkeypatch.setattr(data_module, "FederatedDataset", FakeFederatedDataset)

    manager = MixFedMoEDataManager(
        config=_build_config(test_samples=3),
        tokenizer=object(),  # type: ignore[arg-type]
    )
    manager._get_federated_dataset()

    assert captured["partitioner_kwargs"] == {
        "num_partitions": 2,
        "partition_by": "label",
        "alpha": 0.5,
        "min_partition_size": 10,
        "self_balancing": False,
        "shuffle": True,
        "seed": 7,
    }
    assert captured["dataset"] == "ag_news"
    assert isinstance(captured["partitioners"]["train"], FakeDirichletPartitioner)


def test_federated_dataset_uses_iid_partitioner(monkeypatch) -> None:
    captured = {}

    class FakeIidPartitioner:
        def __init__(self, **kwargs) -> None:
            captured["partitioner_kwargs"] = kwargs

    class FakeFederatedDataset:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(data_module, "IidPartitioner", FakeIidPartitioner)
    monkeypatch.setattr(data_module, "FederatedDataset", FakeFederatedDataset)

    manager = MixFedMoEDataManager(
        config=replace(_build_config(test_samples=3), data_distribution="iid"),
        tokenizer=object(),  # type: ignore[arg-type]
    )
    manager._get_federated_dataset()

    assert captured["partitioner_kwargs"] == {"num_partitions": 2}
    assert captured["dataset"] == "ag_news"
    assert isinstance(captured["partitioners"]["train"], FakeIidPartitioner)


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


def test_client_can_use_global_random_examples_as_fixed_size_poison_source(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        data_module,
        "_tokenize_split",
        lambda split, tokenizer, max_length: split,
    )
    manager = _build_data_manager(test_samples=4)
    manager.config = replace(
        manager.config,
        attack_enabled=True,
        global_random_poison_source=True,
        poison_rate=0.5,
        target_label=0,
        trigger="cf",
        trigger_token_ids=(),
    )
    local_partition = Dataset.from_dict(
        {
            "text": ["local-a", "local-b", "local-c", "local-d"],
            "label": [1, 1, 1, 1],
        }
    )

    class FakeFederatedDataset:
        def load_partition(self, partition_id, split):
            assert partition_id == 0
            assert split == "train"
            return local_partition

    manager._fds = FakeFederatedDataset()

    bundle = manager.load_client_dataset(partition_id=0, apply_badnet=True)

    poisoned_texts = [
        text for text in bundle.train_dataset["text"] if text.endswith(" cf")
    ]
    assert bundle.num_train_examples == len(local_partition)
    assert bundle.num_poisoned_examples == 2
    assert sorted(poisoned_texts) == ["b cf", "d cf"]
    assert all(label == 0 for text, label in zip(
        bundle.train_dataset["text"], bundle.train_dataset["label"]
    ) if text.endswith(" cf"))


def test_only_malicious_client_can_use_an_iid_like_global_random_partition(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        data_module,
        "_tokenize_split",
        lambda split, tokenizer, max_length: split,
    )
    manager = _build_data_manager(test_samples=4)
    manager.config = replace(
        manager.config,
        attack_enabled=True,
        iid_malicious_client_data=True,
    )
    malicious_partition = Dataset.from_dict(
        {
            "text": ["mal-a", "mal-b", "mal-c", "mal-d"],
            "label": [1, 1, 1, 1],
        }
    )
    benign_partition = Dataset.from_dict(
        {
            "text": ["benign-a", "benign-b", "benign-c", "benign-d"],
            "label": [1, 1, 1, 1],
        }
    )

    class FakeFederatedDataset:
        def load_partition(self, partition_id, split):
            assert split == "train"
            return malicious_partition if partition_id == 0 else benign_partition

    manager._fds = FakeFederatedDataset()

    malicious = manager.load_client_dataset(partition_id=0)
    benign = manager.load_client_dataset(partition_id=1)

    assert malicious.num_train_examples == len(malicious_partition)
    assert set(malicious.train_dataset["text"]) == {"a", "b", "c", "d"}
    assert malicious.train_dataset["label"].count(0) == 2
    assert malicious.train_dataset["label"].count(1) == 2
    assert benign.train_dataset["text"] == benign_partition["text"]
    assert benign.train_dataset["label"] == benign_partition["label"]


def test_monitor_asr_builds_only_a_server_side_triggered_test_set(monkeypatch) -> None:
    monkeypatch.setattr(
        data_module,
        "_tokenize_split",
        lambda split, tokenizer, max_length: split,
    )
    manager = _build_data_manager(test_samples=3)
    manager.config = replace(manager.config, monitor_asr=True, attack_enabled=False)

    bundle = manager.load_server_test_dataset()

    assert bundle.triggered_test_dataset is not None
    assert 0 < len(bundle.triggered_test_dataset) <= 3
    assert all(label != manager.config.target_label for label in bundle.triggered_test_dataset["label"])


def test_a3fl_does_not_build_a_server_side_triggered_test_set(monkeypatch) -> None:
    monkeypatch.setattr(
        data_module,
        "_tokenize_split",
        lambda split, tokenizer, max_length: split,
    )
    manager = _build_data_manager(test_samples=3)
    manager.config = replace(
        manager.config,
        attack_enabled=True,
        attack_mode="a3fl",
        monitor_asr=True,
    )

    bundle = manager.load_server_test_dataset()

    assert bundle.triggered_test_dataset is None


def test_a3fl_builds_dynamic_triggered_test_set_from_reported_token_ids(monkeypatch) -> None:
    monkeypatch.setattr(
        data_module,
        "_tokenize_split",
        lambda split, tokenizer, max_length: split,
    )
    manager = _build_data_manager(test_samples=4)
    manager.config = replace(
        manager.config,
        attack_enabled=True,
        attack_mode="a3fl",
        target_label=1,
    )

    triggered = manager.load_triggered_server_test_dataset(
        trigger="unused-text-fallback",
        trigger_token_ids=(70, 71),
    )

    assert 0 < len(triggered) <= 4
    assert all(label == 0 for label in triggered["label"])
    assert all(token_ids == [70, 71] for token_ids in triggered["_mixfedmoe_trigger_ids"])


def test_tokenize_split_reserves_space_and_appends_exact_trigger_ids() -> None:
    class FakeTokenizer:
        eos_token_id = 99

        def __call__(self, text, **kwargs):
            assert kwargs["max_length"] == 3
            return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

    split = Dataset.from_dict(
        {
            "text": ["a very long source text"],
            "label": [1],
            "_mixfedmoe_trigger_ids": [[70, 71]],
            "_mixfedmoe_is_poisoned": [True],
        }
    )

    tokenized = _tokenize_split(split, FakeTokenizer(), max_length=6)  # type: ignore[arg-type]

    assert tokenized[0]["input_ids"].tolist() == [1, 2, 3, 70, 71, 99]
    assert tokenized[0]["attention_mask"].tolist() == [1, 1, 1, 1, 1, 1]
    assert bool(tokenized[0]["is_poisoned"].item()) is True
