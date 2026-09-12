from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from mixfedmoe_fl.attack import build_triggered_test_dataset, poison_text_classification_dataset
from mixfedmoe_fl.config import MixFedMoEConfig, SUPPORTED_DATASET_NAME_TO_HF, supported_dataset_names

try:
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import DirichletPartitioner
except ImportError:
    FederatedDataset = None  # type: ignore[assignment]
    DirichletPartitioner = None  # type: ignore[assignment]


@dataclass(frozen=True)
class LabelInfo:
    label_names: List[str]
    num_labels: int
    id2label: Dict[int, str]
    label2id: Dict[str, int]


@dataclass(frozen=True)
class ClientDatasetBundle:
    train_dataset: Dataset
    eval_dataset: Dataset
    num_train_examples: int
    num_eval_examples: int
    label_info: LabelInfo
    num_poisoned_examples: int = 0


@dataclass(frozen=True)
class ServerDatasetBundle:
    test_dataset: Dataset
    triggered_test_dataset: Optional[Dataset]
    effective_test_samples: int
    label_info: LabelInfo


def resolve_hf_dataset_name(dataset_name: str) -> str:
    if dataset_name not in SUPPORTED_DATASET_NAME_TO_HF:
        allowed = ", ".join(supported_dataset_names())
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            f"Please switch dataset via --dataset_name in [{allowed}]."
        )
    return SUPPORTED_DATASET_NAME_TO_HF[dataset_name]


def build_tokenizer(model_name_or_path: str, max_length: int) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, model_max_length=max_length)
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer must define eos_token for EOS-based sequence classification.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _assert_dataset_compatibility(
    train_split: Dataset,
    test_split: Dataset,
    dataset_name: str,
    hf_dataset_name: str,
) -> None:
    required_columns: List[str] = ["text", "label"]
    train_missing_cols: List[str] = [col for col in required_columns if col not in train_split.column_names]
    test_missing_cols: List[str] = [col for col in required_columns if col not in test_split.column_names]
    if train_missing_cols or test_missing_cols:
        raise RuntimeError(
            f"Dataset '{dataset_name}' (HF: '{hf_dataset_name}') is incompatible. "
            f"Missing train columns={train_missing_cols}, test columns={test_missing_cols}. "
            f"Expected columns: {required_columns}. "
            f"Please switch dataset via --dataset_name in [{', '.join(supported_dataset_names())}]."
        )

    if len(train_split) == 0 or len(test_split) == 0:
        raise RuntimeError(
            f"Dataset '{dataset_name}' (HF: '{hf_dataset_name}') is incompatible: empty split detected. "
            f"Please switch dataset via --dataset_name in [{', '.join(supported_dataset_names())}]."
        )

    sample_row: Dict[str, Any] = train_split[0]
    sample_text = sample_row.get("text")
    if not isinstance(sample_text, str):
        raise RuntimeError(
            f"Dataset '{dataset_name}' (HF: '{hf_dataset_name}') is incompatible: train['text'] is not string "
            f"(got type={type(sample_text)}). Please switch dataset via --dataset_name in "
            f"[{', '.join(supported_dataset_names())}]."
        )
    sample_label = sample_row.get("label")
    int(sample_label)


def _append_eos_to_batch(
    input_ids_batch: Sequence[Sequence[int]],
    attention_mask_batch: Sequence[Sequence[int]],
    eos_token_id: int,
    max_length: int,
) -> Tuple[List[List[int]], List[List[int]]]:
    output_input_ids: List[List[int]] = []
    output_attention_mask: List[List[int]] = []
    for input_ids, attention_mask in zip(input_ids_batch, attention_mask_batch):
        fixed_ids: List[int] = list(input_ids)[: max_length - 1]
        fixed_mask: List[int] = list(attention_mask)[: max_length - 1]
        fixed_ids.append(eos_token_id)
        fixed_mask.append(1)
        output_input_ids.append(fixed_ids)
        output_attention_mask.append(fixed_mask)
    return output_input_ids, output_attention_mask


def _tokenize_split(
    split: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dataset:
    eos_token_id: Optional[int] = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id for EOS-based sequence classification.")
    if max_length < 2:
        raise ValueError("--max_length must be >= 2.")

    def tokenize_batch(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        encodings: Dict[str, Any] = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length - 1,
            padding=False,
        )
        input_ids_batch: Sequence[Sequence[int]] = encodings["input_ids"]
        attention_mask_batch: Sequence[Sequence[int]] = encodings["attention_mask"]
        fixed_ids, fixed_masks = _append_eos_to_batch(
            input_ids_batch=input_ids_batch,
            attention_mask_batch=attention_mask_batch,
            eos_token_id=eos_token_id,
            max_length=max_length,
        )
        labels: List[int] = [int(x) for x in batch["label"]]
        return {"input_ids": fixed_ids, "attention_mask": fixed_masks, "labels": labels}

    tokenized = split.map(tokenize_batch, batched=True, remove_columns=split.column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def _build_label_info(train_split: Dataset) -> LabelInfo:
    label_feature: Any = train_split.features.get("label")
    label_names: List[str]
    if label_feature is not None and hasattr(label_feature, "names") and label_feature.names:
        label_names = [str(name) for name in label_feature.names]
    else:
        num_labels = int(max(train_split["label"])) + 1  # type: ignore[index]
        label_names = [str(i) for i in range(num_labels)]

    id2label: Dict[int, str] = {idx: name for idx, name in enumerate(label_names)}
    label2id: Dict[str, int] = {name: idx for idx, name in enumerate(label_names)}
    return LabelInfo(
        label_names=label_names,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
    )


@dataclass
class MixFedMoEDataManager:
    config: MixFedMoEConfig
    tokenizer: PreTrainedTokenizerBase
    _fds: Optional[Any] = field(default=None, init=False, repr=False)
    _full_train_split: Optional[Dataset] = field(default=None, init=False, repr=False)
    _full_test_split: Optional[Dataset] = field(default=None, init=False, repr=False)
    _label_info: Optional[LabelInfo] = field(default=None, init=False, repr=False)

    @classmethod
    def from_config(cls, config: MixFedMoEConfig) -> "MixFedMoEDataManager":
        tokenizer = build_tokenizer(config.model_name_or_path, max_length=config.max_length)
        return cls(config=config, tokenizer=tokenizer)

    def _get_federated_dataset(self) -> Any:
        if self._fds is not None:
            return self._fds
        if FederatedDataset is None or DirichletPartitioner is None:
            raise ImportError(
                "flwr_datasets is required for MixFedMoE data loading. "
                "Please use the `flwr` conda environment."
            )

        hf_dataset_name = resolve_hf_dataset_name(self.config.dataset_name)
        partitioner = DirichletPartitioner(
            num_partitions=self.config.num_clients,
            partition_by="label",
            alpha=self.config.alpha,
            min_partition_size=2,
            self_balancing=False,
            shuffle=True,
            seed=self.config.seed,
        )
        self._fds = FederatedDataset(
            dataset=hf_dataset_name,
            partitioners={"train": partitioner},
            shuffle=True,
            seed=self.config.seed,
        )
        return self._fds

    def _load_full_splits(self) -> Tuple[Dataset, Dataset]:
        if self._full_train_split is not None and self._full_test_split is not None:
            return self._full_train_split, self._full_test_split

        fds = self._get_federated_dataset()
        train_split = fds.load_split("train")
        test_split = fds.load_split("test")
        hf_dataset_name = resolve_hf_dataset_name(self.config.dataset_name)
        _assert_dataset_compatibility(
            train_split=train_split,
            test_split=test_split,
            dataset_name=self.config.dataset_name,
            hf_dataset_name=hf_dataset_name,
        )
        self._full_train_split = train_split
        self._full_test_split = test_split
        return train_split, test_split

    def get_label_info(self) -> LabelInfo:
        if self._label_info is not None:
            return self._label_info
        train_split, _ = self._load_full_splits()
        self._label_info = _build_label_info(train_split)
        return self._label_info

    def load_client_dataset(self, partition_id: int, apply_badnet: bool = False) -> ClientDatasetBundle:
        if partition_id < 0 or partition_id >= self.config.num_clients:
            raise ValueError(
                f"partition_id={partition_id} is invalid for num_clients={self.config.num_clients}."
            )

        partition = self._get_federated_dataset().load_partition(partition_id, split="train")
        if len(partition) < 1:
            raise RuntimeError(
                f"Client partition {partition_id} has {len(partition)} sample(s); requires at least 1."
            )

        num_poisoned_examples = 0
        if apply_badnet:
            label_info = self.get_label_info()
            if self.config.target_label >= label_info.num_labels:
                raise ValueError(
                    f"--target_label={self.config.target_label} is invalid for dataset "
                    f"'{self.config.dataset_name}' with {label_info.num_labels} labels."
                )
            partition, num_poisoned_examples = poison_text_classification_dataset(
                dataset=partition,
                poison_rate=self.config.poison_rate,
                target_label=self.config.target_label,
                trigger=self.config.trigger,
                seed=self.config.seed + partition_id,
            )

        # No client-side local evaluation split: use the full client partition for training.
        tokenized_train = _tokenize_split(
            split=partition,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
        )
        tokenized_eval = tokenized_train.select([])
        label_info = self.get_label_info()
        return ClientDatasetBundle(
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            num_train_examples=len(tokenized_train),
            num_eval_examples=0,
            label_info=label_info,
            num_poisoned_examples=num_poisoned_examples,
        )

    def load_server_test_dataset(self) -> ServerDatasetBundle:
        _, test_split = self._load_full_splits()
        test_split = test_split.shuffle(seed=self.config.seed)
        if self.config.test_samples > 0:
            effective_test_samples = min(self.config.test_samples, len(test_split))
            test_split = test_split.select(range(effective_test_samples))
        else:
            effective_test_samples = len(test_split)

        label_info = self.get_label_info()
        if self.config.attack_enabled and self.config.target_label >= label_info.num_labels:
            raise ValueError(
                f"--target_label={self.config.target_label} is invalid for dataset "
                f"'{self.config.dataset_name}' with {label_info.num_labels} labels."
            )
        tokenized_test = _tokenize_split(
            split=test_split,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
        )
        tokenized_triggered_test: Optional[Dataset] = None
        if self.config.attack_enabled:
            triggered_test_split = build_triggered_test_dataset(test_split, trigger=self.config.trigger)
            tokenized_triggered_test = _tokenize_split(
                split=triggered_test_split,
                tokenizer=self.tokenizer,
                max_length=self.config.max_length,
            )
        return ServerDatasetBundle(
            test_dataset=tokenized_test,
            triggered_test_dataset=tokenized_triggered_test,
            effective_test_samples=effective_test_samples,
            label_info=label_info,
        )
