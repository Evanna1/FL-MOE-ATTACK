from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
from flwr.common import RecordDict

from mixfedmoe_fl.a3fl import (
    A3FLSearchConfig,
    A3FLTriggerOptimizer,
    _model_parameter_cosine_similarity,
    append_token_trigger,
)
from mixfedmoe_fl.client import _load_a3fl_trigger, _save_a3fl_trigger


class _TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(12, 5)
        self.classifier = nn.Linear(5, 2)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(self, input_ids, attention_mask, **_kwargs):
        hidden = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return SimpleNamespace(logits=self.classifier(pooled))


class _EncoderPathClassifier(nn.Module):
    """Mimic Switch: get_input_embeddings() is not called by forward()."""

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Embedding(12, 5)
        self.encoder = nn.Module()
        self.encoder.embed_tokens = nn.Embedding(12, 5)
        self.encoder.embed_tokens.weight = self.shared.weight
        self.classifier = nn.Linear(5, 2)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.shared

    def forward(self, input_ids, attention_mask, **_kwargs):
        hidden = self.encoder.embed_tokens(input_ids)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return SimpleNamespace(logits=self.classifier(pooled))


def test_append_token_trigger_places_exact_ids_before_eos() -> None:
    result = append_token_trigger(
        input_ids=torch.tensor([[4, 5, 1, 0], [6, 1, 0, 0]]),
        attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        labels=torch.tensor([1, 0]),
        trigger_token_ids=[8, 9],
        eos_token_id=1,
        pad_token_id=0,
        max_length=8,
    )

    assert result.input_ids.tolist() == [[4, 5, 8, 9, 1], [6, 8, 9, 1, 0]]
    assert result.attention_mask.tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]
    assert result.trigger_mask.tolist() == [
        [False, False, True, True, False],
        [False, True, True, False, False],
    ]
    assert torch.equal(result.labels, torch.tensor([1, 0]))


def test_a3fl_search_restores_downloaded_global_model() -> None:
    torch.manual_seed(7)
    model = _TinyClassifier()
    global_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    def restore_global() -> None:
        model.load_state_dict(global_state, strict=True)

    optimizer = A3FLTriggerOptimizer(
        model,
        A3FLSearchConfig(
            outer_steps=1,
            trigger_steps=1,
            candidate_batch_size=4,
            top_k=3,
            adversarial_steps=1,
            adversarial_lr=0.05,
            lambda0=0.01,
            early_stop_asr=1.0,
            max_length=8,
            seed=9,
        ),
        eos_token_id=1,
        pad_token_id=0,
        target_label=0,
        restore_global=restore_global,
        excluded_token_ids=(0, 1),
        decode_trigger=lambda ids: " ".join(str(value) for value in ids),
    )
    batch = {
        "input_ids": torch.tensor([[3, 4, 1], [5, 6, 1]]),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "labels": torch.tensor([1, 1]),
    }

    result = optimizer.optimize(batch, initial_trigger_token_ids=(2, 2))

    assert len(result.trigger_token_ids) == 2
    assert result.search_time >= 0.0
    assert 0.0 <= result.global_metrics.attack_success_rate <= 1.0
    assert 0.0 <= result.adversarial_metrics.attack_success_rate <= 1.0
    for name, value in model.state_dict().items():
        assert torch.equal(value, global_state[name])


def test_a3fl_uses_the_embedding_module_on_the_actual_encoder_path() -> None:
    torch.manual_seed(11)
    model = _EncoderPathClassifier()
    global_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    def restore_global() -> None:
        model.load_state_dict(global_state, strict=True)

    optimizer = A3FLTriggerOptimizer(
        model,
        A3FLSearchConfig(
            outer_steps=1,
            trigger_steps=1,
            candidate_batch_size=4,
            top_k=3,
            adversarial_steps=1,
            adversarial_lr=0.05,
            lambda0=0.01,
            early_stop_asr=1.0,
            max_length=8,
            seed=13,
        ),
        eos_token_id=1,
        pad_token_id=0,
        target_label=0,
        restore_global=restore_global,
        excluded_token_ids=(0, 1),
        decode_trigger=lambda ids: " ".join(str(value) for value in ids),
    )
    batch = {
        "input_ids": torch.tensor([[3, 4, 1], [5, 6, 1]]),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "labels": torch.tensor([1, 1]),
    }

    result = optimizer.optimize(batch, initial_trigger_token_ids=(2, 2))

    assert len(result.trigger_token_ids) == 2
    assert 0.0 <= result.global_metrics.attack_success_rate <= 1.0
    for name, value in model.state_dict().items():
        assert torch.equal(value, global_state[name])


def test_a3fl_trigger_is_private_to_single_client_state() -> None:
    malicious_client_state = RecordDict()
    other_client_state = RecordDict()

    _save_a3fl_trigger(malicious_client_state, (7, 8, 9))

    assert _load_a3fl_trigger(malicious_client_state) == (7, 8, 9)
    assert _load_a3fl_trigger(other_client_state) == ()


def test_a3fl_model_similarity_matches_parameter_cosine() -> None:
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 0.0]]))

    identical = {"weight": torch.tensor([[1.0, 0.0]])}
    orthogonal = {"weight": torch.tensor([[0.0, 2.0]])}

    assert _model_parameter_cosine_similarity(model, identical) == 1.0
    assert _model_parameter_cosine_similarity(model, orthogonal) == 0.0

