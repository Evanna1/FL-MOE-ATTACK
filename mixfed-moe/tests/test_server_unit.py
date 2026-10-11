from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from mixfedmoe_fl.config import parse_config
from mixfedmoe_fl.server import _NaturalTriggerRoutingTracker, _make_eval_loader, _should_evaluate_asr


def test_make_eval_loader_returns_none_for_missing_optional_dataset() -> None:
    assert _make_eval_loader(None, batch_size=2, collator=lambda rows: rows) is None  # type: ignore[arg-type]


def test_make_eval_loader_builds_loader_for_present_dataset() -> None:
    loader = _make_eval_loader(
        dataset=[{"value": 1}, {"value": 2}],
        batch_size=2,
        collator=lambda rows: rows,  # type: ignore[arg-type]
    )
    assert isinstance(loader, DataLoader)
    assert next(iter(loader)) == [{"value": 1}, {"value": 2}]


def test_monitor_asr_runs_only_after_completed_training_rounds() -> None:
    cfg = parse_config(["--mode", "full", "--monitor_asr"])

    assert _should_evaluate_asr(cfg, server_round=0) is False
    assert _should_evaluate_asr(cfg, server_round=1) is True
    assert _should_evaluate_asr(cfg, server_round=10) is True


def test_natural_trigger_routing_tracker_measures_probability_top1_and_capacity() -> None:
    class IdentityClassifier(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    class FakeRouter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.classifier = IdentityClassifier()

        def forward(self, hidden_states):
            logits = self.classifier(hidden_states)
            probabilities = torch.softmax(logits, dim=-1)
            selected = torch.nn.functional.one_hot(logits.argmax(dim=-1), num_classes=3)
            selected[3] = 0  # Simulate capacity overflow for one trigger token.
            return probabilities.max(dim=-1).values.unsqueeze(-1), selected, logits

    router = FakeRouter()
    tracker = _NaturalTriggerRoutingTracker(router, target_expert=1, trigger_token_ids=(7, 8))
    input_ids = torch.tensor([[10, 11, 7, 8, 1]])
    attention_mask = torch.ones_like(input_ids)
    logits = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    try:
        tracker.begin_batch(input_ids, attention_mask)
        router(logits)
        metrics = tracker.metrics()
    finally:
        tracker.remove()

    expected_probability = torch.softmax(torch.tensor([0.0, 2.0, 0.0]), dim=-1)[1].item()
    assert metrics["trigger_target_probability"] == pytest.approx(expected_probability)
    assert metrics["trigger_target_hit_rate"] == 1.0
    assert metrics["trigger_target_execution_rate"] == 0.5
    assert metrics["trigger_routing_tokens"] == 2


def test_natural_trigger_routing_tracker_shifts_mask_for_decoder_router() -> None:
    class IdentityClassifier(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    class FakeRouter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.classifier = IdentityClassifier()

        def forward(self, hidden_states):
            logits = self.classifier(hidden_states)
            probabilities = torch.softmax(logits, dim=-1)
            selected = torch.nn.functional.one_hot(logits.argmax(dim=-1), num_classes=3)
            return probabilities.max(dim=-1).values.unsqueeze(-1), selected, logits

    router = FakeRouter()
    tracker = _NaturalTriggerRoutingTracker(
        router,
        target_expert=1,
        trigger_token_ids=(7, 8),
        decoder=True,
    )
    # Encoder positions 2 and 3 contain the trigger. After shift_right(), the
    # decoder consumes those two trigger tokens at positions 3 and 4.
    input_ids = torch.tensor([[10, 11, 7, 8, 1]])
    attention_mask = torch.ones_like(input_ids)
    decoder_router_logits = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )
    try:
        tracker.begin_batch(input_ids, attention_mask)
        router(decoder_router_logits)
        metrics = tracker.metrics()
    finally:
        tracker.remove()

    assert metrics["trigger_target_hit_rate"] == 1.0
    assert metrics["trigger_target_execution_rate"] == 1.0
    assert metrics["trigger_routing_tokens"] == 2
