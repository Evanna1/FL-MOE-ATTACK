from __future__ import annotations

import numpy as np
import pytest
import torch
from flwr.common import ndarrays_to_parameters
from flwr.server import SimpleClientManager
from flwr.server.strategy import FedAvg
from torch.utils.data import DataLoader

from mixfedmoe_fl.config import parse_config
from mixfedmoe_fl.server import (
    ExpertProbeEarlyStoppingServer,
    _NaturalTriggerRoutingTracker,
    _make_eval_loader,
    _should_evaluate_asr,
)


def test_expert_probe_server_stops_before_evaluation_and_next_round() -> None:
    parameters = ndarrays_to_parameters([np.array([1.0], dtype=np.float32)])

    class StopStrategy(FedAvg):
        def __init__(self) -> None:
            super().__init__(initial_parameters=parameters)
            self.expert_probe_stop_requested = False
            self.found_low_frequency_expert = None

        def evaluate(self, server_round, parameters):
            _ = server_round, parameters
            return None

    class StopServer(ExpertProbeEarlyStoppingServer):
        def __init__(self, strategy) -> None:
            super().__init__(
                client_manager=SimpleClientManager(),
                strategy=strategy,
            )
            self.completed_rounds = []

        def fit_round(self, server_round, timeout):
            _ = timeout
            self.completed_rounds.append(server_round)
            self.strategy.expert_probe_stop_requested = True
            self.strategy.found_low_frequency_expert = {
                "expert": "decoder.1/expert_7"
            }
            return self.parameters, {}, ([], [])

        def evaluate_round(self, server_round, timeout):
            raise AssertionError(
                f"round {server_round} evaluation should not run after discovery"
            )

    strategy = StopStrategy()
    server = StopServer(strategy)

    server.fit(num_rounds=20, timeout=None)

    assert server.completed_rounds == [1]


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


def test_monitor_asr_requests_evaluation_from_round_zero() -> None:
    cfg = parse_config(["--mode", "full", "--monitor_asr"])

    assert _should_evaluate_asr(cfg, server_round=0) is True
    assert _should_evaluate_asr(cfg, server_round=1) is True
    assert _should_evaluate_asr(cfg, server_round=10) is True


def test_a3fl_requests_asr_evaluation_from_round_zero() -> None:
    cfg = parse_config(
        ["--mode", "full", "--attack_enabled", "--attack_mode", "a3fl"]
    )

    assert _should_evaluate_asr(cfg, server_round=0) is True
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
    tracker = _NaturalTriggerRoutingTracker(router, target_experts=[1], trigger_token_ids=(7, 8))
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


def test_natural_trigger_routing_tracker_measures_joint_multi_expert_hits() -> None:
    import json

    class TopKRouter(torch.nn.Module):
        """OLMoE-style router returning (probabilities, scores, top-k indices)."""

        def forward(self, logits):
            probabilities = torch.softmax(logits, dim=-1)
            indices = logits.topk(2, dim=-1).indices
            return probabilities, probabilities, indices

    router = TopKRouter()
    tracker = _NaturalTriggerRoutingTracker(
        router, target_experts=[1, 3], trigger_token_ids=(7, 8)
    )
    # Trigger (7, 8) sits immediately before the final EOS token.
    input_ids = torch.tensor([[10, 2, 7, 8, 1]])
    attention_mask = torch.ones_like(input_ids)
    # Rows 2 and 3 are the two trigger tokens. Row 2 routes to experts {1, 3}
    # (joint hit); row 3 routes to {0, 1} (expert 1 only, so no joint hit).
    logits = torch.tensor(
        [
            [5.0, 0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 2.0],
            [1.0, 3.0, 0.0, 0.0],
            [5.0, 0.0, 0.0, 0.0],
        ]
    )
    try:
        tracker.begin_batch(input_ids, attention_mask)
        router(logits)
        metrics = tracker.metrics()
    finally:
        tracker.remove()

    assert metrics["trigger_routing_tokens"] == 2
    assert metrics["trigger_target_num_experts"] == 2
    assert metrics["trigger_target_experts"] == "1,3"
    # Expert 1 is routed to by both trigger tokens; expert 3 by only one.
    per_expert = json.loads(metrics["trigger_target_per_expert_json"])
    assert per_expert["1"]["hit_rate"] == 1.0
    assert per_expert["3"]["hit_rate"] == 0.5
    # Both experts fire together for only one of the two trigger tokens.
    assert metrics["trigger_target_joint_hit_rate"] == 0.5


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
        target_experts=[1],
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
