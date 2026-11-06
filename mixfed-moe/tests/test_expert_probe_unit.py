from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from datasets import Dataset
from flwr.common import RecordDict

from mixfedmoe_fl.client import MoeHandle, _screen_probe_candidates_across_layers
from mixfedmoe_fl.config import parse_config
from mixfedmoe_fl.expert_probe import (
    advance_batch_probe_confirmation,
    advance_probe_confirmation,
    build_probe_candidate_queue,
    choose_batch_probe_candidates,
    choose_lazy_probe_expert,
    boundary_distance_lower_bounds,
    inject_paired_probe,
    load_probe_state,
    observe_global_and_update_history,
    promote_probe_expert_in_activation_profile,
    promote_probe_experts_in_activation_profile,
    rank_candidates_by_boundary_distance,
    rank_lazy_candidates_across_layers,
    rank_lazy_expert_candidates,
    save_probe_state,
    map_outgoing_parameter_names,
)
from mixfedmoe_fl.strategy import MixFedMoEStrategy


PARAMETER_NAME = "encoder.block.5.layer.1.mlp.experts.expert_3.wi.weight"


def _warm_state(state, values: np.ndarray, rounds: int = 3) -> None:
    for server_round in range(1, rounds + 2):
        observe_global_and_update_history(
            state=state,
            parameter_names=[PARAMETER_NAME],
            parameters=[values.copy()],
            server_round=server_round,
            target_layer="encoder.5",
            candidate_pool=16,
            repetitions=5,
        )


def test_paired_probe_decodes_known_aggregation_retention() -> None:
    state = {"version": 1, "trackers": {}, "pending": [], "results": {}}
    baseline = np.linspace(-0.5, 0.5, 32, dtype=np.float32)
    _warm_state(state, baseline)
    outgoing = [baseline.copy()]

    injected = inject_paired_probe(
        state=state,
        incoming_names=[PARAMETER_NAME],
        incoming_parameters=[baseline],
        outgoing_global_names=[PARAMETER_NAME],
        outgoing_parameters=outgoing,
        server_round=5,
        target_layer="encoder.5",
        target_experts=[3],
        warmup_observations=3,
        pairs=4,
        relative_scale=1e-3,
        max_relative_scale=1e-2,
        noise_multiplier=0.0,
        seed=7,
        client_id=0,
    )

    assert len(injected) == 1
    # Simulate one malicious contributor among four equal-weight contributors.
    aggregated = baseline + 0.25 * (outgoing[0] - baseline)
    report = observe_global_and_update_history(
        state=state,
        parameter_names=[PARAMETER_NAME],
        parameters=[aggregated],
        server_round=6,
        target_layer="encoder.5",
        candidate_pool=16,
        repetitions=5,
    )

    assert report["decoded"][0]["valid"] is True
    assert np.isclose(report["decoded"][0]["retention"], 0.25, atol=2e-4)
    assert report["decoded"][0]["consistency_error"] < 2e-4


def test_nonconsecutive_probe_is_not_used() -> None:
    state = {"version": 1, "trackers": {}, "pending": [], "results": {}}
    baseline = np.linspace(-0.5, 0.5, 32, dtype=np.float32)
    _warm_state(state, baseline)
    outgoing = [baseline.copy()]
    inject_paired_probe(
        state=state,
        incoming_names=[PARAMETER_NAME],
        incoming_parameters=[baseline],
        outgoing_global_names=[PARAMETER_NAME],
        outgoing_parameters=outgoing,
        server_round=5,
        target_layer="encoder.5",
        target_experts=[3],
        warmup_observations=3,
        pairs=4,
        relative_scale=1e-3,
        max_relative_scale=1e-2,
        noise_multiplier=0.0,
        seed=7,
        client_id=0,
    )

    report = observe_global_and_update_history(
        state=state,
        parameter_names=[PARAMETER_NAME],
        parameters=[outgoing[0]],
        server_round=7,
        target_layer="encoder.5",
        candidate_pool=16,
        repetitions=5,
    )
    assert report["decoded"][0]["valid"] is False
    assert report["decoded"][0]["round_gap"] == 2
    assert state["results"] == {}


def test_inconsistent_probe_is_decoded_but_excluded_from_rankings() -> None:
    state = {"version": 1, "trackers": {}, "pending": [], "results": {}}
    baseline = np.linspace(-0.5, 0.5, 32, dtype=np.float32)
    _warm_state(state, baseline)
    outgoing = [baseline.copy()]
    inject_paired_probe(
        state=state,
        incoming_names=[PARAMETER_NAME],
        incoming_parameters=[baseline],
        outgoing_global_names=[PARAMETER_NAME],
        outgoing_parameters=outgoing,
        server_round=5,
        target_layer="encoder.5",
        target_experts=[3],
        warmup_observations=3,
        pairs=4,
        relative_scale=1e-3,
        max_relative_scale=1e-2,
        noise_multiplier=0.0,
        seed=7,
        client_id=0,
    )
    pending = state["pending"][0]
    asymmetric = baseline.copy()
    positive = np.asarray(pending["positive_indices"], dtype=np.int64)
    asymmetric[positive] = outgoing[0][positive]

    report = observe_global_and_update_history(
        state=state,
        parameter_names=[PARAMETER_NAME],
        parameters=[asymmetric],
        server_round=6,
        target_layer="encoder.5",
        candidate_pool=16,
        repetitions=5,
        max_consistency_error=0.1,
    )

    assert report["decoded"][0]["valid"] is True
    assert report["decoded"][0]["measurement_valid"] is False
    assert report["decoded"][0]["quality"] == "distorted"
    assert report["rankings"] == []
    assert state["results"] == {}


def test_probe_state_round_trip() -> None:
    state_record = RecordDict()
    state = {
        "version": 1,
        "trackers": {},
        "pending": [],
        "results": {"e": [0.8]},
        "attempts": {},
        "scheduled_expert": None,
        "scheduled_target": None,
        "phase": "scan",
        "confirmation_expert": None,
        "confirmation_expert_id": None,
        "confirmation_target": None,
        "confirmation_successes": 0,
        "candidate_queue": [],
        "rejected_experts": [],
        "rejected_candidates": [],
        "found_low_frequency_expert": None,
        "scheduled_targets": [],
        "batch_candidates": {},
        "boundary_screen": None,
    }
    save_probe_state(state_record, state)
    assert load_probe_state(state_record) == state



def test_probe_target_is_promoted_by_swapping_activation_counts() -> None:
    reported, report = promote_probe_expert_in_activation_profile(
        activation_profile={"encoder.5": [100, 80, 7, 3]},
        target_layer="encoder.5",
        target_expert=3,
        k=2,
    )

    assert reported["encoder.5"] == [100, 3, 7, 80]
    assert report["displaced_expert"] == 1
    assert report["target_in_top_k"] is True


def test_probe_target_promotion_breaks_boundary_tie_minimally() -> None:
    reported, report = promote_probe_expert_in_activation_profile(
        activation_profile={"encoder.5": [10, 5, 5]},
        target_layer="encoder.5",
        target_expert=2,
        k=2,
    )

    assert reported["encoder.5"] == [10, 5, 6]
    assert report["tie_break_increment"] == 1
    assert report["target_in_top_k"] is True


def test_multiple_probe_targets_are_promoted_jointly_without_evicting_each_other() -> None:
    reported, report = promote_probe_experts_in_activation_profile(
        activation_profile={"encoder.5": [100, 80, 7, 3]},
        target_layer="encoder.5",
        target_experts=[2, 3],
        k=2,
    )

    ranking = sorted(range(4), key=lambda expert: (-reported["encoder.5"][expert], expert))
    assert set(ranking[:2]) == {2, 3}
    assert report["targets_in_top_k"] is True
    assert {item["target_expert"] for item in report["swaps"]} == {2, 3}


def test_batch_probe_injects_independent_pairs_for_two_experts() -> None:
    second_name = PARAMETER_NAME.replace("expert_3", "expert_4")
    state = {"version": 1, "trackers": {}, "pending": [], "results": {}}
    baseline = np.linspace(-0.5, 0.5, 32, dtype=np.float32)
    for server_round in range(1, 5):
        observe_global_and_update_history(
            state=state,
            parameter_names=[PARAMETER_NAME, second_name],
            parameters=[baseline.copy(), baseline.copy()],
            server_round=server_round,
            target_layer="encoder.5",
            candidate_pool=16,
            repetitions=5,
        )
    outgoing = [baseline.copy(), baseline.copy()]

    injected = inject_paired_probe(
        state=state,
        incoming_names=[PARAMETER_NAME, second_name],
        incoming_parameters=[baseline, baseline],
        outgoing_global_names=[PARAMETER_NAME, second_name],
        outgoing_parameters=outgoing,
        server_round=5,
        target_layer="encoder.5",
        target_experts=[3, 4],
        warmup_observations=3,
        pairs=4,
        relative_scale=1e-3,
        max_relative_scale=1e-2,
        noise_multiplier=0.0,
        seed=7,
        client_id=0,
    )

    assert {item["expert"] for item in injected} == {
        "encoder.5/expert_3",
        "encoder.5/expert_4",
    }
    assert len(state["pending"]) == 2
    assert not np.array_equal(outgoing[0], baseline)
    assert not np.array_equal(outgoing[1], baseline)


def test_boundary_ranking_compares_same_trigger_bank_across_experts() -> None:
    weight = np.asarray([[1.0], [0.0], [-1.0]], dtype=np.float32)
    trigger_logits = [
        np.asarray([[0.1, 0.2, 1.0], [0.0, 0.3, 0.8]], dtype=np.float32),
        np.asarray([[0.2, 0.5, 0.6]], dtype=np.float32),
    ]
    candidates = [
        {"expert": "encoder.5/expert_0", "expert_id": 0},
        {"expert": "encoder.5/expert_1", "expert_id": 1},
    ]

    bounds = boundary_distance_lower_bounds(trigger_logits[0], weight)
    ranked = rank_candidates_by_boundary_distance(
        candidates=candidates,
        trigger_router_logits=trigger_logits,
        router_weight=weight,
        quantile=0.8,
    )

    assert bounds.shape == (2, 3)
    assert [item["expert_id"] for item in ranked] == [1, 0]
    assert all("boundary_best_trigger_index" in item for item in ranked)


def test_model_wide_boundary_screen_captures_multiple_layers_per_forward() -> None:
    class TinyMLP(nn.Module):
        def __init__(self, weight: list[float]) -> None:
            super().__init__()
            self.router = nn.Module()
            self.router.classifier = nn.Linear(1, len(weight), bias=False)
            with torch.no_grad():
                self.router.classifier.weight.copy_(
                    torch.tensor(weight, dtype=torch.float32).reshape(-1, 1)
                )

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = TinyMLP([1.0, 0.0, -1.0])
            self.second = TinyMLP([0.5, -0.5, 1.5])

        def forward(self, input_ids, attention_mask, **_kwargs):
            hidden = input_ids.float().unsqueeze(-1)
            self.first.router.classifier(hidden)
            self.second.router.classifier(hidden)
            return object()

    class TinyTokenizer:
        all_special_ids = [0, 99]
        eos_token_id = 99
        pad_token_id = 0

    def collate(examples):
        return {
            "input_ids": torch.tensor(
                [example["input_ids"] for example in examples], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [example["attention_mask"] for example in examples], dtype=torch.long
            ),
            "labels": torch.tensor(
                [example["label"] for example in examples], dtype=torch.long
            ),
        }

    model = TinyModel()
    handles = [
        MoeHandle(
            layer_id="encoder.1",
            stack_name="encoder",
            layer_idx=1,
            layer_path="first",
            layer_module=model.first,
            ff_module=model.first,
            mlp_module=model.first,
            experts_attr="experts",
            gate_linear_path="router.classifier",
        ),
        MoeHandle(
            layer_id="encoder.5",
            stack_name="encoder",
            layer_idx=5,
            layer_path="second",
            layer_module=model.second,
            ff_module=model.second,
            mlp_module=model.second,
            experts_attr="experts",
            gate_linear_path="router.classifier",
        ),
    ]
    dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 99], [3, 4, 99]],
            "attention_mask": [[1, 1, 1], [1, 1, 1]],
            "label": [0, 1],
        }
    )
    candidates = [
        {"expert": "encoder.1/expert_0", "target_layer": "encoder.1", "expert_id": 0},
        {"expert": "encoder.5/expert_2", "target_layer": "encoder.5", "expert_id": 2},
    ]

    ranked, trigger_bank = _screen_probe_candidates_across_layers(
        model=model,
        moe_handles=handles,
        candidates=candidates,
        dataset=dataset,
        collator=collate,  # type: ignore[arg-type]
        tokenizer=TinyTokenizer(),
        device=torch.device("cpu"),
        max_length=6,
        samples=2,
        trigger_bank_size=2,
        trigger_length=1,
        margin=0.0,
        quantile=0.8,
        seed=3,
    )

    assert {item["target_layer"] for item in ranked} == {"encoder.1", "encoder.5"}
    assert len(trigger_bank) == 2


def test_batch_scheduler_keeps_confirming_candidates_and_confirms_independently() -> None:
    state = {
        "phase": "batch_scan",
        "attempts": {},
        "batch_candidates": {
            "encoder.5/expert_1": {
                "status": "confirming",
                "confirmation_successes": 0,
                "retentions": [],
            }
        },
        "rejected_candidates": [],
    }
    ranked = [
        {
            "expert": "encoder.5/expert_2",
            "expert_id": 2,
            "boundary_distance_quantile": 0.1,
        },
        {
            "expert": "encoder.5/expert_1",
            "expert_id": 1,
            "boundary_distance_quantile": 0.2,
        },
    ]
    chosen = choose_batch_probe_candidates(
        state=state,
        ranked_candidates=ranked,
        batch_size=2,
        currently_assigned=[1, 2],
    )
    assert [item["expert_id"] for item in chosen] == [1, 2]

    events = advance_batch_probe_confirmation(
        state=state,
        decoded=[
            _decoded(expert_id=1, observed_round=8, retention=1.0),
            _decoded(expert_id=2, observed_round=8, retention=0.25),
        ],
        confirmation_rounds=1,
        single_retention_min=0.9,
        single_retention_max=1.1,
    )

    assert state["phase"] == "complete"
    assert state["found_low_frequency_expert"]["expert_id"] == 1
    assert state["batch_candidates"]["encoder.5/expert_2"]["status"] == "rejected"
    assert any(event["event"] == "low_frequency_expert_found" for event in events)


def test_completed_batch_probe_cannot_resume_on_a_later_round() -> None:
    found = {
        "expert": "encoder.5/expert_1",
        "layer": "encoder.5",
        "expert_id": 1,
        "found_round": 8,
    }
    state = {
        "phase": "complete",
        "found_low_frequency_expert": found,
        "scheduled_targets": [{"layer": "encoder.5", "expert_id": 2}],
        "scheduled_expert": 2,
        "scheduled_target": {"layer": "encoder.5", "expert_id": 2},
        "pending": [{"expert": "encoder.5/expert_2"}],
        "batch_candidates": {
            "encoder.5/expert_2": {
                "status": "confirming",
                "confirmation_successes": 1,
                "retentions": [1.0],
            }
        },
    }

    events = advance_batch_probe_confirmation(
        state=state,
        decoded=[],
        confirmation_rounds=5,
        single_retention_min=0.9,
        single_retention_max=1.1,
    )

    assert events == []
    assert state["phase"] == "complete"
    assert state["found_low_frequency_expert"] == found
    assert state["scheduled_targets"] == []
    assert state["scheduled_expert"] is None
    assert state["scheduled_target"] is None
    assert state["pending"] == []


def test_strategy_persists_discovery_and_requests_server_stop(tmp_path) -> None:
    strategy = object.__new__(MixFedMoEStrategy)
    strategy.runtime_config = SimpleNamespace(
        output_dir=str(tmp_path),
        checkpoint_interval=5,
        checkpoint_rounds=(),
    )
    strategy.parameter_names = ["shared.weight"]
    strategy._expert_probe_stop_requested = False
    strategy._found_low_frequency_expert = None
    found = {
        "expert": "decoder.1/expert_7",
        "layer": "decoder.1",
        "expert_id": 7,
        "found_round": 17,
        "confirmation_rounds": 5,
        "final_retention": 1.0001,
        "final_consistency_error": 0.0001,
    }
    report = {
        "client_id": 0,
        "found_low_frequency_expert": found,
        "state_events": [
            {"event": "low_frequency_expert_found", **found},
        ],
    }

    strategy._record_expert_probe_discovery(
        server_round=17,
        reports=[report],
        arrays=[np.array([1.0], dtype=np.float32)],
    )

    assert strategy.expert_probe_stop_requested is True
    assert strategy.found_low_frequency_expert is not None
    assert strategy.found_low_frequency_expert["expert"] == "decoder.1/expert_7"
    discovery_path = tmp_path / "expert_probe_discovery.json"
    assert json.loads(discovery_path.read_text(encoding="utf-8"))["expert_id"] == 7
    assert (tmp_path / "checkpoints" / "round_0017.pt").exists()


def test_model_wide_batch_scheduler_caps_each_layer_at_k_minus_two() -> None:
    state = {
        "phase": "batch_scan",
        "attempts": {},
        "batch_candidates": {},
        "rejected_candidates": [],
    }
    ranked = [
        {
            "expert": f"encoder.5/expert_{expert_id}",
            "target_layer": "encoder.5",
            "expert_id": expert_id,
            "boundary_distance_quantile": 0.01 + expert_id * 0.01,
        }
        for expert_id in range(4)
    ] + [
        {
            "expert": f"encoder.9/expert_{expert_id}",
            "target_layer": "encoder.9",
            "expert_id": expert_id,
            "boundary_distance_quantile": 0.1 + expert_id * 0.01,
        }
        for expert_id in range(4)
    ]

    chosen = choose_batch_probe_candidates(
        state=state,
        ranked_candidates=ranked,
        batch_size=6,
        currently_assigned={
            "encoder.5": [0, 1, 2, 3],
            "encoder.9": [0, 1, 2, 3],
        },
        max_per_layer=2,  # K=4 => K-2
    )

    assert len(chosen) == 4
    assert sum(item["target_layer"] == "encoder.5" for item in chosen) == 2
    assert sum(item["target_layer"] == "encoder.9" for item in chosen) == 2


def test_cross_layer_pending_probes_are_appended_for_one_round_decode() -> None:
    first_name = PARAMETER_NAME
    second_name = "encoder.block.9.layer.1.mlp.experts.expert_4.wi.weight"
    state = {"version": 1, "trackers": {}, "pending": [], "results": {}}
    baseline = np.linspace(-0.5, 0.5, 32, dtype=np.float32)
    for server_round in range(1, 5):
        observe_global_and_update_history(
            state=state,
            parameter_names=[first_name, second_name],
            parameters=[baseline.copy(), baseline.copy()],
            server_round=server_round,
            target_layer=None,
            candidate_pool=16,
            repetitions=5,
        )
    outgoing = [baseline.copy(), baseline.copy()]

    inject_paired_probe(
        state=state,
        incoming_names=[first_name, second_name],
        incoming_parameters=[baseline, baseline],
        outgoing_global_names=[first_name, second_name],
        outgoing_parameters=outgoing,
        server_round=5,
        target_layer="encoder.5",
        target_experts=[3],
        warmup_observations=3,
        pairs=4,
        relative_scale=1e-3,
        max_relative_scale=1e-2,
        noise_multiplier=0.0,
        seed=7,
        client_id=0,
    )
    inject_paired_probe(
        state=state,
        incoming_names=[first_name, second_name],
        incoming_parameters=[baseline, baseline],
        outgoing_global_names=[first_name, second_name],
        outgoing_parameters=outgoing,
        server_round=5,
        target_layer="encoder.9",
        target_experts=[4],
        warmup_observations=3,
        pairs=4,
        relative_scale=1e-3,
        max_relative_scale=1e-2,
        noise_multiplier=0.0,
        seed=7,
        client_id=0,
        append_pending=True,
    )

    assert {item["expert"] for item in state["pending"]} == {
        "encoder.5/expert_3",
        "encoder.9/expert_4",
    }


def test_spoofed_profile_is_used_by_server_hot_preference_order() -> None:
    reported, _report = promote_probe_expert_in_activation_profile(
        activation_profile={"encoder.5": [100, 80, 7, 3]},
        target_layer="encoder.5",
        target_expert=3,
        k=2,
    )
    strategy = object.__new__(MixFedMoEStrategy)
    strategy._last_activation_profiles = {"0": reported}

    assert strategy._preference_order("0", "encoder.5", 4)[:2] == [0, 3]


def test_drop_parameter_names_are_mapped_to_global_experts() -> None:
    local_name = "encoder.block.5.layer.1.mlp.experts.expert_0.wi.weight"
    assert map_outgoing_parameter_names(
        [local_name],
        mode="drop",
        assignments={"encoder.5": [3, 7]},
    ) == [PARAMETER_NAME]


def test_probe_cli_configuration_is_parsed() -> None:
    config = parse_config(
        [
            "--mode",
            "mix",
            "--alpha",
            "0.2",
            "--expert_probe_enabled",
            "--expert_probe_target_layer",
            "encoder.5",
            "--expert_probe_pairs",
            "4",
        ]
    )
    assert config.alpha == 0.2
    assert config.expert_probe_enabled is True
    assert config.expert_probe_target_layer == "encoder.5"
    assert config.expert_probe_auto_layer is True
    assert config.expert_probe_layer_top_experts == 2
    assert config.expert_probe_drift_window == 5
    assert config.expert_probe_max_pilot_attempts == 2
    assert config.expert_probe_batch_size == 2
    assert config.expert_probe_screen_samples == 32
    assert config.expert_probe_trigger_bank_size == 4
    assert config.expert_probe_trigger_length == 6
    assert config.expert_probe_boundary_quantile == 0.8
    assert config.expert_probe_pairs == 4
    assert config.expert_probe_confirmation_rounds == 5
    assert config.expert_probe_single_retention_min == 0.9
    assert config.expert_probe_single_retention_max == 1.1
    assert config.expert_probe_max_consistency_error == 0.1


def test_cross_layer_multi_expert_probe_requires_two_reserved_hot_slots() -> None:
    config = parse_config(
        [
            "--mode",
            "mix",
            "--K",
            "4",
            "--expert_probe_enabled",
            "--expert_probe_batch_size",
            "4",
        ]
    )
    assert config.expert_probe_target_layer is None
    assert config.k - 2 == 2

    with pytest.raises(ValueError, match="K >= 3"):
        parse_config(
            [
                "--mode",
                "mix",
                "--K",
                "2",
                "--expert_probe_enabled",
            ]
        )

    with pytest.raises(ValueError, match="batch_size >= 2"):
        parse_config(
            [
                "--mode",
                "mix",
                "--K",
                "4",
                "--expert_probe_enabled",
                "--expert_probe_batch_size",
                "1",
            ]
        )


def _decoded(
    *,
    expert_id: int,
    observed_round: int,
    retention: float,
    consistency_error: float = 0.001,
    measurement_valid: bool = True,
):
    return {
        "expert": f"encoder.5/expert_{expert_id}",
        "parameter_name": PARAMETER_NAME,
        "injected_round": observed_round - 1,
        "observed_round": observed_round,
        "round_gap": 1,
        "valid": True,
        "measurement_valid": measurement_valid,
        "retention": retention,
        "positive_retention": retention + consistency_error / 2,
        "negative_retention": retention - consistency_error / 2,
        "consistency_error": consistency_error,
        "delta": 1e-4,
    }


def test_single_contributor_candidate_requires_five_additional_confirmations() -> None:
    state = {
        "phase": "scan",
        "scheduled_expert": 3,
        "confirmation_successes": 0,
        "candidate_queue": [1, 2],
        "rejected_experts": [],
        "found_low_frequency_expert": None,
    }
    events = advance_probe_confirmation(
        state=state,
        decoded=[_decoded(expert_id=3, observed_round=6, retention=1.0)],
        confirmation_rounds=5,
        single_retention_min=0.9,
        single_retention_max=1.1,
    )
    assert events[0]["event"] == "candidate_locked"
    assert state["phase"] == "confirm"
    assert state["confirmation_successes"] == 0

    for confirmation in range(1, 6):
        events = advance_probe_confirmation(
            state=state,
            decoded=[
                _decoded(
                    expert_id=3,
                    observed_round=6 + confirmation,
                    retention=0.99,
                )
            ],
            confirmation_rounds=5,
            single_retention_min=0.9,
            single_retention_max=1.1,
        )
        if confirmation < 5:
            assert state["phase"] == "confirm"
            assert state["found_low_frequency_expert"] is None

    assert state["phase"] == "complete"
    assert state["scheduled_expert"] is None
    assert state["found_low_frequency_expert"]["expert_id"] == 3
    assert state["found_low_frequency_expert"]["layer"] == "encoder.5"
    assert events[-1]["event"] == "low_frequency_expert_found"


def test_distorted_confirmation_rejects_candidate_and_requests_fallback() -> None:
    state = {
        "phase": "confirm",
        "scheduled_expert": 3,
        "confirmation_expert": "encoder.5/expert_3",
        "confirmation_expert_id": 3,
        "confirmation_successes": 2,
        "candidate_queue": [1, 2],
        "rejected_experts": [],
        "found_low_frequency_expert": None,
    }
    events = advance_probe_confirmation(
        state=state,
        decoded=[
            _decoded(
                expert_id=3,
                observed_round=9,
                retention=-1.5,
                consistency_error=2.0,
                measurement_valid=False,
            )
        ],
        confirmation_rounds=5,
        single_retention_min=0.9,
        single_retention_max=1.1,
    )

    assert events[-1]["event"] == "candidate_rejected"
    assert events[-1]["reason"] == "distorted"
    assert state["phase"] == "fallback"
    assert state["scheduled_expert"] is None
    assert state["confirmation_successes"] == 0
    assert state["rejected_experts"] == [3]


def test_reliable_non_single_pilot_rejects_full_layer_expert_key() -> None:
    state = {
        "phase": "scan",
        "attempts": {"encoder.5/expert_3": 1},
        "rejected_candidates": [],
    }
    events = advance_probe_confirmation(
        state=state,
        decoded=[_decoded(expert_id=3, observed_round=6, retention=0.25)],
        confirmation_rounds=5,
        single_retention_min=0.9,
        single_retention_max=1.1,
    )

    assert events[-1]["event"] == "pilot_candidate_rejected"
    assert state["rejected_candidates"] == ["encoder.5/expert_3"]
    assert state["phase"] == "scan"


def test_lazy_expert_selection_prefers_unassigned_low_drift_candidate() -> None:
    def tracker(drift: float):
        return {
            "size": 32,
            "indices": list(range(16)),
            "last": [0.0] * 16,
            "max_drift": [drift] * 16,
            "mean_drift": [drift] * 16,
            "observations": 3,
        }

    state = {
        "version": 1,
        "trackers": {
            "encoder.block.5.layer.1.mlp.experts.expert_0.wi.weight": tracker(1e-3),
            "encoder.block.5.layer.1.mlp.experts.expert_1.wi.weight": tracker(2e-5),
            "encoder.block.5.layer.1.mlp.experts.expert_2.wi.weight": tracker(5e-5),
        },
        "pending": [],
        "results": {},
        "attempts": {},
        "scheduled_expert": None,
    }

    selected = choose_lazy_probe_expert(
        state=state,
        target_layer="encoder.5",
        num_experts=3,
        warmup_observations=3,
        pairs=4,
        currently_assigned=[0],
    )

    assert selected is not None
    assert selected["expert_id"] == 1
    assert selected["lazy_parameter_drift"] == 2e-5

    excluded = choose_lazy_probe_expert(
        state=state,
        target_layer="encoder.5",
        num_experts=3,
        warmup_observations=3,
        pairs=4,
        currently_assigned=[0],
        excluded_experts=["encoder.5/expert_1"],
    )
    assert excluded is not None
    assert excluded["expert_id"] == 2


def test_automatic_layer_ranking_balances_candidates_across_layers() -> None:
    def tracker(drift: float, scale: float = 1.0):
        return {
            "size": 32,
            "indices": list(range(16)),
            "last": [0.0] * 16,
            "max_drift": [drift] * 16,
            "mean_drift": [drift] * 16,
            "scale_rms": scale,
            "observations": 3,
        }

    state = {
        "trackers": {
            "encoder.block.5.layer.1.mlp.experts.expert_0.wi.weight": tracker(1e-5),
            "encoder.block.5.layer.1.mlp.experts.expert_1.wi.weight": tracker(1e-3),
            # Raw drift is larger, but equal after RMS normalization.
            "encoder.block.9.layer.1.mlp.experts.expert_0.wi.weight": tracker(1e-4, 10.0),
            "encoder.block.9.layer.1.mlp.experts.expert_1.wi.weight": tracker(2e-3, 10.0),
        },
        "attempts": {},
    }

    candidates, layers = rank_lazy_candidates_across_layers(
        state=state,
        layer_num_experts={"encoder.5": 2, "encoder.9": 2},
        warmup_observations=3,
        pairs=4,
        top_experts_per_layer=2,
    )

    assert {candidate["target_layer"] for candidate in candidates[:2]} == {
        "encoder.5",
        "encoder.9",
    }
    assert {item["layer"] for item in layers} == {"encoder.5", "encoder.9"}
    assert all("layer_contrast" in candidate for candidate in candidates)


def test_cross_layer_shortlist_retains_an_inflight_candidate_within_layer_cap() -> None:
    def tracker(drift: float):
        return {
            "size": 32,
            "indices": list(range(16)),
            "last": [0.0] * 16,
            "max_drift": [drift] * 16,
            "mean_drift": [drift] * 16,
            "scale_rms": 1.0,
            "observations": 3,
        }

    state = {
        "trackers": {
            "encoder.block.5.layer.1.mlp.experts.expert_0.wi.weight": tracker(1e-5),
            "encoder.block.5.layer.1.mlp.experts.expert_1.wi.weight": tracker(1e-3),
        },
        "attempts": {},
        "scheduled_targets": [{"layer": "encoder.5", "expert_id": 1}],
        "batch_candidates": {
            "encoder.5/expert_1": {
                "status": "confirming",
                "confirmation_successes": 1,
            }
        },
    }

    candidates, _layers = rank_lazy_candidates_across_layers(
        state=state,
        layer_num_experts={"encoder.5": 2},
        warmup_observations=3,
        pairs=4,
        top_experts_per_layer=1,
    )

    assert [candidate["expert_id"] for candidate in candidates] == [1]


def test_expert_ranking_uses_all_sampled_coordinates_not_probe_tail() -> None:
    def tracker(drifts):
        values = [float(value) for value in drifts]
        return {
            "size": len(values),
            "indices": list(range(len(values))),
            "last": [0.0] * len(values),
            "max_drift": values,
            "mean_drift": values,
            "scale_rms": 1.0,
            "observations": 3,
        }

    state = {
        "trackers": {
            # Its eight safest probe coordinates are perfect zeros, but most
            # of the expert is moving substantially.
            "encoder.block.5.layer.1.mlp.experts.expert_0.wi.weight": tracker(
                [0.0] * 8 + [1e-3] * 24
            ),
            # No exact-zero tail, but the whole expert is consistently quiet.
            "encoder.block.5.layer.1.mlp.experts.expert_1.wi.weight": tracker(
                [1e-4] * 32
            ),
        },
        "attempts": {},
    }

    candidates = rank_lazy_expert_candidates(
        state=state,
        target_layer="encoder.5",
        num_experts=2,
        warmup_observations=3,
        pairs=4,
    )

    assert candidates[0]["expert_id"] == 1
    moving = next(item for item in candidates if item["expert_id"] == 0)
    assert moving["probe_coordinate_drift"] == 0.0
    assert moving["expert_activity_drift"] == 1e-3


def test_fallback_queue_preserves_cross_layer_priority_and_is_bounded() -> None:
    candidates = [
        {"expert": "encoder.1/expert_2"},
        {"expert": "encoder.5/expert_4"},
        {"expert": "decoder.11/expert_0"},
        {"expert": "encoder.1/expert_2"},
    ]

    queue = build_probe_candidate_queue(
        candidates,
        current_expert="encoder.5/expert_4",
        rejected_candidates=["decoder.11/expert_0"],
    )

    assert queue == ["encoder.1/expert_2"]


def test_automatic_observation_tracks_every_layer() -> None:
    state = {"version": 1, "trackers": {}, "pending": [], "results": {}}
    names = [
        "encoder.block.5.layer.1.mlp.experts.expert_0.wi.weight",
        "encoder.block.9.layer.1.mlp.experts.expert_0.wi.weight",
    ]
    arrays = [np.ones(32, dtype=np.float32), np.ones(32, dtype=np.float32)]

    observe_global_and_update_history(
        state=state,
        parameter_names=names,
        parameters=arrays,
        server_round=1,
        target_layer=None,
        candidate_pool=16,
        repetitions=5,
        drift_window=3,
    )

    assert set(state["trackers"]) == set(names)


def test_server_uses_its_authoritative_assignment_for_current_round_aggregation() -> None:
    strategy = object.__new__(MixFedMoEStrategy)
    strategy._last_round_assignments = {"0": {"encoder.5": [0, 1]}}
    parsed = strategy._parse_assignments_metric(
        metrics={
            "assigned_experts_json": '{"encoder.5": [0, 1]}',
        },
        client_cid="0",
    )
    assert parsed == {"encoder.5": [0, 1]}


def test_server_rejects_client_assignment_spoofing() -> None:
    strategy = object.__new__(MixFedMoEStrategy)
    strategy._last_round_assignments = {"0": {"encoder.5": [0, 1]}}

    with pytest.raises(ValueError, match="does not match the server-issued assignment"):
        strategy._parse_assignments_metric(
            metrics={
                "assigned_experts_json": '{"encoder.5": [0, 3]}',
            },
            client_cid="0",
        )
