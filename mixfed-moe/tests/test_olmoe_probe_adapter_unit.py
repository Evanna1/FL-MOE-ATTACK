from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from flwr.common import RecordDict

from mixfedmoe_fl.client import MoeHandle, _screen_probe_candidates_across_layers
from mixfedmoe_fl.expert_probe import load_probe_state, parse_expert_parameter, save_probe_state
from mixfedmoe_fl.olmoe_adapter import make_olmoe_expert_slice_name
from mixfedmoe_fl.olmoe_probe_adapter import (
    inject_olmoe_paired_probe,
    load_olmoe_probe_state,
    observe_olmoe_global_and_update_history,
    parse_olmoe_probe_parameter,
    project_packed_olmoe_experts,
    rank_olmoe_candidates_by_boundary_distance,
    save_olmoe_probe_state,
)


GATE_NAME = "model.layers.2.mlp.experts.gate_up_proj"
DOWN_NAME = "model.layers.2.mlp.experts.down_proj"


def test_olmoe_probe_projection_is_isolated_from_switch_namespace() -> None:
    gate = np.arange(3 * 4 * 8, dtype=np.float32).reshape(3, 4, 8)
    down = np.arange(3 * 8 * 4, dtype=np.float32).reshape(3, 8, 4)

    names, arrays = project_packed_olmoe_experts(
        [GATE_NAME, DOWN_NAME],
        [gate, down],
    )

    expected = make_olmoe_expert_slice_name("decoder.2", 1, "gate_up_proj")
    index = names.index(expected)
    assert np.shares_memory(arrays[index], gate)
    assert parse_olmoe_probe_parameter(expected) == ("decoder.2", 1)
    assert parse_expert_parameter(expected) is None
    switch_name = "decoder.block.2.layer.2.mlp.experts.expert_1.wi.weight"
    assert parse_olmoe_probe_parameter(switch_name) is None


def test_olmoe_and_switch_probe_states_use_different_record_keys() -> None:
    records = RecordDict()
    switch_state = load_probe_state(records)
    switch_state["phase"] = "switch"
    save_probe_state(records, switch_state)
    olmoe_state = load_olmoe_probe_state(records)
    olmoe_state["phase"] = "olmoe"
    save_olmoe_probe_state(records, olmoe_state)

    assert load_probe_state(records)["phase"] == "switch"
    assert load_olmoe_probe_state(records)["phase"] == "olmoe"
    assert len(records.array_records) == 2


def test_olmoe_probe_round_trip_decodes_sparse_slice_after_packed_aggregation() -> None:
    state = {"version": 1, "trackers": {}, "pending": [], "results": {}}
    gate = np.linspace(-0.5, 0.5, 3 * 4 * 8, dtype=np.float32).reshape(3, 4, 8)
    down = np.linspace(-0.25, 0.25, 3 * 8 * 4, dtype=np.float32).reshape(3, 8, 4)
    for server_round in range(1, 5):
        observe_olmoe_global_and_update_history(
            state=state,
            parameter_names=[GATE_NAME, DOWN_NAME],
            parameters=[gate.copy(), down.copy()],
            server_round=server_round,
            target_layer="decoder.2",
            candidate_pool=16,
            repetitions=5,
        )

    outgoing_names = [
        make_olmoe_expert_slice_name("decoder.2", 1, "gate_up_proj"),
        make_olmoe_expert_slice_name("decoder.2", 1, "down_proj"),
    ]
    outgoing = [gate[1].copy(), down[1].copy()]
    injected = inject_olmoe_paired_probe(
        state=state,
        incoming_names=[GATE_NAME, DOWN_NAME],
        incoming_parameters=[gate, down],
        outgoing_global_names=outgoing_names,
        outgoing_parameters=outgoing,
        server_round=5,
        target_layer="decoder.2",
        target_experts=[1],
        warmup_observations=3,
        pairs=4,
        relative_scale=1e-3,
        max_relative_scale=1e-2,
        noise_multiplier=0.0,
        seed=7,
        client_id=0,
    )

    assert len(injected) == 1
    injected_name = str(injected[0]["parameter_name"])
    outgoing_index = outgoing_names.index(injected_name)
    aggregated_gate = gate.copy()
    aggregated_down = down.copy()
    if injected_name.endswith(".gate_up_proj"):
        aggregated_gate[1] += 0.25 * (outgoing[outgoing_index] - gate[1])
    else:
        aggregated_down[1] += 0.25 * (outgoing[outgoing_index] - down[1])

    report = observe_olmoe_global_and_update_history(
        state=state,
        parameter_names=[GATE_NAME, DOWN_NAME],
        parameters=[aggregated_gate, aggregated_down],
        server_round=6,
        target_layer="decoder.2",
        candidate_pool=16,
        repetitions=5,
    )

    assert report["decoded"][0]["valid"] is True
    assert np.isclose(report["decoded"][0]["retention"], 0.25, atol=2e-4)


def test_olmoe_boundary_ranking_uses_topk_admission_not_top1() -> None:
    weight = np.eye(4, dtype=np.float32)
    logits = [np.asarray([[10.0, 9.0, 0.0, -1.0]], dtype=np.float32)]
    ranked = rank_olmoe_candidates_by_boundary_distance(
        candidates=[
            {
                "expert": "decoder.2/expert_1",
                "target_layer": "decoder.2",
                "expert_id": 1,
            }
        ],
        trigger_router_logits=logits,
        router_weight=weight,
        routing_top_k=2,
    )

    assert ranked[0]["boundary_router_top_k"] == 2
    assert ranked[0]["boundary_topk_hit_rate"] == 1.0
    assert ranked[0]["boundary_distance_quantile"] == 0.0
    assert ranked[0]["boundary_margin_q10"] == 9.0


def test_boundary_screen_accepts_olmoe_tuple_router_output() -> None:
    class TinyTopKRouter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(
                torch.tensor([[1.0], [0.5], [-0.5], [-1.0]], dtype=torch.float32)
            )
            self.top_k = 2

        def forward(self, hidden_states: torch.Tensor):
            logits = F.linear(hidden_states, self.weight)
            scores, indices = torch.topk(logits, self.top_k, dim=-1)
            probabilities = torch.softmax(scores, dim=-1)
            return probabilities, scores, indices

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.router = TinyTopKRouter()

        def forward(self, input_ids, attention_mask, **_kwargs):
            self.router(input_ids.float().unsqueeze(-1))
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
    handle = MoeHandle(
        layer_id="decoder.2",
        stack_name="decoder",
        layer_idx=2,
        layer_path="router",
        layer_module=model.router,
        ff_module=model.router,
        mlp_module=model.router,
        experts_attr="experts",
        gate_linear_path="",
        backend="olmoe",
        num_experts=4,
        router_top_k=2,
        router_module=model.router,
    )
    dataset = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 99], [3, 4, 99]],
            "attention_mask": [[1, 1, 1], [1, 1, 1]],
            "label": [0, 1],
        }
    )
    candidates = [
        {
            "expert": "decoder.2/expert_1",
            "target_layer": "decoder.2",
            "expert_id": 1,
        }
    ]

    ranked, trigger_bank = _screen_probe_candidates_across_layers(
        model=model,
        moe_handles=[handle],
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

    assert len(trigger_bank) == 2
    assert ranked[0]["boundary_router_top_k"] == 2
    assert "boundary_topk_hit_rate" in ranked[0]

