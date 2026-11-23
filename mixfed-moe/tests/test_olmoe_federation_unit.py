from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Dict, List, Sequence

import numpy as np
import pytest
import torch
from flwr.common import (
    Code,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from transformers import OlmoeConfig

from mixfedmoe_fl.client import (
    _build_trigger_router_loss,
    _to_numpy_ndarrays,
    apply_mode_mix,
    collect_moe_handles,
    restrict_malicious_updates_to_target_expert_and_router,
)
from mixfedmoe_fl.a3fl import A3FLSearchConfig, A3FLTriggerOptimizer
from mixfedmoe_fl.config import MixFedMoEConfig
from mixfedmoe_fl.expert_profiler import TrainingLFERoutingTracker
from mixfedmoe_fl.igcg_trigger_optimization import (
    IGCGTriggerConfig,
    IGCGTriggerOptimizer,
)
from mixfedmoe_fl.olmoe_adapter import OlmoeAdapter
from mixfedmoe_fl.strategy import MixFedMoEStrategy
from mixfedmoe_fl.server import _NaturalTriggerRoutingTracker
from models.olmoe import OlmoeForSequenceClassification


def _tiny_config() -> OlmoeConfig:
    return OlmoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=32,
        pad_token_id=0,
        eos_token_id=2,
        num_labels=3,
    )


def _batch() -> Dict[str, torch.Tensor]:
    input_ids = torch.tensor([[3, 4, 2, 0], [5, 6, 7, 2]])
    return {
        "input_ids": input_ids,
        "attention_mask": input_ids.ne(0).long(),
        "labels": torch.tensor([1, 2]),
    }


def _runtime_config(model_path: str, output_dir: str) -> MixFedMoEConfig:
    return MixFedMoEConfig(
        mode="mix",
        model_name_or_path=model_path,
        model_backend="olmoe",
        model_dtype="float32",
        dataset_name="ag_news",
        num_clients=2,
        num_rounds=1,
        k=2,
        alpha=0.5,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        local_epochs=1.0,
        learning_rate=1e-3,
        weight_decay=0.0,
        train_batch_size=2,
        eval_batch_size=2,
        max_length=16,
        client_eval_ratio=0.0,
        seed=7,
        num_cpus_per_client=1,
        num_gpus_per_client=0.0,
        output_dir=output_dir,
        test_samples=4,
    )


@dataclass(frozen=True)
class _DummyClient:
    cid: str


class _DummyManager:
    def __init__(self, clients: Sequence[_DummyClient]):
        self.clients = list(clients)

    def num_available(self) -> int:
        return len(self.clients)

    def sample(self, num_clients: int, min_num_clients: int) -> List[_DummyClient]:
        assert len(self.clients) >= min_num_clients
        return self.clients[:num_clients]


class _TinyTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    all_special_ids = [0, 2]

    def __call__(self, text: str, add_special_tokens: bool = False, **_kwargs):
        del add_special_tokens
        base = 3 + (sum(ord(char) for char in text) % 5)
        return {"input_ids": [base, base + 1]}

    def decode(self, token_ids, **_kwargs) -> str:
        return " ".join(str(int(value)) for value in token_ids)


def test_olmoe_mix_trains_only_selected_expert_copies() -> None:
    model = OlmoeForSequenceClassification(_tiny_config())
    handles = collect_moe_handles(model)
    assignments = {handle.layer_id: [0, 2] for handle in handles}
    unselected_before = (
        model.model.layers[0].mlp.experts.gate_up_proj[1].detach().clone()
    )

    apply_mode_mix(model, handles, assignments)
    names = OlmoeAdapter.sparse_upload_names(model, handles, assignments)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=0.05)
    selected_before = [
        parameter.detach().clone()
        for parameter in model.model.layers[0].mlp.experts.selected_gate_up_proj
    ]

    optimizer.zero_grad(set_to_none=True)
    outputs = model(**_batch())
    assert outputs.loss is not None
    outputs.loss.backward()
    optimizer.step()

    experts = model.model.layers[0].mlp.experts
    assert any(
        not torch.equal(parameter, before)
        for parameter, before in zip(experts.selected_gate_up_proj, selected_before)
    )
    assert torch.equal(experts.gate_up_proj[1], unselected_before)
    assert experts.gate_up_proj.grad is None
    assert "model.layers.0.mlp.experts.gate_up_proj" not in names
    assert "__olmoe_expert__.decoder.0.0.gate_up_proj" in names
    arrays = _to_numpy_ndarrays(model, names)
    sparse_idx = names.index("__olmoe_expert__.decoder.0.0.gate_up_proj")
    assert arrays[sparse_idx].shape == (16, 16)


def test_olmoe_training_tracker_counts_topk_memberships() -> None:
    model = OlmoeForSequenceClassification(_tiny_config())
    model.gradient_checkpointing_enable()
    model.train()
    handles = collect_moe_handles(model)
    tracker = TrainingLFERoutingTracker(handles, top_k=2)
    batch = _batch()

    tracker.begin_batch(batch["attention_mask"])
    outputs = model(**batch)
    assert outputs.loss is not None
    outputs.loss.backward()
    profile = tracker.finish()

    layer = profile["layers"]["decoder.0"]
    assert layer["router_top_k"] == 2
    assert layer["valid_tokens"] == 7
    assert sum(layer["counts"]) == 14
    assert sum(layer["frequencies"]) == pytest.approx(2.0)


def test_olmoe_neurotoxin_global_delta_follows_selected_experts() -> None:
    model = OlmoeForSequenceClassification(_tiny_config())
    handles = collect_moe_handles(model)
    assignments = {handle.layer_id: [0, 2] for handle in handles}
    source = model.model.layers[0].mlp.experts
    source.gate_up_proj.grad = torch.stack(
        [torch.full_like(row, float(index + 1)) for index, row in enumerate(source.gate_up_proj)]
    )
    source.down_proj.grad = torch.stack(
        [torch.full_like(row, float(10 + index)) for index, row in enumerate(source.down_proj)]
    )

    apply_mode_mix(model, handles, assignments)
    adapted = model.model.layers[0].mlp.experts

    assert torch.all(adapted.selected_gate_up_proj[0].grad == 1)
    assert torch.all(adapted.selected_gate_up_proj[1].grad == 3)
    assert torch.all(adapted.selected_down_proj[0].grad == 10)
    assert torch.all(adapted.selected_down_proj[1].grad == 12)


def test_olmoe_target_expert_attack_isolates_one_parameter_pair() -> None:
    model = OlmoeForSequenceClassification(_tiny_config())
    handles = collect_moe_handles(model)
    assignments = {handle.layer_id: [0, 2] for handle in handles}
    apply_mode_mix(model, handles, assignments)

    available = restrict_malicious_updates_to_target_expert_and_router(
        model=model,
        moe_handles=handles,
        target_layer="decoder.0",
        target_expert=2,
        mode="mix",
        assignments=assignments,
    )

    assert available is True
    experts = model.model.layers[0].mlp.experts
    assert experts.selected_gate_up_proj[0].requires_grad is False
    assert experts.selected_down_proj[0].requires_grad is False
    assert experts.selected_gate_up_proj[1].requires_grad is True
    assert experts.selected_down_proj[1].requires_grad is True
    assert model.model.layers[0].mlp.gate.weight.requires_grad is True
    assert model.model.layers[1].mlp.gate.weight.requires_grad is False


def test_olmoe_target_router_loss_uses_unshifted_causal_tokens() -> None:
    model = OlmoeForSequenceClassification(_tiny_config())
    handles = collect_moe_handles(model)
    assignments = {handle.layer_id: [0, 2] for handle in handles}
    apply_mode_mix(model, handles, assignments)
    objective = _build_trigger_router_loss(
        moe_handles=handles,
        target_layer="decoder.0",
        target_experts=[2],
        mode="mix",
        assignments=assignments,
        trigger_token_ids=(4,),
    )
    batch = _batch()

    objective.begin_batch(batch["input_ids"], batch["attention_mask"])
    outputs = model(**batch)
    router_loss = objective.loss()
    router_loss.backward()
    objective.close()

    assert objective.decoder is False
    assert torch.isfinite(router_loss)
    assert model.model.layers[0].mlp.gate.weight.grad is not None


def test_olmoe_igcg_computes_trigger_gradient(tmp_path) -> None:
    model = OlmoeForSequenceClassification(_tiny_config())
    checkpoint_path = tmp_path / "olmoe.pt"
    torch.save(model.state_dict(), checkpoint_path)
    config = IGCGTriggerConfig(
        checkpoint_path=str(checkpoint_path),
        target_layer="decoder.0",
        target_expert=1,
        trigger_length=1,
        gradient_batch_size=1,
        evaluation_batch_size=1,
        max_length=8,
    )
    model.eval().requires_grad_(False)
    optimizer = IGCGTriggerOptimizer(
        model=model,
        tokenizer=_TinyTokenizer(),  # type: ignore[arg-type]
        config=config,
        calibration_texts=["alpha"],
        heldout_texts=["beta"],
        device=torch.device("cpu"),
    )

    gradient = optimizer.gradient([9])

    assert gradient.shape == (1, model.config.hidden_size)
    assert torch.isfinite(gradient).all()


def test_olmoe_a3fl_uses_memory_bounded_adversarial_state() -> None:
    model = OlmoeForSequenceClassification(_tiny_config())
    pristine = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    def restore() -> None:
        model.load_state_dict(pristine, strict=True)

    optimizer = A3FLTriggerOptimizer(
        model=model,
        config=A3FLSearchConfig(
            outer_steps=1,
            trigger_steps=1,
            candidate_batch_size=2,
            top_k=2,
            adversarial_steps=1,
            max_length=8,
            seed=3,
        ),
        eos_token_id=2,
        pad_token_id=0,
        target_label=1,
        restore_global=restore,
        excluded_token_ids=(0, 2),
        decode_trigger=lambda ids: " ".join(str(value) for value in ids),
    )

    assert optimizer._adversarial_parameter_names
    assert not any(
        ".experts." in name for name in optimizer._adversarial_parameter_names
    )
    full_scope_optimizer = A3FLTriggerOptimizer(
        model=model,
        config=A3FLSearchConfig(
            outer_steps=1,
            trigger_steps=1,
            candidate_batch_size=1,
            top_k=1,
            adversarial_steps=1,
            max_length=8,
            olmoe_adversarial_scope="full",
        ),
        eos_token_id=2,
        pad_token_id=0,
        target_label=1,
        restore_global=restore,
    )
    assert any(
        ".experts." in name
        for name in full_scope_optimizer._adversarial_parameter_names
    )
    result = optimizer.optimize(_batch(), initial_trigger_token_ids=(9,))

    assert len(result.trigger_token_ids) == 1
    for name, value in model.state_dict().items():
        assert torch.equal(value, pristine[name])


def test_olmoe_server_tracker_measures_topk_execution() -> None:
    model = OlmoeForSequenceClassification(_tiny_config())
    router = model.model.layers[0].mlp.gate
    tracker = _NaturalTriggerRoutingTracker(
        router,
        target_experts=[1],
        trigger_token_ids=(4,),
        decoder=False,
    )
    input_ids = torch.tensor([[3, 4, 2], [5, 4, 2]])
    attention_mask = torch.ones_like(input_ids)

    tracker.begin_batch(input_ids, attention_mask)
    model(input_ids=input_ids, attention_mask=attention_mask)
    metrics = tracker.metrics()
    tracker.remove()

    assert metrics["trigger_routing_tokens"] == 2
    assert 0.0 <= metrics["trigger_target_probability"] <= 1.0
    assert 0.0 <= metrics["trigger_target_hit_rate"] <= 1.0
    assert 0.0 <= metrics["trigger_target_execution_rate"] <= 1.0


def test_olmoe_strategy_aggregates_sparse_expert_slices(tmp_path) -> None:
    model_config = _tiny_config()
    model_config.save_pretrained(tmp_path)
    model = OlmoeForSequenceClassification(model_config)
    names = list(model.state_dict().keys())
    initial_arrays = _to_numpy_ndarrays(model, names)
    initial_parameters = ndarrays_to_parameters(initial_arrays)
    runtime_config = replace(
        _runtime_config(str(tmp_path), str(tmp_path / "output")),
        measure_local_asr=True,
        lfe_target_layer="decoder.0",
        lfe_target_expert=0,
        malicious_clients=(0,),
    )
    strategy = MixFedMoEStrategy(
        runtime_config=runtime_config,
        parameter_names=names,
        initial_parameters=initial_parameters,
    )
    clients = [_DummyClient("0"), _DummyClient("1")]
    fit_instructions = strategy.configure_fit(
        server_round=1,
        parameters=initial_parameters,
        client_manager=_DummyManager(clients),  # type: ignore[arg-type]
    )

    name_to_idx = {name: idx for idx, name in enumerate(names)}
    expected_slices: Dict[tuple[str, int, str], np.ndarray] = {}
    expected_router_rows: Dict[tuple[str, int], np.ndarray] = {}
    results = []
    for client_idx, (client, fit_ins) in enumerate(fit_instructions):
        assignments = json.loads(str(fit_ins.config["assigned_experts_json"]))
        returned_names: List[str] = []
        returned_arrays: List[np.ndarray] = []
        for layer_id, expert_ids in assignments.items():
            layer_idx = int(layer_id.split(".", 1)[1])
            for expert_id in expert_ids:
                for kind in ("gate_up_proj", "down_proj"):
                    packed_name = f"model.layers.{layer_idx}.mlp.experts.{kind}"
                    value = initial_arrays[name_to_idx[packed_name]][expert_id].copy()
                    value += np.float32(1.0 + client_idx + expert_id)
                    sparse_name = (
                        f"__olmoe_expert__.{layer_id}.{expert_id}.{kind}"
                    )
                    returned_names.append(sparse_name)
                    returned_arrays.append(value)
                    expected_slices[(layer_id, expert_id, kind)] = value

            router_name = f"model.layers.{layer_idx}.mlp.gate.weight"
            router = initial_arrays[name_to_idx[router_name]].copy()
            for expert_id in expert_ids:
                router[expert_id] += np.float32(10.0 + client_idx)
                expected_router_rows[(layer_id, expert_id)] = router[expert_id].copy()
            returned_names.append(router_name)
            returned_arrays.append(router)

        metrics = {
            "mode": "mix",
            "assigned_experts_json": fit_ins.config["assigned_experts_json"],
            "returned_parameter_names_json": json.dumps(returned_names),
        }
        if 0 in assignments["decoder.0"]:
            metrics.update(
                {
                    "client_id": client_idx,
                    "is_malicious": True,
                    "attack_active": True,
                    "local_asr_pre": 0.1,
                    "local_asr_post": 0.8,
                    "local_asr_gain": 0.7,
                    "local_asr_eval_examples": 4,
                }
            )
        fit_res = FitRes(
            status=Status(code=Code.OK, message="ok"),
            parameters=ndarrays_to_parameters(returned_arrays),
            num_examples=1,
            metrics=metrics,
        )
        results.append((client, fit_res))

    aggregated_parameters, aggregate_metrics = strategy.aggregate_fit(
        server_round=1,
        results=results,  # type: ignore[arg-type]
        failures=[],
    )
    assert aggregated_parameters is not None
    aggregated = parameters_to_ndarrays(aggregated_parameters)

    for (layer_id, expert_id, kind), expected in expected_slices.items():
        layer_idx = int(layer_id.split(".", 1)[1])
        packed_name = f"model.layers.{layer_idx}.mlp.experts.{kind}"
        actual = aggregated[name_to_idx[packed_name]][expert_id]
        np.testing.assert_allclose(actual, expected)
    for (layer_id, expert_id), expected in expected_router_rows.items():
        layer_idx = int(layer_id.split(".", 1)[1])
        router_name = f"model.layers.{layer_idx}.mlp.gate.weight"
        actual = aggregated[name_to_idx[router_name]][expert_id]
        np.testing.assert_allclose(actual, expected)
    assert float(aggregate_metrics["target_expert_local_update_l2"]) > 0.0
    assert float(aggregate_metrics["target_expert_global_update_l2"]) > 0.0
    assert int(aggregate_metrics["target_expert_contributors"]) == 1
