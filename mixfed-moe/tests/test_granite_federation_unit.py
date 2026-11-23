import json
from dataclasses import replace
from typing import Dict, List, Sequence

import numpy as np
import torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from transformers import GraniteMoeConfig

from mixfedmoe_fl.client import _to_numpy_ndarrays, apply_mode_mix, collect_moe_handles
from mixfedmoe_fl.client import _build_trigger_router_loss, restrict_malicious_updates_to_target_expert_and_router
from mixfedmoe_fl.a3fl import A3FLSearchConfig, A3FLTriggerOptimizer
from mixfedmoe_fl.config import MixFedMoEConfig
from mixfedmoe_fl.expert_profiler import TrainingLFERoutingTracker
from mixfedmoe_fl.granite_adapter import (
    GraniteAdapter,
    granite_probe_parameter_views,
    parse_granite_probe_parameter,
)
from mixfedmoe_fl.expert_probe import (
    observe_global_and_update_history,
    rank_lazy_expert_candidates,
)
from mixfedmoe_fl.igcg_trigger_optimization import IGCGTriggerConfig, IGCGTriggerOptimizer
from mixfedmoe_fl.strategy import MixFedMoEStrategy
from models.granite import GraniteMoeForSequenceClassification


def _config() -> GraniteMoeConfig:
    return GraniteMoeConfig(
        vocab_size=32, hidden_size=16, intermediate_size=8,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        num_local_experts=8, num_experts_per_tok=2,
        max_position_embeddings=32, pad_token_id=0, eos_token_id=0,
        num_labels=3,
    )


def _batch() -> Dict[str, torch.Tensor]:
    ids = torch.tensor([[3, 4, 0, 0], [5, 6, 7, 0]])
    return {"input_ids": ids, "attention_mask": ids.ne(0).long(), "labels": torch.tensor([1, 2])}


def test_granite_mix_trains_selected_expert_rows_and_counts_topk() -> None:
    model = GraniteMoeForSequenceClassification(_config())
    model.model.to(torch.bfloat16)
    model.classification_head.to(torch.float32)
    handles = collect_moe_handles(model)
    assert len(handles) == 2
    assert all(handle.num_experts == 8 and handle.router_top_k == 2 for handle in handles)
    assignments = {handle.layer_id: [0, 3] for handle in handles}
    before = model.model.layers[0].block_sparse_moe.input_linear.weight[1].detach().clone()
    apply_mode_mix(model, handles, assignments)
    tracker = TrainingLFERoutingTracker(handles, top_k=8)
    tracker.begin_batch(_batch()["attention_mask"])
    model(**_batch())
    profile = tracker.finish()
    for layer in profile["layers"].values():
        assert layer["router_top_k"] == 2
        assert sum(layer["counts"]) == 2 * layer["valid_tokens"]
    names = GraniteAdapter.sparse_upload_names(model, handles, assignments)
    selected_before = model.model.layers[0].block_sparse_moe.input_linear.selected_weight[0].detach().clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.05)
    optimizer.zero_grad(set_to_none=True)
    loss = model(**_batch()).loss
    assert loss is not None
    loss.backward()
    optimizer.step()
    selected = model.model.layers[0].block_sparse_moe.input_linear.selected_weight[0]
    assert not torch.equal(selected, selected_before)
    assert torch.equal(model.model.layers[0].block_sparse_moe.input_linear.weight[1], before)
    assert "model.layers.0.block_sparse_moe.input_linear.weight" not in names
    assert "__granite_expert__.decoder.0.0.input_linear" in names
    arrays = _to_numpy_ndarrays(model, names)
    idx = names.index("__granite_expert__.decoder.0.0.input_linear")
    assert arrays[idx].shape == (16, 16)


def test_granite_probe_expands_packed_experts_for_generic_tracker() -> None:
    names = [
        "model.layers.1.block_sparse_moe.input_linear.weight",
        "model.layers.1.block_sparse_moe.output_linear.weight",
    ]
    arrays = [
        np.zeros((4, 3, 2), dtype=np.float32),
        np.zeros((4, 2, 3), dtype=np.float32),
    ]
    probe_names, probe_arrays = granite_probe_parameter_views(names, arrays)
    assert len(probe_names) == 8
    assert parse_granite_probe_parameter(probe_names[0]) == ("decoder.1", 0)
    state = {"trackers": {}, "pending": [], "results": {}, "attempts": {}}
    observe_global_and_update_history(
        state=state,
        parameter_names=probe_names,
        parameters=probe_arrays,
        server_round=1,
        target_layer=None,
        candidate_pool=4,
        repetitions=2,
        automatic_tensor_suffix="",
        parameter_parser=parse_granite_probe_parameter,
    )
    observe_global_and_update_history(
        state=state,
        parameter_names=probe_names,
        parameters=[array + 0.01 for array in probe_arrays],
        server_round=2,
        target_layer=None,
        candidate_pool=4,
        repetitions=2,
        automatic_tensor_suffix="",
        parameter_parser=parse_granite_probe_parameter,
    )
    candidates = rank_lazy_expert_candidates(
        state=state,
        target_layer="decoder.1",
        num_experts=4,
        warmup_observations=1,
        pairs=1,
        parameter_parser=parse_granite_probe_parameter,
    )
    assert len(candidates) == 4


class _Client:
    def __init__(self, cid: str):
        self.cid = cid


class _Manager:
    def __init__(self, clients: Sequence[_Client]):
        self.clients = list(clients)
    def num_available(self) -> int:
        return len(self.clients)
    def sample(self, num_clients: int, min_num_clients: int) -> List[_Client]:
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


def test_granite_strategy_aggregates_expert_slices(tmp_path) -> None:
    cfg = _config()
    cfg.save_pretrained(tmp_path)
    model = GraniteMoeForSequenceClassification(cfg)
    names = list(model.state_dict().keys())
    initial = _to_numpy_ndarrays(model, names)
    parameters = ndarrays_to_parameters(initial)
    runtime = MixFedMoEConfig(
        mode="mix", model_name_or_path=str(tmp_path), model_backend="granite", model_dtype="float32",
        dataset_name="ag_news", num_clients=2, num_rounds=1, k=2, alpha=0.5,
        fraction_fit=1.0, fraction_evaluate=0.0, local_epochs=1.0, learning_rate=1e-3,
        weight_decay=0.0, train_batch_size=2, eval_batch_size=2, max_length=16,
        client_eval_ratio=0.0, seed=7, num_cpus_per_client=1, num_gpus_per_client=0.0,
        output_dir=str(tmp_path / "out"), test_samples=4,
    )
    strategy = MixFedMoEStrategy(runtime_config=runtime, parameter_names=names, initial_parameters=parameters)
    clients = [_Client(str(idx)) for idx in range(4)]
    instructions = strategy.configure_fit(1, parameters, _Manager(clients))
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    results = []
    expected: Dict[tuple[str, int, str], np.ndarray] = {}
    for client_idx, (client, fit_ins) in enumerate(instructions):
        assignments = json.loads(str(fit_ins.config["assigned_experts_json"]))
        returned_names: List[str] = []
        returned_arrays: List[np.ndarray] = []
        for layer_id, expert_ids in assignments.items():
            layer_idx = int(layer_id.split(".", 1)[1])
            for expert_id in expert_ids:
                for kind in ("input_linear", "output_linear"):
                    packed = f"model.layers.{layer_idx}.block_sparse_moe.{kind}.weight"
                    value = initial[name_to_idx[packed]][expert_id].copy()
                    value += np.float32(expert_id + 1)
                    sparse = f"__granite_expert__.{layer_id}.{expert_id}.{kind}"
                    returned_names.append(sparse)
                    returned_arrays.append(value)
                    expected[(layer_id, expert_id, kind)] = value
            router_name = f"model.layers.{layer_idx}.block_sparse_moe.router.layer.weight"
            returned_names.append(router_name)
            returned_arrays.append(initial[name_to_idx[router_name]].copy())
        metrics = {
            "mode": "mix", "assigned_experts_json": fit_ins.config["assigned_experts_json"],
            "returned_parameter_names_json": json.dumps(returned_names),
        }
        results.append((client, FitRes(status=Status(code=Code.OK, message="ok"), parameters=ndarrays_to_parameters(returned_arrays), num_examples=1, metrics=metrics)))
    aggregated, _ = strategy.aggregate_fit(1, results, [])
    assert aggregated is not None
    arrays = parameters_to_ndarrays(aggregated)
    for (layer_id, expert_id, kind), value in expected.items():
        packed = f"model.layers.{int(layer_id.split('.', 1)[1])}.block_sparse_moe.{kind}.weight"
        np.testing.assert_allclose(arrays[name_to_idx[packed]][expert_id], value)


def test_granite_target_expert_router_loss_and_restriction() -> None:
    model = GraniteMoeForSequenceClassification(_config())
    handles = collect_moe_handles(model)
    assignments = {handle.layer_id: [0, 3] for handle in handles}
    apply_mode_mix(model, handles, assignments)

    objective = _build_trigger_router_loss(
        moe_handles=handles,
        target_layer="decoder.0",
        target_experts=[3],
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
    assert model.model.layers[0].block_sparse_moe.router.layer.weight.grad is not None

    model = GraniteMoeForSequenceClassification(_config())
    model.model.to(torch.bfloat16)
    model.classification_head.to(torch.float32)
    handles = collect_moe_handles(model)
    assignments = {handle.layer_id: [0, 3] for handle in handles}
    apply_mode_mix(model, handles, assignments)
    assert restrict_malicious_updates_to_target_expert_and_router(
        model=model,
        moe_handles=handles,
        target_layer="decoder.0",
        target_expert=3,
        mode="mix",
        assignments=assignments,
    )
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert "model.layers.0.block_sparse_moe.router.layer.weight" in trainable
    assert "model.layers.0.block_sparse_moe.input_linear.selected_weight.1" in trainable


def test_granite_igcg_and_a3fl_use_decoder_embeddings(tmp_path) -> None:
    model = GraniteMoeForSequenceClassification(_config())
    tokenizer = _TinyTokenizer()
    checkpoint = str(tmp_path / "granite.pt")
    torch.save(model.state_dict(), checkpoint)
    igcg = IGCGTriggerOptimizer(
        model=model,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        config=IGCGTriggerConfig(
            checkpoint_path=checkpoint,
            target_layer="decoder.0",
            target_expert=1,
            trigger_length=1,
            gradient_batch_size=1,
            evaluation_batch_size=1,
            max_length=8,
        ),
        calibration_texts=["alpha"],
        heldout_texts=["beta"],
        device=torch.device("cpu"),
    )
    gradient = igcg.gradient([9])
    assert gradient.shape == (1, model.config.hidden_size)
    assert torch.isfinite(gradient).all()

    pristine = {name: value.detach().clone() for name, value in model.state_dict().items()}

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
    assert any("router.layer.weight" in name for name in optimizer._adversarial_parameter_names)
    result = optimizer.optimize(_batch(), initial_trigger_token_ids=(9,))
    assert len(result.trigger_token_ids) == 1


def test_granite_igcg_jointly_optimizes_multiple_decoder_layers(tmp_path) -> None:
    model = GraniteMoeForSequenceClassification(_config())
    tokenizer = _TinyTokenizer()
    checkpoint = str(tmp_path / "granite_cross_layer.pt")
    torch.save(model.state_dict(), checkpoint)
    optimizer = IGCGTriggerOptimizer(
        model=model,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        config=IGCGTriggerConfig(
            checkpoint_path=checkpoint,
            target_layer="decoder.0",
            target_expert=1,
            target_specs=(("decoder.0", (1,)), ("decoder.1", (2,))),
            trigger_length=1,
            gradient_batch_size=1,
            evaluation_batch_size=1,
            max_length=8,
        ),
        calibration_texts=["alpha"],
        heldout_texts=["beta"],
        device=torch.device("cpu"),
    )
    gradient = optimizer.gradient([9])
    metrics = optimizer.evaluate([9])
    assert gradient.shape == (1, model.config.hidden_size)
    assert torch.isfinite(gradient).all()
    assert 0.0 <= metrics.target_probability <= 1.0
