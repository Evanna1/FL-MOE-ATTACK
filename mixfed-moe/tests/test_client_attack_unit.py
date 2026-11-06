from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD

from mixfedmoe_fl.client import (
    MixFedMoEClient,
    MoeHandle,
    evaluate_targeted_asr,
    restrict_malicious_updates_to_target_expert_and_router,
)
from mixfedmoe_fl.config import parse_config


class _ToyMlp(nn.Module):
    def __init__(self, num_experts: int) -> None:
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(3, 3) for _ in range(num_experts)])
        self.gate = nn.Linear(3, num_experts)


class _ToyModel(nn.Module):
    def __init__(self, num_experts: int) -> None:
        super().__init__()
        self.mlp = _ToyMlp(num_experts)
        self.embedding = nn.Embedding(8, 3)
        self.classification_head = nn.Linear(3, 2)


class _ASRModel(nn.Module):
    def forward(self, input_ids, attention_mask, **_kwargs):
        del attention_mask
        positive = input_ids[:, 0].eq(1).float()
        logits = torch.stack((1.0 - positive, positive), dim=-1)
        return SimpleNamespace(logits=logits)


def _asr_collator(rows):
    return {
        "input_ids": torch.stack([row["input_ids"] for row in rows]),
        "attention_mask": torch.stack([row["attention_mask"] for row in rows]),
    }


def test_evaluate_targeted_asr_uses_target_predictions_and_restores_train_mode() -> None:
    model = _ASRModel()
    model.train()
    dataset = [
        {
            "input_ids": torch.tensor([value]),
            "attention_mask": torch.tensor([1]),
        }
        for value in (1, 0, 1)
    ]

    asr, examples = evaluate_targeted_asr(
        model=model,
        dataset=dataset,
        collator=_asr_collator,
        batch_size=2,
        target_label=1,
        device=torch.device("cpu"),
    )

    assert asr == pytest.approx(2 / 3)
    assert examples == 3
    assert model.training is True


def _toy_handle(model: _ToyModel) -> MoeHandle:
    return MoeHandle(
        layer_id="encoder.0",
        stack_name="encoder",
        layer_idx=0,
        layer_path="mlp",
        layer_module=model.mlp,
        ff_module=model.mlp,
        mlp_module=model.mlp,
        experts_attr="experts",
        gate_linear_path="gate",
    )


def test_attack_activation_requires_enabled_malicious_client_and_start_round() -> None:
    cfg = parse_config(
        [
            "--mode",
            "full",
            "--num_clients",
            "2",
            "--attack_enabled",
            "--malicious_clients",
            "0",
            "--attack_start_round",
            "3",
        ]
    )
    malicious = MixFedMoEClient(0, cfg, object(), is_malicious=True)  # type: ignore[arg-type]
    benign = MixFedMoEClient(1, cfg, object(), is_malicious=False)  # type: ignore[arg-type]

    assert malicious._is_attack_active(2) is False
    assert malicious._is_attack_active(3) is True
    assert malicious._is_attack_active(3, {"attack_enabled": False}) is False
    assert benign._is_attack_active(3) is False


def test_attack_disabled_keeps_all_clients_clean() -> None:
    cfg = parse_config(["--mode", "full", "--num_clients", "2"])
    client = MixFedMoEClient(0, cfg, object(), is_malicious=False)  # type: ignore[arg-type]
    assert client._is_attack_active(1) is False


def test_monitor_asr_keeps_even_configured_malicious_client_clean() -> None:
    cfg = parse_config(
        ["--mode", "full", "--num_clients", "2", "--monitor_asr", "--malicious_clients", "0"]
    )
    client = MixFedMoEClient(0, cfg, object(), is_malicious=True)  # type: ignore[arg-type]

    assert cfg.attack_enabled is False
    assert client._is_attack_active(1) is False
    assert client._is_attack_active(100) is False


def test_target_expert_router_only_freezes_every_other_parameter() -> None:
    model = _ToyModel(num_experts=3)
    trained_target = restrict_malicious_updates_to_target_expert_and_router(
        model=model,
        moe_handles=[_toy_handle(model)],
        target_layer="encoder.0",
        target_expert=1,
        mode="mix",
        assignments={"encoder.0": [0, 1]},
    )

    assert trained_target is True
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable == {
        "mlp.experts.1.weight",
        "mlp.experts.1.bias",
        "mlp.gate.weight",
        "mlp.gate.bias",
    }

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    optimizer = SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.1,
    )
    loss = sum(parameter.sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    changed = {
        name
        for name, parameter in model.named_parameters()
        if not parameter.detach().equal(before[name])
    }
    assert changed == trainable


def test_target_expert_router_only_trains_only_router_when_target_unassigned() -> None:
    model = _ToyModel(num_experts=3)
    trained_target = restrict_malicious_updates_to_target_expert_and_router(
        model=model,
        moe_handles=[_toy_handle(model)],
        target_layer="encoder.0",
        target_expert=2,
        mode="mix",
        assignments={"encoder.0": [0, 1]},
    )

    assert trained_target is False
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable == {"mlp.gate.weight", "mlp.gate.bias"}


def test_target_expert_router_only_maps_global_target_in_drop_mode() -> None:
    model = _ToyModel(num_experts=2)
    trained_target = restrict_malicious_updates_to_target_expert_and_router(
        model=model,
        moe_handles=[_toy_handle(model)],
        target_layer="encoder.0",
        target_expert=5,
        mode="drop",
        assignments={"encoder.0": [2, 5]},
    )

    assert trained_target is True
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable == {
        "mlp.experts.1.weight",
        "mlp.experts.1.bias",
        "mlp.gate.weight",
        "mlp.gate.bias",
    }


def test_target_expert_router_only_cli_requires_attack_and_target() -> None:
    cfg = parse_config(
        [
            "--mode",
            "full",
            "--attack_enabled",
            "--malicious_target_expert_router_only",
            "--lfe_target_layer",
            "encoder.0",
            "--lfe_target_expert",
            "1",
        ]
    )

    assert cfg.malicious_target_expert_router_only is True
