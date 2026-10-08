from __future__ import annotations

import pytest
import torch

from mixfedmoe_fl.client import (
    MixFedMoEClient,
    MoeHandle,
    TargetRouterTrainingObjective,
    _restrict_injected_training_scope,
)
from mixfedmoe_fl.config import parse_config


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


def test_target_router_objective_updates_only_target_row_and_clips_norm() -> None:
    classifier = torch.nn.Linear(2, 3, bias=True)
    with torch.no_grad():
        classifier.weight.zero_()
        classifier.bias.zero_()
    objective = TargetRouterTrainingObjective(
        classifier=classifier,
        target_expert=1,
        trigger_token_ids=(7, 8),
        margin=4.0,
        max_update_norm=0.05,
    )
    reference_weight = classifier.weight.detach().clone()
    reference_bias = classifier.bias.detach().clone()
    input_ids = torch.tensor([[10, 7, 8, 1], [10, 11, 1, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    hidden_states = torch.tensor(
        [[1.0, 0.0], [1.0, 1.0], [2.0, 1.0], [0.0, 1.0],
         [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]
    )

    objective.begin_batch(input_ids, attention_mask)
    classifier(hidden_states)
    margin_loss, clean_kl, metrics = objective.losses()
    (margin_loss + clean_kl).backward()

    assert metrics["trigger_tokens"] == 2
    assert float(margin_loss.item()) == pytest.approx(4.0)
    assert float(clean_kl.item()) == pytest.approx(0.0, abs=1e-7)
    assert classifier.weight.grad is not None
    assert torch.count_nonzero(classifier.weight.grad[0]).item() == 0
    assert torch.count_nonzero(classifier.weight.grad[1]).item() > 0
    assert torch.count_nonzero(classifier.weight.grad[2]).item() == 0

    with torch.no_grad():
        classifier.weight.add_(1.0)
        classifier.bias.add_(1.0)
    update_norm = objective.project_target_row()

    assert torch.equal(classifier.weight[0], reference_weight[0])
    assert torch.equal(classifier.weight[2], reference_weight[2])
    assert classifier.bias is not None
    assert torch.equal(classifier.bias[0], reference_bias[0])
    assert torch.equal(classifier.bias[2], reference_bias[2])
    assert update_norm == pytest.approx(0.05)
    objective.close()


def test_injected_training_scope_keeps_only_target_expert_and_target_router_trainable() -> None:
    class TinyMlp(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = torch.nn.ModuleDict(
                {
                    "expert_0": torch.nn.Linear(2, 2),
                    "expert_1": torch.nn.Linear(2, 2),
                    "expert_2": torch.nn.Linear(2, 2),
                }
            )
            self.router = torch.nn.Linear(2, 3)

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = TinyMlp()
            self.embedding = torch.nn.Embedding(8, 2)
            self.classifier = torch.nn.Linear(2, 2)

    model = TinyModel()
    handle = MoeHandle(
        layer_id="encoder.0",
        stack_name="encoder",
        layer_idx=0,
        layer_path="mlp",
        layer_module=model.mlp,
        ff_module=model.mlp,
        mlp_module=model.mlp,
        experts_attr="experts",
        gate_linear_path="router",
    )

    router, target_tensors = _restrict_injected_training_scope(
        model=model,
        moe_handles=[handle],
        assignments={"encoder.0": [0, 2]},
        target_layer="encoder.0",
        target_expert=2,
        mode="mix",
    )

    assert router is model.mlp.router
    assert target_tensors > 0
    assert all(param.requires_grad for param in model.mlp.experts["expert_2"].parameters())
    assert all(param.requires_grad for param in model.mlp.router.parameters())
    assert not any(param.requires_grad for param in model.mlp.experts["expert_0"].parameters())
    assert not any(param.requires_grad for param in model.mlp.experts["expert_1"].parameters())
    assert not any(param.requires_grad for param in model.embedding.parameters())
    assert not any(param.requires_grad for param in model.classifier.parameters())
