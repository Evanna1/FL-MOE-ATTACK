from __future__ import annotations

import torch
import pytest

from mixfedmoe_fl.client import restrict_to_target_expert_and_input_embeddings
from mixfedmoe_fl.config import parse_config


class _TinyMlp(torch.nn.Module):
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


class _TinyLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.is_sparse = True
        self.mlp = _TinyMlp()


class _TinyBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.ModuleList([_TinyLayer()])


class _TinyStack(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = torch.nn.ModuleList([_TinyBlock()])


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = torch.nn.Embedding(16, 2)
        self.encoder = _TinyStack()
        self.decoder = _TinyStack()
        self.classification_head = torch.nn.Linear(2, 2)

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.shared


def test_restricts_training_to_target_expert_and_input_embeddings() -> None:
    model = _TinyModel()

    expert_tensors, embedding_tensors = restrict_to_target_expert_and_input_embeddings(
        model,
        target_layer="encoder.0",
        target_expert=1,
        mode="mix",
        assignments={"encoder.0": [1], "decoder.0": [0]},
    )

    assert expert_tensors > 0
    assert embedding_tensors > 0
    assert all(parameter.requires_grad for parameter in model.shared.parameters())
    assert all(
        parameter.requires_grad
        for parameter in model.encoder.block[0].layer[0].mlp.experts["expert_1"].parameters()
    )
    assert not any(
        parameter.requires_grad
        for parameter in model.encoder.block[0].layer[0].mlp.experts["expert_0"].parameters()
    )
    assert not any(parameter.requires_grad for parameter in model.encoder.block[0].layer[0].mlp.router.parameters())
    assert not any(parameter.requires_grad for parameter in model.decoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.classification_head.parameters())


def test_drop_mode_maps_global_target_expert_to_local_index() -> None:
    model = _TinyModel()
    # Simulate retaining global experts [0, 2], with global expert 2 becoming local expert 1.
    first = model.encoder.block[0].layer[0].mlp.experts["expert_0"]
    target = model.encoder.block[0].layer[0].mlp.experts["expert_2"]
    model.encoder.block[0].layer[0].mlp.experts = torch.nn.ModuleDict(
        {"expert_0": first, "expert_1": target}
    )
    model.encoder.block[0].layer[0].mlp.router = torch.nn.Linear(2, 2)

    restrict_to_target_expert_and_input_embeddings(
        model,
        target_layer="encoder.0",
        target_expert=2,
        mode="drop",
        assignments={"encoder.0": [0, 2], "decoder.0": [0, 1]},
    )

    assert all(parameter.requires_grad for parameter in target.parameters())
    assert not any(parameter.requires_grad for parameter in first.parameters())
    assert all(parameter.requires_grad for parameter in model.shared.parameters())
    assert not any(parameter.requires_grad for parameter in model.encoder.block[0].layer[0].mlp.router.parameters())


def test_embedding_training_scope_has_an_explicit_config_switch() -> None:
    config = parse_config(
        [
            "--mode",
            "mix",
            "--attack_enabled",
            "--inject_target_expert",
            "--malicious_train_embedding",
            "--lfe_target_layer",
            "encoder.0",
            "--lfe_target_expert",
            "1",
        ]
    )

    assert config.malicious_train_embedding is True


def test_embedding_training_requires_localized_target_expert_injection() -> None:
    with pytest.raises(ValueError, match="requires --inject_target_expert"):
        parse_config(["--mode", "mix", "--attack_enabled", "--malicious_train_embedding"])


def test_delta_scale_reads_trigger_ids_and_implies_embedding_training(tmp_path) -> None:
    path = tmp_path / "selected_triggers.json"
    path.write_text(
        '{"target":{"layer_id":"encoder.0","expert_id":1},'
        '"best_trigger":{"trigger":"older WWII","trigger_token_ids":[2749,29420]}}',
        encoding="utf-8",
    )
    args = ["--mode", "mix", "--attack_enabled", "--inject_target_expert",
            "--malicious_embedding_delta_scale", "8"]
    config = parse_config(args + ["--selected_triggers_path", str(path)])
    assert config.malicious_train_embedding is True
    assert config.trigger_token_ids == (2749, 29420)

    with pytest.raises(ValueError, match="selected_triggers_path"):
        parse_config(args + ["--lfe_target_layer", "encoder.0", "--lfe_target_expert", "1"])
