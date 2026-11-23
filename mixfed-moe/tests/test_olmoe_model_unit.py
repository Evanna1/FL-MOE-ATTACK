from __future__ import annotations

import torch
from transformers import OlmoeConfig, OlmoeForCausalLM

from mixfedmoe_fl.model_factory import (
    build_sequence_classification_model,
    freeze_for_classification_head_training,
    get_model_backend,
)
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
        id2label={0: "zero", 1: "one", 2: "two"},
        label2id={"zero": 0, "one": 1, "two": 2},
    )


def _batch() -> dict[str, torch.Tensor]:
    input_ids = torch.tensor(
        [
            [3, 4, 2, 0],
            [5, 6, 7, 2],
        ]
    )
    return {
        "input_ids": input_ids,
        "attention_mask": input_ids.ne(0).long(),
        "labels": torch.tensor([1, 2]),
    }


def test_olmoe_classifier_forward_returns_loss_logits_and_topk_routes() -> None:
    model = OlmoeForSequenceClassification(_tiny_config())

    outputs = model(**_batch(), output_router_logits=True)

    assert outputs.loss is not None
    assert outputs.loss.ndim == 0
    assert outputs.logits.shape == (2, 3)
    assert outputs.router_logits is not None
    assert len(outputs.router_logits) == 2
    assert outputs.router_logits[0].shape == (8, 4)
    assert outputs.aux_loss is not None


def test_olmoe_head_only_training_updates_head_but_not_backbone() -> None:
    model = OlmoeForSequenceClassification(_tiny_config())
    trainable = freeze_for_classification_head_training(model)
    optimizer = torch.optim.AdamW(trainable, lr=0.05)
    head_before = model.classification_head.out_proj.weight.detach().clone()
    backbone_before = model.model.embed_tokens.weight.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    outputs = model(**_batch())
    assert outputs.loss is not None
    outputs.loss.backward()
    optimizer.step()

    assert not torch.equal(model.classification_head.out_proj.weight, head_before)
    assert torch.equal(model.model.embed_tokens.weight, backbone_before)
    assert model.model.embed_tokens.weight.grad is None


def test_model_factory_loads_a_local_olmoe_checkpoint(tmp_path) -> None:
    # Match the public checkpoint layout: model.* backbone weights plus lm_head.
    source = OlmoeForCausalLM(_tiny_config())
    source.save_pretrained(tmp_path)

    class TokenizerStub:
        pad_token_id = 0
        eos_token_id = 2

    model = build_sequence_classification_model(
        str(tmp_path),
        num_labels=3,
        id2label={0: "zero", 1: "one", 2: "two"},
        label2id={"zero": 0, "one": 1, "two": 2},
        tokenizer=TokenizerStub(),  # type: ignore[arg-type]
        model_backend="auto",
        model_dtype="float32",
    )

    assert isinstance(model, OlmoeForSequenceClassification)
    assert get_model_backend(model) == "olmoe"
    assert model.config.use_cache is False
    assert model(**_batch()).logits.shape == (2, 3)
