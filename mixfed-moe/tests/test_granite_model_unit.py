import torch
from transformers import GraniteMoeConfig, GraniteMoeForCausalLM

from mixfedmoe_fl.model_factory import (
    build_sequence_classification_model,
    freeze_for_classification_head_training,
    get_model_backend,
    resolve_model_backend,
)
from models.granite import GraniteMoeForSequenceClassification


def _tiny_config() -> GraniteMoeConfig:
    return GraniteMoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=32,
        pad_token_id=0,
        eos_token_id=0,
        num_labels=3,
    )


def _batch() -> dict[str, torch.Tensor]:
    input_ids = torch.tensor([[3, 4, 0, 0], [5, 6, 7, 0]])
    return {
        "input_ids": input_ids,
        "attention_mask": input_ids.ne(0).long(),
        "labels": torch.tensor([1, 2]),
    }


def test_granite_classifier_forward_and_head_training() -> None:
    model = GraniteMoeForSequenceClassification(_tiny_config())
    outputs = model(**_batch())

    assert outputs.loss is not None
    assert outputs.loss.ndim == 0
    assert outputs.logits.shape == (2, 3)

    trainable = freeze_for_classification_head_training(model)
    optimizer = torch.optim.AdamW(trainable, lr=0.05)
    head_before = model.classification_head.out_proj.weight.detach().clone()
    backbone_before = model.model.embed_tokens.weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    model(**_batch()).loss.backward()  # type: ignore[union-attr]
    optimizer.step()

    assert not torch.equal(model.classification_head.out_proj.weight, head_before)
    assert torch.equal(model.model.embed_tokens.weight, backbone_before)
    assert model.model.embed_tokens.weight.grad is None


def test_granite_classifier_casts_bfloat16_backbone_to_float32_head() -> None:
    model = GraniteMoeForSequenceClassification(_tiny_config())
    model.model.to(torch.bfloat16)
    model.classification_head.to(torch.float32)

    outputs = model(**_batch())

    assert outputs.loss is not None
    assert outputs.logits.dtype == torch.float32


def test_granite_backend_is_detected_without_affecting_existing_backends() -> None:
    config = _tiny_config()
    assert resolve_model_backend(config, "auto") == "granite"
    assert resolve_model_backend(config, "granite") == "granite"
    model = GraniteMoeForSequenceClassification(config)
    assert get_model_backend(model) == "granite"


def test_model_factory_loads_a_local_granite_checkpoint(tmp_path) -> None:
    source = GraniteMoeForCausalLM(_tiny_config())
    source.save_pretrained(tmp_path)

    class TokenizerStub:
        pad_token_id = 0
        eos_token_id = 0

    model = build_sequence_classification_model(
        str(tmp_path),
        num_labels=3,
        id2label={0: "zero", 1: "one", 2: "two"},
        label2id={"zero": 0, "one": 1, "two": 2},
        tokenizer=TokenizerStub(),  # type: ignore[arg-type]
        model_backend="auto",
        model_dtype="float32",
    )

    assert isinstance(model, GraniteMoeForSequenceClassification)
    assert get_model_backend(model) == "granite"
    assert model.config.use_cache is False
    assert model(**_batch()).logits.shape == (2, 3)
