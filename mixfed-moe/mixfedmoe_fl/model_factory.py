from __future__ import annotations

from typing import Literal, Mapping, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, PreTrainedModel, PreTrainedTokenizerBase
from transformers.configuration_utils import PretrainedConfig

from models.olmoe import OlmoeForSequenceClassification
from models.granite import GraniteMoeForSequenceClassification
from models.switch_transformers import SwitchTransformersForSequenceClassification

ModelBackend = Literal["auto", "switch", "olmoe", "granite"]
ModelDType = Literal["auto", "float32", "bfloat16", "float16"]

SUPPORTED_MODEL_BACKENDS = ("auto", "switch", "olmoe", "granite")
SUPPORTED_MODEL_DTYPES = ("auto", "float32", "bfloat16", "float16")

# Decoder-only, packed-expert MoE backends.  Unlike the encoder-decoder Switch
# backend, these store every expert in packed tensors, place the router inside
# the decoder MoE block, and default to BF16.  Client, strategy, server and
# trigger code share a single handling path for them, so the membership test is
# centralised here instead of repeating the ``{"olmoe", "granite"}`` literal.
DECODER_ONLY_MOE_BACKENDS = frozenset({"olmoe", "granite"})


def resolve_model_backend(
    config: PretrainedConfig,
    requested_backend: ModelBackend = "auto",
) -> Literal["switch", "olmoe", "granite"]:
    model_type = str(getattr(config, "model_type", "")).lower()
    detected: Optional[Literal["switch", "olmoe", "granite"]]
    if model_type == "olmoe":
        detected = "olmoe"
    elif model_type in {"switch_transformers", "switchtransformers"}:
        detected = "switch"
    elif model_type in {"granitemoe", "granite_moe"}:
        detected = "granite"
    else:
        detected = None

    if requested_backend == "auto":
        if detected is None:
            raise ValueError(
                f"Unsupported model_type={model_type!r}. This stage supports Switch Transformers "
                ", OLMoE, and Granite-MoE checkpoints only."
            )
        return detected
    if requested_backend not in {"switch", "olmoe", "granite"}:
        raise ValueError(f"Unsupported model backend: {requested_backend!r}.")
    if detected is not None and requested_backend != detected:
        raise ValueError(
            f"Requested model_backend={requested_backend!r}, but checkpoint config reports "
            f"model_type={model_type!r} ({detected})."
        )
    return requested_backend


def resolve_model_dtype(
    backend: Literal["switch", "olmoe", "granite"],
    requested_dtype: ModelDType = "auto",
) -> torch.dtype:
    if requested_dtype == "auto":
        return torch.bfloat16 if backend in DECODER_ONLY_MOE_BACKENDS else torch.float32
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    try:
        return mapping[requested_dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported model dtype: {requested_dtype!r}.") from exc


def configure_model_special_tokens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if model.config.eos_token_id is None:
        model.config.eos_token_id = tokenizer.eos_token_id
    if (
        getattr(model.config, "is_encoder_decoder", False)
        and getattr(model.config, "decoder_start_token_id", None) is None
    ):
        model.config.decoder_start_token_id = model.config.pad_token_id
    model.config.use_cache = False


def build_sequence_classification_model(
    model_name_or_path: str,
    *,
    num_labels: int,
    id2label: Mapping[int, str],
    label2id: Mapping[str, int],
    tokenizer: PreTrainedTokenizerBase | None = None,
    model_backend: ModelBackend = "auto",
    model_dtype: ModelDType = "auto",
    ignore_mismatched_sizes: bool = True,
    local_files_only: bool = False,
    router_aux_loss_coef: Optional[float] = None,
) -> PreTrainedModel:
    """Build a Switch, OLMoE, or Granite-MoE sequence classifier."""

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        local_files_only=local_files_only,
    )
    backend = resolve_model_backend(config, model_backend)
    dtype = resolve_model_dtype(backend, model_dtype)

    config.num_labels = int(num_labels)
    config.id2label = {int(key): str(value) for key, value in id2label.items()}
    config.label2id = {str(key): int(value) for key, value in label2id.items()}
    config.tie_word_embeddings = False
    config.use_cache = False
    # Global override of the router load-balancing (auxiliary) loss coefficient.
    # None keeps the checkpoint's value; 0.0 disables load balancing entirely so
    # the caller can test whether routing collapses without it.
    if router_aux_loss_coef is not None:
        config.router_aux_loss_coef = float(router_aux_loss_coef)

    model_class = {
        "switch": SwitchTransformersForSequenceClassification,
        "olmoe": OlmoeForSequenceClassification,
        "granite": GraniteMoeForSequenceClassification,
    }[backend]
    model = model_class.from_pretrained(
        model_name_or_path,
        config=config,
        ignore_mismatched_sizes=ignore_mismatched_sizes,
        dtype=dtype,
        local_files_only=local_files_only,
    )
    if backend == "olmoe":
        # Selected-expert training still backpropagates through every decoder
        # layer. Checkpointing keeps activation memory practical on one A800.
        model.gradient_checkpointing_enable()
    elif backend == "granite":
        model.gradient_checkpointing_enable()
    if tokenizer is not None:
        configure_model_special_tokens(model, tokenizer)
    setattr(model, "_mixfedmoe_backend", backend)
    return model


def get_model_backend(model: nn.Module) -> Literal["switch", "olmoe", "granite"]:
    explicit = getattr(model, "_mixfedmoe_backend", None)
    if explicit in {"switch", "olmoe", "granite"}:
        return explicit
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError(f"Model {type(model).__name__} does not expose a Hugging Face config.")
    return resolve_model_backend(config, "auto")


def freeze_for_classification_head_training(model: nn.Module) -> list[nn.Parameter]:
    """Freeze the backbone and return only classification-head parameters."""

    model.requires_grad_(False)
    head = getattr(model, "classification_head", None)
    if not isinstance(head, nn.Module):
        raise ValueError(
            f"Model {type(model).__name__} does not expose a classification_head module."
        )
    head.to(torch.float32)
    head.requires_grad_(True)
    return [parameter for parameter in head.parameters() if parameter.requires_grad]
