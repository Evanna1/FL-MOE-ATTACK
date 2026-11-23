from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import OlmoeConfig, OlmoeModel, OlmoePreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.models.olmoe.modeling_olmoe import load_balancing_loss_func


@dataclass
class OlmoeSequenceClassifierOutput(SequenceClassifierOutputWithPast):
    """Sequence-classification output with OLMoE routing diagnostics."""

    aux_loss: torch.FloatTensor | None = None
    router_logits: tuple[torch.FloatTensor, ...] | None = None


class OlmoeClassificationHead(nn.Module):
    """Last-token classification head used on top of the decoder-only backbone."""

    def __init__(self, config: OlmoeConfig):
        super().__init__()
        dropout_p = getattr(config, "classifier_dropout", None)
        if dropout_p is None:
            dropout_p = getattr(config, "attention_dropout", 0.0)
        self.dropout = nn.Dropout(float(dropout_p))
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The OLMoE backbone normally stays in BF16 on an A800 while the small
        # classification head can be trained in FP32.
        hidden_states = hidden_states.to(self.out_proj.weight.dtype)
        return self.out_proj(self.dropout(hidden_states))


class OlmoeForSequenceClassification(OlmoePreTrainedModel):
    """OLMoE backbone with a sentence-level classification head.

    The representation is taken from the final non-padding token, which works
    with both left- and right-padded batches.  The project data pipeline appends
    EOS, so in normal training this is the final EOS representation.
    """

    _keys_to_ignore_on_load_unexpected = [r"lm_head.weight"]

    def __init__(self, config: OlmoeConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = OlmoeModel(config)
        self.classification_head = OlmoeClassificationHead(config)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, new_embeddings: nn.Module) -> None:
        self.model.embed_tokens = new_embeddings

    def get_output_embeddings(self) -> None:
        # This is a classifier, not a causal language model.
        return None

    @staticmethod
    def _last_non_padding_indices(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.LongTensor:
        batch_size, sequence_length, _ = hidden_states.shape
        if attention_mask is None:
            return torch.full(
                (batch_size,),
                sequence_length - 1,
                dtype=torch.long,
                device=hidden_states.device,
            )
        if attention_mask.ndim != 2 or tuple(attention_mask.shape) != (
            batch_size,
            sequence_length,
        ):
            raise ValueError(
                "attention_mask must have shape [batch_size, sequence_length], "
                f"got {tuple(attention_mask.shape)}."
            )
        valid = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
        if not bool(valid.any(dim=1).all()):
            raise ValueError("Every sequence must contain at least one non-padding token.")
        positions = torch.arange(sequence_length, device=hidden_states.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        return positions.masked_fill(~valid, -1).max(dim=1).values

    def _classification_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.problem_type is None:
            if self.num_labels == 1:
                self.config.problem_type = "regression"
            elif self.num_labels > 1 and labels.dtype in (torch.long, torch.int):
                self.config.problem_type = "single_label_classification"
            else:
                self.config.problem_type = "multi_label_classification"

        if self.config.problem_type == "regression":
            loss_fct = MSELoss()
            if self.num_labels == 1:
                return loss_fct(logits.squeeze(), labels.squeeze().to(logits.dtype))
            return loss_fct(logits, labels.to(logits.dtype))
        if self.config.problem_type == "single_label_classification":
            return CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))
        return BCEWithLogitsLoss()(logits, labels.to(logits.dtype))

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> tuple | OlmoeSequenceClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )
        if labels is not None:
            use_cache = False

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        last_indices = self._last_non_padding_indices(hidden_states, attention_mask)
        batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        sentence_representation = hidden_states[batch_indices, last_indices]
        logits = self.classification_head(sentence_representation)

        loss = self._classification_loss(logits, labels) if labels is not None else None
        router_logits = getattr(outputs, "router_logits", None)
        aux_loss = None
        if output_router_logits and router_logits:
            aux_loss = load_balancing_loss_func(
                router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if loss is not None and isinstance(aux_loss, torch.Tensor):
                loss = loss + self.router_aux_loss_coef * aux_loss.to(loss.device)

        if not return_dict:
            result = (
                logits,
                outputs.past_key_values,
                outputs.hidden_states,
                outputs.attentions,
                router_logits,
            )
            if aux_loss is not None:
                result = (aux_loss,) + result
            return ((loss,) + result) if loss is not None else result

        return OlmoeSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            aux_loss=aux_loss,
            router_logits=router_logits,
        )
