from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.cache_utils import EncoderDecoderCache
from transformers.modeling_outputs import (
    MoEModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from transformers.models.switch_transformers.configuration_switch_transformers import (
    SwitchTransformersConfig,
)
from transformers.models.switch_transformers.modeling_switch_transformers import (
    SwitchTransformersPreTrainedModel,
    SwitchTransformersStack,
    load_balancing_loss_func,
    router_z_loss_func,
)


@dataclass
class Seq2SeqMoESequenceClassifierOutput(Seq2SeqSequenceClassifierOutput):
    encoder_z_loss: torch.FloatTensor | None = None
    encoder_aux_loss: torch.FloatTensor | None = None
    decoder_z_loss: torch.FloatTensor | None = None
    decoder_aux_loss: torch.FloatTensor | None = None
    decoder_router_logits: tuple[torch.FloatTensor, ...] | None = None
    encoder_router_logits: tuple[torch.FloatTensor, ...] | None = None


class SwitchTransformersClassificationHead(nn.Module):
    """T5-style sentence-level classification head."""

    def __init__(self, config: SwitchTransformersConfig):
        super().__init__()
        dropout_p = getattr(config, "classifier_dropout", None)
        if dropout_p is None:
            dropout_p = config.dropout_rate

        self.dropout = nn.Dropout(dropout_p)
        # self.dense = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states = self.dropout(hidden_states)
        # hidden_states = self.dense(hidden_states)
        # hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class SwitchTransformersForSequenceClassification(SwitchTransformersPreTrainedModel):
    _tied_weights_keys = {
        "encoder.embed_tokens.weight": "shared.weight",
        "decoder.embed_tokens.weight": "shared.weight",
    }
    _input_embed_layer = "shared"

    def __init__(self, config: SwitchTransformersConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        self.encoder = SwitchTransformersStack(encoder_config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = SwitchTransformersStack(decoder_config)

        self.classification_head = SwitchTransformersClassificationHead(config)
        self.router_z_loss_coef = config.router_z_loss_coef
        self.router_aux_loss_coef = config.router_aux_loss_coef

        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        super()._init_weights(module)
        if isinstance(module, SwitchTransformersClassificationHead):
            factor: float = float(getattr(self.config, "initializer_factor", 1.0))
            std: float = factor * (float(self.config.d_model) ** -0.5)
            # nn.init.normal_(module.dense.weight, mean=0.0, std=std)
            # if module.dense.bias is not None:
            #     nn.init.zeros_(module.dense.bias)
            nn.init.normal_(module.out_proj.weight, mean=0.0, std=std)
            if module.out_proj.bias is not None:
                nn.init.zeros_(module.out_proj.bias)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    @staticmethod
    def _coerce_encoder_outputs(encoder_outputs: Any) -> MoEModelOutput:
        if isinstance(encoder_outputs, MoEModelOutput):
            return encoder_outputs
        if isinstance(encoder_outputs, tuple):
            return MoEModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                router_logits=encoder_outputs[3] if len(encoder_outputs) > 3 else None,
            )
        if isinstance(encoder_outputs, dict):
            return MoEModelOutput(**encoder_outputs)
        raise TypeError(f"Unsupported encoder_outputs type: {type(encoder_outputs)}")

    @staticmethod
    def _unpack_router_logits(
        router_outputs: tuple[torch.Tensor, ...] | tuple[tuple[torch.Tensor, ...], ...] | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        total_router_logits = []
        total_expert_indexes = []

        if router_outputs is None:
            return None, None

        for router_output in router_outputs:
            if isinstance(router_output, tuple) and len(router_output) >= 2:
                router_logits, expert_indexes = router_output[0], router_output[1]
                if getattr(router_logits, "ndim", 0) > 1:
                    total_router_logits.append(router_logits)
                    total_expert_indexes.append(expert_indexes)

        if not total_router_logits:
            return None, None
        return torch.cat(total_router_logits, dim=1), torch.cat(total_expert_indexes, dim=1)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.LongTensor | None = None,
        encoder_outputs: tuple[tuple[torch.Tensor]] | MoEModelOutput | dict[str, Any] | None = None,
        past_key_values: EncoderDecoderCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        decoder_inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple | Seq2SeqMoESequenceClassifierOutput:
        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings without input_ids is not supported for {self.__class__.__name__}"
            )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else getattr(self.config, "add_router_probs", False)
        )
        if labels is not None:
            use_cache = False

        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are passed, `input_ids` cannot be `None`."
                )
            decoder_input_ids = self._shift_right(input_ids)

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                output_router_logits=output_router_logits,
                **kwargs,
            )
        else:
            encoder_outputs = self._coerce_encoder_outputs(encoder_outputs)

        hidden_states = encoder_outputs.last_hidden_state
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            cache_position=cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            **kwargs,
        )

        sequence_output = decoder_outputs.last_hidden_state
        if input_ids is None:
            raise ValueError("`input_ids` must be provided to compute EOS-based sentence representations.")

        eos_mask = input_ids.eq(self.config.eos_token_id).to(sequence_output.device)
        if torch.unique_consecutive(eos_mask.sum(1)).numel() != 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")

        batch_size, _, hidden_size = sequence_output.shape
        sentence_representation = sequence_output[eos_mask, :].view(batch_size, -1, hidden_size)[:, -1, :]
        logits = self.classification_head(sentence_representation)

        loss = None
        encoder_z_loss = None
        encoder_aux_loss = None
        decoder_z_loss = None
        decoder_aux_loss = None

        if output_router_logits:
            zero = logits.new_zeros(())
            encoder_z_loss = zero
            encoder_aux_loss = zero
            decoder_z_loss = zero
            decoder_aux_loss = zero

            if self.encoder.config.encoder_sparse_step > 1:
                encoder_router_logits, encoder_expert_indexes = self._unpack_router_logits(
                    encoder_outputs.router_logits
                )
                if encoder_router_logits is not None and encoder_expert_indexes is not None:
                    encoder_z_loss = router_z_loss_func(encoder_router_logits)
                    encoder_router_probs = nn.Softmax(dim=-1)(encoder_router_logits)
                    encoder_aux_loss = load_balancing_loss_func(encoder_router_probs, encoder_expert_indexes)

            if self.decoder.config.decoder_sparse_step > 1:
                decoder_router_logits, decoder_expert_indexes = self._unpack_router_logits(
                    decoder_outputs.router_logits
                )
                if decoder_router_logits is not None and decoder_expert_indexes is not None:
                    decoder_z_loss = router_z_loss_func(decoder_router_logits)
                    decoder_router_probs = nn.Softmax(dim=-1)(decoder_router_logits)
                    decoder_aux_loss = load_balancing_loss_func(decoder_router_probs, decoder_expert_indexes)

        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

            if output_router_logits:
                z_loss = self.router_z_loss_coef * (encoder_z_loss + decoder_z_loss)
                aux_loss = self.router_aux_loss_coef * (encoder_aux_loss + decoder_aux_loss)
                loss = loss + z_loss + aux_loss

        if not return_dict:
            output = (logits,)
            if output_router_logits:
                output += (encoder_z_loss, encoder_aux_loss, decoder_z_loss, decoder_aux_loss)
            output += (
                decoder_outputs.past_key_values,
                decoder_outputs.hidden_states,
                decoder_outputs.attentions,
                decoder_outputs.cross_attentions,
                encoder_outputs.last_hidden_state,
                encoder_outputs.hidden_states,
                encoder_outputs.attentions,
            )
            if output_router_logits:
                output += (decoder_outputs.router_logits, encoder_outputs.router_logits)
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqMoESequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            encoder_z_loss=encoder_z_loss,
            encoder_aux_loss=encoder_aux_loss,
            decoder_z_loss=decoder_z_loss,
            decoder_aux_loss=decoder_aux_loss,
            decoder_router_logits=decoder_outputs.router_logits,
            encoder_router_logits=encoder_outputs.router_logits,
        )


__all__ = [
    "Seq2SeqMoESequenceClassifierOutput",
    "SwitchTransformersClassificationHead",
    "SwitchTransformersForSequenceClassification",
]
