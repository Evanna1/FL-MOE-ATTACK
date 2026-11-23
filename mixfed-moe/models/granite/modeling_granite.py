from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.granitemoe.modeling_granitemoe import (
    GraniteMoeModel,
    GraniteMoePreTrainedModel,
    load_balancing_loss_func,
)


class GraniteMoeClassificationHead(nn.Module):
    """Classification head for the decoder-only Granite-MoE backbone."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        dropout = getattr(config, "classifier_dropout", None)
        if dropout is None:
            dropout = getattr(config, "hidden_dropout", 0.0)
        self.dropout = nn.Dropout(float(dropout))
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        # Granite backbones are commonly loaded in BF16 while the classifier
        # head is kept in FP32 for stable local training.  ``nn.Linear`` does
        # not perform this promotion itself, so align the activation with the
        # projection parameters at this boundary.
        projection_dtype = self.out_proj.weight.dtype
        if hidden_states.dtype != projection_dtype:
            hidden_states = hidden_states.to(projection_dtype)
        return self.out_proj(hidden_states)


class GraniteMoeForSequenceClassification(GraniteMoePreTrainedModel):
    """Load a Granite causal checkpoint and expose sequence classification."""

    _tied_weights_keys: set[str] = set()
    _input_embed_layer = "model.embed_tokens"

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.num_labels = int(getattr(config, "num_labels", 2))
        self.model = GraniteMoeModel(config)
        self.classification_head = GraniteMoeClassificationHead(config)
        # Load-balancing (router auxiliary) loss parameters.  The decoder-only
        # ``GraniteMoeModel`` never emits router logits, so unlike the causal LM
        # the classifier cannot rely on ``outputs.router_logits``; the forward
        # pass captures them through router hooks instead (see ``forward``).
        self.router_aux_loss_coef = float(getattr(config, "router_aux_loss_coef", 0.0) or 0.0)
        self.num_experts = int(getattr(config, "num_local_experts", 0) or 0)
        self.num_experts_per_tok = int(getattr(config, "num_experts_per_tok", 0) or 0)
        # Detached magnitude of the most recent auxiliary loss, exposed purely
        # for monitoring routing collapse during training.
        self.last_router_aux_loss: float = 0.0
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value  # type: ignore[assignment]

    def _last_non_padding_indices(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.LongTensor:
        batch_size, sequence_length, _ = hidden_states.shape
        if attention_mask is None:
            return torch.full((batch_size,), sequence_length - 1, dtype=torch.long, device=hidden_states.device)
        if attention_mask.ndim != 2 or tuple(attention_mask.shape) != (batch_size, sequence_length):
            raise ValueError(
                "attention_mask must have shape [batch_size, sequence_length], "
                f"got {tuple(attention_mask.shape)}."
            )
        valid = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
        if not bool(valid.any(dim=1).all()):
            raise ValueError("Every sequence must contain at least one non-padding token.")
        positions = torch.arange(sequence_length, device=hidden_states.device).unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        return positions.masked_fill(~valid, -1).max(dim=1).values

    def _router_aux_loss_enabled(self) -> bool:
        return (
            self.router_aux_loss_coef > 0.0
            and self.num_experts > 1
            and self.num_experts_per_tok > 0
        )

    def _iter_router_linears(self) -> list[nn.Module]:
        """Return each decoder layer's router linear in execution order.

        The linear is the module whose output is the per-token expert logits,
        both before and after the adapter rewires the packed expert matrices.
        """
        linears: list[nn.Module] = []
        for layer in self.model.layers:
            router = getattr(getattr(layer, "block_sparse_moe", None), "router", None)
            router_linear = getattr(router, "layer", None)
            if not isinstance(router_linear, nn.Module):
                raise RuntimeError("Granite decoder layer is missing its router linear.")
            linears.append(router_linear)
        return linears

    def _classification_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        problem_type = getattr(self.config, "problem_type", None)
        if problem_type is None:
            if self.num_labels == 1:
                problem_type = "regression"
            elif self.num_labels > 1 and labels.dtype in (torch.long, torch.int):
                problem_type = "single_label_classification"
            else:
                problem_type = "multi_label_classification"
            self.config.problem_type = problem_type
        if problem_type == "regression":
            loss_fct = MSELoss()
            if self.num_labels == 1:
                return loss_fct(logits.squeeze(), labels.squeeze().to(logits.dtype))
            return loss_fct(logits, labels.to(logits.dtype))
        if problem_type == "single_label_classification":
            return CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))
        return BCEWithLogitsLoss()(logits, labels.to(logits.dtype))

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> tuple | SequenceClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        collect_router = labels is not None and self._router_aux_loss_enabled()
        router_logits: list[torch.Tensor] = []
        hook_handles: list[Any] = []
        if collect_router:
            def _capture(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
                # [num_tokens, num_experts]; keep the autograd link so the
                # auxiliary loss backpropagates into the router linear.
                router_logits.append(output)

            for router_linear in self._iter_router_linears():
                hook_handles.append(router_linear.register_forward_hook(_capture))

        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
        finally:
            for handle in hook_handles:
                handle.remove()

        hidden_states = outputs.last_hidden_state
        last_indices = self._last_non_padding_indices(hidden_states, attention_mask)
        batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        logits = self.classification_head(hidden_states[batch_indices, last_indices])
        loss = self._classification_loss(logits, labels) if labels is not None else None

        # The auxiliary (load-balancing) loss is exposed as a separate, already
        # scaled tensor instead of being folded into ``loss``.  Callers decide
        # whether to add it: training loops do, while gradient-attribution
        # probes keep a pure classification signal.  See the client trainer.
        router_aux_loss: torch.Tensor | None = None
        if loss is not None and collect_router:
            if len(router_logits) != len(self.model.layers):
                raise RuntimeError(
                    "Granite router capture expected one logits tensor per decoder "
                    f"layer, got {len(router_logits)} of {len(self.model.layers)}."
                )
            aux_loss = load_balancing_loss_func(
                tuple(layer_logits.float() for layer_logits in router_logits),
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            self.last_router_aux_loss = float(aux_loss.detach())
            router_aux_loss = self.router_aux_loss_coef * aux_loss.to(loss.device)
        else:
            self.last_router_aux_loss = 0.0

        if not return_dict:
            result = (logits, outputs.past_key_values, None, None)
            return ((loss,) + result) if loss is not None else result
        output = SequenceClassifierOutput(loss=loss, logits=logits)
        if router_aux_loss is not None:
            # Attribute-only: does not appear in ``keys()``/``to_tuple()``.
            output.router_aux_loss = router_aux_loss
        return output


__all__ = ["GraniteMoeClassificationHead", "GraniteMoeForSequenceClassification"]
