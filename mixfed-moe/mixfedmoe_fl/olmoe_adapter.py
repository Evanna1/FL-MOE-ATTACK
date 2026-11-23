from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


OLMOE_EXPERT_SLICE_PATTERN = re.compile(
    r"^__olmoe_expert__\.(decoder\.\d+)\.(\d+)\.(gate_up_proj|down_proj)$"
)
OLMOE_PACKED_EXPERT_PATTERN = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$"
)
OLMOE_ROUTER_WEIGHT_PATTERN = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.gate\.weight$"
)


def make_olmoe_expert_slice_name(
    layer_id: str,
    expert_id: int,
    kind: str,
) -> str:
    if kind not in {"gate_up_proj", "down_proj"}:
        raise ValueError(f"Unsupported OLMoE expert tensor kind: {kind!r}.")
    return f"__olmoe_expert__.{layer_id}.{int(expert_id)}.{kind}"


def parse_olmoe_expert_slice_name(name: str) -> Tuple[str, int, str] | None:
    match = OLMOE_EXPERT_SLICE_PATTERN.match(name)
    if match is None:
        return None
    return match.group(1), int(match.group(2)), match.group(3)


@dataclass(frozen=True)
class OlmoeLayerDescriptor:
    layer_id: str
    layer_idx: int
    layer_path: str
    layer_module: nn.Module
    mlp_module: nn.Module
    experts_module: nn.Module
    router_module: nn.Module
    num_experts: int
    router_top_k: int
    parameter_prefix: str


class OlmoePackedExpertsForMix(nn.Module):
    """Packed OLMoE experts with trainable copies for selected expert rows only."""

    def __init__(self, source: nn.Module, selected_expert_ids: Sequence[int]):
        super().__init__()
        selected = sorted({int(value) for value in selected_expert_ids})
        num_experts = int(getattr(source, "num_experts"))
        if not selected:
            raise ValueError("OLMoE mix requires at least one selected expert.")
        if selected[0] < 0 or selected[-1] >= num_experts:
            raise ValueError(
                f"Selected OLMoE experts are outside [0, {num_experts - 1}]: {selected}."
            )

        self.config = getattr(source, "config", None)
        self.num_experts = num_experts
        self.hidden_dim = int(getattr(source, "hidden_dim"))
        self.intermediate_dim = int(getattr(source, "intermediate_dim"))
        self.act_fn = getattr(source, "act_fn")

        # Keep the complete global expert bank in its checkpoint dtype (normally
        # BF16) and frozen.  Reusing the Parameter objects avoids a second full
        # copy of the 64-expert tensors.
        self.gate_up_proj = getattr(source, "gate_up_proj")
        self.down_proj = getattr(source, "down_proj")
        self.gate_up_proj.requires_grad_(False)
        self.down_proj.requires_grad_(False)

        selected_tensor = torch.tensor(
            selected,
            dtype=torch.long,
            device=self.gate_up_proj.device,
        )
        self.register_buffer("selected_expert_ids", selected_tensor, persistent=False)
        # Keep every assigned expert as an independent Parameter.  Besides
        # making the upload schema genuinely expert-granular, this lets target
        # attacks train exactly one expert without AdamW moving the other
        # assigned experts through weight decay.
        self.selected_gate_up_proj = nn.ParameterList(
            [
                nn.Parameter(self.gate_up_proj[expert_id].detach().float().clone())
                for expert_id in selected
            ]
        )
        self.selected_down_proj = nn.ParameterList(
            [
                nn.Parameter(self.down_proj[expert_id].detach().float().clone())
                for expert_id in selected
            ]
        )
        # Neurotoxin stores the previous global delta temporarily in grad
        # before the mix adapter is installed.  Transfer the assigned rows to
        # their new expert-local Parameters so mask construction sees the real
        # global movement instead of an all-zero fallback.
        if self.gate_up_proj.grad is not None:
            for local_idx, expert_id in enumerate(selected):
                self.selected_gate_up_proj[local_idx].grad = (
                    self.gate_up_proj.grad[expert_id].detach().float().clone()
                )
        if self.down_proj.grad is not None:
            for local_idx, expert_id in enumerate(selected):
                self.selected_down_proj[local_idx].grad = (
                    self.down_proj.grad[expert_id].detach().float().clone()
                )
        self._selected_lookup = {
            expert_id: local_idx for local_idx, expert_id in enumerate(selected)
        }

    def get_selected_slice(self, expert_id: int, kind: str) -> torch.Tensor:
        expert_id = int(expert_id)
        if expert_id not in self._selected_lookup:
            raise KeyError(
                f"Expert {expert_id} is not trainable in this client; "
                f"selected={self.selected_expert_ids.tolist()}."
            )
        local_idx = self._selected_lookup[expert_id]
        if kind == "gate_up_proj":
            return self.selected_gate_up_proj[local_idx]
        if kind == "down_proj":
            return self.selected_down_proj[local_idx]
        raise ValueError(f"Unsupported OLMoE expert tensor kind: {kind!r}.")

    def get_selected_expert(self, expert_id: int) -> nn.Module:
        expert_id = int(expert_id)
        if expert_id not in self._selected_lookup:
            raise KeyError(
                f"Expert {expert_id} is not trainable in this client; "
                f"selected={self.selected_expert_ids.tolist()}."
            )
        local_idx = self._selected_lookup[expert_id]
        return OlmoeSelectedExpertView(
            self.selected_gate_up_proj[local_idx],
            self.selected_down_proj[local_idx],
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for raw_expert_idx in expert_hit:
            expert_idx = int(raw_expert_idx[0].item())
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            local_idx = self._selected_lookup.get(expert_idx)
            if local_idx is None:
                gate_up_weight = self.gate_up_proj[expert_idx]
                down_weight = self.down_proj[expert_idx]
            else:
                gate_up_weight = self.selected_gate_up_proj[local_idx]
                down_weight = self.selected_down_proj[local_idx]

            compute_state = current_state.to(gate_up_weight.dtype)
            gate, up = F.linear(compute_state, gate_up_weight).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, down_weight)
            current_hidden_states = current_hidden_states.to(final_hidden_states.dtype)
            current_hidden_states = (
                current_hidden_states
                * top_k_weights[token_idx, top_k_pos, None].to(
                    current_hidden_states.dtype
                )
            )
            final_hidden_states.index_add_(0, token_idx, current_hidden_states)

        return final_hidden_states


class OlmoeSelectedExpertView(nn.Module):
    """A lightweight view exposing one packed OLMoE expert's two Parameters."""

    def __init__(
        self,
        gate_up_proj: nn.Parameter,
        down_proj: nn.Parameter,
    ) -> None:
        super().__init__()
        self.gate_up_proj = gate_up_proj
        self.down_proj = down_proj


class OlmoeAdapter:
    """Architecture adapter for Transformers 5 packed-tensor OLMoE models."""

    @staticmethod
    def supports(model: nn.Module) -> bool:
        return str(getattr(getattr(model, "config", None), "model_type", "")) == "olmoe"

    @staticmethod
    def collect_layers(model: nn.Module) -> List[OlmoeLayerDescriptor]:
        if not OlmoeAdapter.supports(model):
            raise ValueError(f"Model {type(model).__name__} is not an OLMoE model.")
        backbone = getattr(model, "model", None)
        layers = getattr(backbone, "layers", None)
        if not isinstance(layers, nn.ModuleList):
            raise RuntimeError("OLMoE backbone must expose model.layers as ModuleList.")

        descriptors: List[OlmoeLayerDescriptor] = []
        for layer_idx, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            experts = getattr(mlp, "experts", None)
            router = getattr(mlp, "gate", None)
            if not isinstance(mlp, nn.Module) or not isinstance(experts, nn.Module):
                raise RuntimeError(f"OLMoE layer {layer_idx} has no packed MoE experts.")
            if not isinstance(router, nn.Module):
                raise RuntimeError(f"OLMoE layer {layer_idx} has no gate/router module.")
            num_experts = int(getattr(experts, "num_experts"))
            router_top_k = int(getattr(router, "top_k"))
            descriptors.append(
                OlmoeLayerDescriptor(
                    layer_id=f"decoder.{layer_idx}",
                    layer_idx=layer_idx,
                    layer_path=f"model.layers.{layer_idx}",
                    layer_module=layer,
                    mlp_module=mlp,
                    experts_module=experts,
                    router_module=router,
                    num_experts=num_experts,
                    router_top_k=router_top_k,
                    parameter_prefix=f"model.layers.{layer_idx}.mlp",
                )
            )
        return descriptors

    @staticmethod
    def apply_mix(
        model: nn.Module,
        layers: Sequence[Any],
        assignments: Mapping[str, Sequence[int]],
    ) -> None:
        model.requires_grad_(False)
        for layer in layers:
            selected = sorted({int(value) for value in assignments[layer.layer_id]})
            if len(selected) < int(layer.router_top_k):
                raise ValueError(
                    f"OLMoE layer {layer.layer_id} uses Top-{layer.router_top_k} routing, "
                    f"but the client received only {len(selected)} experts."
                )
            adapted = OlmoePackedExpertsForMix(
                source=layer.mlp_module.experts,
                selected_expert_ids=selected,
            )
            layer.mlp_module.experts = adapted
            layer.router_module.requires_grad_(True)

        input_embeddings = model.get_input_embeddings()
        if isinstance(input_embeddings, nn.Module):
            input_embeddings.requires_grad_(True)
        classification_head = getattr(model, "classification_head", None)
        if not isinstance(classification_head, nn.Module):
            raise RuntimeError("OLMoE classifier must expose classification_head.")
        classification_head.to(torch.float32)
        classification_head.requires_grad_(True)

    @staticmethod
    def sparse_upload_names(
        model: nn.Module,
        layers: Sequence[Any],
        assignments: Mapping[str, Sequence[int]],
    ) -> List[str]:
        names = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and ".experts.selected_" not in name
        ]
        for layer in layers:
            for expert_id in sorted({int(value) for value in assignments[layer.layer_id]}):
                names.append(
                    make_olmoe_expert_slice_name(
                        layer.layer_id, expert_id, "gate_up_proj"
                    )
                )
                names.append(
                    make_olmoe_expert_slice_name(
                        layer.layer_id, expert_id, "down_proj"
                    )
                )
        if len(set(names)) != len(names):
            raise RuntimeError("OLMoE sparse upload contains duplicate parameter names.")
        return names

    @staticmethod
    def export_sparse_tensor(model: nn.Module, name: str) -> torch.Tensor | None:
        parsed = parse_olmoe_expert_slice_name(name)
        if parsed is None:
            return None
        layer_id, expert_id, kind = parsed
        layer_idx = int(layer_id.split(".", 1)[1])
        experts = model.model.layers[layer_idx].mlp.experts
        if not isinstance(experts, OlmoePackedExpertsForMix):
            raise RuntimeError(
                f"OLMoE layer {layer_id} is not configured for selected-expert mix training."
            )
        return experts.get_selected_slice(expert_id, kind)

    @staticmethod
    def selected_expert_module(
        model: nn.Module,
        layer_id: str,
        expert_id: int,
    ) -> nn.Module | None:
        layer_idx = int(layer_id.split(".", 1)[1])
        experts = model.model.layers[layer_idx].mlp.experts
        if not isinstance(experts, OlmoePackedExpertsForMix):
            raise RuntimeError(
                f"OLMoE layer {layer_id} is not configured for mix training."
            )
        if int(expert_id) not in experts._selected_lookup:
            return None
        return experts.get_selected_expert(int(expert_id))

