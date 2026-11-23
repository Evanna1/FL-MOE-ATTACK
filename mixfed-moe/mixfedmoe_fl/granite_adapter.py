from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

GRANITE_EXPERT_SLICE_PATTERN = re.compile(r"^__granite_expert__\.(decoder\.\d+)\.(\d+)\.(input_linear|output_linear)$")
GRANITE_PACKED_EXPERT_PATTERN = re.compile(r"^model\.layers\.(\d+)\.block_sparse_moe\.(input_linear|output_linear)\.weight$")
GRANITE_ROUTER_WEIGHT_PATTERN = re.compile(r"^model\.layers\.(\d+)\.block_sparse_moe\.router\.layer\.weight$")
GRANITE_PROBE_EXPERT_PATTERN = re.compile(
    r"^__granite_expert__\.(decoder\.\d+)\.(\d+)\.(input_linear|output_linear)$"
)


def make_granite_expert_slice_name(layer_id: str, expert_id: int, kind: str) -> str:
    if kind not in {"input_linear", "output_linear"}:
        raise ValueError(f"Unsupported Granite expert tensor kind: {kind!r}.")
    return f"__granite_expert__.{layer_id}.{int(expert_id)}.{kind}"


def parse_granite_expert_slice_name(name: str) -> Tuple[str, int, str] | None:
    match = GRANITE_EXPERT_SLICE_PATTERN.match(name)
    return None if match is None else (match.group(1), int(match.group(2)), match.group(3))


def parse_granite_probe_parameter(name: str) -> Tuple[str, int] | None:
    """Parse a virtual Granite expert tensor name for the probe tracker.

    Granite stores all experts in the first dimension of two packed matrices.
    The probe code works on one expert tensor at a time, so the client exposes
    stable virtual names while retaining the normal server parameter schema.
    """
    match = GRANITE_PROBE_EXPERT_PATTERN.match(str(name))
    if match is None:
        return None
    return f"{match.group(1)}", int(match.group(2))


def granite_probe_parameter_views(
    parameter_names: Sequence[str],
    parameters: Sequence[np.ndarray],
) -> Tuple[List[str], List[np.ndarray]]:
    """Expand packed Granite expert matrices into virtual probe tensors.

    The returned arrays are detached copies because probe injection modifies
    only the outgoing sparse expert slices.  Non-expert tensors are omitted;
    the probe tracker intentionally consumes only expert coordinates.
    """
    if len(parameter_names) != len(parameters):
        raise ValueError("Granite probe parameter names and arrays have different lengths.")
    names: List[str] = []
    arrays: List[np.ndarray] = []
    for name, array in zip(parameter_names, parameters):
        match = GRANITE_PACKED_EXPERT_PATTERN.match(str(name))
        if match is None:
            continue
        values = np.asarray(array)
        if values.ndim < 1:
            continue
        layer_id = f"decoder.{int(match.group(1))}"
        kind = str(match.group(2))
        for expert_id in range(int(values.shape[0])):
            names.append(make_granite_expert_slice_name(layer_id, expert_id, kind))
            arrays.append(np.asarray(values[expert_id]).copy())
    return names, arrays


@dataclass(frozen=True)
class GraniteLayerDescriptor:
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


class GranitePackedLinearForMix(nn.Module):
    def __init__(self, source: nn.Module, selected: Sequence[int], kind: str):
        super().__init__()
        if kind not in {"input_linear", "output_linear"}:
            raise ValueError(f"Unsupported Granite projection kind: {kind!r}.")
        self.kind = kind
        self.weight = getattr(source, "weight")
        self.weight.requires_grad_(False)
        self.num_experts = int(getattr(source, "num_experts"))
        self.input_size = int(getattr(source, "input_size"))
        self.output_size = int(getattr(source, "output_size"))
        selected_ids = sorted({int(x) for x in selected})
        if not selected_ids:
            raise ValueError("Granite mix requires at least one selected expert.")
        if selected_ids[0] < 0 or selected_ids[-1] >= self.num_experts:
            raise ValueError(
                f"Selected Granite experts are outside [0, {self.num_experts - 1}]: {selected_ids}."
            )
        self.register_buffer("selected_expert_ids", torch.tensor(selected_ids, dtype=torch.long, device=self.weight.device), persistent=False)
        self._selected_lookup = {expert_id: local for local, expert_id in enumerate(selected_ids)}
        self.selected_weight = nn.ParameterList([nn.Parameter(self.weight[x].detach().float().clone()) for x in selected_ids])
        # Neurotoxin attaches the previous-global delta to the packed weight's
        # ``.grad`` before this mix adapter is installed.  Transfer the assigned
        # rows onto their new expert-local Parameters so the mask sees the real
        # global movement instead of an all-zero fallback (mirrors OlmoePackedExpertsForMix).
        if self.weight.grad is not None:
            for local, expert_id in enumerate(selected_ids):
                self.selected_weight[local].grad = self.weight.grad[expert_id].detach().float().clone()

    def selected_slice(self, expert_id: int) -> nn.Parameter:
        return self.selected_weight[self._selected_lookup[int(expert_id)]]

    def forward(self, inputs: torch.Tensor, expert_size: Sequence[int]) -> torch.Tensor:
        input_list = inputs.split(expert_size, dim=0)
        outputs: List[torch.Tensor] = []
        for expert_id, values in enumerate(input_list):
            weight = self.selected_weight[self._selected_lookup[expert_id]] if expert_id in self._selected_lookup else self.weight[expert_id]
            outputs.append(F.linear(values.to(weight.dtype), weight).to(inputs.dtype))
        return torch.cat(outputs, dim=0)


class GraniteSelectedExpertView(nn.Module):
    def __init__(self, input_projection: GranitePackedLinearForMix, output_projection: GranitePackedLinearForMix, expert_id: int):
        super().__init__()
        self.input_linear = input_projection.selected_slice(expert_id)
        self.output_linear = output_projection.selected_slice(expert_id)


class GraniteAdapter:
    @staticmethod
    def supports(model: nn.Module) -> bool:
        return str(getattr(getattr(model, "config", None), "model_type", "")).lower() in {"granitemoe", "granite_moe"}

    @staticmethod
    def collect_layers(model: nn.Module) -> List[GraniteLayerDescriptor]:
        layers = getattr(getattr(model, "model", None), "layers", None)
        if not GraniteAdapter.supports(model) or not isinstance(layers, nn.ModuleList):
            raise ValueError("Granite model must expose model.layers as ModuleList.")
        descriptors: List[GraniteLayerDescriptor] = []
        for idx, layer in enumerate(layers):
            mlp = getattr(layer, "block_sparse_moe", None)
            input_linear = getattr(mlp, "input_linear", None)
            output_linear = getattr(mlp, "output_linear", None)
            router = getattr(mlp, "router", None)
            if (
                not isinstance(mlp, nn.Module)
                or not isinstance(input_linear, nn.Module)
                or not isinstance(output_linear, nn.Module)
                or not isinstance(router, nn.Module)
            ):
                raise RuntimeError(f"Granite layer {idx} has incomplete MoE modules.")
            router_layer = getattr(router, "layer", None)
            if not isinstance(router_layer, nn.Module):
                raise RuntimeError(f"Granite layer {idx} router has no linear layer.")
            descriptors.append(GraniteLayerDescriptor(
                layer_id=f"decoder.{idx}", layer_idx=idx, layer_path=f"model.layers.{idx}",
                layer_module=layer, mlp_module=mlp, experts_module=input_linear,
                router_module=router_layer, num_experts=int(input_linear.num_experts),
                router_top_k=int(router.top_k), parameter_prefix=f"model.layers.{idx}.block_sparse_moe",
            ))
        return descriptors

    @staticmethod
    def apply_mix(model: nn.Module, layers: Sequence[Any], assignments: Mapping[str, Sequence[int]]) -> None:
        model.requires_grad_(False)
        for layer in layers:
            selected = sorted({int(x) for x in assignments[layer.layer_id]})
            if len(selected) < int(layer.router_top_k):
                raise ValueError(f"Granite layer {layer.layer_id} requires at least Top-{layer.router_top_k} assigned experts.")
            mlp = layer.mlp_module
            mlp.input_linear = GranitePackedLinearForMix(mlp.input_linear, selected, "input_linear")
            mlp.output_linear = GranitePackedLinearForMix(mlp.output_linear, selected, "output_linear")
            layer.router_module.requires_grad_(True)
        embeddings = model.get_input_embeddings()
        if isinstance(embeddings, nn.Module):
            embeddings.requires_grad_(True)
        head = getattr(model, "classification_head", None)
        if not isinstance(head, nn.Module):
            raise RuntimeError("Granite classifier must expose classification_head.")
        head.to(torch.float32)
        head.requires_grad_(True)

    @staticmethod
    def sparse_upload_names(model: nn.Module, layers: Sequence[Any], assignments: Mapping[str, Sequence[int]]) -> List[str]:
        names = [name for name, parameter in model.named_parameters() if parameter.requires_grad and ".selected_weight." not in name]
        for layer in layers:
            for expert_id in sorted({int(x) for x in assignments[layer.layer_id]}):
                names.extend(make_granite_expert_slice_name(layer.layer_id, expert_id, kind) for kind in ("input_linear", "output_linear"))
        if len(names) != len(set(names)):
            raise RuntimeError("Granite sparse upload contains duplicate parameter names.")
        return names

    @staticmethod
    def export_sparse_tensor(model: nn.Module, name: str) -> torch.Tensor | None:
        parsed = parse_granite_expert_slice_name(name)
        if parsed is None:
            return None
        layer_id, expert_id, kind = parsed
        mlp = model.model.layers[int(layer_id.split(".", 1)[1])].block_sparse_moe
        projection = getattr(mlp, kind)
        if not isinstance(projection, GranitePackedLinearForMix):
            raise RuntimeError(f"Granite layer {layer_id} is not configured for mix training.")
        return projection.selected_slice(expert_id)

    @staticmethod
    def selected_expert_module(model: nn.Module, layer_id: str, expert_id: int) -> nn.Module | None:
        mlp = model.model.layers[int(layer_id.split(".", 1)[1])].block_sparse_moe
        return GraniteAdapter.selected_expert_module_from_mlp(mlp, layer_id, expert_id)

    @staticmethod
    def selected_expert_module_from_mlp(mlp: nn.Module, layer_id: str, expert_id: int) -> nn.Module | None:
        input_projection = getattr(mlp, "input_linear")
        output_projection = getattr(mlp, "output_linear")
        if not isinstance(input_projection, GranitePackedLinearForMix) or not isinstance(output_projection, GranitePackedLinearForMix):
            raise RuntimeError(f"Granite layer {layer_id} is not configured for mix training.")
        if int(expert_id) not in input_projection._selected_lookup:
            return None
        return GraniteSelectedExpertView(input_projection, output_projection, int(expert_id))
