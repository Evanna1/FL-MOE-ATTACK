from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from flwr.common import RecordDict, ndarrays_to_parameters, parameters_to_ndarrays

from mixfedmoe_fl.client import (
    _apply_neurotoxin_gradient_mask,
    _attach_global_update_to_parameter_grads,
    _build_neurotoxin_mask_from_global_update,
    _project_neurotoxin_update,
    _restore_neurotoxin_blocked_values,
    _load_neurotoxin_previous_global,
    _save_neurotoxin_previous_global,
    _scale_neurotoxin_update,
    _snapshot_neurotoxin_blocked_values,
    _snapshot_trainable_parameters,
)
from mixfedmoe_fl.config import parse_config
from mixfedmoe_fl.strategy import MixFedMoEStrategy


def test_global_mask_keeps_exact_coordinate_count_when_gradients_tie() -> None:
    model = torch.nn.Linear(4, 1, bias=False)
    model.weight.grad = torch.zeros_like(model.weight)

    masks = _build_neurotoxin_mask_from_global_update(
        model=model,
        ratio=0.5,
        aggregate_all_layers=True,
    )

    assert masks["weight"].dtype == torch.bool
    assert masks["weight"].sum().item() == 2


def test_per_tensor_mask_keeps_bottom_fraction_in_each_tensor() -> None:
    model = torch.nn.Linear(4, 2, bias=True)
    model.weight.grad = torch.arange(8, dtype=torch.float32).reshape_as(model.weight)
    model.bias.grad = torch.tensor([10.0, 9.0])

    masks = _build_neurotoxin_mask_from_global_update(
        model=model,
        ratio=0.5,
        aggregate_all_layers=False,
    )

    assert masks["weight"].sum().item() == 4
    assert masks["bias"].sum().item() == 1
    assert masks["bias"].tolist() == [False, True]


def test_unrouted_parameter_without_gradient_is_treated_as_benign_quiet() -> None:
    model = torch.nn.Linear(4, 1, bias=True)
    model.weight.grad = torch.ones_like(model.weight)
    model.bias.grad = None

    masks = _build_neurotoxin_mask_from_global_update(
        model=model,
        ratio=0.4,
        aggregate_all_layers=True,
    )

    assert masks["bias"].item() is True
    assert sum(mask.sum().item() for mask in masks.values()) == 2


def test_mask_uses_difference_between_consecutive_global_models() -> None:
    model = torch.nn.Linear(3, 1, bias=True)
    current = [
        np.array([[1.0, 4.0, 2.0]], dtype=np.float32),
        np.array([3.0], dtype=np.float32),
    ]
    previous = [
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
    ]
    update_norm = _attach_global_update_to_parameter_grads(
        model=model,
        parameter_names=["weight", "bias"],
        current_parameters=current,
        previous_parameters=previous,
    )
    masks = _build_neurotoxin_mask_from_global_update(
        model=model,
        ratio=0.5,
        aggregate_all_layers=True,
    )

    assert update_norm == pytest.approx(30.0**0.5)
    assert sum(mask.sum().item() for mask in masks.values()) == 2
    assert masks["weight"].tolist() == [[True, False, True]]
    assert masks["bias"].item() is False


@dataclass(frozen=True)
class _Client:
    cid: str


class _ClientManager:
    def __init__(self, clients):
        self.clients = clients

    def num_available(self):
        return len(self.clients)

    def sample(self, num_clients, min_num_clients):
        assert num_clients <= len(self.clients)
        assert min_num_clients <= len(self.clients)
        return self.clients[:num_clients]


def test_strategy_sends_only_current_global_model_to_neurotoxin_client() -> None:
    cfg = parse_config(
        [
            "--mode",
            "full",
            "--num_clients",
            "1",
            "--attack_enabled",
            "--attack_mode",
            "neurotoxin",
        ]
    )
    current = [np.array([2.0], dtype=np.float32)]
    strategy = MixFedMoEStrategy(
        cfg,
        ["weight"],
        initial_parameters=ndarrays_to_parameters(current),
    )

    instructions = strategy.configure_fit(
        2,
        ndarrays_to_parameters(current),
        _ClientManager([_Client("0")]),  # type: ignore[arg-type]
    )

    payload = parameters_to_ndarrays(instructions[0][1].parameters)
    assert "neurotoxin_reference_parameter_count" not in instructions[0][1].config
    assert len(payload) == 1
    assert payload[0].item() == pytest.approx(2.0)


def test_neurotoxin_global_history_is_persisted_and_replaced_in_client_state() -> None:
    state = RecordDict()
    first = [np.array([1.0], dtype=np.float32)]
    later = [np.array([4.0], dtype=np.float32)]

    assert _load_neurotoxin_previous_global(state) is None

    _save_neurotoxin_previous_global(state, first)
    loaded_first = _load_neurotoxin_previous_global(state)
    assert loaded_first is not None
    assert loaded_first[0].item() == pytest.approx(1.0)

    _save_neurotoxin_previous_global(state, later)
    loaded_later = _load_neurotoxin_previous_global(state)
    assert loaded_later is not None
    assert loaded_later[0].item() == pytest.approx(4.0)
    # ArrayRecord owns a serialized copy rather than aliasing the input array.
    later[0][0] = 99.0
    reloaded = _load_neurotoxin_previous_global(state)
    assert reloaded is not None
    assert reloaded[0].item() == pytest.approx(4.0)


def test_masked_coordinates_do_not_move_under_adamw_weight_decay() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    model.weight.grad = torch.tensor([[1.0, 1.0]])
    masks = {"weight": torch.tensor([[True, False]])}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.1)

    _apply_neurotoxin_gradient_mask(model, masks)
    blocked = _snapshot_neurotoxin_blocked_values(model, masks)
    optimizer.step()
    _restore_neurotoxin_blocked_values(model, masks, blocked)

    assert model.weight[0, 0].item() != pytest.approx(1.0)
    assert model.weight[0, 1].item() == pytest.approx(1.0)


def test_projection_and_upload_scaling_operate_on_model_delta() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    origin = _snapshot_trainable_parameters(model)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3.0, 4.0]]))

    projected_norm = _project_neurotoxin_update(model, origin, max_norm=1.0)
    assert projected_norm == pytest.approx(1.0)
    assert torch.linalg.vector_norm(model.weight).item() == pytest.approx(1.0)

    _scale_neurotoxin_update(model, origin, scale=2.0)
    assert torch.linalg.vector_norm(model.weight).item() == pytest.approx(2.0)


def test_parse_config_accepts_complete_neurotoxin_options() -> None:
    cfg = parse_config(
        [
            "--mode",
            "full",
            "--attack_enabled",
            "--attack_mode",
            "neurotoxin",
            "--neurotoxin_mask_ratio",
            "0.9",
            "--neurotoxin_mask_per_layer",
            "--neurotoxin_max_update_norm",
            "0.5",
            "--neurotoxin_update_scale",
            "3",
        ]
    )

    assert cfg.neurotoxin_mask_ratio == pytest.approx(0.9)
    assert cfg.neurotoxin_aggregate_all_layers is False
    assert cfg.neurotoxin_max_update_norm == pytest.approx(0.5)
    assert cfg.neurotoxin_update_scale == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--neurotoxin_max_update_norm", "-1", "max_update_norm"),
        ("--neurotoxin_update_scale", "0", "update_scale"),
    ],
)
def test_parse_config_rejects_invalid_neurotoxin_options(
    option: str, value: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_config(["--mode", "full", option, value])
