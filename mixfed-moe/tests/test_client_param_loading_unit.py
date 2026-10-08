from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from mixfedmoe_fl.client import (
    _compute_target_train_steps,
    _load_from_numpy_ndarrays,
    _parameter_update_l2,
    _state_dict_names,
    _target_expert_parameter_snapshot,
)


def _build_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(3, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )


def _state_arrays(model: nn.Module):
    names = _state_dict_names(model)
    arrays = [model.state_dict()[name].detach().cpu().numpy().copy() for name in names]
    return names, arrays


def test_load_from_numpy_ndarrays_accepts_full_matching_state() -> None:
    model = _build_model()
    names, arrays = _state_arrays(model)

    _load_from_numpy_ndarrays(model=model, param_names=names, parameters=arrays)

    state = model.state_dict()
    for name, array in zip(names, arrays):
        loaded = state[name].detach().cpu().numpy()
        np.testing.assert_allclose(loaded, array)


def test_load_from_numpy_ndarrays_rejects_length_mismatch() -> None:
    model = _build_model()
    names, arrays = _state_arrays(model)

    with pytest.raises(ValueError, match="Parameter name count mismatch"):
        _load_from_numpy_ndarrays(model=model, param_names=names, parameters=arrays[:-1])


def test_load_from_numpy_ndarrays_rejects_shape_mismatch() -> None:
    model = _build_model()
    names, arrays = _state_arrays(model)
    bad_arrays = list(arrays)
    bad_arrays[0] = np.zeros((1,), dtype=np.float32)

    with pytest.raises(ValueError, match="Shape mismatch"):
        _load_from_numpy_ndarrays(model=model, param_names=names, parameters=bad_arrays)


def test_load_from_numpy_ndarrays_casts_dtype_to_model_target() -> None:
    model = _build_model()
    names, arrays = _state_arrays(model)
    float64_arrays = [arr.astype(np.float64, copy=True) for arr in arrays]

    _load_from_numpy_ndarrays(model=model, param_names=names, parameters=float64_arrays)

    for name in names:
        assert model.state_dict()[name].dtype == torch.float32


def test_compute_target_train_steps_supports_fractional_epochs() -> None:
    assert _compute_target_train_steps(local_epochs=0.5, steps_per_epoch=10) == 5
    assert _compute_target_train_steps(local_epochs=0.1, steps_per_epoch=10) == 1
    assert _compute_target_train_steps(local_epochs=1.5, steps_per_epoch=10) == 15


def test_compute_target_train_steps_requires_non_empty_dataloader() -> None:
    with pytest.raises(ValueError, match="steps_per_epoch must be > 0"):
        _compute_target_train_steps(local_epochs=0.5, steps_per_epoch=0)


class _TinyExpertModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.block = nn.ModuleList([nn.Identity() for _ in range(12)])
        target_block = nn.Module()
        target_block.layer = nn.ModuleList([nn.Module()])
        target_block.layer[0].mlp = nn.Module()
        target_block.layer[0].mlp.experts = nn.ModuleDict(
            {
                "expert_2": nn.Linear(2, 2, bias=False),
                "expert_3": nn.Linear(2, 2, bias=False),
            }
        )
        self.encoder.block[11] = target_block


def test_target_expert_snapshot_measures_only_selected_expert_update() -> None:
    model = _TinyExpertModel()
    with torch.no_grad():
        model.encoder.block[11].layer[0].mlp.experts["expert_2"].weight.fill_(2.0)
        model.encoder.block[11].layer[0].mlp.experts["expert_3"].weight.fill_(3.0)

    snapshot = _target_expert_parameter_snapshot(
        model=model,
        layer_id="encoder.11",
        expert_idx=2,
    )
    assert list(snapshot) == [
        "encoder.block.11.layer.0.mlp.experts.expert_2.weight"
    ]

    with torch.no_grad():
        model.encoder.block[11].layer[0].mlp.experts["expert_2"].weight.add_(1.0)
        model.encoder.block[11].layer[0].mlp.experts["expert_3"].weight.add_(100.0)

    absolute, relative = _parameter_update_l2(model=model, before=snapshot)
    assert absolute == pytest.approx(2.0)
    assert relative == pytest.approx(0.5)
