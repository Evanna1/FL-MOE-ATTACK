from __future__ import annotations

import torch

from mixfedmoe_fl.client import MoeHandle, TriggerRouterLoss, _build_trigger_router_loss


def test_trigger_router_loss_supervises_exact_trigger_tokens() -> None:
    gate = torch.nn.Linear(2, 3)
    objective = TriggerRouterLoss(gate, token_ids=(7, 8), target_expert=2, decoder=False)
    try:
        input_ids = torch.tensor([[10, 7, 8, 1], [7, 8, 0, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]])
        objective.begin_batch(input_ids, attention_mask)

        assert objective.mask is not None
        assert objective.mask.tolist() == [
            [False, True, True, False],
            [False, False, False, False],
        ]
        assert objective.trigger_tokens == 2

        gate(torch.randn(2, 4, 2))
        loss = objective.loss()
        loss.backward()

        assert loss.item() > 0.0
        assert gate.weight.grad is not None
        assert torch.count_nonzero(gate.weight.grad).item() > 0
    finally:
        objective.close()

    assert not gate._forward_hooks


def test_trigger_router_loss_returns_differentiable_zero_for_clean_batch() -> None:
    gate = torch.nn.Linear(2, 3)
    objective = TriggerRouterLoss(gate, token_ids=(7, 8), target_expert=1, decoder=True)
    try:
        input_ids = torch.tensor([[10, 11, 12, 1]])
        objective.begin_batch(input_ids, torch.ones_like(input_ids))
        gate(torch.randn(1, 4, 2))

        loss = objective.loss()
        loss.backward()

        assert loss.item() == 0.0
        assert objective.trigger_tokens == 0
        assert gate.weight.grad is not None
        assert torch.count_nonzero(gate.weight.grad).item() == 0
    finally:
        objective.close()


def test_drop_mode_maps_global_target_expert_to_local_router_row() -> None:
    class TinyMlp(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.router = torch.nn.Linear(2, 2)

    mlp = TinyMlp()
    handle = MoeHandle(
        layer_id="encoder.0",
        stack_name="encoder",
        layer_idx=0,
        layer_path="encoder.block.0",
        layer_module=mlp,
        ff_module=mlp,
        mlp_module=mlp,
        experts_attr="experts",
        gate_linear_path="router",
    )

    objective = _build_trigger_router_loss(
        moe_handles=[handle],
        target_layer="encoder.0",
        target_expert=2,
        mode="drop",
        assignments={"encoder.0": [0, 2]},
        trigger_token_ids=(7, 8),
    )
    try:
        assert objective.target_expert == 1
    finally:
        objective.close()
