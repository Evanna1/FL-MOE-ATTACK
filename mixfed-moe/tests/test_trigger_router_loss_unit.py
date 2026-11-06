from __future__ import annotations

import pytest
import torch

from mixfedmoe_fl.client import (
    MoeHandle,
    TargetExpertPoisonGradientState,
    TriggerRouterLoss,
    _backward_with_isolated_poison_gradients,
    _build_trigger_router_loss,
)


def test_isolated_poison_backward_preserves_example_weights_and_gradient_scope() -> None:
    shared = torch.nn.Parameter(torch.tensor(0.2))
    target = torch.nn.Parameter(torch.tensor(-0.1))
    features = torch.tensor([1.0, 2.0])
    labels = torch.tensor([0, 1])
    poison_mask = torch.tensor([False, True])

    scores = (shared + target) * features
    logits = torch.stack((scores, -scores), dim=1)
    per_example = torch.nn.functional.cross_entropy(
        logits,
        labels,
        reduction="none",
    )
    clean_loss = per_example[0] / 2
    poison_loss = per_example[1] / 2
    expected_shared_clean = torch.autograd.grad(
        clean_loss,
        shared,
        retain_graph=True,
    )[0]
    expected_target_clean = torch.autograd.grad(
        clean_loss,
        target,
        retain_graph=True,
    )[0]
    expected_target_poison = torch.autograd.grad(
        poison_loss,
        target,
        retain_graph=True,
    )[0]

    loss, clean_count, poison_count = _backward_with_isolated_poison_gradients(
        logits=logits,
        labels=labels,
        poison_mask=poison_mask,
        poison_parameters=[target],
    )

    assert clean_count == 1
    assert poison_count == 1
    assert loss.item() == pytest.approx(per_example.mean().item())
    assert shared.grad is not None
    assert target.grad is not None
    assert shared.grad.item() == pytest.approx(expected_shared_clean.item())
    assert target.grad.item() == pytest.approx(
        (expected_target_clean + expected_target_poison).item()
    )


def test_isolated_poison_backward_routes_auxiliary_loss_only_to_target_scope() -> None:
    shared = torch.nn.Parameter(torch.tensor(0.2))
    target = torch.nn.Parameter(torch.tensor(-0.1))
    router = torch.nn.Parameter(torch.tensor(0.3))
    labels = torch.tensor([0])
    logits = torch.stack((shared + target, -(shared + target))).reshape(1, 2)
    router_loss = router.square()

    _backward_with_isolated_poison_gradients(
        logits=logits,
        labels=labels,
        poison_mask=torch.tensor([True]),
        poison_parameters=[target, router],
        router_loss=router_loss,
        router_loss_weight=0.25,
    )

    assert shared.grad is None
    assert target.grad is not None
    assert router.grad is not None
    assert router.grad.item() == pytest.approx(0.15)


def test_unified_nontarget_mask_only_filters_poison_gradient() -> None:
    target = torch.nn.Parameter(torch.tensor(0.1))
    head = torch.nn.Parameter(torch.tensor(-0.2))
    shared = torch.nn.Parameter(torch.tensor([0.3, -0.4]))
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    labels = torch.tensor([0, 1])
    poison_mask = torch.tensor([False, True])

    scores = features @ shared + target + head
    logits = torch.stack((scores, -scores), dim=1)
    per_example = torch.nn.functional.cross_entropy(
        logits,
        labels,
        reduction="none",
    )
    clean_loss = per_example[0] / 2
    poison_loss = per_example[1] / 2
    clean_target, clean_head, clean_shared = torch.autograd.grad(
        clean_loss,
        (target, head, shared),
        retain_graph=True,
    )
    poison_target, poison_head, poison_shared = torch.autograd.grad(
        poison_loss,
        (target, head, shared),
        retain_graph=True,
    )

    _backward_with_isolated_poison_gradients(
        logits=logits,
        labels=labels,
        poison_mask=poison_mask,
        poison_parameters=[target, head, shared],
        poison_parameter_masks={
            id(head): torch.tensor(True),
            id(shared): torch.tensor([True, False]),
        },
    )

    assert target.grad is not None
    assert head.grad is not None
    assert shared.grad is not None
    assert target.grad.tolist() == pytest.approx(
        (clean_target + poison_target).tolist()
    )
    assert head.grad.tolist() == pytest.approx(
        (clean_head + poison_head).tolist()
    )
    expected_shared = clean_shared + poison_shared * torch.tensor(
        [1.0, 0.0]
    )
    assert shared.grad.tolist() == pytest.approx(expected_shared.tolist())


def test_target_expert_gradient_gain_preserves_masked_poison_budget() -> None:
    target = torch.nn.Parameter(torch.tensor(0.1))
    other = torch.nn.Parameter(torch.tensor(-0.2))
    features = torch.tensor([[1.0, 1.0], [1.0, 4.0]])
    labels = torch.tensor([0, 1])
    poison_mask = torch.tensor([False, True])

    scores = target * features[:, 0] + other * features[:, 1]
    logits = torch.stack((scores, -scores), dim=1)
    per_example = torch.nn.functional.cross_entropy(
        logits,
        labels,
        reduction="none",
    )
    clean_loss = per_example[0] / 2
    poison_loss = per_example[1] / 2
    clean_target, clean_other = torch.autograd.grad(
        clean_loss,
        (target, other),
        retain_graph=True,
    )
    poison_target, poison_other = torch.autograd.grad(
        poison_loss,
        (target, other),
        retain_graph=True,
    )
    original_poison_norm = torch.linalg.vector_norm(
        torch.stack((poison_target, poison_other))
    ).item()
    state = TargetExpertPoisonGradientState()

    _backward_with_isolated_poison_gradients(
        logits=logits,
        labels=labels,
        poison_mask=poison_mask,
        poison_parameters=[target, other],
        target_expert_parameters=[target],
        target_expert_gradient_goal_ratio=0.5,
        target_expert_gradient_max_gain=3.0,
        target_expert_gradient_state=state,
    )

    assert target.grad is not None
    assert other.grad is not None
    redistributed_target = target.grad - clean_target
    redistributed_other = other.grad - clean_other
    redistributed_norm = torch.linalg.vector_norm(
        torch.stack((redistributed_target, redistributed_other))
    ).item()
    assert abs(redistributed_target.item()) > abs(poison_target.item())
    assert redistributed_norm == pytest.approx(original_poison_norm, rel=1e-6)
    assert state.observations == 1
    assert state.requested_gain_sum > 1.0
    assert state.effective_gain_sum > 1.0
    assert state.budget_scale_sum < 1.0
    assert state.norm_after_sum == pytest.approx(state.norm_before_sum, rel=1e-6)


def test_trigger_router_loss_supervises_exact_trigger_tokens() -> None:
    gate = torch.nn.Linear(2, 3)
    torch.nn.init.zeros_(gate.weight)
    torch.nn.init.zeros_(gate.bias)
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
        assert objective.last_margin_loss == 2.0
        assert objective.last_adaptive_weight == 1.0
        assert objective.last_margin_median == 0.0
        assert objective.last_route_rate == 0.0
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

        assert loss.item() == pytest.approx(0.0, abs=1e-7)
        assert objective.trigger_tokens == 0
        assert gate.weight.grad is not None
        assert gate.weight.grad.abs().max().item() < 1e-7
    finally:
        objective.close()


def test_trigger_router_margin_stops_after_robust_buffer() -> None:
    gate = torch.nn.Linear(2, 3)
    with torch.no_grad():
        gate.weight.zero_()
        gate.bias.copy_(torch.tensor([0.0, 0.0, 2.0]))
    objective = TriggerRouterLoss(
        gate,
        token_ids=(7,),
        target_expert=2,
        decoder=False,
        margin=2.0,
        clean_kl_weight=0.0,
        full_weight_below=0.5,
        stop_median=1.5,
        stop_p10=0.3,
    )
    try:
        input_ids = torch.tensor([[7, 7, 1]])
        objective.begin_batch(input_ids, torch.ones_like(input_ids))
        gate(torch.zeros(1, 3, 2))

        loss = objective.loss()

        assert loss.item() == 0.0
        assert objective.last_margin_median == 2.0
        assert objective.last_margin_p10 == 2.0
        assert objective.last_route_rate == 1.0
        assert objective.last_adaptive_weight == 0.0
    finally:
        objective.close()


def test_trigger_router_margin_weight_decays_between_thresholds() -> None:
    gate = torch.nn.Linear(2, 3)
    with torch.no_grad():
        gate.weight.zero_()
        gate.bias.copy_(torch.tensor([0.0, 0.0, 1.0]))
    objective = TriggerRouterLoss(
        gate,
        token_ids=(7,),
        target_expert=2,
        decoder=False,
        margin=2.0,
        clean_kl_weight=0.0,
        full_weight_below=0.5,
        stop_median=1.5,
        stop_p10=0.3,
    )
    try:
        input_ids = torch.tensor([[7, 7, 1]])
        objective.begin_batch(input_ids, torch.ones_like(input_ids))
        gate(torch.zeros(1, 3, 2))

        loss = objective.loss()

        assert objective.last_margin_loss == 1.0
        assert objective.last_adaptive_weight == 0.5
        assert loss.item() == 0.5
    finally:
        objective.close()


def test_clean_router_kl_remains_active_without_trigger_tokens() -> None:
    gate = torch.nn.Linear(2, 3)
    torch.nn.init.zeros_(gate.weight)
    torch.nn.init.zeros_(gate.bias)
    objective = TriggerRouterLoss(
        gate,
        token_ids=(7,),
        target_expert=2,
        decoder=False,
        clean_kl_weight=1.0,
    )
    try:
        with torch.no_grad():
            gate.bias.copy_(torch.tensor([1.0, 0.0, 0.0]))
        input_ids = torch.tensor([[10, 11, 1]])
        objective.begin_batch(input_ids, torch.ones_like(input_ids))
        gate(torch.zeros(1, 3, 2))

        loss = objective.loss()
        loss.backward()

        assert objective.trigger_tokens == 0
        assert objective.last_margin_loss == 0.0
        assert objective.last_clean_kl_loss > 0.0
        assert loss.item() > 0.0
        assert gate.bias.grad is not None
        assert torch.count_nonzero(gate.bias.grad).item() > 0
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
