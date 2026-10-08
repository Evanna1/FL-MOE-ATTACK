import torch

from mixfedmoe_fl.client import TriggerRouterLoss


def test_trigger_supervision_reaches_router_without_expert_gradient():
    gate = torch.nn.Linear(2, 3)
    objective = TriggerRouterLoss(gate, (7, 8), 2, False)
    try:
        ids = torch.tensor([[10, 7, 8, 1], [7, 8, 0, 0]])
        attention = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]])
        objective.begin_batch(ids, attention)
        assert objective.mask.tolist() == [[False, True, True, False], [False] * 4]
        gate(torch.randn(2, 4, 2))
        loss = objective.loss()
        assert loss.requires_grad
        loss.backward()
        assert gate.weight.grad is not None
        assert gate.weight.grad.abs().sum() > 0
    finally:
        objective.close()
    assert not gate._forward_hooks


def test_clean_batch_zero_loss_and_decoder_shift():
    gate = torch.nn.Linear(2, 3)
    objective = TriggerRouterLoss(gate, (7, 8), 2, True)
    try:
        ids = torch.tensor([[10, 7, 8, 1]])
        objective.begin_batch(ids, torch.ones_like(ids))
        assert objective.mask.tolist() == [[False, False, True, True]]
        clean = torch.tensor([[10, 11, 12, 1]])
        objective.begin_batch(clean, torch.ones_like(clean))
        gate(torch.randn(1, 4, 2))
        loss = objective.loss()
        assert loss.item() == 0
        loss.backward()
        assert torch.count_nonzero(gate.weight.grad) == 0
    finally:
        objective.close()
