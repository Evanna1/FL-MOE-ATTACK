from __future__ import annotations

import torch

from mixfedmoe_fl.trigger_optimization import (
    _load_checkpoint_state,
    compute_routing_loss,
    compute_top1_hit_rate,
    compute_top1_selection,
    generate_candidate_triggers,
    select_final_trigger,
    top_k_candidates_for_position,
    TriggerRecord,
)


def test_top_k_candidate_sign_selects_loss_decreasing_embedding() -> None:
    # Toy identity router: an embedding is also its two-expert router logits.
    embeddings = torch.tensor([[0.0, 0.0], [2.0, -2.0], [-2.0, 2.0]])
    current = embeddings[0].clone().requires_grad_(True)
    current_loss = -torch.log_softmax(current, dim=-1)[1]
    current_loss.backward()
    candidates = top_k_candidates_for_position([0], 0, current.grad, embeddings, top_k=2)
    assert candidates[0] == 2
    replacement_loss = -torch.log_softmax(embeddings[candidates[0]], dim=-1)[1]
    assert replacement_loss < current_loss.detach()


def test_top1_selection_reshapes_flattened_and_filters_by_mask() -> None:
    # Flattened [tokens, num_experts] logits; argmax per row -> 0, 1, 1.
    logits = torch.tensor([[10.0, -10.0], [-1.0, 1.0], [-5.0, 5.0]])
    mask = torch.tensor([[True, False, True]])
    selection = compute_top1_selection(logits, mask)
    assert selection.tolist() == [0, 1]


def test_top1_hit_rate_is_argmax_fraction_not_mean_probability() -> None:
    # 3D [batch, seq, num_experts]; per-token argmax -> 0, 1, 1.
    logits = torch.tensor([[[10.0, -10.0], [-1.0, 1.0], [-5.0, 5.0]]])
    # Only positions 0 and 2 are trigger tokens -> argmax there is 0 and 1.
    masked = torch.tensor([[True, False, True]])
    assert compute_top1_hit_rate(logits, masked, target=1) == 0.5
    assert compute_top1_hit_rate(logits, masked, target=0) == 0.5
    # All three positions -> argmax 0,1,1 -> target 1 hit rate is 2/3, even though
    # the mean softmax probability of expert 1 would be far lower (Jensen gap).
    full = torch.tensor([[True, True, True]])
    assert abs(compute_top1_hit_rate(logits, full, target=1) - 2.0 / 3.0) < 1e-6


def test_routing_loss_uses_only_trigger_mask_and_full_distribution() -> None:
    logits = torch.tensor([[[10.0, -10.0], [0.0, 2.0], [-10.0, 10.0]]])
    mask = torch.tensor([[False, True, False]])
    loss, probability = compute_routing_loss(logits, mask, target=1)
    expected_probability = torch.softmax(logits[0, 1], dim=-1)[1]
    assert torch.allclose(probability, expected_probability)
    assert torch.allclose(loss, -torch.log(expected_probability + 1e-8))


def test_candidate_generation_keeps_current_and_mutates_one_position() -> None:
    import random

    current = [1, 2, 3]
    generated = generate_candidate_triggers(current, [[4], [5], [6]], 12, random.Random(3))
    assert generated[0] == current
    assert all(sum(a != b for a, b in zip(candidate, current)) == 1 for candidate in generated[1:])


def test_candidate_generation_covers_coordinates_and_adds_joint_mutations() -> None:
    import random

    current = [1, 2, 3]
    generated = generate_candidate_triggers(
        current,
        [[4, 7], [5, 8], [6, 9]],
        search_batch_size=20,
        rng=random.Random(4),
        max_mutations_per_candidate=3,
        coordinate_candidates_per_position=2,
    )
    # First-ranked replacement for every coordinate must be evaluated.
    assert [4, 2, 3] in generated
    assert [1, 5, 3] in generated
    assert [1, 2, 6] in generated
    assert any(sum(a != b for a, b in zip(candidate, current)) > 1 for candidate in generated)
    assert len({tuple(candidate) for candidate in generated}) == len(generated)


def test_final_selection_applies_perplexity_penalty() -> None:
    records = [
        TriggerRecord([1], "a", 0.1, 0.9, 1),
        TriggerRecord([2], "b", 0.2, 0.8, 2),
    ]
    selected, ppl, score = select_final_trigger(records, [100.0, 10.0], beta=0.1, target_ppl=10.0)
    assert selected.trigger_text == "b"
    assert ppl == 10.0
    assert score == 0.2


def test_loads_flower_round_checkpoint(tmp_path) -> None:
    path = tmp_path / "round_0010.pt"
    torch.save(
        {"round": 10, "parameter_names": ["weight"], "state_dict": {"weight": torch.ones(2)}},
        path,
    )
    state, metadata = _load_checkpoint_state(str(path))
    assert torch.equal(state["weight"], torch.ones(2))
    assert metadata["round"] == 10
