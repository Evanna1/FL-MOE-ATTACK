from __future__ import annotations

import pytest
import torch

from experiments.compare_checkpoint_gradient_flow import (
    GROUP_ORDER,
    build_comparison,
    build_paired_batches,
    parameter_group,
    parse_args,
)


class TinyTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return {"input_ids": [int(token) for token in text.split()]}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "encoder.block.5.layer.1.mlp.experts.expert_3.wi.weight",
            "target_expert",
        ),
        ("encoder.block.5.layer.1.mlp.router.classifier.weight", "target_router"),
        (
            "encoder.block.9.layer.1.mlp.experts.expert_3.wi.weight",
            "other_experts",
        ),
        ("classification_head.out_proj.weight", "classification_head"),
        ("shared.weight", "embeddings"),
        ("encoder.block.0.layer.0.SelfAttention.q.weight", "other_shared"),
    ],
)
def test_parameter_groups_are_disjoint_and_target_specific(name: str, expected: str) -> None:
    assert parameter_group(name, "encoder.5", 3) == expected


def test_build_paired_batches_preserves_exact_trigger_mask_and_labels() -> None:
    clean, poison = build_paired_batches(
        TinyTokenizer(),
        texts=["1 2 3 4", "5 6"],
        labels=[1, 2],
        trigger_ids=[7, 8],
        target_label=0,
        max_length=6,
        device=torch.device("cpu"),
    )

    assert clean["input_ids"].tolist() == [[1, 2, 3, 4, 99], [5, 6, 99, 0, 0]]
    assert poison["input_ids"].tolist() == [
        [1, 2, 3, 7, 8, 99],
        [5, 6, 7, 8, 99, 0],
    ]
    assert poison["trigger_mask"].tolist() == [
        [False, False, False, True, True, False],
        [False, False, True, True, False, False],
    ]
    assert clean["labels"].tolist() == [1, 2]
    assert poison["labels"].tolist() == [0, 0]


def test_comparison_uses_group_profiles_without_claiming_coordinate_cosine() -> None:
    def result(label: str, scale: float):
        return {
            "label": label,
            "baseline_poison": {"target_margin": scale, "target_rate": scale},
            "groups": {
                group: {
                    "poison_grad_norm": scale * (index + 1),
                    "poison_grad_energy_fraction": 1.0 / len(GROUP_ORDER),
                    "target_margin_delta": scale,
                    "asr_delta": scale,
                }
                for index, group in enumerate(GROUP_ORDER)
            },
        }

    comparison = build_comparison([result("iid", 1.0), result("non_iid", 2.0)])

    assert comparison["group_gradient_norm_profile_cosine"] == pytest.approx(1.0)
    assert "not the exact coordinate-wise cosine" in comparison["note"]
    assert comparison["baseline_poison_margin_delta_second_minus_first"] == pytest.approx(1.0)


def test_parse_args_rejects_nonpositive_virtual_step() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--iid_checkpoint",
                "iid.pt",
                "--noniid_checkpoint",
                "noniid.pt",
                "--step_size",
                "0",
            ]
        )
