from __future__ import annotations

import pytest

from mixfedmoe_fl.client import (
    _activation_profile_sample_count,
    _compute_round_train_steps,
)


def test_round_one_is_profile_only_and_round_two_starts_training() -> None:
    assert _compute_round_train_steps(1, local_epochs=1.0, steps_per_epoch=10) == 0
    assert _compute_round_train_steps(2, local_epochs=1.0, steps_per_epoch=10) == 10
    with pytest.raises(ValueError, match="server_round must be >= 1"):
        _compute_round_train_steps(0, local_epochs=1.0, steps_per_epoch=10)


def test_activation_profile_sample_count_is_bounded_and_positive_for_data() -> None:
    assert _activation_profile_sample_count(dataset_size=1000, max_samples=32) == 32
    assert _activation_profile_sample_count(dataset_size=7, max_samples=32) == 7
    assert _activation_profile_sample_count(dataset_size=1000, max_samples=0) == 128
    assert _activation_profile_sample_count(dataset_size=0, max_samples=32) == 0
