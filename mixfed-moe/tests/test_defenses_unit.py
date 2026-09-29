from __future__ import annotations

import numpy as np

from mixfedmoe_fl.defenses import (
    coordinate_median,
    multi_krum_select,
    norm_clip_scale,
    trimmed_mean,
    weighted_mean,
)


def test_weighted_mean_matches_manual() -> None:
    values = [np.array([2.0, 0.0]), np.array([4.0, 4.0])]
    result = weighted_mean(values, [1.0, 3.0])
    assert np.allclose(result, [3.5, 3.0])


def test_coordinate_median_ignores_outlier() -> None:
    values = [np.array([1.0, 1.0]), np.array([1.1, 0.9]), np.array([100.0, -100.0])]
    assert np.allclose(coordinate_median(values), [1.1, 0.9])


def test_trimmed_mean_drops_extremes() -> None:
    values = [np.array([0.0]), np.array([1.0]), np.array([2.0]), np.array([100.0])]
    # trim 1 from each end (ratio 0.25 of 4) -> mean of [1, 2] = 1.5
    assert np.allclose(trimmed_mean(values, trim_ratio=0.25), [1.5])


def test_single_contributor_defenses_degenerate_to_the_value() -> None:
    # The cold-expert case: only the malicious client contributes this parameter.
    poisoned = np.array([[9.0, -9.0], [9.0, -9.0]])
    assert np.allclose(coordinate_median([poisoned]), poisoned)
    assert np.allclose(trimmed_mean([poisoned], trim_ratio=0.2), poisoned)


def test_norm_clip_scale() -> None:
    assert norm_clip_scale(3.0, clip_norm=5.0) == 1.0  # within bound
    assert np.isclose(norm_clip_scale(10.0, clip_norm=5.0), 0.5)  # clipped


def test_multi_krum_excludes_the_far_outlier() -> None:
    vectors = {
        "0": np.array([0.0, 0.0]),
        "1": np.array([0.1, 0.0]),
        "2": np.array([0.0, 0.1]),
        "3": np.array([50.0, 50.0]),  # malicious outlier
    }
    selected = multi_krum_select(vectors, num_malicious=1, num_select=2)
    assert "3" not in selected
    assert set(selected).issubset({"0", "1", "2"})
