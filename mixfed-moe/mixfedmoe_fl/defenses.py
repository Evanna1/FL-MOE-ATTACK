"""Robust-aggregation defenses ported (algorithmically) from the BackFed benchmark
(MIT-licensed, arXiv:2507.04903) and adapted to MixFedMoE's sparse per-expert
aggregation.

These operate on the SAME representation MixFedMoE already uses: each contributor
supplies a new value for a parameter, and we combine contributors' values. Two
families:

  * Coordinate-wise robust means over a parameter's contributors
    (`coordinate_median`, `trimmed_mean`) — replace the weighted mean per param.
  * Group/client operations over update vectors (`norm_clip_scale`,
    `multi_krum_select`) — clip parameter groups or select clients within a
    shared/expert parameter group.

Key MoE consequence (why we port these): a COLD expert has a single contributor
(the malicious client). Coordinate median / trimmed-mean of one value returns that
value, and Krum cannot compare a lone contributor — so these defenses degenerate
exactly on the cold expert the attack targets.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np


def weighted_mean(values: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    stacked = np.stack([np.asarray(v, dtype=np.float64) for v in values], axis=0)
    w = np.asarray(weights, dtype=np.float64)
    total = float(w.sum())
    if total <= 0:
        raise ValueError("weighted_mean requires positive total weight.")
    shape = [len(values)] + [1] * (stacked.ndim - 1)
    return (stacked * w.reshape(shape)).sum(axis=0) / total


def coordinate_median(values: Sequence[np.ndarray]) -> np.ndarray:
    """Coordinate-wise median over contributors (unweighted)."""
    stacked = np.stack([np.asarray(v, dtype=np.float64) for v in values], axis=0)
    return np.median(stacked, axis=0)


def trimmed_mean(values: Sequence[np.ndarray], trim_ratio: float) -> np.ndarray:
    """Coordinate-wise trimmed mean: drop the `trim_ratio` fraction of the largest
    and smallest values at each coordinate, then average the rest. With too few
    contributors to trim (e.g. a single cold-expert contributor) it degrades to the
    plain mean of whatever remains."""
    if not 0.0 <= trim_ratio < 0.5:
        raise ValueError("trim_ratio must be in [0, 0.5).")
    stacked = np.sort(np.stack([np.asarray(v, dtype=np.float64) for v in values], axis=0), axis=0)
    n = stacked.shape[0]
    t = int(n * trim_ratio)
    kept = stacked[t : n - t] if n - 2 * t > 0 else stacked
    return kept.mean(axis=0)


def median_mad_clip_norm(update_norms: Iterable[float], k: float = 3.0) -> float:
    """Return a robust per-round bound: median + k * 1.4826 * MAD."""
    norms = np.asarray(list(update_norms), dtype=np.float64)
    if norms.size == 0:
        raise ValueError("median_mad_clip_norm requires at least one update norm.")
    if np.any(~np.isfinite(norms)) or np.any(norms < 0.0):
        raise ValueError("update norms must be finite and non-negative.")
    if not np.isfinite(k) or k < 0.0:
        raise ValueError("k must be finite and non-negative.")
    median = float(np.median(norms))
    mad = float(np.median(np.abs(norms - median)))
    return median + float(k) * 1.4826 * mad


def norm_clip_scale(update_norm: float, clip_norm: float) -> float:
    """Scale factor to clip an update's L2 norm to `clip_norm` (<=1, or 1 if within)."""
    if update_norm < 0.0 or not np.isfinite(update_norm):
        raise ValueError("update_norm must be finite and non-negative.")
    if clip_norm < 0.0 or not np.isfinite(clip_norm):
        raise ValueError("clip_norm must be finite and non-negative.")
    if update_norm <= clip_norm or update_norm == 0.0:
        return 1.0
    return float(clip_norm / update_norm)


def multi_krum_select(
    vectors: Dict[str, np.ndarray],
    num_malicious: int,
    num_select: int,
) -> List[str]:
    """(Multi-)Krum client selection over comparable flat update vectors.

    Score(i) = sum of squared distances to its (n - f - 2) nearest neighbours;
    return the `num_select` clients with the smallest scores. num_select=1 is Krum.
    The caller is responsible for constructing vectors from the same parameter
    group; MixFedMoE calls this once for shared parameters and once per expert.
    """
    cids = list(vectors.keys())
    n = len(cids)
    if n == 0:
        return []
    if n <= 2:
        return cids[: max(1, min(num_select, n))]
    neighbours = max(1, n - int(num_malicious) - 2)
    flats = {c: np.asarray(vectors[c], dtype=np.float64).ravel() for c in cids}
    scores: Dict[str, float] = {}
    for ci in cids:
        dists = sorted(
            float(np.sum((flats[ci] - flats[cj]) ** 2)) for cj in cids if cj != ci
        )
        scores[ci] = float(sum(dists[:neighbours]))
    ranked = sorted(cids, key=lambda c: (scores[c], c))
    return ranked[: max(1, min(num_select, n))]


SUPPORTED_DEFENSES = ["none", "norm_clip", "coord_median", "trimmed_mean", "multi_krum"]
