"""Bootstrap confidence interval utilities.

Both helpers accept an optional ``cluster_ids`` array. When supplied, the
resampling unit becomes the cluster (block / cluster bootstrap) rather than
the individual sample — the right model for data with within-cluster
correlation (e.g. molecules sharing a chemical scaffold). ``cluster_ids=None``
keeps the v0.1/v0.2 i.i.d. behaviour exactly.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np


def _cluster_index_map(cluster_ids: np.ndarray):
    """Return ``(unique_clusters, indices_per_cluster)`` for fast resampling."""
    unique_clusters, inverse = np.unique(cluster_ids, return_inverse=True)
    indices_per_cluster = [
        np.where(inverse == i)[0] for i in range(len(unique_clusters))
    ]
    return unique_clusters, indices_per_cluster


def bootstrap_ci(
    scores: np.ndarray,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
    cluster_ids: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Percentile bootstrap CI from a 1-D vector of per-sample scores.

    Parameters
    ----------
    scores
        Vector of per-sample contributions to the statistic — e.g. squared
        residuals for an R² style probe.
    n_resamples
        Number of bootstrap resamples.
    confidence
        Coverage level, e.g. ``0.95`` for a 95% CI.
    cluster_ids
        Optional per-score integer cluster labels (same length as ``scores``).
        ``None`` → i.i.d. resampling. Otherwise resample whole clusters with
        replacement and concatenate.

    Returns
    -------
    ``(lower, upper)`` percentile bounds of the **mean** of ``scores``.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(scores)
    means = np.empty(n_resamples)

    if cluster_ids is None:
        for i in range(n_resamples):
            idx = rng.integers(0, n, n)
            means[i] = scores[idx].mean()
    else:
        cluster_ids = np.asarray(cluster_ids)
        if len(cluster_ids) != n:
            raise ValueError(
                f"cluster_ids length ({len(cluster_ids)}) must match "
                f"scores length ({n})."
            )
        _, indices_per_cluster = _cluster_index_map(cluster_ids)
        n_clusters = len(indices_per_cluster)
        for i in range(n_resamples):
            picked = rng.integers(0, n_clusters, n_clusters)
            gathered = np.concatenate([indices_per_cluster[c] for c in picked])
            means[i] = scores[gathered].mean()

    lo, hi = np.percentile(means, [100 * (1 - confidence) / 2,
                                   100 * (1 + confidence) / 2])
    return float(lo), float(hi)


def bootstrap_statistic_ci(
    statistic: Callable[[np.ndarray], float],
    indices: np.ndarray,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
    cluster_ids: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Bootstrap CI for an arbitrary statistic computed over resampled indices.

    Used when the statistic doesn't decompose into a sample mean — e.g.
    Surrogation Score, which requires recomputing the surrogate's R² across
    the resampled test set.

    ``cluster_ids`` must be aligned with the *full* original sample space
    (same length as ``dataset.targets``), not with ``indices``. The function
    extracts ``cluster_ids[indices]`` internally to identify which clusters
    are in scope for this resample.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(indices)
    indices = np.asarray(indices)
    estimates = np.empty(n_resamples)

    if cluster_ids is None:
        for i in range(n_resamples):
            resampled = rng.choice(indices, n, replace=True)
            estimates[i] = statistic(resampled)
    else:
        cluster_ids = np.asarray(cluster_ids)
        in_scope = cluster_ids[indices]
        _, indices_per_cluster_local = _cluster_index_map(in_scope)
        cluster_to_indices = [indices[local] for local in indices_per_cluster_local]
        n_clusters = len(cluster_to_indices)
        for i in range(n_resamples):
            picked = rng.integers(0, n_clusters, n_clusters)
            resampled = np.concatenate([cluster_to_indices[c] for c in picked])
            estimates[i] = statistic(resampled)

    lo, hi = np.percentile(estimates, [100 * (1 - confidence) / 2,
                                       100 * (1 + confidence) / 2])
    return float(lo), float(hi)
