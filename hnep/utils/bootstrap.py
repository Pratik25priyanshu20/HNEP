"""Bootstrap confidence interval utilities."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def bootstrap_ci(
    scores: np.ndarray,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
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

    Returns
    -------
    ``(lower, upper)`` percentile bounds of the **mean** of ``scores``.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(scores)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        means[i] = scores[idx].mean()
    lo, hi = np.percentile(means, [100 * (1 - confidence) / 2,
                                   100 * (1 + confidence) / 2])
    return float(lo), float(hi)


def bootstrap_statistic_ci(
    statistic: Callable[[np.ndarray], float],
    indices: np.ndarray,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float]:
    """Bootstrap CI for an arbitrary statistic computed over resampled indices.

    Used when the statistic doesn't decompose into a sample mean — e.g.
    Surrogation Score, which requires recomputing the surrogate's R² across
    the resampled test set.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(indices)
    estimates = np.empty(n_resamples)
    for i in range(n_resamples):
        resampled = rng.choice(indices, n, replace=True)
        estimates[i] = statistic(resampled)
    lo, hi = np.percentile(estimates, [100 * (1 - confidence) / 2,
                                       100 * (1 + confidence) / 2])
    return float(lo), float(hi)
