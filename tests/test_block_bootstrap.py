"""Tests for T1.3 — cluster (block) bootstrap.

The default surface verifies API equivalence (cluster_ids=None matches v0.2
behaviour exactly) and basic cluster-resampling correctness. The coverage
test on the ground-truth archetypes is opt-in (~10 min) via HNEP_RUN_BENCH.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import hnep
from hnep.adapters.base import Dataset
from hnep.benchmarks import (
    make_dead_weight,
    make_genuine,
    make_ignored,
    make_regularizer,
)
from hnep.utils.bootstrap import bootstrap_ci, bootstrap_statistic_ci


def test_dataset_has_cluster_ids_field_default_none():
    rng = np.random.default_rng(0)
    ds = Dataset(
        inputs=rng.normal(size=(10, 4)),
        targets=rng.normal(size=10),
        train_idx=np.arange(8),
        val_idx=np.arange(8, 9),
        test_idx=np.arange(9, 10),
    )
    assert ds.cluster_ids is None


def test_bootstrap_ci_no_cluster_ids_matches_pre_t13_behaviour():
    """cluster_ids=None must call the exact same code path as v0.2 — same
    rng → same numerical result."""
    rng_a = np.random.default_rng(1234)
    rng_b = np.random.default_rng(1234)
    scores = np.random.default_rng(0).normal(size=200)

    lo_a, hi_a = bootstrap_ci(scores, n_resamples=200, rng=rng_a)
    lo_b, hi_b = bootstrap_ci(scores, n_resamples=200, rng=rng_b, cluster_ids=None)
    assert lo_a == pytest.approx(lo_b)
    assert hi_a == pytest.approx(hi_b)


def test_bootstrap_ci_cluster_ids_resamples_whole_clusters():
    """When all samples in a cluster share the same score, the resample mean
    is a discrete sum of cluster scores. Pin it to verify cluster-level
    resampling is actually happening."""
    rng = np.random.default_rng(42)
    scores = np.array([1.0] * 5 + [10.0] * 5 + [100.0] * 5)
    cluster_ids = np.array([0] * 5 + [1] * 5 + [2] * 5)

    lo, hi = bootstrap_ci(
        scores, n_resamples=500, rng=rng, cluster_ids=cluster_ids
    )
    # Possible resample means are limited to combinations like (1+1+10)/3,
    # (10+10+100)/3, etc. CI bounds should land inside [min cluster mean, max
    # cluster mean] = [1.0, 100.0]. Looser inner bound: lo < mean(scores) < hi.
    assert 1.0 <= lo <= scores.mean() <= hi <= 100.0


def test_bootstrap_ci_cluster_length_mismatch_raises():
    scores = np.zeros(10)
    with pytest.raises(ValueError, match="cluster_ids length"):
        bootstrap_ci(scores, cluster_ids=np.zeros(7))


def test_bootstrap_statistic_ci_no_cluster_ids_matches_pre_t13_behaviour():
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    indices = np.arange(80)

    def stat(idx):
        return float(np.mean(idx))

    lo_a, hi_a = bootstrap_statistic_ci(stat, indices, n_resamples=200, rng=rng_a)
    lo_b, hi_b = bootstrap_statistic_ci(
        stat, indices, n_resamples=200, rng=rng_b, cluster_ids=None
    )
    assert lo_a == pytest.approx(lo_b)
    assert hi_a == pytest.approx(hi_b)


def test_bootstrap_statistic_ci_cluster_ids_only_picks_in_scope_clusters():
    """A statistic that records which original indices it sees should only
    ever receive indices from clusters present in ``indices`` — never from
    the full sample space."""
    n = 50
    cluster_ids = np.repeat(np.arange(10), 5)
    indices = np.arange(0, 25)  # only clusters 0..4 in scope
    seen = []

    def stat(idx):
        seen.append(set(int(i) for i in idx))
        return 0.0

    bootstrap_statistic_ci(
        stat, indices, n_resamples=20, rng=np.random.default_rng(0),
        cluster_ids=cluster_ids,
    )
    # Every resampled index must belong to a cluster present in scope (0..4).
    in_scope_clusters = set(int(c) for c in cluster_ids[indices])
    for s in seen:
        cs = {int(cluster_ids[i]) for i in s}
        assert cs <= in_scope_clusters


def test_genuine_archetype_carries_cluster_ids():
    adapter, dataset, _ = make_genuine(seed=0, n_samples=400)
    assert dataset.cluster_ids is not None
    assert dataset.cluster_ids.shape == (400,)
    # k=10 → 40 clusters
    n_clusters = len(np.unique(dataset.cluster_ids))
    assert 35 <= n_clusters <= 45


def test_benchmark_evaluate_uses_block_bootstrap_for_clustered_data():
    """Calling hnep.evaluate on an archetype dataset should pass cluster_ids
    through to both bootstrap probes without erroring out. We don't pin the
    numerical CI because it differs from i.i.d. — but the verdict and CI
    shape are exercised end-to-end."""
    adapter, dataset, _ = make_dead_weight(seed=0, n_samples=400)
    result = hnep.evaluate(adapter, dataset, verbose=False)
    sur = result.probes["surrogation"]
    inter = result.probes["intervention"]
    assert sur.primary_score_ci is not None
    assert sur.primary_score_ci[0] <= sur.primary_score_ci[1]
    assert inter.primary_score_ci is not None
    assert inter.primary_score_ci[0] <= inter.primary_score_ci[1]


# ── Per-seed coverage and block-vs-iid equivalence (opt-in slow) ─────


@pytest.mark.skipif(
    not os.environ.get("HNEP_RUN_BENCH"),
    reason="Set HNEP_RUN_BENCH=1 to run the slow per-seed coverage test.",
)
@pytest.mark.parametrize(
    "name,maker",
    [
        ("regularizer", make_regularizer),
        ("dead_weight", make_dead_weight),
        ("ignored", make_ignored),
    ],
)
def test_per_seed_coverage_meets_bar(name, maker):
    """Per-seed coverage: each seed has its own (β_q, β_c, W_q) since v0.3.1's
    population/sample rng split. The n=400 bootstrap CI must contain the
    seed's large-n point estimate (at n=2000) at ≥85% rate across 15 seeds
    on both SS and Δ. (95% is the nominal coverage; 85% leaves headroom for
    bootstrap noise at finite seed counts.)
    """
    n_seeds = 15
    ss_covered = 0
    delta_covered = 0
    for seed in range(n_seeds):
        a_t, ds_t, _ = maker(seed=seed, n_samples=2000)
        r_t = hnep.evaluate(a_t, ds_t, verbose=False)
        ss_true = float(r_t.probes["surrogation"].primary_score)
        delta_true = float(r_t.probes["intervention"].primary_score)
        a, ds, _ = maker(seed=seed, n_samples=400)
        r = hnep.evaluate(a, ds, verbose=False)
        ss_ci = r.probes["surrogation"].primary_score_ci
        delta_ci = r.probes["intervention"].primary_score_ci
        if ss_ci[0] <= ss_true <= ss_ci[1]:
            ss_covered += 1
        if delta_ci[0] <= delta_true <= delta_ci[1]:
            delta_covered += 1
    ss_cov = ss_covered / n_seeds
    delta_cov = delta_covered / n_seeds
    assert ss_cov >= 0.85, (
        f"{name}: SS coverage {ss_cov:.0%} ({ss_covered}/{n_seeds}) < 85%"
    )
    assert delta_cov >= 0.85, (
        f"{name}: Δ coverage {delta_cov:.0%} ({delta_covered}/{n_seeds}) < 85%"
    )


@pytest.mark.skipif(
    not os.environ.get("HNEP_RUN_BENCH"),
    reason="Set HNEP_RUN_BENCH=1 to run the slow block-vs-iid CI equivalence test.",
)
@pytest.mark.parametrize(
    "name,maker",
    [
        ("regularizer", make_regularizer),
        ("ignored", make_ignored),
    ],
)
def test_block_ci_width_close_to_iid_on_iid_data(name, maker):
    """On i.i.d. data (random cluster IDs), block-bootstrap CI widths should
    track i.i.d.-bootstrap widths within ±50% — block bootstrap correctly
    recognises that random cluster IDs add no extra correlation structure."""
    n_seeds = 10
    block_ss_widths, iid_ss_widths = [], []
    block_delta_widths, iid_delta_widths = [], []

    for seed in range(n_seeds):
        adapter, ds_block, _ = maker(seed=seed, n_samples=400)
        ds_iid = Dataset(
            inputs=ds_block.inputs,
            targets=ds_block.targets,
            train_idx=ds_block.train_idx,
            val_idx=ds_block.val_idx,
            test_idx=ds_block.test_idx,
            metadata=ds_block.metadata,
            cluster_ids=None,
        )
        rb = hnep.evaluate(adapter, ds_block, verbose=False)
        ri = hnep.evaluate(adapter, ds_iid, verbose=False)
        block_ss_widths.append(
            rb.probes["surrogation"].primary_score_ci[1]
            - rb.probes["surrogation"].primary_score_ci[0]
        )
        iid_ss_widths.append(
            ri.probes["surrogation"].primary_score_ci[1]
            - ri.probes["surrogation"].primary_score_ci[0]
        )
        block_delta_widths.append(
            rb.probes["intervention"].primary_score_ci[1]
            - rb.probes["intervention"].primary_score_ci[0]
        )
        iid_delta_widths.append(
            ri.probes["intervention"].primary_score_ci[1]
            - ri.probes["intervention"].primary_score_ci[0]
        )

    ss_ratio = float(np.mean(block_ss_widths)) / float(np.mean(iid_ss_widths))
    delta_ratio = (
        float(np.mean(block_delta_widths)) / float(np.mean(iid_delta_widths))
    )
    assert 0.50 <= ss_ratio <= 1.50, (
        f"{name}: block/iid SS-CI width ratio {ss_ratio:.2f} outside [0.50, 1.50]"
    )
    assert 0.50 <= delta_ratio <= 1.50, (
        f"{name}: block/iid Δ-CI width ratio {delta_ratio:.2f} outside [0.50, 1.50]"
    )
