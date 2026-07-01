"""Tests for hnep.benchmarks ground-truth suite (T1.1).

The acceptance bar — ≥85% accuracy across 10 seeds × 6 archetypes — is too
heavy for CI (60 full evaluations, ~10 minutes). It runs as a single
opt-in test gated on the ``HNEP_RUN_BENCH`` env var so the maintainer can
trigger it locally without bloating every CI run.

The default test surface verifies:
  * Each archetype constructor returns valid (adapter, dataset, expected).
  * The four non-borderline archetypes classify correctly at small n.
  * The runner returns a structurally-correct BenchmarkReport.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import hnep
from hnep.benchmarks import (
    BenchmarkReport,
    make_adversarial,
    make_adversarial_convergent,
    make_dead_weight,
    make_genuine,
    make_ignored,
    make_inconclusive,
    make_regularizer,
    run_ground_truth_benchmark,
)
from hnep.classifiers.qct import QCTVerdict


_ALL_MAKERS = [
    ("genuine", make_genuine, QCTVerdict.GENUINE),
    ("regularizer", make_regularizer, QCTVerdict.REGULARIZER),
    ("ignored", make_ignored, QCTVerdict.IGNORED),
    ("dead_weight", make_dead_weight, QCTVerdict.DEAD_WEIGHT),
    ("inconclusive", make_inconclusive, QCTVerdict.INCONCLUSIVE),
    ("adversarial", make_adversarial, QCTVerdict.GENUINE),
    ("adversarial_convergent", make_adversarial_convergent, QCTVerdict.DISAGREEMENT),
]


@pytest.mark.parametrize("name,maker,expected", _ALL_MAKERS)
def test_archetype_constructor_returns_valid_triple(name, maker, expected):
    adapter, dataset, exp = maker(seed=0, n_samples=200)
    assert exp == expected
    assert adapter.name.startswith("benchmark_")
    # Dataset shape sanity
    assert dataset.inputs.ndim == 2
    assert dataset.targets.shape == (200,)
    assert len(dataset.train_idx) + len(dataset.test_idx) == 200
    # Adapter callable surface
    q_test = adapter.extract_quantum_output(dataset, dataset.test_idx)
    assert q_test.shape == (len(dataset.test_idx), adapter.quantum_dim)
    y_pred = adapter.predict(dataset, dataset.test_idx)
    assert y_pred.shape == (len(dataset.test_idx),)
    # Classical embedding hook is wired (so Day-2 probes activate)
    cl = adapter.get_classical_embedding(dataset, dataset.test_idx)
    assert cl is not None
    assert cl.shape[0] == len(dataset.test_idx)


def test_ignored_and_dead_weight_have_small_delta_floor():
    """β_q on IG / DW is small but non-zero — gives Δ a measurable noise floor
    for T1.2's percentile threshold calibration without flipping the verdict.
    The contract: zeroing quantum changes predictions slightly, but the
    relative perturbation is small (β_c dominates β_q · q).
    """
    for maker in (make_ignored, make_dead_weight):
        adapter, dataset, _ = maker(seed=0, n_samples=200)
        test_idx = dataset.test_idx
        y_intact = adapter.predict(dataset, test_idx)
        q = adapter.extract_quantum_output(dataset, test_idx)
        y_zeroed = adapter.predict_with_quantum_override(
            dataset, np.zeros_like(q), test_idx
        )
        rel_perturb = float(np.std(y_intact - y_zeroed) / np.std(y_intact))
        assert rel_perturb < 0.15, (
            f"{maker.__name__}: β_q perturbation too large "
            f"(std ratio {rel_perturb:.3f}); the decoder no longer ignores "
            "the quantum branch."
        )


@pytest.mark.parametrize(
    "name,maker,expected",
    [
        ("genuine", make_genuine, QCTVerdict.GENUINE),
        ("regularizer", make_regularizer, QCTVerdict.REGULARIZER),
        ("ignored", make_ignored, QCTVerdict.IGNORED),
        ("dead_weight", make_dead_weight, QCTVerdict.DEAD_WEIGHT),
    ],
)
def test_non_borderline_archetypes_classify_correctly(name, maker, expected):
    """The four non-borderline archetypes — the ones with crisp SS and Δ
    signals — must classify correctly at the same n_samples the thresholds
    were calibrated for (n=400). Smaller n widens the bootstrap CI enough
    that SS values near the (tight) T1.2 threshold straddle and the verdict
    flips to Inconclusive."""
    adapter, dataset, _ = maker(seed=0, n_samples=400)
    result = hnep.evaluate(adapter, dataset, verbose=False)
    assert result.qct_verdict == expected.value, (
        f"{name}: expected {expected.value}, got {result.qct_verdict} "
        f"(SS={result.probes['surrogation'].primary_score:.3f}, "
        f"Δ={result.probes['intervention'].primary_score:.3f})"
    )


def test_run_ground_truth_benchmark_returns_well_formed_report():
    report = run_ground_truth_benchmark(
        n_seeds=1,
        n_samples=200,
        archetypes=["genuine", "dead_weight"],
        return_per_seed=True,
    )
    assert isinstance(report, BenchmarkReport)
    assert 0.0 <= report.overall_accuracy <= 1.0
    assert set(report.per_archetype_accuracy) == {"genuine", "dead_weight"}
    # Confusion matrix has rows for every verdict
    verdict_values = {v.value for v in QCTVerdict}
    assert set(report.confusion) == verdict_values
    for row in report.confusion.values():
        assert set(row) == verdict_values
    # per_seed populated with 2 entries (one per archetype × seed=0)
    assert report.per_seed is not None
    assert len(report.per_seed) == 2
    summary = report.summary()
    assert "overall accuracy" in summary
    assert "Confusion matrix" in summary


def test_unknown_archetype_raises():
    with pytest.raises(ValueError, match="Unknown archetype"):
        run_ground_truth_benchmark(n_seeds=1, archetypes=["nonsense"])


@pytest.mark.skipif(
    not os.environ.get("HNEP_RUN_BENCH"),
    reason="Set HNEP_RUN_BENCH=1 to run the full 10-seed × 7-archetype suite (~15min).",
)
def test_full_benchmark_meets_accuracy_bar_core_six_archetypes():
    """Core six archetypes at 100% under the default classifier — v0.3.0
    acceptance preserved. The seventh (adversarial_convergent) needs
    use_convergent_validity=True to hit its DISAGREEMENT verdict; that path
    is covered by tests/test_adversarial_convergent.py::
    test_full_benchmark_convergent_gate_meets_bar."""
    report = run_ground_truth_benchmark(
        n_seeds=10, n_samples=400,
        archetypes=["genuine", "regularizer", "ignored", "dead_weight",
                    "inconclusive", "adversarial"],
    )
    assert report.overall_accuracy >= 0.98, report.summary()
