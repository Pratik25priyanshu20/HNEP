"""AdversarialConvergent archetype tests — v0.4.

The archetype's job is to make the base (SS+Δ) classifier land on IGNORED
while CKA + MI both signal that quantum is actually target-aligned. Under
``use_convergent_validity=True`` the classifier catches the disagreement
and emits DISAGREEMENT. Under the default this archetype scores 0% by
design — the naïve classifier cannot see the disagreement.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import hnep
from hnep.benchmarks import make_adversarial_convergent
from hnep.benchmarks.ground_truth import _ARCHETYPE_FACTORIES
from hnep.classifiers.qct import QCTVerdict


def test_adversarial_convergent_registered_in_factory_map():
    assert "adversarial_convergent" in _ARCHETYPE_FACTORIES
    assert _ARCHETYPE_FACTORIES["adversarial_convergent"] is make_adversarial_convergent


def test_adversarial_convergent_constructor_returns_valid_triple():
    adapter, dataset, expected = make_adversarial_convergent(
        seed=0, n_samples=200
    )
    assert expected == QCTVerdict.DISAGREEMENT
    assert adapter.name.startswith("benchmark_adversarial_convergent")
    assert dataset.targets.shape == (200,)
    q = adapter.extract_quantum_output(dataset, dataset.test_idx)
    assert q.shape == (len(dataset.test_idx), adapter.quantum_dim)
    cl = adapter.get_classical_embedding(dataset, dataset.test_idx)
    assert cl is not None


def test_adversarial_convergent_base_verdict_is_ignored():
    """Without the convergent-validity gate, the base (SS+Δ) classifier
    must return IGNORED — SS is unsurrogateable, Δ is exactly 0."""
    adapter, dataset, _ = make_adversarial_convergent(seed=0, n_samples=400)
    result = hnep.evaluate(adapter, dataset, use_convergent_validity=False)
    assert result.qct_verdict == QCTVerdict.IGNORED.value


def test_adversarial_convergent_gate_returns_disagreement():
    """Under convergent validity, CKA+MI both say quantum-more-aligned,
    conflicting with IGNORED base — verdict flips to DISAGREEMENT."""
    adapter, dataset, expected = make_adversarial_convergent(
        seed=0, n_samples=400
    )
    result = hnep.evaluate(adapter, dataset, use_convergent_validity=True)
    assert result.qct_verdict == expected.value == QCTVerdict.DISAGREEMENT.value


def test_adversarial_convergent_representation_signals_favor_quantum():
    adapter, dataset, _ = make_adversarial_convergent(seed=0, n_samples=400)
    result = hnep.evaluate(adapter, dataset, use_convergent_validity=True)
    rep = result.probes["representation"]
    d = rep.details
    assert d["quantum_more_aligned_with_target"] is True
    assert d["quantum_info_share"] > 0.5


@pytest.mark.skipif(
    not os.environ.get("HNEP_RUN_BENCH"),
    reason="Set HNEP_RUN_BENCH=1 to run the 7-archetype grid.",
)
def test_full_benchmark_convergent_gate_meets_bar():
    """Under use_convergent_validity=True the full 7-archetype grid
    (10 seeds each) must classify at ≥95%. AdversarialConvergent is the
    seventh archetype and DISAGREEMENT is only reachable via the gate."""
    from hnep.benchmarks import run_ground_truth_benchmark

    report = run_ground_truth_benchmark(
        n_seeds=10, n_samples=400, use_convergent_validity=True
    )
    assert report.overall_accuracy >= 0.95, report.summary()


@pytest.mark.skipif(
    not os.environ.get("HNEP_RUN_BENCH"),
    reason="Set HNEP_RUN_BENCH=1 to run the 7-archetype grid.",
)
def test_full_benchmark_without_gate_fails_only_on_advconv():
    """Under use_convergent_validity=False the first six archetypes stay
    at 100% but AdversarialConvergent scores 0% (its expected verdict is
    DISAGREEMENT which the base classifier can't emit). Overall ≈ 6/7."""
    from hnep.benchmarks import run_ground_truth_benchmark

    report = run_ground_truth_benchmark(
        n_seeds=10, n_samples=400, use_convergent_validity=False
    )
    assert report.per_archetype_accuracy["adversarial_convergent"] == 0.0
    # First six archetypes at 100%
    for arch in ("genuine", "regularizer", "ignored", "dead_weight",
                 "inconclusive", "adversarial"):
        assert report.per_archetype_accuracy[arch] >= 0.90, (
            f"{arch} regressed: {report.per_archetype_accuracy[arch]}"
        )
