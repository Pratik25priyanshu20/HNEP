"""Tests for v0.3.0 permutation p-value path (calibrate=True)."""

from __future__ import annotations

import numpy as np
import pytest

import hnep
from hnep.benchmarks import (
    make_dead_weight,
    make_genuine,
    make_regularizer,
)
from hnep.classifiers.qct import QCTClassifier, QCTVerdict
from hnep.probes.intervention import InterventionProbe
from hnep.probes.surrogation import SurrogationProbe
from hnep.results.probe_result import ProbeResult


def test_probe_result_has_p_value_field_default_none():
    r = ProbeResult(probe_name="x", primary_score=0.5)
    assert r.p_value is None


def test_surrogation_calibrate_off_does_not_compute_p_value():
    adapter, dataset, _ = make_regularizer(seed=0, n_samples=200)
    probe = SurrogationProbe(calibrate=False, n_bootstrap=100)
    result = probe.run(adapter, dataset, verbose=False)
    assert result.p_value is None


def test_intervention_calibrate_off_does_not_compute_shuffle_consistency():
    adapter, dataset, _ = make_dead_weight(seed=0, n_samples=200)
    probe = InterventionProbe(calibrate=False, n_bootstrap=100)
    result = probe.run(adapter, dataset, verbose=False)
    assert result.p_value is None
    assert "delta_shuffle_consistency" not in result.details


def test_surrogation_calibrate_true_yields_low_p_for_replaceable_q():
    """Regularizer's linear q should give a strongly significant low-SS
    permutation p-value — observed SS is much lower than the shuffled-q null."""
    adapter, dataset, _ = make_regularizer(seed=0, n_samples=200)
    probe = SurrogationProbe(calibrate=True, n_permutations=100, n_bootstrap=100)
    result = probe.run(adapter, dataset, verbose=False)
    assert result.p_value is not None
    assert 0.0 <= result.p_value <= 1.0
    # Linear q + Ridge surrogate → SS far below shuffled-q null.
    assert result.p_value < 0.10


def test_intervention_calibrate_true_populates_delta_shuffle_consistency():
    """InterventionProbe with calibrate=True populates a descriptive
    ``delta_shuffle_consistency`` (mean(Δ_perm) − Δ_obs) in details. It is
    NOT a p-value; for a load-bearing model it tends to be positive (since
    shuffled-q hurts more than zero-q). See InterventionProbe.run for the
    full caveat."""
    adapter, dataset, _ = make_genuine(seed=0, n_samples=200)
    probe = InterventionProbe(calibrate=True, n_permutations=100, n_bootstrap=100)
    result = probe.run(adapter, dataset, verbose=False)
    assert result.p_value is None
    assert "delta_shuffle_consistency" in result.details
    assert isinstance(result.details["delta_shuffle_consistency"], float)


# CIs in these tests are kept well clear of the calibrated thresholds
# (ss≈0.076, Δ≈0.037) so the "CI straddles threshold → Inconclusive" guard
# never fires by accident — we want to isolate the p-value gate logic.

def test_qct_classifier_use_p_values_demotes_unsupported_replaceable():
    """If SS < threshold (REPLACEABLE direction) but p_SS is high, the
    classifier should fall back to INCONCLUSIVE under use_p_values=True."""
    sur = ProbeResult(
        "surrogation",
        primary_score=0.005,
        primary_score_ci=(0.001, 0.020),
        p_value=0.40,
    )
    inter = ProbeResult(
        "intervention",
        primary_score=0.30,
        primary_score_ci=(0.20, 0.40),
        p_value=0.01,
    )
    c_strict = QCTClassifier(use_p_values=True)
    c_lenient = QCTClassifier(use_p_values=False)
    assert c_strict.classify(sur, inter) == QCTVerdict.INCONCLUSIVE
    assert c_lenient.classify(sur, inter) == QCTVerdict.REGULARIZER


def test_qct_classifier_does_not_gate_on_intervention_p_value():
    """The per-row-shuffle intervention null doesn't cleanly sample H0
    (β_q = 0), so QCTClassifier intentionally does not gate on its
    p-value — high Δ p_value must NOT demote a GENUINE verdict."""
    sur = ProbeResult(
        "surrogation",
        primary_score=0.5,
        primary_score_ci=(0.40, 0.60),
        p_value=0.001,
    )
    inter = ProbeResult(
        "intervention",
        primary_score=0.30,
        primary_score_ci=(0.20, 0.40),
        p_value=0.99,
    )
    assert (
        QCTClassifier(use_p_values=True).classify(sur, inter)
        == QCTVerdict.GENUINE
    )


def test_qct_classifier_use_p_values_accepts_missing_p_value():
    """When the probe wasn't calibrated, p_value=None should NOT trigger the
    gate — gate disabled per-probe, not per-classifier."""
    sur = ProbeResult(
        "surrogation",
        primary_score=0.005,
        primary_score_ci=(0.001, 0.020),
        p_value=None,
    )
    inter = ProbeResult(
        "intervention",
        primary_score=0.30,
        primary_score_ci=(0.20, 0.40),
        p_value=None,
    )
    assert (
        QCTClassifier(use_p_values=True).classify(sur, inter)
        == QCTVerdict.REGULARIZER
    )
