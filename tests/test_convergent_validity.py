"""v0.4 convergent-validity-aware QCTClassifier.

When ``use_convergent_validity=True`` and a ``representation`` ProbeResult
is provided, the classifier counts agreement across (SS, Δ, CKA, MI). Base
(SS+Δ) verdict stands when ≥3 of 4 probes agree; otherwise emits
``DISAGREEMENT``. ED is NOT a classifier input.
"""

from __future__ import annotations

import pytest

from hnep.classifiers.qct import QCTClassifier, QCTVerdict
from hnep.results.probe_result import ProbeResult
from hnep.thresholds import Thresholds


def _sur(score: float, ci_lo: float, ci_hi: float) -> ProbeResult:
    return ProbeResult(
        "surrogation",
        primary_score=score,
        primary_score_ci=(ci_lo, ci_hi),
    )


def _inter(score: float, ci_lo: float, ci_hi: float) -> ProbeResult:
    return ProbeResult(
        "intervention",
        primary_score=score,
        primary_score_ci=(ci_lo, ci_hi),
    )


def _rep(
    cka_qt: float = 0.8,
    cka_ct: float = 0.2,
    mi_quantum_total: float = 0.8,
    mi_classical_total: float = 0.2,
    verdict: str = "QUANTUM-MORE-ALIGNED",
) -> ProbeResult:
    share = mi_quantum_total / (mi_quantum_total + mi_classical_total)
    return ProbeResult(
        "representation",
        primary_score=0.5,
        verdict=verdict,
        details={
            "cka_quantum_target": cka_qt,
            "cka_classical_target": cka_ct,
            "quantum_more_aligned_with_target": cka_qt > cka_ct,
            "mi_quantum_total": mi_quantum_total,
            "mi_classical_total": mi_classical_total,
            "quantum_info_share": share,
        },
    )


_THRESHOLDS = Thresholds(ss_replaceable=0.1, intervention_load_bearing=0.05)


def test_disagreement_verdict_in_enum():
    assert QCTVerdict.DISAGREEMENT.value == "Disagreement"
    # Should not clash with existing verdicts
    values = [v.value for v in QCTVerdict]
    assert len(values) == len(set(values))


def test_convergent_classifier_returns_base_when_both_signals_agree():
    """SS NECESSARY + Δ LOAD-BEARING + CKA quantum-aligned + MI quantum-dominant
    → all 4 probes agree on GENUINE → returns GENUINE."""
    sur = _sur(0.5, 0.4, 0.6)
    inter = _inter(0.3, 0.2, 0.4)
    rep = _rep(cka_qt=0.8, cka_ct=0.2, mi_quantum_total=0.8, mi_classical_total=0.2)
    classifier = QCTClassifier(
        thresholds=_THRESHOLDS, use_convergent_validity=True
    )
    assert classifier.classify(sur, inter, rep) == QCTVerdict.GENUINE


def test_convergent_classifier_returns_base_on_3_of_4_agreement():
    """One of CKA/MI disagreeing still keeps base verdict (3 of 4 majority)."""
    sur = _sur(0.5, 0.4, 0.6)
    inter = _inter(0.3, 0.2, 0.4)
    rep = _rep(
        cka_qt=0.2, cka_ct=0.8,  # CKA says CLASSICAL-MORE-ALIGNED (disagrees)
        mi_quantum_total=0.8, mi_classical_total=0.2,  # MI agrees
        verdict="CLASSICAL-MORE-ALIGNED",
    )
    classifier = QCTClassifier(
        thresholds=_THRESHOLDS, use_convergent_validity=True
    )
    # Base = GENUINE; SS+Δ agree (2), MI agrees (3), CKA disagrees → 3/4
    assert classifier.classify(sur, inter, rep) == QCTVerdict.GENUINE


def test_convergent_classifier_emits_disagreement_when_both_signals_disagree():
    """Base GENUINE but CKA AND MI both say classical-favored → 2/4 agreement
    → DISAGREEMENT."""
    sur = _sur(0.5, 0.4, 0.6)
    inter = _inter(0.3, 0.2, 0.4)
    rep = _rep(
        cka_qt=0.2, cka_ct=0.8,
        mi_quantum_total=0.2, mi_classical_total=0.8,
        verdict="CLASSICAL-MORE-ALIGNED",
    )
    classifier = QCTClassifier(
        thresholds=_THRESHOLDS, use_convergent_validity=True
    )
    assert classifier.classify(sur, inter, rep) == QCTVerdict.DISAGREEMENT


def test_convergent_classifier_disagreement_on_ignored_base():
    """Base IGNORED (SS necessary, Δ not load-bearing), but CKA+MI both
    quantum-favored → 2/4 agreement → DISAGREEMENT."""
    sur = _sur(0.5, 0.4, 0.6)            # SS NECESSARY
    inter = _inter(0.01, -0.01, 0.03)    # Δ NOT-LOAD-BEARING
    rep = _rep(
        cka_qt=0.8, cka_ct=0.2,
        mi_quantum_total=0.8, mi_classical_total=0.2,
    )
    classifier = QCTClassifier(
        thresholds=_THRESHOLDS, use_convergent_validity=True
    )
    assert classifier.classify(sur, inter, rep) == QCTVerdict.DISAGREEMENT


def test_convergent_classifier_skipped_when_representation_missing():
    """No representation supplied → behavior identical to base classifier."""
    sur = _sur(0.5, 0.4, 0.6)
    inter = _inter(0.3, 0.2, 0.4)
    classifier_on = QCTClassifier(
        thresholds=_THRESHOLDS, use_convergent_validity=True
    )
    classifier_off = QCTClassifier(thresholds=_THRESHOLDS)
    assert classifier_on.classify(sur, inter) == classifier_off.classify(sur, inter)
    assert classifier_on.classify(sur, inter) == QCTVerdict.GENUINE


def test_convergent_classifier_skipped_when_representation_unavailable():
    """If RepresentationProbe returned UNAVAILABLE (no classical embedding),
    the convergent gate is skipped — verdict falls back to base SS+Δ."""
    sur = _sur(0.5, 0.4, 0.6)
    inter = _inter(0.3, 0.2, 0.4)
    rep = ProbeResult(
        "representation", primary_score=float("nan"),
        verdict="UNAVAILABLE", details={},
    )
    classifier = QCTClassifier(
        thresholds=_THRESHOLDS, use_convergent_validity=True
    )
    assert classifier.classify(sur, inter, rep) == QCTVerdict.GENUINE


def test_convergent_classifier_does_not_change_inconclusive_on_ci_straddle():
    """CI-straddle Inconclusive must be preserved even under convergent gate."""
    # SS CI straddles 0.1 threshold
    sur = _sur(0.10, 0.05, 0.15)
    inter = _inter(0.3, 0.2, 0.4)
    rep = _rep(
        cka_qt=0.8, cka_ct=0.2,
        mi_quantum_total=0.8, mi_classical_total=0.2,
    )
    classifier = QCTClassifier(
        thresholds=_THRESHOLDS, use_convergent_validity=True
    )
    assert classifier.classify(sur, inter, rep) == QCTVerdict.INCONCLUSIVE


def test_convergent_classifier_off_by_default_preserves_v030_behavior():
    """Without use_convergent_validity=True, representation is ignored."""
    sur = _sur(0.5, 0.4, 0.6)
    inter = _inter(0.3, 0.2, 0.4)
    rep_disagrees = _rep(
        cka_qt=0.2, cka_ct=0.8,
        mi_quantum_total=0.2, mi_classical_total=0.8,
        verdict="CLASSICAL-MORE-ALIGNED",
    )
    classifier = QCTClassifier(thresholds=_THRESHOLDS, use_convergent_validity=False)
    # Even though rep disagrees, with use_convergent_validity=False we get
    # the base GENUINE verdict (representation argument is simply ignored).
    assert classifier.classify(sur, inter, rep_disagrees) == QCTVerdict.GENUINE
