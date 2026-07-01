"""QCT — Quantum Contribution Taxonomy classifier.

Takes a surrogation result + an intervention result and emits one of four
verdicts (or Inconclusive). The full implementation including bootstrap CIs
lands in Phase 3; this file defines the data types and a stub classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from hnep.results.probe_result import ProbeResult
from hnep.thresholds import DEFAULT_THRESHOLDS, Thresholds


class QCTVerdict(str, Enum):
    """The four QCT roles plus two fallback verdicts."""

    GENUINE = "Genuine"           # unique output AND structurally necessary
    REGULARIZER = "Regularizer"   # mimicable output BUT structurally necessary
    IGNORED = "Ignored"           # unique output BUT not structurally necessary
    DEAD_WEIGHT = "Dead Weight"   # mimicable AND not structurally necessary
    INCONCLUSIVE = "Inconclusive"
    DISAGREEMENT = "Disagreement"  # v0.4: probes disagree under convergent-validity gating

    def __str__(self) -> str:
        return self.value


def _expected_signals(corner: QCTVerdict) -> Tuple[bool, bool, bool]:
    """Return ``(unique_expected, load_bearing_expected, q_favored_expected)``
    for a given QCT corner — the signal pattern a probe should report when
    the model genuinely sits at that corner."""
    return (
        corner in (QCTVerdict.GENUINE, QCTVerdict.IGNORED),
        corner in (QCTVerdict.GENUINE, QCTVerdict.REGULARIZER),
        corner in (QCTVerdict.GENUINE, QCTVerdict.REGULARIZER),
    )


@dataclass
class QCTClassifier:
    """Apply QCT thresholds to a (surrogation, intervention) probe pair.

    The default thresholds match the empirical T1.2 calibration; users may
    pass custom :class:`Thresholds`.

    ``use_p_values=True`` adds a convergent-validity gate: when a
    REPLACEABLE verdict (SS < threshold) is not significant at p < 0.05
    on the surrogation permutation test, the verdict falls back to
    ``Inconclusive``. Specifically REGULARIZER / DEAD_WEIGHT need
    ``surrogation.p_value < 0.05`` (SS is significantly lower than the
    shuffled-q null).

    The intervention probe also reports a permutation p-value, but the QCT
    classifier does **not** gate on it: the per-row-shuffle null does not
    cleanly sample "β_q = 0" because the model's trained β_q stays fixed,
    so a high p_value on Δ does not have a clean reject-H0 interpretation.
    See InterventionProbe.run for details.

    Probes constructed without ``calibrate=True`` report ``p_value=None``,
    which is treated as "p-value gate disabled for that probe".

    ``use_convergent_validity=True`` (v0.4) enables a 4-probe vote:
    surrogation (SS), intervention (Δ), CKA (from a representation
    ``ProbeResult`` passed as the optional ``representation`` argument),
    and MI (also from that probe's details). Each probe casts a vote for
    one of the four QCT corners. If ≥3 of 4 votes are consistent with
    the base (SS+Δ) verdict, that verdict is returned. Otherwise the
    classifier emits ``DISAGREEMENT``. When ``representation`` is None or
    its verdict is ``UNAVAILABLE``, the convergent gate is skipped and
    only the base (SS+Δ) verdict applies. The error-diversity probe is
    NOT a classifier input — it is diagnostic only.
    """

    thresholds: Thresholds = DEFAULT_THRESHOLDS
    use_p_values: bool = False
    p_value_threshold: float = 0.05
    use_convergent_validity: bool = False
    convergent_agreement_min: int = 3

    def classify(
        self,
        surrogation: ProbeResult,
        intervention: ProbeResult,
        representation: Optional[ProbeResult] = None,
    ) -> QCTVerdict:
        """Return a verdict; falls back to ``Inconclusive`` if a CI straddles
        a threshold or, when ``use_p_values=True``, if a REPLACEABLE verdict
        fails its surrogation permutation-test significance gate. When
        ``use_convergent_validity=True``, returns ``DISAGREEMENT`` if CKA + MI
        from ``representation`` don't agree with the base verdict at the
        configured ``convergent_agreement_min`` threshold (default 3 of 4)."""
        if self._straddles_threshold(surrogation, self.thresholds.ss_replaceable):
            return QCTVerdict.INCONCLUSIVE
        if self._straddles_threshold(
            intervention, self.thresholds.intervention_load_bearing
        ):
            return QCTVerdict.INCONCLUSIVE

        unique = surrogation.primary_score >= self.thresholds.ss_replaceable
        load_bearing = intervention.primary_score >= self.thresholds.intervention_load_bearing

        if self.use_p_values:
            if (
                not unique
                and surrogation.p_value is not None
                and surrogation.p_value >= self.p_value_threshold
            ):
                return QCTVerdict.INCONCLUSIVE

        base = self._base_corner(unique, load_bearing)

        if self.use_convergent_validity:
            cka_q_favored, mi_q_favored = self._read_representation(representation)
            if cka_q_favored is not None or mi_q_favored is not None:
                votes = self._count_votes_for_corner(
                    base, unique, load_bearing, cka_q_favored, mi_q_favored
                )
                if votes < self.convergent_agreement_min:
                    return QCTVerdict.DISAGREEMENT

        return base

    @staticmethod
    def _base_corner(unique: bool, load_bearing: bool) -> QCTVerdict:
        if unique and load_bearing:
            return QCTVerdict.GENUINE
        if not unique and load_bearing:
            return QCTVerdict.REGULARIZER
        if unique and not load_bearing:
            return QCTVerdict.IGNORED
        return QCTVerdict.DEAD_WEIGHT

    @staticmethod
    def _read_representation(
        representation: Optional[ProbeResult],
    ) -> Tuple[Optional[bool], Optional[bool]]:
        """Extract (CKA quantum-favored?, MI quantum-favored?) from a
        representation ProbeResult. Returns (None, None) when unavailable."""
        if representation is None or representation.verdict == "UNAVAILABLE":
            return None, None
        d = representation.details
        cka = d.get("quantum_more_aligned_with_target")
        mi_share = d.get("quantum_info_share")
        mi = bool(mi_share > 0.5) if isinstance(mi_share, (int, float)) and mi_share == mi_share else None
        cka = bool(cka) if cka is not None else None
        return cka, mi

    @staticmethod
    def _count_votes_for_corner(
        corner: QCTVerdict,
        unique: bool,
        load_bearing: bool,
        cka_q_favored: Optional[bool],
        mi_q_favored: Optional[bool],
    ) -> int:
        unique_expected, load_expected, q_favored_expected = _expected_signals(corner)
        n = 0
        n += int(unique == unique_expected)
        n += int(load_bearing == load_expected)
        if cka_q_favored is not None:
            n += int(cka_q_favored == q_favored_expected)
        if mi_q_favored is not None:
            n += int(mi_q_favored == q_favored_expected)
        return n

    @staticmethod
    def _straddles_threshold(probe: ProbeResult, threshold: float) -> bool:
        ci = probe.primary_score_ci
        if ci is None:
            return False
        lower, upper = ci
        return lower < threshold < upper
