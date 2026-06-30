"""QCT — Quantum Contribution Taxonomy classifier.

Takes a surrogation result + an intervention result and emits one of four
verdicts (or Inconclusive). The full implementation including bootstrap CIs
lands in Phase 3; this file defines the data types and a stub classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from hnep.results.probe_result import ProbeResult
from hnep.thresholds import DEFAULT_THRESHOLDS, Thresholds


class QCTVerdict(str, Enum):
    """The four QCT roles plus an honest fallback."""

    GENUINE = "Genuine"           # unique output AND structurally necessary
    REGULARIZER = "Regularizer"   # mimicable output BUT structurally necessary
    IGNORED = "Ignored"           # unique output BUT not structurally necessary
    DEAD_WEIGHT = "Dead Weight"   # mimicable AND not structurally necessary
    INCONCLUSIVE = "Inconclusive"

    def __str__(self) -> str:
        return self.value


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
    """

    thresholds: Thresholds = DEFAULT_THRESHOLDS
    use_p_values: bool = False
    p_value_threshold: float = 0.05

    def classify(
        self,
        surrogation: ProbeResult,
        intervention: ProbeResult,
    ) -> QCTVerdict:
        """Return a verdict; falls back to ``Inconclusive`` if a CI straddles
        a threshold or, when ``use_p_values=True``, if a REPLACEABLE verdict
        fails its surrogation permutation-test significance gate."""
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

        if unique and load_bearing:
            return QCTVerdict.GENUINE
        if not unique and load_bearing:
            return QCTVerdict.REGULARIZER
        if unique and not load_bearing:
            return QCTVerdict.IGNORED
        return QCTVerdict.DEAD_WEIGHT

    @staticmethod
    def _straddles_threshold(probe: ProbeResult, threshold: float) -> bool:
        ci = probe.primary_score_ci
        if ci is None:
            return False
        lower, upper = ci
        return lower < threshold < upper
