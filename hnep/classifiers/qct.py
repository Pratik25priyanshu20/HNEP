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

    The default thresholds match the values used throughout the thesis
    benchmarks; users may pass custom :class:`Thresholds`.

    Phase 3 will extend this to consume confidence intervals and emit
    ``Inconclusive`` whenever the intervals straddle a boundary.
    """

    thresholds: Thresholds = DEFAULT_THRESHOLDS

    def classify(
        self,
        surrogation: ProbeResult,
        intervention: ProbeResult,
    ) -> QCTVerdict:
        """Return a verdict; falls back to ``Inconclusive`` if a CI straddles
        a threshold."""
        if self._straddles_threshold(surrogation, self.thresholds.ss_replaceable):
            return QCTVerdict.INCONCLUSIVE
        if self._straddles_threshold(
            intervention, self.thresholds.intervention_load_bearing
        ):
            return QCTVerdict.INCONCLUSIVE

        unique = surrogation.primary_score >= self.thresholds.ss_replaceable
        load_bearing = intervention.primary_score >= self.thresholds.intervention_load_bearing

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
