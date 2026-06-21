"""Threshold configuration for QCT classification.

These thresholds were chosen by inspection of the SS distribution in our
benchmarks. A future version of HNEP will derive them from bootstrap CIs or
permutation null distributions; for now they are user-overridable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    """Cutoff values for binarising probe scores into QCT verdicts.

    Attributes
    ----------
    ss_replaceable
        Surrogation Score below this value classifies the quantum output as
        REPLACEABLE (a classical surrogate succeeds at mimicking it).
    intervention_load_bearing
        Relative R² drop above this value classifies the quantum component as
        LOAD-BEARING (removing it crashes the model).
    """

    ss_replaceable: float = 0.2
    intervention_load_bearing: float = 0.05

    def describe(self) -> str:
        return (
            f"Thresholds(SS<{self.ss_replaceable:.2f} ⇒ REPLACEABLE, "
            f"Δ≥{self.intervention_load_bearing:.2f} ⇒ LOAD-BEARING)"
        )


DEFAULT_THRESHOLDS = Thresholds()
