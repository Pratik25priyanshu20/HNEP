"""Diagnostic probes — supplementary evidence, NOT verdict-driving.

v0.4 tiers HNEP's probe API into a **core** tier that drives QCT verdicts
(``SurrogationProbe`` + ``InterventionProbe`` + ``RepresentationProbe``,
importable from the top-level ``hnep`` namespace) and a **diagnostic** tier
housed here. Diagnostic probes run to completion, return valid
``ProbeResult`` values, and provide supplementary convergent-validity
signals — but ``QCTClassifier`` deliberately does not consume them.

The diagnostic tier exists because these probes are methodologically sound
but hit limitations during real-data evaluation:

* ``NoiseProbe`` and ``TemporalProbe`` returned null results (no verdict
  flip) on the thesis datasets — informative for stability sanity but not
  discriminating enough to gate a verdict.
* ``ErrorDiversityProbe`` uses a Ridge readout that fails on non-linearly
  informative branches, producing spurious REDUNDANT verdicts. See its
  docstring for the ``low_readout_strength`` caveat.

Diagnostic probe results attached to ``HNEPResult.probes`` do not
influence ``qct_verdict``. Users may still inspect them for their own
convergent-validity picture, but should not override the core QCT verdict
based on diagnostic signals.
"""

from __future__ import annotations

from hnep.probes.error_diversity import ErrorDiversityProbe
from hnep.probes.noise import (
    NoiseProbe,
    bit_flip_noise,
    depolarizing_approx,
    gaussian_noise,
)
from hnep.probes.temporal import TemporalProbe

__all__ = [
    "NoiseProbe",
    "TemporalProbe",
    "ErrorDiversityProbe",
    # Noise-model helpers exposed for users writing custom NoiseProbe fns
    "gaussian_noise",
    "depolarizing_approx",
    "bit_flip_noise",
]
