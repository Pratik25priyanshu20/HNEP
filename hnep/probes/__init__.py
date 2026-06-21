"""HNEP probes — independent evaluation methods for hybrid models.

Each probe answers one question about a hybrid model. Multiple probes are
designed to converge (or disagree) on the same underlying quantity. The
:mod:`hnep.classifiers` module turns probe outputs into QCT verdicts.
"""

from hnep.probes.base import Probe
from hnep.probes.surrogation import (
    SurrogateSpec,
    SurrogationProbe,
    default_surrogate_ladder,
)
from hnep.probes.intervention import InterventionProbe

__all__ = [
    "Probe",
    "SurrogationProbe",
    "SurrogateSpec",
    "default_surrogate_ladder",
    "InterventionProbe",
]
