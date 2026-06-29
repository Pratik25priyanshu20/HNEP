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
from hnep.probes.noise import (
    BUILTIN_NOISE,
    NoiseProbe,
    bit_flip_noise,
    depolarizing_approx,
    gaussian_noise,
)
from hnep.probes.temporal import TemporalProbe
from hnep.probes.error_diversity import ErrorDiversityProbe
from hnep.probes.representation import (
    RepresentationProbe,
    cka_with_scalar_target,
    linear_cka,
)

__all__ = [
    "Probe",
    "SurrogationProbe",
    "SurrogateSpec",
    "default_surrogate_ladder",
    "InterventionProbe",
    "NoiseProbe",
    "TemporalProbe",
    "ErrorDiversityProbe",
    "RepresentationProbe",
    "BUILTIN_NOISE",
    "gaussian_noise",
    "depolarizing_approx",
    "bit_flip_noise",
    "linear_cka",
    "cka_with_scalar_target",
]
