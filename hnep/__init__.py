"""HNEP — Hybrid Network Evaluation Protocol.

A multi-method evaluation framework that answers: *does the quantum component
in a hybrid quantum-classical model actually contribute meaningful computation,
or could it be replaced by classical surrogates?*

Quick start
-----------

>>> import hnep
>>> result = hnep.evaluate(my_model, my_dataset)
>>> print(result.qct_verdict)
'Regularizer'
>>> result.to_html("report.html")

See the README and the examples/ directory for more.
"""

from __future__ import annotations

# Version (kept in sync with pyproject.toml)
__version__ = "0.2.0"

# Top-level public API — populated as Phase 2/3 lands
from hnep.api import evaluate
from hnep.results.hnep_result import HNEPResult
from hnep.results.probe_result import ProbeResult
from hnep.adapters.base import Dataset, ModelInterface
from hnep.adapters.functional import FunctionalAdapter
from hnep.adapters.precomputed import PrecomputedAdapter
from hnep.probes.base import Probe
from hnep.probes.surrogation import SurrogationProbe
from hnep.probes.intervention import InterventionProbe
from hnep.probes.noise import NoiseProbe
from hnep.probes.temporal import TemporalProbe
from hnep.probes.error_diversity import ErrorDiversityProbe
from hnep.probes.representation import RepresentationProbe
from hnep.classifiers.qct import QCTClassifier, QCTVerdict
from hnep.thresholds import Thresholds
from hnep.cost_utility import (
    PointMeasurement,
    compute_qus,
    estimate_hardware_cost,
)
from hnep.visualizations import (
    plot_qct_plane,
    plot_convergent_validity_radar,
    plot_pareto_with_hardware_cost,
    plot_disagreement_heatmap,
    plot_activation_atlas,
    plot_activation_atlas_grid,
)
from hnep.reports import render_html_report
from hnep.gallery import (
    MoleculeRecord,
    build_gallery,
    render_gallery_html,
)
from hnep.card import (
    HNEPCard,
    compare_cards_html,
    compare_cards_markdown,
    compare_cards_text,
    load_result_from_json,
)
from hnep.explain import explain_result, explain_result_html
from hnep.exports import (
    compare_to_latex,
    compare_to_markdown,
    to_latex,
    to_markdown,
)

__all__ = [
    "__version__",
    # Top-level functions
    "evaluate",
    # Public dataclasses
    "Dataset",
    "HNEPResult",
    "ProbeResult",
    "QCTVerdict",
    # Public classes
    "ModelInterface",
    "FunctionalAdapter",
    "PrecomputedAdapter",
    "Probe",
    "SurrogationProbe",
    "InterventionProbe",
    "NoiseProbe",
    "TemporalProbe",
    "ErrorDiversityProbe",
    "RepresentationProbe",
    "QCTClassifier",
    "Thresholds",
    # Cost-utility
    "PointMeasurement",
    "compute_qus",
    "estimate_hardware_cost",
    # Visualisations
    "plot_qct_plane",
    "plot_convergent_validity_radar",
    "plot_pareto_with_hardware_cost",
    "plot_disagreement_heatmap",
    "plot_activation_atlas",
    "plot_activation_atlas_grid",
    # Reports
    "render_html_report",
    # Gallery
    "MoleculeRecord",
    "build_gallery",
    "render_gallery_html",
    # Card / CLI helpers
    "HNEPCard",
    "compare_cards_text",
    "compare_cards_markdown",
    "compare_cards_html",
    "load_result_from_json",
    # Explainer + exports
    "explain_result",
    "explain_result_html",
    "to_latex",
    "compare_to_latex",
    "to_markdown",
    "compare_to_markdown",
]
