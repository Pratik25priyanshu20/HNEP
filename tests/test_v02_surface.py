"""v0.2 public-surface smoke tests.

Pins the public API: anything imported here is part of the v0.2 contract.
If a symbol is renamed or removed, this test fails — a deliberate signal
that the change deserves a major-version bump or a deprecation notice.
"""

from __future__ import annotations

import inspect

import hnep


EXPECTED_VERSION = "0.4.0"

# Everything we promise users can import from the top-level `hnep` namespace.
EXPECTED_TOP_LEVEL = {
    # Core
    "evaluate",
    "Dataset", "HNEPResult", "ProbeResult",
    "ModelInterface", "FunctionalAdapter", "PrecomputedAdapter",
    "QCTClassifier", "QCTVerdict", "Thresholds",
    # Probes (v0.1 + v0.2)
    "Probe",
    "SurrogationProbe", "InterventionProbe",
    "NoiseProbe", "TemporalProbe",
    "ErrorDiversityProbe", "RepresentationProbe",
    # Cost-utility
    "PointMeasurement", "compute_qus", "estimate_hardware_cost",
    # Visualisations (v0.1 + v0.2)
    "plot_qct_plane",
    "plot_convergent_validity_radar",
    "plot_pareto_with_hardware_cost",
    "plot_disagreement_heatmap",
    "plot_activation_atlas",
    "plot_activation_atlas_grid",
    # Reports
    "render_html_report",
    # Gallery (v0.2)
    "MoleculeRecord", "build_gallery", "render_gallery_html",
    # Card + CLI helpers (v0.2)
    "HNEPCard",
    "compare_cards_text", "compare_cards_markdown", "compare_cards_html",
    "load_result_from_json",
    # Explainer + exports (v0.2)
    "explain_result", "explain_result_html",
    "to_latex", "compare_to_latex",
    "to_markdown", "compare_to_markdown",
}


def test_version_matches_expected():
    assert hnep.__version__ == EXPECTED_VERSION


def test_public_surface_is_present():
    missing = sorted(s for s in EXPECTED_TOP_LEVEL if not hasattr(hnep, s))
    assert not missing, f"Missing top-level exports: {missing}"


def test_all_export_list_covers_expected():
    missing_from_all = sorted(EXPECTED_TOP_LEVEL - set(hnep.__all__))
    assert not missing_from_all, (
        f"Symbols present but absent from __all__: {missing_from_all}"
    )


def test_result_has_v02_helper_methods():
    """The result methods that were added in v0.2 must exist on HNEPResult."""
    for name in ("explain", "to_latex", "to_markdown",
                 "to_html", "to_json", "to_csv"):
        assert hasattr(hnep.HNEPResult, name), name
        assert callable(getattr(hnep.HNEPResult, name)), name


def test_cli_advertises_v02_commands():
    """`hnep` with no arguments must mention `card` and `compare`."""
    from io import StringIO
    import sys

    from hnep.cli import main

    buf = StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        rc = main([])
    finally:
        sys.stdout = old
    assert rc == 0
    out = buf.getvalue()
    assert "card" in out
    assert "compare" in out
    assert EXPECTED_VERSION in out


def test_probe_classes_share_run_signature():
    """Every probe should expose a `.run(model, dataset, ...)` method."""
    for name in ("SurrogationProbe", "InterventionProbe", "NoiseProbe",
                 "ErrorDiversityProbe", "RepresentationProbe"):
        cls = getattr(hnep, name)
        sig = inspect.signature(cls.run)
        params = list(sig.parameters)
        assert params[:3] == ["self", "model", "dataset"], (
            f"{name}.run signature drifted: {params}"
        )
