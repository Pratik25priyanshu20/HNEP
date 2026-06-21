"""Phase 3 tests — cost-utility, visualisations, HTML report."""

from __future__ import annotations

import numpy as np
import pytest

import hnep
from hnep.adapters.base import Dataset
from hnep.cost_utility import (
    PointMeasurement,
    compute_qus,
    estimate_hardware_cost,
    pareto_frontier,
)
from hnep.reports.html import render_html_report
from hnep.visualizations.pareto import plot_pareto_with_hardware_cost
from hnep.visualizations.qct_plane import plot_qct_plane
from hnep.visualizations.radar import plot_convergent_validity_radar


# ─────────────────────────────────────────────────────────────────────
# Cost-utility tests
# ─────────────────────────────────────────────────────────────────────

def test_qus_basic_math():
    s = compute_qus(
        quantum_model="Hybrid",
        classical_model="MPNN",
        r2_quantum=0.83,
        r2_classical=0.65,
        time_quantum_s=3000,
        time_classical_s=20,
    )
    assert s.delta_r2 == pytest.approx(0.18, abs=1e-3)
    assert s.compute_multiplier == pytest.approx(150.0, abs=1e-1)
    assert s.qus == pytest.approx(0.18 / 150.0, abs=1e-5)


def test_qus_negative_when_quantum_loses():
    s = compute_qus("VQC", "MPNN",
                    r2_quantum=-0.04, r2_classical=0.83,
                    time_quantum_s=1600, time_classical_s=15)
    assert s.delta_r2 < 0
    assert s.qus < 0
    assert "loses" in s.verdict


def test_pareto_frontier_basic():
    pts = [
        PointMeasurement("GNN",   accuracy=0.43, compute_cost=12),
        PointMeasurement("MPNN",  accuracy=0.83, compute_cost=15),
        PointMeasurement("Hybrid",accuracy=0.83, compute_cost=7000, is_quantum=True),
        PointMeasurement("VQC",   accuracy=-0.04, compute_cost=1600, is_quantum=True),
    ]
    keep = pareto_frontier(pts)
    # MPNN dominates VQC and Hybrid on accuracy. GNN is cheapest.
    # Frontier should keep GNN (cheapest) and MPNN (highest accuracy + low cost).
    names = {pts[i].model_name for i in keep}
    assert "MPNN" in names
    # Hybrid ties MPNN on accuracy but is much more expensive — dominated.
    assert "Hybrid" not in names
    # VQC has lower accuracy AND higher cost than GNN — dominated.
    assert "VQC" not in names


def test_hardware_cost_estimates_sorted():
    estimates = estimate_hardware_cost()
    assert len(estimates) >= 2
    costs = [e.estimated_usd_per_run for e in estimates]
    assert costs == sorted(costs)


# ─────────────────────────────────────────────────────────────────────
# Visualisation tests — render, don't crash, dimensions sensible
# ─────────────────────────────────────────────────────────────────────

def _build_real_result():
    """Run a quick HNEP eval on a synthetic Regularizer model so we have a
    populated HNEPResult to visualise."""
    from hnep.probes.surrogation import default_surrogate_ladder

    rng = np.random.default_rng(0)
    n, d = 200, 12
    inputs = rng.normal(size=(n, d))
    targets = inputs @ rng.normal(size=d) + 0.1 * rng.normal(size=n)

    dataset = Dataset(
        inputs=inputs, targets=targets,
        train_idx=np.arange(0, 140),
        val_idx=np.arange(140, 170),
        test_idx=np.arange(170, 200),
        metadata={"name": "test_set"},
    )

    from hnep.examples.quickstart_helpers import RegularizerAdapter  # type: ignore[attr-defined]
    model = RegularizerAdapter(dataset)
    return hnep.evaluate(model, dataset, probes=[
        hnep.SurrogationProbe(surrogates=default_surrogate_ladder()[:4],
                              n_bootstrap=30),
        hnep.InterventionProbe(n_bootstrap=30),
    ])


def test_qct_plane_renders():
    result = _build_real_result()
    fig = plot_qct_plane(result)
    assert fig is not None
    assert len(fig.axes) >= 1
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_radar_renders_with_two_probes():
    result = _build_real_result()
    fig = plot_convergent_validity_radar(result)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_pareto_plot_renders():
    pts = [
        PointMeasurement("GNN", 0.43, 12),
        PointMeasurement("MPNN", 0.83, 15),
        PointMeasurement("Hybrid", 0.85, 7000, is_quantum=True),
    ]
    fig = plot_pareto_with_hardware_cost(pts)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Report tests
# ─────────────────────────────────────────────────────────────────────

def test_html_report_contains_key_elements(tmp_path):
    result = _build_real_result()
    out = tmp_path / "report.html"
    html = render_html_report(result, path=out)
    assert out.exists()
    assert result.qct_verdict in html
    assert "QCT" in html
    assert "Convergent Validity" in html
    assert "data:image/png;base64" in html  # figures embedded


def test_hnep_result_to_html_uses_real_renderer(tmp_path):
    result = _build_real_result()
    out = tmp_path / "report.html"
    result.to_html(out)
    text = out.read_text()
    assert "Probe Results" in text
    assert "surrogation" in text
    assert "intervention" in text
