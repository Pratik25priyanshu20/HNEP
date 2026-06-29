"""Day-3 visualisation tests: disagreement heatmap + activation atlas."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # noqa: E402  (must precede pyplot import)

import matplotlib.pyplot as plt
import numpy as np
import pytest

import hnep
from hnep.results.hnep_result import HNEPResult
from hnep.results.probe_result import ProbeResult
from hnep.visualizations.activation_atlas import (
    _project_2d,
    plot_activation_atlas,
    plot_activation_atlas_grid,
)
from hnep.visualizations.disagreement_heatmap import (
    VERDICT_COLORS,
    plot_disagreement_heatmap,
)


# ── helpers ───────────────────────────────────────────────────────────

def make_probe_result(name, verdict, confidence=0.9):
    return ProbeResult(
        probe_name=name,
        primary_score=0.5,
        verdict=verdict,
        confidence=confidence,
    )


def make_hnep_result(name, probes_dict):
    return HNEPResult(
        model_name=name,
        dataset_name=name,
        probes=probes_dict,
    )


# ── disagreement heatmap ──────────────────────────────────────────────

def test_disagreement_heatmap_runs_with_multi_dataset():
    results = {
        "ESOL": make_hnep_result("esol", {
            "surrogation": make_probe_result("surrogation", "REPLACEABLE"),
            "error_diversity": make_probe_result("error_diversity", "REDUNDANT"),
        }),
        "FreeSolv": make_hnep_result("freesolv", {
            "surrogation": make_probe_result("surrogation", "NECESSARY"),
            "error_diversity": make_probe_result("error_diversity", "DIVERSE"),
        }),
        "QM9": make_hnep_result("qm9", {
            # The headline QM9 disagreement — should be visually distinct.
            "surrogation": make_probe_result("surrogation", "NECESSARY"),
            "error_diversity": make_probe_result("error_diversity", "REDUNDANT"),
        }),
    }
    fig = plot_disagreement_heatmap(results)
    assert fig is not None
    ax = fig.axes[0]
    # 3 datasets on x-axis
    assert len(ax.get_xticklabels()) == 3
    # 2 probes on y-axis
    assert len(ax.get_yticklabels()) == 2
    plt.close(fig)


def test_disagreement_heatmap_handles_missing_probes():
    """If a probe is missing for one dataset, the cell should be blank."""
    results = {
        "A": make_hnep_result("a", {
            "surrogation": make_probe_result("surrogation", "REPLACEABLE"),
        }),
        "B": make_hnep_result("b", {
            "surrogation": make_probe_result("surrogation", "NECESSARY"),
            "noise": make_probe_result("noise", "STABLE"),
        }),
    }
    fig = plot_disagreement_heatmap(results)
    # Union of probes = 2
    ax = fig.axes[0]
    assert len(ax.get_yticklabels()) == 2
    plt.close(fig)


def test_disagreement_heatmap_empty_input():
    fig = plot_disagreement_heatmap({})
    assert fig is not None
    plt.close(fig)


def test_disagreement_heatmap_explicit_probe_order():
    results = {
        "A": make_hnep_result("a", {
            "surrogation": make_probe_result("surrogation", "REPLACEABLE"),
            "noise": make_probe_result("noise", "STABLE"),
        }),
    }
    fig = plot_disagreement_heatmap(results, probes=["noise", "surrogation"])
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    # Top row = first probe in list → "noise" appears at top.
    assert labels == ["noise", "surrogation"]
    plt.close(fig)


def test_disagreement_color_palette_covers_core_verdicts():
    for v in ("REPLACEABLE", "NECESSARY", "DIVERSE", "REDUNDANT",
              "STABLE", "VERDICT-FLIPS"):
        assert v in VERDICT_COLORS


# ── activation atlas ──────────────────────────────────────────────────

def test_project_2d_pca_fallback():
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(60, 8))
    emb, used = _project_2d(Q, method="pca", seed=0)
    assert emb.shape == (60, 2)
    assert used == "pca"


def test_project_2d_raw_1d_keeps_data():
    Q = np.linspace(0, 1, 50).reshape(50, 1)
    emb, used = _project_2d(Q, method="auto")
    assert used == "raw-1d"
    np.testing.assert_allclose(emb[:, 0], Q[:, 0])


def test_project_2d_raw_2d_passthrough():
    Q = np.random.default_rng(0).normal(size=(20, 2))
    emb, used = _project_2d(Q, method="auto")
    assert used == "raw-2d"
    np.testing.assert_allclose(emb, Q)


def test_activation_atlas_with_colors():
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(80, 6))
    y = Q[:, 0] + 0.5 * Q[:, 1]
    fig = plot_activation_atlas(Q, color_by=y, method="pca",
                                color_label="solubility")
    assert fig is not None
    # one main axis + one colorbar axis
    assert len(fig.axes) == 2
    plt.close(fig)


def test_activation_atlas_no_colors():
    rng = np.random.default_rng(1)
    Q = rng.normal(size=(50, 4))
    fig = plot_activation_atlas(Q, color_by=None, method="pca")
    assert fig is not None
    assert len(fig.axes) == 1
    plt.close(fig)


def test_activation_atlas_rejects_wrong_color_length():
    Q = np.random.default_rng(0).normal(size=(30, 4))
    with pytest.raises(ValueError):
        plot_activation_atlas(Q, color_by=np.zeros(5), method="pca")


def test_activation_atlas_rejects_non_2d_input():
    with pytest.raises(ValueError):
        plot_activation_atlas(np.zeros(20), method="pca")


def test_activation_atlas_grid_multi_dataset():
    rng = np.random.default_rng(2)
    quantum = {
        "ESOL": rng.normal(size=(40, 4)),
        "FreeSolv": rng.normal(size=(35, 4)),
        "QM9": rng.normal(size=(60, 4)),
    }
    targets = {k: rng.normal(size=v.shape[0]) for k, v in quantum.items()}
    fig = plot_activation_atlas_grid(quantum, targets, method="pca")
    assert fig is not None
    # 3 datasets × cmap → 3 scatter axes + 3 colorbar axes
    assert len(fig.axes) >= 3
    plt.close(fig)


def test_activation_atlas_grid_empty():
    fig = plot_activation_atlas_grid({})
    assert fig is not None
    plt.close(fig)


# ── top-level exports ─────────────────────────────────────────────────

def test_day3_viz_exported_at_top_level():
    assert hasattr(hnep, "plot_disagreement_heatmap")
    assert hasattr(hnep, "plot_activation_atlas")
    assert hasattr(hnep, "plot_activation_atlas_grid")
