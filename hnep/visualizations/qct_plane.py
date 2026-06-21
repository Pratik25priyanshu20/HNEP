"""QCT Plane — the headline HNEP visualisation.

Plots Surrogation Score (x) vs Intervention Δ (y) with the four QCT quadrants
shaded. Each model is a dot with a confidence rectangle drawn from its
bootstrap CIs. The plane includes labels, citation footer, and HNEP branding.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from hnep.results.hnep_result import HNEPResult
from hnep.thresholds import DEFAULT_THRESHOLDS, Thresholds


QUADRANT_COLORS = {
    "Genuine":      "#2ca02c",   # green
    "Regularizer":  "#1f77b4",   # blue
    "Ignored":      "#ff7f0e",   # orange
    "Dead Weight":  "#d62728",   # red
}


def plot_qct_plane(
    results: HNEPResult | Sequence[HNEPResult],
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    xlim: tuple[float, float] = (0.0, 1.0),
    ylim: tuple[float, float] = (-0.1, 1.1),
    figsize: tuple[float, float] = (7.5, 6.5),
    title: str = "HNEP — Quantum Contribution Taxonomy",
):
    """Render the QCT plane figure.

    Parameters
    ----------
    results
        A single :class:`HNEPResult` or a list of them. Each result becomes
        a dot on the plane with a CI rectangle.
    thresholds
        The classification thresholds (drawn as dashed quadrant boundaries).
    xlim, ylim
        Plot bounds. Default covers the typical SS / Δ ranges.
    figsize
        Matplotlib figure size, inches.
    title
        Title text.

    Returns
    -------
    The matplotlib :class:`~matplotlib.figure.Figure`.
    """
    if isinstance(results, HNEPResult):
        results = [results]

    fig, ax = plt.subplots(figsize=figsize)

    x_th = thresholds.ss_replaceable
    y_th = thresholds.intervention_load_bearing

    # ── Shaded quadrants ──
    # Layout:
    #   (low SS, high Δ)  = REGULARIZER       (upper-left)
    #   (high SS, high Δ) = GENUINE            (upper-right)
    #   (low SS, low Δ)   = DEAD WEIGHT       (lower-left)
    #   (high SS, low Δ)  = IGNORED            (lower-right)
    quads = [
        ((xlim[0], x_th), (y_th, ylim[1]), "Regularizer"),
        ((x_th, xlim[1]), (y_th, ylim[1]), "Genuine"),
        ((xlim[0], x_th), (ylim[0], y_th), "Dead Weight"),
        ((x_th, xlim[1]), (ylim[0], y_th), "Ignored"),
    ]
    for (x0, x1), (y0, y1), label in quads:
        ax.add_patch(mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            facecolor=QUADRANT_COLORS[label], alpha=0.10, linewidth=0,
        ))
        # quadrant label in the corner
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        ax.text(cx, cy, label,
                color=QUADRANT_COLORS[label], alpha=0.6,
                fontsize=12, fontweight="bold",
                ha="center", va="center")

    # ── Threshold lines ──
    ax.axvline(x_th, color="#444", lw=0.8, ls="--", alpha=0.7)
    ax.axhline(y_th, color="#444", lw=0.8, ls="--", alpha=0.7)
    ax.text(x_th, ylim[1] - 0.02, f" SS = {x_th:.2f}",
            fontsize=8, color="#444", va="top", ha="left")
    ax.text(xlim[1] - 0.02, y_th + 0.02, f"Δ = {y_th:.2f} ",
            fontsize=8, color="#444", va="bottom", ha="right")

    # ── Model dots ──
    palette = plt.get_cmap("tab10")
    for i, res in enumerate(results):
        sur = res.probes.get("surrogation")
        inter = res.probes.get("intervention")
        if sur is None or inter is None:
            continue

        x = sur.primary_score
        y = inter.primary_score
        color = palette(i % 10)

        # Confidence rectangle
        if sur.primary_score_ci and inter.primary_score_ci:
            xlo, xhi = sur.primary_score_ci
            ylo, yhi = inter.primary_score_ci
            ax.add_patch(mpatches.Rectangle(
                (xlo, ylo), xhi - xlo, yhi - ylo,
                facecolor=color, alpha=0.15,
                edgecolor=color, linewidth=1.0,
            ))

        ax.scatter([x], [y], color=color, s=180, zorder=5,
                    edgecolor="black", linewidth=1.0, marker="o")
        ax.annotate(
            res.model_name, (x, y), xytext=(8, 6),
            textcoords="offset points",
            fontsize=10, fontweight="bold", color=color,
        )

    # ── Axes & decoration ──
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Surrogation Score (SS)", fontsize=11)
    ax.set_ylabel("Intervention Δ (relative R² drop)", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    ax.grid(True, alpha=0.2, linestyle=":")
    ax.text(
        0.99, 0.01,
        "HNEP — Hybrid Network Evaluation Protocol",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=8, color="#888", style="italic",
    )
    fig.tight_layout()
    return fig
