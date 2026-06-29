"""Probe Disagreement Heatmap.

Grid where each cell is one (probe, dataset) verdict, colour-coded by verdict
class. Surfaces convergent-validity disagreements at a glance — the QM9
NECESSARY-but-REDUNDANT pattern, for example, immediately stands out as an
off-colour cell next to its peers.

Designed for multi-dataset comparison; for a single dataset use the
convergent-validity radar instead.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from hnep.results.hnep_result import HNEPResult


# Colour palette per verdict — designed for at-a-glance disagreement spotting.
VERDICT_COLORS = {
    # Surrogation verdicts
    "REPLACEABLE":         "#1f77b4",   # blue
    "NECESSARY":           "#d62728",   # red

    # Intervention verdicts
    "LOAD-BEARING":        "#d62728",   # red
    "NOT-LOAD-BEARING":    "#1f77b4",   # blue

    # Error diversity verdicts
    "DIVERSE":             "#2ca02c",   # green
    "REDUNDANT":           "#ff7f0e",   # orange

    # Representation verdicts
    "QUANTUM-MORE-ALIGNED":   "#9467bd",   # purple
    "CLASSICAL-MORE-ALIGNED": "#8c564b",   # brown

    # Noise verdicts
    "STABLE":              "#2ca02c",
    "MOSTLY-STABLE":       "#9acd32",
    "PERFORMANCE-DEGRADES":"#ff7f0e",
    "VERDICT-FLIPS":       "#d62728",

    # Temporal verdicts
    "TRANSIENT-DRIFT":     "#9acd32",
    "VERDICT-DRIFTS":      "#d62728",

    # QCT classification verdicts
    "Genuine":             "#2ca02c",
    "Regularizer":         "#1f77b4",
    "Ignored":             "#ff7f0e",
    "Dead Weight":         "#d62728",
    "Inconclusive":        "#888888",

    # Fallbacks
    "UNAVAILABLE":         "#cccccc",
    "INCONCLUSIVE":        "#888888",
    "":                    "#ffffff",
}


def plot_disagreement_heatmap(
    results_by_dataset: Mapping[str, HNEPResult],
    probes: Sequence[str] | None = None,
    figsize: tuple[float, float] | None = None,
    title: str = "HNEP — Probe verdicts across datasets",
    show_confidence: bool = True,
):
    """Render the disagreement heatmap.

    Parameters
    ----------
    results_by_dataset
        Mapping from dataset name to :class:`HNEPResult`. Each result's
        probes provide one row in the grid.
    probes
        Optional explicit list of probe names. If ``None``, uses the union of
        probes across all results in the order they were first seen.
    figsize
        Matplotlib figure size. Defaults scale with grid dimensions.
    title
        Figure title.
    show_confidence
        If True, dim cells with low confidence (alpha proportional to
        ``probe.confidence``).
    """
    if not results_by_dataset:
        fig, ax = plt.subplots(figsize=figsize or (6, 3))
        ax.text(0.5, 0.5, "No results to plot", ha="center", va="center")
        ax.set_axis_off()
        return fig

    # Collect probes from the union of results in the order seen
    if probes is None:
        seen: list[str] = []
        for r in results_by_dataset.values():
            for name in r.probes.keys():
                if name not in seen:
                    seen.append(name)
        probes = seen

    datasets = list(results_by_dataset.keys())
    n_rows = len(probes)
    n_cols = len(datasets)

    if figsize is None:
        figsize = (max(6, 1.5 * n_cols + 2), max(2.5, 0.7 * n_rows + 1.5))

    fig, ax = plt.subplots(figsize=figsize)

    # Build the colour grid + verdict labels
    for i, probe_name in enumerate(probes):
        for j, ds_name in enumerate(datasets):
            result = results_by_dataset[ds_name]
            probe_result = result.probes.get(probe_name)

            if probe_result is None:
                # Probe didn't run on this dataset
                color = VERDICT_COLORS[""]
                label = "—"
                alpha = 1.0
            else:
                verdict = probe_result.verdict
                color = VERDICT_COLORS.get(verdict, "#aaaaaa")
                label = verdict
                alpha = (
                    max(0.4, probe_result.confidence)
                    if show_confidence else 1.0
                )

            rgba = mcolors.to_rgba(color, alpha=alpha)
            rect = plt.Rectangle((j, n_rows - 1 - i), 1, 1,
                                  facecolor=rgba, edgecolor="white",
                                  linewidth=2.5)
            ax.add_patch(rect)
            # Centre text
            text_color = _readable_text_color(color)
            ax.text(j + 0.5, n_rows - 1 - i + 0.5, label,
                    ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    # Axes — datasets on x, probes on y (rows)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([j + 0.5 for j in range(n_cols)])
    ax.set_xticklabels(datasets, fontsize=10, rotation=0)
    ax.set_yticks([n_rows - 1 - i + 0.5 for i in range(n_rows)])
    ax.set_yticklabels(probes, fontsize=10)
    ax.set_aspect("equal")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(title, fontsize=12, pad=10)
    ax.text(
        n_cols, -0.5,
        "HNEP — Hybrid Network Evaluation Protocol",
        ha="right", va="top",
        fontsize=7, color="#888", style="italic",
    )
    fig.tight_layout()
    return fig


def _readable_text_color(bg_hex: str) -> str:
    """Pick black or white text depending on background luminance."""
    rgb = mcolors.to_rgb(bg_hex)
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return "black" if luminance > 0.55 else "white"
