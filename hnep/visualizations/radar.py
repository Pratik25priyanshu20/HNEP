"""Convergent Validity Radar.

A radar / spider chart with one axis per probe. Each model's profile is
traced as a polygon. Jaggedness = disagreement = "convergent validity gap".
"""

from __future__ import annotations

from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np

from hnep.results.hnep_result import HNEPResult


def plot_convergent_validity_radar(
    results: HNEPResult | Sequence[HNEPResult],
    probes: List[str] | None = None,
    figsize: tuple[float, float] = (7, 6.5),
    title: str = "HNEP — Convergent Validity Profile",
):
    """Render the convergent-validity radar.

    Parameters
    ----------
    results
        Single result or a list of results (overlay).
    probes
        Which probes to show as axes. Defaults to the union of probes present
        across the supplied results.
    figsize
        Matplotlib figure size.
    title
        Figure title.

    Returns
    -------
    Matplotlib figure.
    """
    if isinstance(results, HNEPResult):
        results = [results]

    if probes is None:
        seen: List[str] = []
        for r in results:
            for name in r.probes.keys():
                if name not in seen:
                    seen.append(name)
        probes = seen

    if not probes:
        # No probes to plot — graceful empty radar
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
        ax.set_title(title)
        ax.text(0, 0, "No probes available", ha="center", va="center")
        return fig

    angles = np.linspace(0, 2 * np.pi, len(probes), endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    palette = plt.get_cmap("tab10")

    for i, res in enumerate(results):
        # Normalise each probe's primary_score into [0, 1] — most HNEP probes
        # already report scores in [0, 1] (SS, |Δ|). We clamp to [0, 1] for
        # the radar, which is purely a visual normalisation.
        values = []
        for name in probes:
            probe = res.probes.get(name)
            if probe is None:
                values.append(0.0)
            else:
                v = abs(probe.primary_score)
                values.append(float(min(1.0, max(0.0, v))))
        values += values[:1]

        color = palette(i % 10)
        ax.fill(angles, values, alpha=0.18, color=color)
        ax.plot(angles, values, color=color, linewidth=2.0,
                marker="o", markersize=6, label=res.model_name)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(probes, fontsize=10)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=12, pad=18)
    ax.grid(True, linestyle=":", alpha=0.5)
    if len(results) > 1:
        ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.05), fontsize=9, frameon=False)
    fig.tight_layout()
    return fig
