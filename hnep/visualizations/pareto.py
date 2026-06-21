"""Cost-utility Pareto plot with hardware-cost annotation.

R² vs wall-clock, log-x axis, with the Pareto frontier highlighted and
optional NISQ hardware-cost callouts on the right margin.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt

from hnep.cost_utility.hardware_cost import (
    HardwareCostEstimate,
    estimate_hardware_cost,
)
from hnep.cost_utility.pareto import (
    ParetoAnalysis,
    PointMeasurement,
    pareto_frontier,
)


def plot_pareto_with_hardware_cost(
    points: Sequence[PointMeasurement],
    figsize: tuple[float, float] = (9.5, 5.5),
    show_hardware_cost: bool = True,
    hardware_estimates: Sequence[HardwareCostEstimate] | None = None,
    title: str = "HNEP — Cost-Utility Pareto",
):
    """Render the Pareto plot.

    Parameters
    ----------
    points
        A list of :class:`PointMeasurement`. The Pareto frontier among these
        is highlighted.
    figsize
        Matplotlib figure size.
    show_hardware_cost
        If ``True``, annotate the right-hand margin with NISQ hardware-cost
        estimates extrapolated for the most-expensive quantum model.
    hardware_estimates
        Explicit hardware-cost estimates to display. If ``None`` and
        ``show_hardware_cost`` is true, defaults are computed via
        :func:`estimate_hardware_cost`.
    title
        Figure title.

    Returns
    -------
    Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    frontier_idx = set(pareto_frontier(points))
    palette = plt.get_cmap("tab10")

    for i, p in enumerate(points):
        color = "#cc3333" if p.is_quantum else "#1f77b4"
        on_frontier = i in frontier_idx
        marker = "D" if p.is_quantum else "s" if "MPNN" in p.model_name else "o"
        edge = "black" if on_frontier else "none"

        ax.errorbar(
            p.compute_cost, p.accuracy,
            xerr=p.compute_cost_std if p.compute_cost_std > 0 else None,
            yerr=p.accuracy_std if p.accuracy_std > 0 else None,
            fmt=marker, color=color, markersize=11 if on_frontier else 9,
            elinewidth=1.0, capsize=3,
            markeredgecolor=edge, markeredgewidth=1.0 if on_frontier else 0.5,
            label=p.model_name, alpha=0.95,
        )
        ax.annotate(
            p.model_name,
            (p.compute_cost, p.accuracy),
            xytext=(7, 6), textcoords="offset points",
            fontsize=9, fontweight="bold" if on_frontier else "normal",
        )

    # Draw Pareto frontier
    frontier_points = sorted(
        [points[i] for i in frontier_idx], key=lambda p: p.compute_cost
    )
    if len(frontier_points) >= 2:
        xs = [p.compute_cost for p in frontier_points]
        ys = [p.accuracy for p in frontier_points]
        ax.plot(xs, ys, "--", color="#888", lw=1.2, alpha=0.7, zorder=1)

    ax.set_xscale("log")
    ax.set_xlabel("Wall-clock per training run (seconds, log scale)", fontsize=10)
    ax.set_ylabel("R² (test set)", fontsize=10)
    ax.set_title(title, fontsize=12, pad=10)
    ax.grid(True, alpha=0.25, linestyle=":")

    # Hardware cost annotations
    if show_hardware_cost:
        estimates = (
            list(hardware_estimates) if hardware_estimates
            else estimate_hardware_cost()
        )
        if estimates:
            lines = ["NISQ deployment (indicative):"]
            for e in estimates[:3]:
                lines.append(f"  {e.provider.upper():<8s} ≈ ${e.estimated_usd_per_run:,.0f}/run")
            ax.text(
                0.99, 0.05, "\n".join(lines),
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff8e6",
                          edgecolor="#e0c060", lw=0.8, alpha=0.95),
            )

    ax.text(
        0.99, 0.01, "HNEP — Hybrid Network Evaluation Protocol",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7, color="#888", style="italic",
    )
    fig.tight_layout()
    return fig
