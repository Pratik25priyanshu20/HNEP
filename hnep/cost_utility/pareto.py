"""Pareto frontier analysis for cost-utility plots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class PointMeasurement:
    """A single point in the (compute_cost, accuracy) plane."""

    model_name: str
    accuracy: float           # higher = better (e.g. R²)
    compute_cost: float       # lower = better (e.g. wall-clock seconds)
    accuracy_std: float = 0.0
    compute_cost_std: float = 0.0
    is_quantum: bool = False  # for styling / annotations


@dataclass
class ParetoAnalysis:
    """Result of a Pareto-frontier analysis across a set of models."""

    points: List[PointMeasurement] = field(default_factory=list)
    frontier_indices: List[int] = field(default_factory=list)

    @property
    def frontier_points(self) -> List[PointMeasurement]:
        return [self.points[i] for i in self.frontier_indices]


def pareto_frontier(points: Sequence[PointMeasurement]) -> List[int]:
    """Indices of points on the Pareto frontier (higher accuracy, lower cost).

    A point is dominated if some other point has strictly higher accuracy AND
    strictly lower compute cost — those are filtered out; non-dominated points
    are returned, sorted by cost ascending.
    """
    indices = list(range(len(points)))
    keep = []
    for i in indices:
        dominated = False
        for j in indices:
            if i == j:
                continue
            if (points[j].accuracy >= points[i].accuracy
                    and points[j].compute_cost <= points[i].compute_cost
                    and (points[j].accuracy > points[i].accuracy
                         or points[j].compute_cost < points[i].compute_cost)):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return sorted(keep, key=lambda k: points[k].compute_cost)


def pareto_dominated(points: Sequence[PointMeasurement]) -> List[int]:
    """Indices of points dominated by the frontier (the complement)."""
    frontier = set(pareto_frontier(points))
    return [i for i in range(len(points)) if i not in frontier]


def build_pareto_analysis(points: Sequence[PointMeasurement]) -> ParetoAnalysis:
    return ParetoAnalysis(
        points=list(points),
        frontier_indices=pareto_frontier(points),
    )
