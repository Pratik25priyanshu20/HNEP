"""Ground-truth benchmark suite for HNEP.

Six archetypal synthetic hybrid models with known-correct QCT verdicts. The
suite is HNEP's measuring stick — any methodology change (permutation-derived
thresholds, block bootstrap, ...) must be validated against it.
"""

from __future__ import annotations

from hnep.benchmarks.ground_truth import (
    BenchmarkReport,
    make_adversarial,
    make_dead_weight,
    make_genuine,
    make_ignored,
    make_inconclusive,
    make_regularizer,
    run_ground_truth_benchmark,
)

__all__ = [
    "BenchmarkReport",
    "make_genuine",
    "make_regularizer",
    "make_ignored",
    "make_dead_weight",
    "make_inconclusive",
    "make_adversarial",
    "run_ground_truth_benchmark",
]
