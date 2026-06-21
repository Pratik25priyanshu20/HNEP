"""Quantum Utility Score (QUS).

QUS = (R²_quantum − R²_classical) / (time_quantum / time_classical)

The numerator is the R² gain from using the quantum model.
The denominator is the compute multiplier paid for that gain.

Positive QUS means quantum buys you R² at a worthwhile per-FLOP rate.
Near-zero QUS means quantum's R² gain is roughly proportional to its compute cost.
Negative QUS means the classical model wins outright.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QuantumUtilityScore:
    """Result of a pairwise QUS computation."""

    quantum_model: str
    classical_model: str
    r2_quantum: float
    r2_classical: float
    time_quantum_s: float
    time_classical_s: float
    delta_r2: float
    compute_multiplier: float
    qus: float

    @property
    def verdict(self) -> str:
        if self.delta_r2 <= 0:
            return "quantum loses on accuracy"
        if self.qus > 1e-3:
            return "quantum buys R² at a worthwhile rate"
        if self.qus > 0:
            return "quantum buys R² marginally per-FLOP"
        return "quantum is dominated"


def compute_qus(
    quantum_model: str,
    classical_model: str,
    r2_quantum: float,
    r2_classical: float,
    time_quantum_s: float,
    time_classical_s: float,
) -> QuantumUtilityScore:
    """Compute the QUS for one (quantum, classical) pairing."""
    delta = r2_quantum - r2_classical
    multiplier = time_quantum_s / max(time_classical_s, 1e-9)
    qus = delta / max(multiplier, 1e-9)
    return QuantumUtilityScore(
        quantum_model=quantum_model,
        classical_model=classical_model,
        r2_quantum=round(r2_quantum, 4),
        r2_classical=round(r2_classical, 4),
        time_quantum_s=round(time_quantum_s, 2),
        time_classical_s=round(time_classical_s, 2),
        delta_r2=round(delta, 4),
        compute_multiplier=round(multiplier, 2),
        qus=round(qus, 8),
    )
