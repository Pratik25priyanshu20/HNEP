"""Hardware cost projection — what would the same training cost on real
NISQ hardware?

The numbers below are *indicative*. Real per-shot prices change frequently
with provider, queue priority, circuit depth, and contract terms. We expose
the constants as overridable parameters so users can plug in their own
quote when they have one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


# Indicative USD per shot — adjust to match the user's current vendor quote.
DEFAULT_PROVIDER_PRICES_USD_PER_SHOT: Mapping[str, float] = {
    "ibm": 0.0015,
    "ionq": 0.30,
    "rigetti": 0.0008,
    "iqm": 0.005,
}

# Default circuit-evaluation accounting. Adjust based on training schedule.
DEFAULT_FORWARD_PASSES_PER_RUN: int = 60_000
DEFAULT_SHOTS_PER_FORWARD: int = 1_000


@dataclass
class HardwareCostEstimate:
    """Per-provider cost estimate for a single training run."""

    provider: str
    shots_per_run: int
    usd_per_shot: float
    estimated_usd_per_run: float
    note: str = ""


def estimate_hardware_cost(
    forward_passes_per_run: int = DEFAULT_FORWARD_PASSES_PER_RUN,
    shots_per_forward: int = DEFAULT_SHOTS_PER_FORWARD,
    provider_prices: Mapping[str, float] = DEFAULT_PROVIDER_PRICES_USD_PER_SHOT,
) -> list[HardwareCostEstimate]:
    """Return per-provider cost estimates for a single training run.

    Parameters
    ----------
    forward_passes_per_run
        Total quantum forward passes the training schedule performs. A typical
        molecular benchmark at N=1000 samples × 20 epochs × ~3 passes/sample
        sits around 60,000.
    shots_per_forward
        Shots needed per forward pass for stable gradient estimation.
    provider_prices
        Mapping ``provider → USD per shot``.

    Returns
    -------
    A list of :class:`HardwareCostEstimate`, one per provider, ordered cheapest first.
    """
    total_shots = forward_passes_per_run * shots_per_forward
    estimates = []
    for provider, price in provider_prices.items():
        total_cost = total_shots * price
        estimates.append(HardwareCostEstimate(
            provider=provider,
            shots_per_run=total_shots,
            usd_per_shot=price,
            estimated_usd_per_run=round(total_cost, 2),
            note=("Indicative only — real pricing varies by provider, queue "
                  "priority, circuit depth, and contract terms."),
        ))
    estimates.sort(key=lambda e: e.estimated_usd_per_run)
    return estimates
