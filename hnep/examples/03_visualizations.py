"""Generate the three killer HNEP figures from an evaluation result.

    python -m hnep.examples.03_visualizations
"""

from __future__ import annotations

import numpy as np

import hnep
from hnep.adapters.base import Dataset
from hnep.examples.quickstart_helpers import IgnoredAdapter, RegularizerAdapter


def main() -> None:
    rng = np.random.default_rng(0)
    n, d = 300, 12
    inputs = rng.normal(size=(n, d))
    targets = inputs @ rng.normal(size=d) + 0.1 * rng.normal(size=n)

    dataset = Dataset(
        inputs=inputs, targets=targets,
        train_idx=np.arange(0, 210),
        val_idx=np.arange(210, 255),
        test_idx=np.arange(255, 300),
        metadata={"name": "synth"},
    )

    r1 = hnep.evaluate(RegularizerAdapter(dataset), dataset)
    r2 = hnep.evaluate(IgnoredAdapter(dataset), dataset)

    # QCT plane — both models overlaid
    fig_qct = hnep.plot_qct_plane([r1, r2])
    fig_qct.savefig("qct_plane.png", dpi=120, bbox_inches="tight")

    # Convergent validity radar
    fig_radar = hnep.plot_convergent_validity_radar([r1, r2])
    fig_radar.savefig("convergent_validity_radar.png", dpi=120, bbox_inches="tight")

    # Pareto plot (synthetic numbers showing the typical Hybrid vs MPNN vs GNN story)
    pts = [
        hnep.PointMeasurement("GNN",   accuracy=0.43, compute_cost=12,
                              accuracy_std=0.05),
        hnep.PointMeasurement("MPNN",  accuracy=0.83, compute_cost=15,
                              accuracy_std=0.03),
        hnep.PointMeasurement("VQC-4q", accuracy=-0.04, compute_cost=1600,
                              is_quantum=True, accuracy_std=0.20),
        hnep.PointMeasurement("Hybrid-V1", accuracy=0.83, compute_cost=7000,
                              is_quantum=True, accuracy_std=0.04),
    ]
    fig_pareto = hnep.plot_pareto_with_hardware_cost(pts)
    fig_pareto.savefig("pareto_with_hw_cost.png", dpi=120, bbox_inches="tight")

    print("Generated three figures:")
    print("  • qct_plane.png")
    print("  • convergent_validity_radar.png")
    print("  • pareto_with_hw_cost.png")


if __name__ == "__main__":
    main()
