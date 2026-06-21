"""A/B comparison of two hybrid models on the same dataset.

Shows how to overlay multiple HNEP results in one HTML report and one
combined figure.

    python -m hnep.examples.04_compare_models
"""

from __future__ import annotations

import numpy as np

import hnep
from hnep.adapters.base import Dataset
from hnep.examples.quickstart_helpers import IgnoredAdapter, RegularizerAdapter


def main() -> None:
    rng = np.random.default_rng(0)
    n, d = 300, 14
    dataset = Dataset(
        inputs=rng.normal(size=(n, d)),
        targets=rng.normal(size=n),
        train_idx=np.arange(0, 210),
        val_idx=np.arange(210, 255),
        test_idx=np.arange(255, 300),
        metadata={"name": "synth_AB"},
    )

    result_a = hnep.evaluate(RegularizerAdapter(dataset), dataset)
    result_b = hnep.evaluate(IgnoredAdapter(dataset), dataset)

    print("Model A (RegularizerToy):")
    print(f"  Verdict:  {result_a.qct_verdict}   ({result_a.qct_confidence:.2f})")
    print("Model B (IgnoredToy):")
    print(f"  Verdict:  {result_b.qct_verdict}   ({result_b.qct_confidence:.2f})")

    # Combined HTML report — model A as primary, B overlaid on the figures
    result_a.to_html("comparison.html", other_results=[result_b])
    print("\nA/B HTML report → comparison.html")


if __name__ == "__main__":
    main()
