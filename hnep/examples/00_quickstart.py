"""Quickstart — full HNEP evaluation on a synthetic hybrid model.

Two synthetic adapters illustrate two QCT verdicts:

1. ``RegularizerAdapter``  — quantum is a linear projection of input (so a
   classical surrogate trivially mimics it = REPLACEABLE) but the decoder
   weights it heavily, so removing it crashes performance = LOAD-BEARING.
   Expected verdict: **Regularizer**.

2. ``IgnoredAdapter``  — quantum is a non-trivial sinusoidal transformation
   (no classical surrogate captures it = NECESSARY) but the decoder weighs
   it near zero, so removing it barely matters = NOT LOAD-BEARING.
   Expected verdict: **Ignored**.

Run:

    python -m hnep.examples.00_quickstart
"""

from __future__ import annotations

import numpy as np

import hnep
from hnep.adapters.base import Dataset, ModelInterface


# ─────────────────────────────────────────────────────────────────────
# Synthetic dataset
# ─────────────────────────────────────────────────────────────────────
def make_dataset(n: int = 400, d: int = 16, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    inputs = rng.normal(size=(n, d))
    targets = inputs @ rng.normal(size=d) + 0.1 * rng.normal(size=n)
    n_train, n_val = int(0.7 * n), int(0.15 * n)
    return Dataset(
        inputs=inputs,
        targets=targets,
        train_idx=np.arange(0, n_train),
        val_idx=np.arange(n_train, n_train + n_val),
        test_idx=np.arange(n_train + n_val, n),
        metadata={"name": "synth"},
    )


# ─────────────────────────────────────────────────────────────────────
# Adapter 1 — Regularizer
# ─────────────────────────────────────────────────────────────────────
class RegularizerAdapter(ModelInterface):
    name = "RegularizerToy"
    _quantum_dim = 4

    def __init__(self, dataset: Dataset, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.proj = rng.normal(size=(dataset.inputs.shape[1], self._quantum_dim))
        self.classical_w = rng.normal(size=dataset.inputs.shape[1]) * 0.05
        self.quantum_w = rng.normal(size=self._quantum_dim) * 1.5

    def _quantum(self, X):
        return X @ self.proj

    def _predict(self, X, q):
        return X @ self.classical_w + q @ self.quantum_w

    def predict(self, dataset, indices=None):
        idx = _idx(dataset, indices)
        X = dataset.inputs[idx]
        return self._predict(X, self._quantum(X))

    def extract_quantum_output(self, dataset, indices=None):
        idx = _idx(dataset, indices)
        return self._quantum(dataset.inputs[idx])

    def predict_with_quantum_override(self, dataset, q_override, indices=None):
        idx = _idx(dataset, indices)
        return self._predict(dataset.inputs[idx], q_override)


# ─────────────────────────────────────────────────────────────────────
# Adapter 2 — Ignored
# ─────────────────────────────────────────────────────────────────────
class IgnoredAdapter(ModelInterface):
    name = "IgnoredToy"
    _quantum_dim = 4

    def __init__(self, dataset: Dataset, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.freq = rng.uniform(0.5, 3.0,
                                 size=(dataset.inputs.shape[1], self._quantum_dim))
        self.phase = rng.uniform(0, np.pi, size=(1, self._quantum_dim))
        self.classical_w = rng.normal(size=dataset.inputs.shape[1])
        self.quantum_w = np.full(self._quantum_dim, 1e-4)

    def _quantum(self, X):
        return np.sin(X @ self.freq + self.phase)

    def _predict(self, X, q):
        return X @ self.classical_w + q @ self.quantum_w

    def predict(self, dataset, indices=None):
        idx = _idx(dataset, indices)
        X = dataset.inputs[idx]
        return self._predict(X, self._quantum(X))

    def extract_quantum_output(self, dataset, indices=None):
        idx = _idx(dataset, indices)
        return self._quantum(dataset.inputs[idx])

    def predict_with_quantum_override(self, dataset, q_override, indices=None):
        idx = _idx(dataset, indices)
        return self._predict(dataset.inputs[idx], q_override)


def _idx(dataset, indices):
    return (np.arange(len(dataset.targets)) if indices is None
            else np.asarray(indices))


def main() -> None:
    dataset = make_dataset()

    for adapter_cls in (RegularizerAdapter, IgnoredAdapter):
        model = adapter_cls(dataset)
        print(f"\n{'=' * 70}")
        print(f"  Evaluating: {model.name}")
        print(f"{'=' * 70}")
        result = hnep.evaluate(model, dataset)
        print(result.summary())

    print(f"\nhnep version: {hnep.__version__}")


if __name__ == "__main__":
    main()
