"""Shared synthetic adapters used by the quickstart and tests."""

from __future__ import annotations

import numpy as np

from hnep.adapters.base import Dataset, ModelInterface


def _idx(dataset: Dataset, indices):
    return (np.arange(len(dataset.targets)) if indices is None
            else np.asarray(indices))


class RegularizerAdapter(ModelInterface):
    """Quantum is a linear projection (REPLACEABLE) but decoder weights it
    heavily (LOAD-BEARING). Expected verdict: Regularizer."""

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


class IgnoredAdapter(ModelInterface):
    """Quantum is sinusoidal (NECESSARY) but decoder weighs it near-zero
    (NOT load-bearing). Expected verdict: Ignored."""

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
