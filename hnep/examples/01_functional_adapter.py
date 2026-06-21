"""How to wrap any model with FunctionalAdapter.

This example shows the recommended path for first-time HNEP users — wrap your
trained model in three callbacks and run a full evaluation.

    python -m hnep.examples.01_functional_adapter
"""

from __future__ import annotations

import numpy as np

import hnep
from hnep.adapters.base import Dataset


# ─────────────────────────────────────────────────────────────────────
# Imagine these are your already-trained model components.
# In a real workflow, they could be a JAX model, a PyTorch module,
# Qiskit ansatz + classical readout — anything callable.
# ─────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(0)
N, D = 400, 16

inputs = rng.normal(size=(N, D))
targets = inputs @ rng.normal(size=D) + 0.1 * rng.normal(size=N)

dataset = Dataset(
    inputs=inputs, targets=targets,
    train_idx=np.arange(0, 280),
    val_idx=np.arange(280, 340),
    test_idx=np.arange(340, 400),
    metadata={"name": "synthetic"},
)

# Mock trained weights for a hybrid model: quantum is a linear projection,
# decoder weights quantum heavily so removing it crashes performance.
proj = rng.normal(size=(D, 4))
classical_w = rng.normal(size=D) * 0.05
quantum_w = rng.normal(size=4) * 1.5


def _quantum(X):
    return X @ proj


def _predict(X, q):
    return X @ classical_w + q @ quantum_w


# ─────────────────────────────────────────────────────────────────────
# Step 1 — define three callables matching the HNEP adapter contract.
# ─────────────────────────────────────────────────────────────────────
def predict_fn(ds, indices):
    indices = np.arange(len(ds.targets)) if indices is None else np.asarray(indices)
    X = ds.inputs[indices]
    return _predict(X, _quantum(X))


def extract_quantum_fn(ds, indices):
    indices = np.arange(len(ds.targets)) if indices is None else np.asarray(indices)
    return _quantum(ds.inputs[indices])


def predict_with_override_fn(ds, q_override, indices):
    indices = np.arange(len(ds.targets)) if indices is None else np.asarray(indices)
    X = ds.inputs[indices]
    return _predict(X, q_override)


# ─────────────────────────────────────────────────────────────────────
# Step 2 — wrap them in FunctionalAdapter.
# ─────────────────────────────────────────────────────────────────────
adapter = hnep.FunctionalAdapter(
    name="MyHybridModel",
    predict_fn=predict_fn,
    extract_quantum_fn=extract_quantum_fn,
    predict_with_override_fn=predict_with_override_fn,
    quantum_dim=4,
)


# ─────────────────────────────────────────────────────────────────────
# Step 3 — run HNEP.
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    result = hnep.evaluate(adapter, dataset)
    print(result.summary())
    print()
    print(f"hnep version: {hnep.__version__}")


if __name__ == "__main__":
    main()
