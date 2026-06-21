"""Phase 4 tests — adapter classes."""

from __future__ import annotations

import numpy as np
import pytest

import hnep
from hnep.adapters.base import Dataset


# ─────────────────────────────────────────────────────────────────────
# Common dataset fixture
# ─────────────────────────────────────────────────────────────────────
def make_dataset(n: int = 200, d: int = 12, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    inputs = rng.normal(size=(n, d))
    targets = inputs @ rng.normal(size=d) + 0.1 * rng.normal(size=n)
    return Dataset(
        inputs=inputs, targets=targets,
        train_idx=np.arange(0, 140),
        val_idx=np.arange(140, 170),
        test_idx=np.arange(170, 200),
        metadata={"name": "t"},
    )


# ─────────────────────────────────────────────────────────────────────
# FunctionalAdapter
# ─────────────────────────────────────────────────────────────────────

def test_functional_adapter_round_trip():
    ds = make_dataset()
    proj = np.random.default_rng(0).normal(size=(ds.inputs.shape[1], 4))

    def predict(d, idx):
        idx = np.arange(len(d.targets)) if idx is None else np.asarray(idx)
        X = d.inputs[idx]
        return X @ np.ones(d.inputs.shape[1]) * 0.1

    def extract(d, idx):
        idx = np.arange(len(d.targets)) if idx is None else np.asarray(idx)
        return d.inputs[idx] @ proj

    def override(d, q, idx):
        idx = np.arange(len(d.targets)) if idx is None else np.asarray(idx)
        return d.inputs[idx] @ np.ones(d.inputs.shape[1]) * 0.1 + q.mean(axis=1)

    adapter = hnep.FunctionalAdapter(
        name="fn", predict_fn=predict,
        extract_quantum_fn=extract,
        predict_with_override_fn=override,
        quantum_dim=4,
    )
    assert adapter.name == "fn"
    assert adapter.quantum_dim == 4
    p = adapter.predict(ds, ds.test_idx)
    q = adapter.extract_quantum_output(ds, ds.test_idx)
    o = adapter.predict_with_quantum_override(ds, q, ds.test_idx)
    assert p.shape == (len(ds.test_idx),)
    assert q.shape == (len(ds.test_idx), 4)
    assert o.shape == (len(ds.test_idx),)


def test_functional_adapter_can_run_through_evaluate():
    ds = make_dataset()
    rng = np.random.default_rng(0)
    proj = rng.normal(size=(ds.inputs.shape[1], 4))
    classical_w = rng.normal(size=ds.inputs.shape[1]) * 0.05
    quantum_w = rng.normal(size=4) * 1.5

    def _q(X):  # linear projection (REPLACEABLE)
        return X @ proj

    def _y(X, q):  # decoder weights quantum heavily (LOAD-BEARING)
        return X @ classical_w + q @ quantum_w

    def predict(d, idx):
        idx = np.arange(len(d.targets)) if idx is None else np.asarray(idx)
        X = d.inputs[idx]
        return _y(X, _q(X))

    def extract(d, idx):
        idx = np.arange(len(d.targets)) if idx is None else np.asarray(idx)
        return _q(d.inputs[idx])

    def override(d, q, idx):
        idx = np.arange(len(d.targets)) if idx is None else np.asarray(idx)
        return _y(d.inputs[idx], q)

    adapter = hnep.FunctionalAdapter(
        "fn_eval", predict, extract, override, quantum_dim=4
    )

    result = hnep.evaluate(adapter, ds)
    assert result.qct_verdict == "Regularizer"


# ─────────────────────────────────────────────────────────────────────
# PrecomputedAdapter
# ─────────────────────────────────────────────────────────────────────

def test_precomputed_adapter_round_trip():
    ds = make_dataset()
    rng = np.random.default_rng(0)

    gnn = rng.normal(size=(len(ds.targets), 16))
    q = rng.normal(size=(len(ds.targets), 4))

    # Trained decoder: linear combination
    decoder_w_g = rng.normal(size=16) * 0.05
    decoder_w_q = rng.normal(size=4) * 1.5

    def decoder(g_batch, q_batch):
        return g_batch @ decoder_w_g + q_batch @ decoder_w_q

    intact = decoder(gnn, q)

    adapter = hnep.PrecomputedAdapter(
        name="pre", gnn_embeddings=gnn, quantum_outputs=q,
        intact_predictions=intact, decoder_fn=decoder,
    )
    assert adapter.quantum_dim == 4
    p = adapter.predict(ds, ds.test_idx)
    q_out = adapter.extract_quantum_output(ds, ds.test_idx)
    np.testing.assert_allclose(p, intact[ds.test_idx])
    np.testing.assert_allclose(q_out, q[ds.test_idx])


def test_precomputed_adapter_evaluate_to_regularizer():
    ds = make_dataset()
    rng = np.random.default_rng(0)

    # Quantum = linear projection of input → REPLACEABLE
    proj = rng.normal(size=(ds.inputs.shape[1], 4))
    q = ds.inputs @ proj

    # GNN embedding = small noise (so classical surrogate from inputs can't
    # over-explain predictions through GNN bypass)
    gnn = rng.normal(size=(len(ds.targets), 8)) * 0.5

    decoder_w_g = rng.normal(size=8) * 0.05
    decoder_w_q = rng.normal(size=4) * 1.5

    def decoder(g_batch, q_batch):
        return g_batch @ decoder_w_g + q_batch @ decoder_w_q

    intact = decoder(gnn, q)
    adapter = hnep.PrecomputedAdapter(
        name="pre_eval", gnn_embeddings=gnn, quantum_outputs=q,
        intact_predictions=intact, decoder_fn=decoder,
    )
    result = hnep.evaluate(adapter, ds)
    assert result.qct_verdict == "Regularizer"


def test_precomputed_adapter_size_validation():
    ds = make_dataset(n=50)
    with pytest.raises(ValueError):
        hnep.PrecomputedAdapter(
            name="bad", gnn_embeddings=np.zeros((49, 8)),
            quantum_outputs=np.zeros((50, 4)),
            intact_predictions=np.zeros(50),
            decoder_fn=lambda g, q: g[:, 0] + q[:, 0],
        )


# ─────────────────────────────────────────────────────────────────────
# JaxFlax and PyTorch adapters — lazy-loaded
# ─────────────────────────────────────────────────────────────────────

def test_jax_flax_adapter_lazy_import():
    from hnep import adapters
    assert "JaxFlaxAdapter" in adapters.__all__
    # Module-level lazy import should succeed (the class is defined in
    # hnep.adapters.jax_flax; importing it does not require running jax).
    JaxFlaxAdapter = adapters.JaxFlaxAdapter
    assert hasattr(JaxFlaxAdapter, "predict")


def test_pytorch_adapter_lazy_import():
    from hnep import adapters
    assert "PyTorchAdapter" in adapters.__all__
    PyTorchAdapter = adapters.PyTorchAdapter
    # We don't try to instantiate — that requires torch to be installed and
    # an actual model. Just verify the class is loadable.
    assert hasattr(PyTorchAdapter, "predict")
    assert hasattr(PyTorchAdapter, "_forward_with_override")
