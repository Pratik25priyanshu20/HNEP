"""Tests for ErrorDiversityProbe + RepresentationProbe."""

from __future__ import annotations

import numpy as np

import hnep
from hnep.adapters.base import Dataset
from hnep.probes.error_diversity import ErrorDiversityProbe
from hnep.probes.representation import (
    RepresentationProbe,
    cka_with_scalar_target,
    linear_cka,
)


def make_dataset(n=200, d=12, seed=0):
    rng = np.random.default_rng(seed)
    inputs = rng.normal(size=(n, d))
    targets = inputs @ rng.normal(size=d) + 0.1 * rng.normal(size=n)
    return Dataset(
        inputs=inputs, targets=targets,
        train_idx=np.arange(0, int(0.7 * n)),
        val_idx=np.arange(int(0.7 * n), int(0.85 * n)),
        test_idx=np.arange(int(0.85 * n), n),
        metadata={"name": "synth_day2"},
    )


class _AdapterWithEmbedding(hnep.ModelInterface):
    """Adapter that exposes get_classical_embedding (uses inputs as the
    classical 'embedding' for testing)."""

    name = "ToyWithEmbedding"
    _quantum_dim = 4

    def __init__(self, dataset, seed=0):
        rng = np.random.default_rng(seed)
        self.proj = rng.normal(size=(dataset.inputs.shape[1], self._quantum_dim))
        self.classical_w = rng.normal(size=dataset.inputs.shape[1]) * 0.3
        self.quantum_w = rng.normal(size=self._quantum_dim) * 0.7

    def _idx(self, dataset, indices):
        return (np.arange(len(dataset.targets)) if indices is None
                else np.asarray(indices))

    def _quantum(self, X):
        return X @ self.proj

    def _predict(self, X, q):
        return X @ self.classical_w + q @ self.quantum_w

    def predict(self, dataset, indices=None):
        idx = self._idx(dataset, indices)
        X = dataset.inputs[idx]
        return self._predict(X, self._quantum(X))

    def extract_quantum_output(self, dataset, indices=None):
        idx = self._idx(dataset, indices)
        return self._quantum(dataset.inputs[idx])

    def predict_with_quantum_override(self, dataset, q_override, indices=None):
        idx = self._idx(dataset, indices)
        return self._predict(dataset.inputs[idx], q_override)

    def get_classical_embedding(self, dataset, indices=None):
        idx = self._idx(dataset, indices)
        return dataset.inputs[idx]


class _AdapterNoEmbedding(hnep.ModelInterface):
    """Adapter that does NOT expose get_classical_embedding."""

    name = "ToyNoEmbedding"
    _quantum_dim = 4

    def __init__(self, dataset, seed=0):
        rng = np.random.default_rng(seed)
        self.proj = rng.normal(size=(dataset.inputs.shape[1], self._quantum_dim))
        self.classical_w = rng.normal(size=dataset.inputs.shape[1]) * 0.3
        self.quantum_w = rng.normal(size=self._quantum_dim) * 0.7

    def _idx(self, dataset, indices):
        return (np.arange(len(dataset.targets)) if indices is None
                else np.asarray(indices))

    def predict(self, dataset, indices=None):
        idx = self._idx(dataset, indices)
        X = dataset.inputs[idx]
        return X @ self.classical_w + (X @ self.proj) @ self.quantum_w

    def extract_quantum_output(self, dataset, indices=None):
        idx = self._idx(dataset, indices)
        return dataset.inputs[idx] @ self.proj

    def predict_with_quantum_override(self, dataset, q_override, indices=None):
        idx = self._idx(dataset, indices)
        return dataset.inputs[idx] @ self.classical_w + q_override @ self.quantum_w


# ── linear_cka unit tests ────────────────────────────────────────────

def test_linear_cka_self_identity():
    """CKA(X, X) should be 1.0 (or very close)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 8))
    assert linear_cka(X, X) > 0.999


def test_linear_cka_uncorrelated_low():
    """CKA(X, Y) with independent X and Y should be low."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 8))
    Y = rng.normal(size=(500, 8))
    cka = linear_cka(X, Y)
    assert 0.0 <= cka < 0.2


def test_linear_cka_rotation_invariant():
    """CKA(X, X @ rotation_matrix) should still be ~1."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4))
    # Random orthogonal matrix (rotation)
    A = rng.normal(size=(4, 4))
    Q, _ = np.linalg.qr(A)
    Y = X @ Q
    assert linear_cka(X, Y) > 0.999


def test_cka_with_scalar_target_runs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 4))
    y = X @ rng.normal(size=4)
    cka = cka_with_scalar_target(X, y)
    assert 0.0 <= cka <= 1.0


# ── ErrorDiversityProbe tests ────────────────────────────────────────

def test_error_diversity_runs_when_embedding_available():
    ds = make_dataset()
    model = _AdapterWithEmbedding(ds)
    probe = ErrorDiversityProbe()
    result = probe.run(model, ds)
    assert result.probe_name == "error_diversity"
    assert result.verdict in {"DIVERSE", "REDUNDANT"}
    assert "error_correlation" in result.details
    assert -1.0 <= result.details["error_correlation"] <= 1.0
    assert "quantum_only_r2" in result.details
    assert "classical_only_r2" in result.details


def test_error_diversity_unavailable_without_embedding():
    ds = make_dataset()
    model = _AdapterNoEmbedding(ds)
    probe = ErrorDiversityProbe()
    result = probe.run(model, ds)
    assert result.verdict == "UNAVAILABLE"
    assert any("classical_embedding" in n.lower() or "embedding" in n.lower()
               for n in result.notes)


def test_error_diversity_custom_threshold():
    ds = make_dataset()
    model = _AdapterWithEmbedding(ds)
    # Force REDUNDANT verdict by setting threshold to 0
    probe = ErrorDiversityProbe(redundancy_threshold=0.0)
    result = probe.run(model, ds)
    assert result.verdict == "REDUNDANT"


def test_error_diversity_exports_at_top_level():
    assert hasattr(hnep, "ErrorDiversityProbe")
    assert hnep.ErrorDiversityProbe is ErrorDiversityProbe


# ── RepresentationProbe tests ────────────────────────────────────────

def test_representation_probe_runs_with_embedding():
    ds = make_dataset()
    model = _AdapterWithEmbedding(ds)
    probe = RepresentationProbe()
    result = probe.run(model, ds)
    assert result.probe_name == "representation"
    d = result.details
    # All three CKA values present
    assert all(k in d for k in [
        "cka_quantum_classical", "cka_quantum_target", "cka_classical_target",
    ])
    # All three MI values present
    assert all(k in d for k in [
        "mi_quantum_total", "mi_classical_total", "quantum_info_share",
    ])
    # Verdict is one of the two valid options
    assert result.verdict in {"QUANTUM-MORE-ALIGNED", "CLASSICAL-MORE-ALIGNED"}


def test_representation_probe_cka_values_bounded():
    ds = make_dataset()
    model = _AdapterWithEmbedding(ds)
    probe = RepresentationProbe()
    result = probe.run(model, ds)
    d = result.details
    for key in ["cka_quantum_classical", "cka_quantum_target", "cka_classical_target"]:
        assert 0.0 <= d[key] <= 1.0


def test_representation_probe_info_share_in_unit_interval():
    ds = make_dataset()
    model = _AdapterWithEmbedding(ds)
    probe = RepresentationProbe()
    result = probe.run(model, ds)
    s = result.details["quantum_info_share"]
    assert 0.0 <= s <= 1.0


def test_representation_unavailable_without_embedding():
    ds = make_dataset()
    model = _AdapterNoEmbedding(ds)
    probe = RepresentationProbe()
    result = probe.run(model, ds)
    assert result.verdict == "UNAVAILABLE"


def test_representation_probe_exports_at_top_level():
    assert hasattr(hnep, "RepresentationProbe")
    assert hnep.RepresentationProbe is RepresentationProbe
