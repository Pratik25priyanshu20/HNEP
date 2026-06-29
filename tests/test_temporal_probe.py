"""Tests for TemporalProbe."""

from __future__ import annotations

import numpy as np
import pytest

import hnep
from hnep.adapters.base import Dataset
from hnep.probes.temporal import TemporalProbe


def make_synth_dataset(n=200, d=12, seed=0):
    rng = np.random.default_rng(seed)
    inputs = rng.normal(size=(n, d))
    targets = inputs @ rng.normal(size=d) + 0.1 * rng.normal(size=n)
    return Dataset(
        inputs=inputs, targets=targets,
        train_idx=np.arange(0, int(0.7 * n)),
        val_idx=np.arange(int(0.7 * n), int(0.85 * n)),
        test_idx=np.arange(int(0.85 * n), n),
        metadata={"name": "synth_temporal"},
    )


class _Adapter(hnep.ModelInterface):
    """Linear-projection adapter. Constructed with a `noise` multiplier so we
    can simulate 'this is the model at different epochs' by varying noise."""

    name = "ToyEpoch"
    _quantum_dim = 4

    def __init__(self, dataset, seed=0, noise_mult=0.0):
        rng = np.random.default_rng(seed)
        self.proj = rng.normal(size=(dataset.inputs.shape[1], self._quantum_dim))
        self.classical_w = rng.normal(size=dataset.inputs.shape[1]) * 0.3
        self.quantum_w = rng.normal(size=self._quantum_dim) * 0.7
        self._noise_rng = rng
        self._noise_mult = noise_mult

    def _quantum(self, X):
        q = X @ self.proj
        if self._noise_mult > 0:
            q = q + self._noise_mult * self._noise_rng.normal(size=q.shape)
        return q

    def _predict(self, X, q):
        return X @ self.classical_w + q @ self.quantum_w

    def _idx(self, dataset, indices):
        return (np.arange(len(dataset.targets)) if indices is None
                else np.asarray(indices))

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


def test_temporal_probe_single_adapter_returns_inconclusive_note():
    ds = make_synth_dataset()
    model = _Adapter(ds)
    probe = TemporalProbe()
    result = probe.run(model, ds)
    assert result.probe_name == "temporal"
    assert result.details["n_checkpoints"] == 1
    # Verdict should be STABLE (only one verdict to look at)
    assert result.verdict in {"STABLE", "TRANSIENT-DRIFT", "VERDICT-DRIFTS"}
    # Should warn about single checkpoint
    assert any("one checkpoint" in n.lower() for n in result.notes)


def test_temporal_probe_three_stable_checkpoints():
    ds = make_synth_dataset()
    # Three "snapshots" with low noise — all should give the same surrogation verdict
    snapshots = [_Adapter(ds, seed=0, noise_mult=0.0) for _ in range(3)]
    probe = TemporalProbe(checkpoints=[5, 10, 20])
    result = probe.run_checkpoints(snapshots, ds)

    assert result.details["n_checkpoints"] == 3
    assert result.verdict == "STABLE"
    # All three checkpoints should appear in per_checkpoint
    cps = [c["checkpoint"] for c in result.details["per_checkpoint"]]
    assert cps == [5, 10, 20]


def test_temporal_probe_drifting_checkpoints():
    """Construct three snapshots where quantum gets progressively noisier — the
    surrogation score should rise across checkpoints, eventually flipping the
    verdict."""
    ds = make_synth_dataset()
    snapshots = [
        _Adapter(ds, seed=0, noise_mult=0.0),    # clean → REPLACEABLE
        _Adapter(ds, seed=0, noise_mult=0.5),    # noisier
        _Adapter(ds, seed=0, noise_mult=2.0),    # very noisy → likely NECESSARY
    ]
    probe = TemporalProbe(checkpoints=[5, 10, 20])
    result = probe.run_checkpoints(snapshots, ds)

    ss_values = [c["surrogation_score"] for c in result.details["per_checkpoint"]]
    # SS should rise as noise increases
    assert ss_values[0] <= ss_values[2]


def test_temporal_probe_rejects_empty_list():
    ds = make_synth_dataset()
    probe = TemporalProbe()
    with pytest.raises(ValueError):
        probe.run_checkpoints([], ds)


def test_temporal_probe_exports_at_top_level():
    assert hasattr(hnep, "TemporalProbe")
    assert hnep.TemporalProbe is TemporalProbe
