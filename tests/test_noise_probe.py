"""Tests for NoiseProbe."""

from __future__ import annotations

import numpy as np
import pytest

import hnep
from hnep.adapters.base import Dataset
from hnep.probes.noise import (
    NoiseProbe,
    bit_flip_noise,
    depolarizing_approx,
    gaussian_noise,
)


def make_synth_dataset(n=200, d=12, seed=0):
    rng = np.random.default_rng(seed)
    inputs = rng.normal(size=(n, d))
    targets = inputs @ rng.normal(size=d) + 0.1 * rng.normal(size=n)
    return Dataset(
        inputs=inputs, targets=targets,
        train_idx=np.arange(0, int(0.7 * n)),
        val_idx=np.arange(int(0.7 * n), int(0.85 * n)),
        test_idx=np.arange(int(0.85 * n), n),
        metadata={"name": "synth_noise"},
    )


# ── Noise-model unit tests ───────────────────────────────────────────

def test_gaussian_noise_preserves_shape_and_dtype():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(50, 4))
    noisy = gaussian_noise(q, 0.1, rng)
    assert noisy.shape == q.shape
    assert noisy.dtype == q.dtype


def test_gaussian_noise_at_zero_level_is_identity():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(50, 4))
    noisy = gaussian_noise(q, 0.0, rng)
    np.testing.assert_allclose(noisy, q, atol=1e-10)


def test_depolarizing_approx_shrinks_to_zero():
    rng = np.random.default_rng(0)
    q = np.ones((20, 4))            # all 1s
    noisy = depolarizing_approx(q, 1.0, rng)   # level=1 → full depolarization
    # mean should be ~0 (within isotropic noise) since shrinkage = (1-1)·q
    assert abs(noisy.mean()) < 0.5


def test_bit_flip_at_zero_level_is_identity():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(50, 4))
    noisy = bit_flip_noise(q, 0.0, rng)
    np.testing.assert_allclose(noisy, q)


def test_bit_flip_at_full_level_flips_signs():
    rng = np.random.default_rng(0)
    q = np.ones((50, 4))
    noisy = bit_flip_noise(q, 1.0, rng)
    np.testing.assert_allclose(noisy, -q)


# ── NoiseProbe integration tests ─────────────────────────────────────

class _LinearAdapter(hnep.ModelInterface):
    """Toy adapter where quantum is a linear projection and decoder uses it
    moderately, so noise should hurt but not collapse the model."""

    name = "LinearToy"
    _quantum_dim = 4

    def __init__(self, dataset, seed=0):
        rng = np.random.default_rng(seed)
        self.proj = rng.normal(size=(dataset.inputs.shape[1], self._quantum_dim))
        self.classical_w = rng.normal(size=dataset.inputs.shape[1]) * 0.3
        self.quantum_w = rng.normal(size=self._quantum_dim) * 0.7

    def _quantum(self, X):
        return X @ self.proj

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


def test_noise_probe_runs_end_to_end():
    ds = make_synth_dataset()
    model = _LinearAdapter(ds)
    probe = NoiseProbe(noise_levels=(0.01, 0.05, 0.10), noise_type="gaussian")
    result = probe.run(model, ds)

    assert result.probe_name == "noise"
    assert result.verdict in {
        "STABLE", "MOSTLY-STABLE", "PERFORMANCE-DEGRADES", "VERDICT-FLIPS",
    }
    assert "per_level" in result.details
    # 0.0 is always prepended, so 4 levels total
    assert len(result.details["per_level"]) == 4
    # All levels should appear
    assert {lvl["level"] for lvl in result.details["per_level"]} == {0.0, 0.01, 0.05, 0.10}


def test_noise_probe_clean_baseline_matches_intact():
    """At noise=0 the noisy R² should exactly match the intact R²."""
    ds = make_synth_dataset()
    model = _LinearAdapter(ds)
    probe = NoiseProbe(noise_levels=(0.05,), noise_type="gaussian")
    result = probe.run(model, ds)

    clean_level = next(l for l in result.details["per_level"] if l["level"] == 0.0)
    assert abs(clean_level["delta_r2_from_intact"]) < 1e-6


def test_noise_probe_higher_noise_hurts_more():
    """A larger noise level should produce at least as large an R² drop as smaller noise (in expectation)."""
    ds = make_synth_dataset()
    model = _LinearAdapter(ds)
    probe = NoiseProbe(noise_levels=(0.05, 0.30, 0.80), noise_type="gaussian", seed=7)
    result = probe.run(model, ds)

    levels = result.details["per_level"]
    by_level = {l["level"]: l["delta_r2_from_intact"] for l in levels}
    # at 0.80 noise the drop should be larger than at 0.05
    assert by_level[0.80] >= by_level[0.05] - 1e-6


def test_noise_probe_accepts_custom_noise_fn():
    """User-supplied noise function should be invoked."""
    called = {"count": 0}

    def my_noise(q, level, rng):
        called["count"] += 1
        return q + level

    ds = make_synth_dataset()
    model = _LinearAdapter(ds)
    probe = NoiseProbe(
        noise_levels=(0.1,),
        noise_type="custom",
        noise_fn=my_noise,
    )
    probe.run(model, ds)
    # noise_fn called once per non-zero level (0.0 short-circuits)
    assert called["count"] >= 1


def test_noise_probe_rejects_unknown_type():
    with pytest.raises(ValueError):
        NoiseProbe(noise_type="not_a_real_type")


def test_noise_probe_custom_requires_callable():
    with pytest.raises(ValueError):
        NoiseProbe(noise_type="custom", noise_fn=None)


def test_noise_probe_exports_at_top_level():
    """NoiseProbe should be importable from the top-level `hnep` namespace."""
    assert hasattr(hnep, "NoiseProbe")
    assert hnep.NoiseProbe is NoiseProbe
