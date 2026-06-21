"""Phase 2 tests — surrogation + intervention probes with synthetic data.

We build two adapters: one where quantum is a deterministic linear function
of the input (REPLACEABLE + load-bearing → Regularizer) and one where the
quantum is genuine noise the decoder needs as a placeholder (NECESSARY +
load-bearing → Genuine).
"""

from __future__ import annotations

import numpy as np

import hnep
from hnep.adapters.base import Dataset, ModelInterface
from hnep.classifiers.qct import QCTVerdict
from hnep.probes.intervention import InterventionProbe
from hnep.probes.surrogation import SurrogationProbe, default_surrogate_ladder


# ─────────────────────────────────────────────────────────────────────
# Synthetic dataset factory
# ─────────────────────────────────────────────────────────────────────
def make_dataset(n: int = 400, d: int = 16, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    inputs = rng.normal(size=(n, d))
    targets = inputs @ rng.normal(size=d) + 0.1 * rng.normal(size=n)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    return Dataset(
        inputs=inputs,
        targets=targets,
        train_idx=np.arange(0, n_train),
        val_idx=np.arange(n_train, n_train + n_val),
        test_idx=np.arange(n_train + n_val, n),
        metadata={"name": "synth"},
    )


# ─────────────────────────────────────────────────────────────────────
# Adapter 1 — quantum = linear projection of input (REPLACEABLE expected)
#             but decoder critically uses it (LOAD-BEARING expected)
# ─────────────────────────────────────────────────────────────────────
class ReplaceableLoadBearingAdapter(ModelInterface):
    """Quantum is a linear projection so trivial surrogates can mimic it.
    The decoder weights heavily on quantum, so removing it crashes performance.
    """

    name = "ReplaceableLB"
    _quantum_dim = 4

    def __init__(self, dataset: Dataset, seed: int = 0):
        rng = np.random.default_rng(seed)
        # Quantum produced as a linear projection — trivial to mimic.
        self.proj = rng.normal(size=(dataset.inputs.shape[1], self._quantum_dim))
        # Decoder: heavily weighted on quantum (≈ 90% of signal).
        self.classical_w = rng.normal(size=dataset.inputs.shape[1]) * 0.05
        self.quantum_w = rng.normal(size=self._quantum_dim) * 1.5
        self.classical_w *= 0.1  # damp the classical contribution

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

    def predict_with_quantum_override(self, dataset, quantum_override, indices=None):
        idx = self._idx(dataset, indices)
        X = dataset.inputs[idx]
        return self._predict(X, quantum_override)

    @staticmethod
    def _idx(dataset, indices):
        return (np.arange(len(dataset.targets)) if indices is None
                else np.asarray(indices))


# ─────────────────────────────────────────────────────────────────────
# Adapter 2 — quantum = irreducible noise (NECESSARY expected)
#             but decoder doesn't use it (NOT load-bearing expected)
# ─────────────────────────────────────────────────────────────────────
class IgnoredAdapter(ModelInterface):
    """Quantum is irreducible noise from input — no classical surrogate can
    mimic it. But the decoder gives it negligible weight, so removing it
    doesn't affect predictions much. Expected QCT: Ignored.
    """

    name = "Ignored"
    _quantum_dim = 4

    def __init__(self, dataset: Dataset, seed: int = 0):
        rng = np.random.default_rng(seed + 1)
        # "Quantum" is sin/cos of a non-trivial transformation — hard for
        # standard surrogates trained from raw input descriptors.
        self.freq = rng.uniform(0.5, 3.0, size=(dataset.inputs.shape[1],
                                                self._quantum_dim))
        self.phase = rng.uniform(0, np.pi, size=(1, self._quantum_dim))
        # Decoder: classical does all the work, quantum gets near-zero weight
        self.classical_w = rng.normal(size=dataset.inputs.shape[1])
        self.quantum_w = np.zeros(self._quantum_dim) + 1e-4  # negligible

    def _quantum(self, X):
        # Sum of sines with random frequencies — not linearly recoverable
        return np.sin(X @ self.freq + self.phase)

    def _predict(self, X, q):
        return X @ self.classical_w + q @ self.quantum_w

    def predict(self, dataset, indices=None):
        idx = self._idx(dataset, indices)
        X = dataset.inputs[idx]
        return self._predict(X, self._quantum(X))

    def extract_quantum_output(self, dataset, indices=None):
        idx = self._idx(dataset, indices)
        return self._quantum(dataset.inputs[idx])

    def predict_with_quantum_override(self, dataset, quantum_override, indices=None):
        idx = self._idx(dataset, indices)
        X = dataset.inputs[idx]
        return self._predict(X, quantum_override)

    @staticmethod
    def _idx(dataset, indices):
        return (np.arange(len(dataset.targets)) if indices is None
                else np.asarray(indices))


# ─────────────────────────────────────────────────────────────────────
# Tests — surrogation probe
# ─────────────────────────────────────────────────────────────────────

def test_surrogation_probe_returns_proper_result():
    dataset = make_dataset()
    model = ReplaceableLoadBearingAdapter(dataset, seed=0)
    # use a small ladder for speed
    ladder = default_surrogate_ladder()[:4]  # linear + KNN + trees
    probe = SurrogationProbe(surrogates=ladder, n_bootstrap=50)

    result = probe.run(model, dataset)
    assert result.probe_name == "surrogation"
    assert 0.0 <= result.primary_score <= 1.0
    assert result.primary_score_ci is not None
    lo, hi = result.primary_score_ci
    assert lo <= hi
    assert result.verdict in {"REPLACEABLE", "NECESSARY"}
    assert "best_surrogate" in result.details
    assert "per_surrogate" in result.details
    assert len(result.details["per_surrogate"]) == 4


def test_surrogation_detects_replaceable_quantum():
    """Linear-projection quantum should be REPLACEABLE — Ridge will succeed."""
    dataset = make_dataset()
    model = ReplaceableLoadBearingAdapter(dataset, seed=0)
    probe = SurrogationProbe(n_bootstrap=50)
    result = probe.run(model, dataset)
    assert result.verdict == "REPLACEABLE"
    assert result.primary_score < 0.2


def test_surrogation_detects_non_replaceable_quantum():
    """sin-of-projection quantum is hard for classical surrogates."""
    dataset = make_dataset()
    model = IgnoredAdapter(dataset, seed=0)
    probe = SurrogationProbe(n_bootstrap=50)
    result = probe.run(model, dataset)
    # We allow either NECESSARY or borderline — sin features can be partially
    # captured by tree models with enough data, but should not trivially win.
    assert result.primary_score > 0.05


# ─────────────────────────────────────────────────────────────────────
# Tests — intervention probe
# ─────────────────────────────────────────────────────────────────────

def test_intervention_probe_returns_proper_result():
    dataset = make_dataset()
    model = ReplaceableLoadBearingAdapter(dataset, seed=0)
    probe = InterventionProbe(n_bootstrap=50)
    result = probe.run(model, dataset)
    assert result.probe_name == "intervention"
    assert result.primary_score_ci is not None
    assert result.verdict in {"LOAD-BEARING", "NOT-LOAD-BEARING"}
    assert "per_intervention" in result.details
    assert set(result.details["per_intervention"].keys()) == {
        "zero_quantum", "random_noise", "permute", "constant"
    }


def test_intervention_detects_load_bearing_quantum():
    """ReplaceableLoadBearingAdapter weights quantum heavily — removing
    should hurt."""
    dataset = make_dataset()
    model = ReplaceableLoadBearingAdapter(dataset, seed=0)
    probe = InterventionProbe(n_bootstrap=50)
    result = probe.run(model, dataset)
    assert result.verdict == "LOAD-BEARING"
    assert result.primary_score > 0.05


def test_intervention_detects_ignored_quantum():
    """IgnoredAdapter has near-zero quantum weight — removing should barely matter."""
    dataset = make_dataset()
    model = IgnoredAdapter(dataset, seed=0)
    probe = InterventionProbe(n_bootstrap=50)
    result = probe.run(model, dataset)
    # The intervention drop should be small; verdict can be NOT-LOAD-BEARING
    # or borderline.
    assert abs(result.primary_score) < 0.2


# ─────────────────────────────────────────────────────────────────────
# End-to-end: hnep.evaluate() with both probes wired up
# ─────────────────────────────────────────────────────────────────────

def test_evaluate_end_to_end_produces_regularizer_verdict():
    dataset = make_dataset()
    model = ReplaceableLoadBearingAdapter(dataset, seed=0)
    # Use a trimmed surrogate ladder for speed
    probes = [
        SurrogationProbe(surrogates=default_surrogate_ladder()[:4], n_bootstrap=50),
        InterventionProbe(n_bootstrap=50),
    ]
    result = hnep.evaluate(model, dataset, probes=probes)

    assert isinstance(result, hnep.HNEPResult)
    assert result.qct_verdict == QCTVerdict.REGULARIZER.value
    assert "surrogation" in result.probes
    assert "intervention" in result.probes
    # Manifest populated
    assert result.manifest["hnep_version"] == hnep.__version__
    assert "elapsed_seconds" in result.manifest


def test_evaluate_uses_default_probes_when_none_given():
    """Just need to confirm the default battery wires up without crashing."""
    dataset = make_dataset(n=100, d=8)
    model = ReplaceableLoadBearingAdapter(dataset, seed=0)
    result = hnep.evaluate(model, dataset)
    assert "surrogation" in result.probes
    assert "intervention" in result.probes
