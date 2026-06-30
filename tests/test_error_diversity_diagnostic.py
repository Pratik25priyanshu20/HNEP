"""ErrorDiversityProbe is diagnostic-only — verify (a) the low_readout_strength
flag fires correctly and (b) QCTClassifier does not consume ED."""

from __future__ import annotations

import inspect

import numpy as np

import hnep
from hnep import FunctionalAdapter
from hnep.adapters.base import Dataset
from hnep.classifiers.qct import QCTClassifier
from hnep.probes.error_diversity import ErrorDiversityProbe
from hnep.results.probe_result import ProbeResult


def _make_simple_adapter(q, c, y, x):
    n = q.shape[0]

    def _resolve(idx):
        return np.arange(n) if idx is None else np.asarray(idx)

    def predict_fn(ds, idx):
        return np.zeros(len(_resolve(idx)))

    def extract_q(ds, idx):
        return q[_resolve(idx)]

    def predict_override(ds, q_ovr, idx):
        return np.zeros(len(_resolve(idx)))

    def classical_emb(ds, idx):
        return c[_resolve(idx)]

    adapter = FunctionalAdapter(
        name="ed_test",
        predict_fn=predict_fn,
        extract_quantum_fn=extract_q,
        predict_with_override_fn=predict_override,
        quantum_dim=q.shape[1],
        get_classical_embedding_fn=classical_emb,
    )
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_train = int(0.75 * n)
    return adapter, Dataset(
        inputs=x,
        targets=y,
        train_idx=np.sort(perm[:n_train]),
        val_idx=np.array([], dtype=int),
        test_idx=np.sort(perm[n_train:]),
    )


def test_ed_details_now_include_low_readout_strength_key():
    rng = np.random.default_rng(0)
    n = 300
    q = rng.normal(size=(n, 4))
    c = rng.normal(size=(n, 12))
    x = rng.normal(size=(n, 8))
    y = rng.normal(size=n)
    adapter, dataset = _make_simple_adapter(q, c, y, x)
    result = ErrorDiversityProbe().run(adapter, dataset)
    assert "low_readout_strength" in result.details


def test_ed_low_readout_strength_fires_on_unpredictable_target():
    """y unrelated to (q, c) → both Ridges score < 0.4 → flag fires."""
    rng = np.random.default_rng(0)
    n = 300
    q = rng.normal(size=(n, 4))
    c = rng.normal(size=(n, 12))
    x = rng.normal(size=(n, 8))
    y = rng.normal(size=n)
    adapter, dataset = _make_simple_adapter(q, c, y, x)
    result = ErrorDiversityProbe(low_readout_strength_threshold=0.4).run(
        adapter, dataset
    )
    assert result.details["low_readout_strength"] is True
    assert any("low_readout_strength" in n for n in result.notes)


def test_ed_low_readout_strength_quiet_on_clean_linear_signal():
    """Both branches linear in y → both Ridges score > 0.4 → flag stays False."""
    rng = np.random.default_rng(0)
    n = 300
    y = rng.normal(size=n)
    q = y[:, None] * rng.normal(scale=0.5, size=(1, 4)) + 0.05 * rng.normal(size=(n, 4))
    c = y[:, None] * rng.normal(scale=0.5, size=(1, 12)) + 0.05 * rng.normal(size=(n, 12))
    x = rng.normal(size=(n, 8))
    adapter, dataset = _make_simple_adapter(q, c, y, x)
    result = ErrorDiversityProbe().run(adapter, dataset)
    assert result.details["low_readout_strength"] is False


def test_qct_classifier_signature_does_not_take_error_diversity():
    """The QCTClassifier.classify signature must not accept an
    error_diversity argument — gating on ED would silently consume the
    diagnostic probe."""
    sig = inspect.signature(QCTClassifier.classify)
    param_names = set(sig.parameters.keys())
    assert "error_diversity" not in param_names
    assert param_names == {"self", "surrogation", "intervention"}


def test_evaluate_verdict_unchanged_when_ed_included_or_excluded():
    """Including ErrorDiversityProbe in the probe list must not change the
    QCT verdict (which is determined by surrogation + intervention only)."""
    from hnep.benchmarks import make_genuine
    from hnep.probes.intervention import InterventionProbe
    from hnep.probes.surrogation import SurrogationProbe

    adapter, dataset, _ = make_genuine(seed=0, n_samples=300)
    sur = SurrogationProbe(n_bootstrap=100)
    inter = InterventionProbe(n_bootstrap=100)
    ed = ErrorDiversityProbe()

    r_without_ed = hnep.evaluate(adapter, dataset, probes=[sur, inter])
    r_with_ed = hnep.evaluate(adapter, dataset, probes=[sur, inter, ed])
    assert r_without_ed.qct_verdict == r_with_ed.qct_verdict
