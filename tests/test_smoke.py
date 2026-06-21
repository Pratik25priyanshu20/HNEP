"""Phase 1 smoke tests — verify imports and basic shape of the public API."""

from __future__ import annotations

import numpy as np

import hnep
from hnep.adapters.base import Dataset, ModelInterface
from hnep.classifiers.qct import QCTClassifier, QCTVerdict
from hnep.probes.base import Probe
from hnep.results.hnep_result import HNEPResult
from hnep.results.probe_result import ProbeResult
from hnep.thresholds import DEFAULT_THRESHOLDS, Thresholds


def test_version_is_string():
    assert isinstance(hnep.__version__, str)
    assert hnep.__version__.startswith("0.1.0")


def test_top_level_exports_exist():
    # The top-level API surface promised in __init__.py
    assert hasattr(hnep, "evaluate")
    assert hasattr(hnep, "HNEPResult")
    assert hasattr(hnep, "ProbeResult")
    assert hasattr(hnep, "ModelInterface")
    assert hasattr(hnep, "Probe")
    assert hasattr(hnep, "QCTClassifier")


def test_thresholds_are_immutable():
    t = Thresholds(ss_replaceable=0.3, intervention_load_bearing=0.1)
    assert t.ss_replaceable == 0.3
    # frozen dataclass — should raise on mutation
    try:
        t.ss_replaceable = 0.5  # type: ignore[misc]
        raised = False
    except Exception:
        raised = True
    assert raised


def test_qct_classifier_basic_logic():
    classifier = QCTClassifier(thresholds=DEFAULT_THRESHOLDS)

    # Surrogation NECESSARY (SS=0.4), intervention LOAD-BEARING (Δ=0.3)
    sur = ProbeResult("surrogation", primary_score=0.4)
    inter = ProbeResult("intervention", primary_score=0.3)
    assert classifier.classify(sur, inter) == QCTVerdict.GENUINE

    # Surrogation REPLACEABLE (SS=0.05), intervention LOAD-BEARING (Δ=0.3)
    sur = ProbeResult("surrogation", primary_score=0.05)
    assert classifier.classify(sur, inter) == QCTVerdict.REGULARIZER

    # Surrogation NECESSARY (SS=0.4), intervention NOT load-bearing (Δ=0.01)
    sur = ProbeResult("surrogation", primary_score=0.4)
    inter = ProbeResult("intervention", primary_score=0.01)
    assert classifier.classify(sur, inter) == QCTVerdict.IGNORED

    # Surrogation REPLACEABLE (SS=0.05), intervention NOT load-bearing (Δ=0.01)
    sur = ProbeResult("surrogation", primary_score=0.05)
    assert classifier.classify(sur, inter) == QCTVerdict.DEAD_WEIGHT


def test_qct_inconclusive_when_ci_straddles_threshold():
    classifier = QCTClassifier(thresholds=DEFAULT_THRESHOLDS)
    sur = ProbeResult(
        "surrogation", primary_score=0.21, primary_score_ci=(0.15, 0.28)
    )
    inter = ProbeResult("intervention", primary_score=0.3)
    assert classifier.classify(sur, inter) == QCTVerdict.INCONCLUSIVE


def test_evaluate_with_explicit_empty_probe_list_returns_inconclusive():
    """If the user passes an empty probe list, no classifier can run and
    the verdict should fall back to Inconclusive."""
    class _Adapter(ModelInterface):
        name = "tiny"
        def predict(self, dataset, indices=None):
            idx = np.arange(len(dataset.targets)) if indices is None else np.asarray(indices)
            return np.zeros(len(idx))
        def extract_quantum_output(self, dataset, indices=None):
            idx = np.arange(len(dataset.targets)) if indices is None else np.asarray(indices)
            return np.zeros((len(idx), 4))
        def predict_with_quantum_override(self, dataset, quantum_override, indices=None):
            idx = np.arange(len(dataset.targets)) if indices is None else np.asarray(indices)
            return np.zeros(len(idx))

    rng = np.random.default_rng(0)
    dataset = Dataset(
        inputs=rng.normal(size=(20, 4)),
        targets=rng.normal(size=20),
        train_idx=np.arange(0, 16),
        val_idx=np.arange(16, 18),
        test_idx=np.arange(18, 20),
        metadata={"name": "tiny"},
    )
    result = hnep.evaluate(_Adapter(), dataset, probes=[])

    assert isinstance(result, HNEPResult)
    assert result.qct_verdict == QCTVerdict.INCONCLUSIVE.value
    assert len(result.probes) == 0


def test_hnep_result_serialisation_roundtrip(tmp_path):
    result = HNEPResult(
        model_name="m", dataset_name="d",
        qct_verdict="Regularizer", qct_confidence=0.9,
    )
    result.probes["surrogation"] = ProbeResult(
        "surrogation", 0.077, primary_score_ci=(0.06, 0.10),
        verdict="REPLACEABLE", confidence=0.95,
    )

    # JSON round-trip
    json_str = result.to_json()
    assert "Regularizer" in json_str
    assert "REPLACEABLE" in json_str

    # CSV
    csv_path = tmp_path / "report.csv"
    result.to_csv(csv_path)
    text = csv_path.read_text()
    assert "surrogation" in text
    assert "0.077" in text

    # HTML
    html_path = tmp_path / "report.html"
    result.to_html(html_path)
    assert "Regularizer" in html_path.read_text()


def test_probe_is_abstract():
    """Probe ABC cannot be instantiated directly."""
    try:
        Probe()  # type: ignore[abstract]
        raised = False
    except TypeError:
        raised = True
    assert raised
