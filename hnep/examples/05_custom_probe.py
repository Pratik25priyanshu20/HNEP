"""Write a custom HNEP probe and use it alongside the defaults.

This example builds a tiny *Error Diversity Probe* that measures the
correlation between quantum-only and decoder-only prediction errors on the
test set — a fourth lens on quantum contribution.

    python -m hnep.examples.05_custom_probe
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import hnep
from hnep.adapters.base import Dataset, ModelInterface
from hnep.examples.quickstart_helpers import RegularizerAdapter
from hnep.probes.base import Probe
from hnep.results.probe_result import ProbeResult


class ErrorDiversityProbe(Probe):
    """Per-sample residual correlation between quantum-only and GNN-only
    Ridge readouts. Low correlation ⇒ branches make different mistakes."""

    name = "error_diversity"

    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self._config = {"seed": seed, "readout": "Ridge(α=1.0)"}

    def run(self, model: ModelInterface, dataset: Dataset, verbose: bool = False) -> ProbeResult:
        train_idx = dataset.train_idx
        test_idx = dataset.test_idx

        q_train = model.extract_quantum_output(dataset, train_idx)
        q_test = model.extract_quantum_output(dataset, test_idx)
        gnn_train = model.get_classical_embedding(dataset, train_idx)
        gnn_test = model.get_classical_embedding(dataset, test_idx)

        if gnn_train is None or gnn_test is None:
            return ProbeResult(
                probe_name=self.name,
                primary_score=float("nan"),
                verdict="UNAVAILABLE",
                confidence=0.0,
                notes=["Adapter does not expose get_classical_embedding."],
            )

        y_train = dataset.targets[train_idx]
        y_test = dataset.targets[test_idx]

        # Quantum-only Ridge
        q_scaler = StandardScaler().fit(q_train)
        q_ridge = Ridge(alpha=1.0).fit(q_scaler.transform(q_train), y_train)
        q_pred = q_ridge.predict(q_scaler.transform(q_test))
        q_err = q_pred - y_test

        # GNN-only Ridge
        g_scaler = StandardScaler().fit(gnn_train)
        g_ridge = Ridge(alpha=1.0).fit(g_scaler.transform(gnn_train), y_train)
        g_pred = g_ridge.predict(g_scaler.transform(gnn_test))
        g_err = g_pred - y_test

        # Pearson correlation
        if q_err.std() < 1e-9 or g_err.std() < 1e-9:
            r = 0.0
        else:
            r = float(np.corrcoef(q_err, g_err)[0, 1])

        if abs(r) < 0.5:
            verdict = "DIVERSE"
        else:
            verdict = "REDUNDANT"

        return ProbeResult(
            probe_name=self.name,
            primary_score=float(r),
            primary_score_ci=None,
            verdict=verdict,
            confidence=float(1.0 - abs(0.5 - abs(r)) * 2),
            details={"q_only_r2": float(np.corrcoef(q_pred, y_test)[0, 1] ** 2),
                     "gnn_only_r2": float(np.corrcoef(g_pred, y_test)[0, 1] ** 2)},
            config=self.config,
        )


class _RegWithEmbedding(RegularizerAdapter):
    """Tiny subclass exposing get_classical_embedding so the new probe can run."""

    def get_classical_embedding(self, dataset, indices=None):
        idx = np.arange(len(dataset.targets)) if indices is None else np.asarray(indices)
        return dataset.inputs[idx]  # use raw input as classical embedding for the example


def main() -> None:
    rng = np.random.default_rng(0)
    n, d = 300, 16
    dataset = Dataset(
        inputs=rng.normal(size=(n, d)),
        targets=rng.normal(size=n),
        train_idx=np.arange(0, 210),
        val_idx=np.arange(210, 255),
        test_idx=np.arange(255, 300),
        metadata={"name": "synth_custom"},
    )

    adapter = _RegWithEmbedding(dataset)

    result = hnep.evaluate(
        adapter, dataset,
        probes=[hnep.SurrogationProbe(), hnep.InterventionProbe(),
                ErrorDiversityProbe()],
    )
    print(result.summary())


if __name__ == "__main__":
    main()
