"""ErrorDiversityProbe — measure whether quantum and classical branches make
different mistakes.

Ensemble-theory perspective on quantum's role. We fit two simple readouts:

  * Quantum-only Ridge: ``quantum_output → target``
  * Classical-only Ridge: ``classical_embedding → target``

Then compute per-molecule errors of each, and the Pearson correlation between
them. Low correlation → the branches make different mistakes → potential
ensemble benefit. High correlation → they make the same mistakes → quantum
adds nothing new at the error level.

This is the probe that revealed the QM9 paradox in the thesis: quantum's
output is informationally unique (high SS) but its errors are highly
correlated with the GNN-only readout's errors (REDUNDANT) — *quantum carries
unique information that doesn't translate to different predictions.*

Requires the adapter to implement ``get_classical_embedding``. Adapters that
don't will get an honest ``UNAVAILABLE`` verdict.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from hnep.adapters.base import Dataset, ModelInterface
from hnep.probes.base import Probe
from hnep.results.probe_result import ProbeResult


# Threshold above which we call the two branches "redundant"
DEFAULT_REDUNDANCY_THRESHOLD = 0.5


class ErrorDiversityProbe(Probe):
    """Pearson correlation between quantum-only and classical-only readout errors.

    Parameters
    ----------
    redundancy_threshold
        |r| above this is classified as REDUNDANT, below as DIVERSE.
    ridge_alpha
        L2 regularisation for both readouts.
    """

    name = "error_diversity"
    description = "Residual correlation between quantum-only and classical-only readouts."

    def __init__(
        self,
        redundancy_threshold: float = DEFAULT_REDUNDANCY_THRESHOLD,
        ridge_alpha: float = 1.0,
        seed: int = 42,
    ) -> None:
        super().__init__(seed=seed)
        self.redundancy_threshold = redundancy_threshold
        self.ridge_alpha = ridge_alpha
        self._config = {
            "seed": seed,
            "readout": f"Ridge(alpha={ridge_alpha})",
            "redundancy_threshold": redundancy_threshold,
        }

    def run(
        self,
        model: ModelInterface,
        dataset: Dataset,
        verbose: bool = False,
    ) -> ProbeResult:
        train_idx = dataset.train_idx
        test_idx = dataset.test_idx

        # Need both branches
        q_train = model.extract_quantum_output(dataset, train_idx)
        q_test = model.extract_quantum_output(dataset, test_idx)
        cl_train = model.get_classical_embedding(dataset, train_idx)
        cl_test = model.get_classical_embedding(dataset, test_idx)

        if cl_train is None or cl_test is None:
            return ProbeResult(
                probe_name=self.name,
                primary_score=float("nan"),
                verdict="UNAVAILABLE",
                confidence=0.0,
                notes=[
                    "Adapter does not expose get_classical_embedding(); cannot "
                    "fit classical-only readout. Implement the hook on your "
                    "adapter to enable this probe."
                ],
                config=self.config,
            )

        y_train = dataset.targets[train_idx]
        y_test = dataset.targets[test_idx]

        # ── Fit two readouts (quiet sklearn convergence chatter) ──
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            q_scaler = StandardScaler().fit(q_train)
            q_ridge = Ridge(alpha=self.ridge_alpha,
                            random_state=self.seed).fit(
                q_scaler.transform(q_train), y_train
            )
            q_pred = q_ridge.predict(q_scaler.transform(q_test))
            q_err = q_pred - y_test

            cl_scaler = StandardScaler().fit(cl_train)
            cl_ridge = Ridge(alpha=self.ridge_alpha,
                             random_state=self.seed).fit(
                cl_scaler.transform(cl_train), y_train
            )
            cl_pred = cl_ridge.predict(cl_scaler.transform(cl_test))
            cl_err = cl_pred - y_test

        # ── Pearson correlation of errors ──
        if q_err.std() < 1e-9 or cl_err.std() < 1e-9:
            r = 0.0
        else:
            r = float(np.corrcoef(q_err, cl_err)[0, 1])

        verdict = (
            "REDUNDANT" if abs(r) >= self.redundancy_threshold else "DIVERSE"
        )

        # Per-branch R² for context
        q_only_r2 = float(r2_score(y_test, q_pred))
        cl_only_r2 = float(r2_score(y_test, cl_pred))

        # Confidence based on distance from threshold
        distance = abs(abs(r) - self.redundancy_threshold)
        confidence = float(min(1.0, distance * 2 + 0.5))

        notes: list[str] = []
        if q_only_r2 < 0:
            notes.append(
                "Quantum-only Ridge has negative R² — quantum branch alone "
                "doesn't predict the target. Error diversity verdict still "
                "informative but interpret per-branch R² with care."
            )

        return ProbeResult(
            probe_name=self.name,
            primary_score=float(r),
            primary_score_ci=None,
            verdict=verdict,
            confidence=confidence,
            details={
                "error_correlation": float(r),
                "quantum_only_r2": q_only_r2,
                "classical_only_r2": cl_only_r2,
                "n_test": int(len(test_idx)),
                "redundancy_threshold": self.redundancy_threshold,
            },
            config=self.config,
            notes=notes,
        )
