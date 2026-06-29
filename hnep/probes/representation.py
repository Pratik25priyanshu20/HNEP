"""RepresentationProbe — CKA + Mutual Information analysis.

To our knowledge, the first application of CKA (Kornblith et al. 2019) to
quantum-classical comparison in QML. Bundled with k-NN mutual information
estimation for a complete representation-level picture.

Three CKA quantities per dataset:

* ``CKA(quantum, classical)`` — geometric similarity between branches.
  High means redundancy, low means genuinely distinct representations.
* ``CKA(quantum, target)`` — how target-aligned the quantum branch is.
* ``CKA(classical, target)`` — same for the classical branch.

Plus two MI quantities:

* Total quantum information share — refines the variance-based "<1% signal"
  claim with a principled information-theoretic measure.
* Per-feature density ratio — controls for the dimensionality gap (quantum
  typically has 4 dims; classical might have 128).

Requires the adapter to expose ``get_classical_embedding``. Returns
``UNAVAILABLE`` gracefully otherwise.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from sklearn.feature_selection import mutual_info_regression

from hnep.adapters.base import Dataset, ModelInterface
from hnep.probes.base import Probe
from hnep.results.probe_result import ProbeResult


# ── Linear CKA implementation ────────────────────────────────────────

def _center(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al. 2019, eq. 5).

        CKA(X, Y) = ‖Yᵀ X‖²_F / (‖Xᵀ X‖_F · ‖Yᵀ Y‖_F)

    Both ``X`` and ``Y`` are first centred along the sample axis.
    Bounded in ``[0, 1]``. Invariant to rotation and rescaling.
    """
    X = _center(np.asarray(X, dtype=np.float64))
    Y = _center(np.asarray(Y, dtype=np.float64))

    YtX = Y.T @ X
    XtX = X.T @ X
    YtY = Y.T @ Y

    num = float(np.linalg.norm(YtX, ord="fro") ** 2)
    den = float(np.linalg.norm(XtX, ord="fro") *
                 np.linalg.norm(YtY, ord="fro"))
    if den < 1e-12:
        return float("nan")
    return num / den


def cka_with_scalar_target(X: np.ndarray, y: np.ndarray) -> float:
    """CKA between feature matrix ``X (N×d)`` and scalar target ``y (N,)``."""
    y_col = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    return linear_cka(X, y_col)


# ── Probe ────────────────────────────────────────────────────────────

class RepresentationProbe(Probe):
    """Representation-level analysis: CKA + Mutual Information.

    Parameters
    ----------
    mi_neighbors
        ``k`` for the Kraskov k-NN mutual-information estimator (sklearn
        default = 3).
    """

    name = "representation"
    description = "CKA + Mutual Information between quantum and classical branches."

    def __init__(
        self,
        mi_neighbors: int = 3,
        seed: int = 42,
    ) -> None:
        super().__init__(seed=seed)
        self.mi_neighbors = mi_neighbors
        self._config = {
            "seed": seed,
            "mi_neighbors": mi_neighbors,
            "cka_kernel": "linear",
        }

    def run(
        self,
        model: ModelInterface,
        dataset: Dataset,
        verbose: bool = False,
    ) -> ProbeResult:
        test_idx = dataset.test_idx

        q_test = model.extract_quantum_output(dataset, test_idx)
        cl_test = model.get_classical_embedding(dataset, test_idx)
        y_test = dataset.targets[test_idx]

        if cl_test is None:
            return ProbeResult(
                probe_name=self.name,
                primary_score=float("nan"),
                verdict="UNAVAILABLE",
                confidence=0.0,
                notes=[
                    "Adapter does not expose get_classical_embedding(); "
                    "cannot compute CKA(quantum, classical) or per-branch MI. "
                    "Implement the hook on your adapter to enable this probe."
                ],
                config=self.config,
            )

        # Drop classical columns with zero variance to avoid numerical issues
        cl_var = cl_test.var(axis=0)
        keep = cl_var > 1e-10
        cl_test_safe = cl_test[:, keep] if not keep.all() else cl_test
        n_classical_kept = int(keep.sum())

        # ── CKA scores ──
        cka_qc = linear_cka(q_test, cl_test_safe)
        cka_qt = cka_with_scalar_target(q_test, y_test)
        cka_ct = cka_with_scalar_target(cl_test_safe, y_test)

        # ── Mutual information ──
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mi_q = mutual_info_regression(
                q_test, y_test, discrete_features=False,
                n_neighbors=self.mi_neighbors, random_state=self.seed,
            )
            mi_cl = mutual_info_regression(
                cl_test_safe, y_test, discrete_features=False,
                n_neighbors=self.mi_neighbors, random_state=self.seed,
            )

        mi_q_total = float(np.sum(mi_q))
        mi_cl_total = float(np.sum(mi_cl))
        mi_q_per_dim = float(np.mean(mi_q))
        mi_cl_per_dim = float(np.mean(mi_cl))
        quantum_info_share = (
            mi_q_total / (mi_q_total + mi_cl_total)
            if (mi_q_total + mi_cl_total) > 1e-12 else float("nan")
        )

        # ── Headline & verdict ──
        # Headline = CKA(quantum, classical). Low values indicate distinct
        # representations between branches. Verdict reads the relative
        # target-alignment of the two branches.
        quantum_more_aligned = (
            not np.isnan(cka_qt)
            and not np.isnan(cka_ct)
            and cka_qt > cka_ct
        )
        verdict = (
            "QUANTUM-MORE-ALIGNED"
            if quantum_more_aligned
            else "CLASSICAL-MORE-ALIGNED"
        )

        notes: list[str] = []
        if not keep.all():
            notes.append(
                f"Dropped {keep.size - n_classical_kept} zero-variance classical "
                "embedding columns before computing CKA/MI."
            )

        return ProbeResult(
            probe_name=self.name,
            primary_score=float(cka_qc),
            primary_score_ci=None,
            verdict=verdict,
            confidence=0.8,  # representation analyses don't have a natural CI
            details={
                "cka_quantum_classical": cka_qc,
                "cka_quantum_target": cka_qt,
                "cka_classical_target": cka_ct,
                "mi_quantum_total": mi_q_total,
                "mi_classical_total": mi_cl_total,
                "mi_quantum_per_dim": mi_q_per_dim,
                "mi_classical_per_dim": mi_cl_per_dim,
                "quantum_info_share": quantum_info_share,
                "n_quantum_dims": int(q_test.shape[1]),
                "n_classical_dims_used": n_classical_kept,
                "quantum_more_aligned_with_target": bool(quantum_more_aligned),
            },
            config=self.config,
            notes=notes,
        )
