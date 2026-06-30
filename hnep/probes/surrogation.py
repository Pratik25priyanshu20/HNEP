"""SurrogationProbe — surrogate-ladder mimicry test.

Trains a battery of classical surrogate models to predict the quantum branch's
output from the dataset's classical features. If any surrogate succeeds, the
quantum output is informationally **replaceable**.

The headline score is the **minimum** Surrogation Score (= 1 − best mimicry R²)
across the surrogate battery. This is intentionally conservative: if even one
classical model can mimic the quantum, we report it as REPLACEABLE rather than
hide behind weaker surrogates.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from hnep.adapters.base import Dataset, ModelInterface
from hnep.probes.base import Probe
from hnep.results.probe_result import ProbeResult
from hnep.thresholds import DEFAULT_THRESHOLDS, Thresholds
from hnep.utils.bootstrap import bootstrap_statistic_ci


@dataclass(frozen=True)
class SurrogateSpec:
    """Description of one surrogate model in the ladder."""

    name: str
    family: str  # "linear" | "tree" | "neural" | "instance"
    factory: Any  # callable returning a fresh sklearn estimator

    def build(self) -> BaseEstimator:
        return self.factory()


def default_surrogate_ladder() -> List[SurrogateSpec]:
    """Eight-surrogate ladder used throughout HNEP."""
    return [
        SurrogateSpec("Linear (Ridge)", "linear",
                      lambda: Ridge(alpha=1.0, random_state=42)),
        SurrogateSpec("KNN (k=5)", "instance",
                      lambda: KNeighborsRegressor(n_neighbors=5)),
        SurrogateSpec("Random Forest", "tree",
                      lambda: RandomForestRegressor(n_estimators=100,
                                                     random_state=42, n_jobs=-1)),
        SurrogateSpec("Gradient Boosting", "tree",
                      lambda: GradientBoostingRegressor(n_estimators=100,
                                                         max_depth=5,
                                                         learning_rate=0.1,
                                                         random_state=42)),
        SurrogateSpec("Small MLP (32)", "neural",
                      lambda: MLPRegressor(hidden_layer_sizes=(32,),
                                            max_iter=500, random_state=42)),
        SurrogateSpec("Medium MLP (64, 64)", "neural",
                      lambda: MLPRegressor(hidden_layer_sizes=(64, 64),
                                            max_iter=500, random_state=42)),
        SurrogateSpec("Large MLP (128, 128, 64)", "neural",
                      lambda: MLPRegressor(hidden_layer_sizes=(128, 128, 64),
                                            max_iter=500, random_state=42)),
        SurrogateSpec("Deep MLP (256, 128, 64, 32)", "neural",
                      lambda: MLPRegressor(
                          hidden_layer_sizes=(256, 128, 64, 32),
                          max_iter=500, random_state=42)),
    ]


def _fit_per_dim(surrogate: BaseEstimator, X: np.ndarray, Y: np.ndarray) -> List[BaseEstimator]:
    """Train one surrogate copy per quantum output dimension."""
    models = []
    for d in range(Y.shape[1]):
        m = clone(surrogate)
        m.fit(X, Y[:, d])
        models.append(m)
    return models


def _predict_per_dim(models: List[BaseEstimator], X: np.ndarray) -> np.ndarray:
    return np.stack([m.predict(X) for m in models], axis=1)


def _surrogation_score(per_dim_r2: Sequence[float]) -> float:
    """SS = 1 − mean R² across dims, clipped to [0, 1]."""
    mean_r2 = float(np.mean([max(0.0, r) for r in per_dim_r2]))
    return float(min(1.0, max(0.0, 1.0 - mean_r2)))


class SurrogationProbe(Probe):
    """Train a battery of classical surrogates to mimic the quantum output.

    Parameters
    ----------
    surrogates
        Custom surrogate ladder. Defaults to :func:`default_surrogate_ladder`.
    n_bootstrap
        Number of bootstrap resamples for the CI on the headline SS.
    n_jobs
        Reserved for parallelisation (not yet used).
    thresholds
        Used to derive the verdict (REPLACEABLE / NECESSARY).
    """

    name = "surrogation"
    description = "Multi-surrogate mimicry test for the quantum branch."

    def __init__(
        self,
        surrogates: Optional[List[SurrogateSpec]] = None,
        n_bootstrap: int = 500,
        n_jobs: int = 1,
        thresholds: Thresholds = DEFAULT_THRESHOLDS,
        seed: int = 42,
        calibrate: bool = False,
        n_permutations: int = 500,
    ) -> None:
        super().__init__(seed=seed)
        self.surrogates = surrogates if surrogates is not None else default_surrogate_ladder()
        self.n_bootstrap = n_bootstrap
        self.n_jobs = n_jobs
        self.thresholds = thresholds
        self.calibrate = calibrate
        self.n_permutations = n_permutations
        self._config = {
            "seed": seed,
            "surrogates": [s.name for s in self.surrogates],
            "n_bootstrap": n_bootstrap,
            "ss_threshold": thresholds.ss_replaceable,
            "calibrate": calibrate,
            "n_permutations": n_permutations if calibrate else 0,
        }

    # ── Probe API ───────────────────────────────────────────────────

    def run(
        self,
        model: ModelInterface,
        dataset: Dataset,
        verbose: bool = False,
    ) -> ProbeResult:
        train_idx = dataset.train_idx
        test_idx = dataset.test_idx

        # Extract quantum output for train/test
        q_train = model.extract_quantum_output(dataset, train_idx)
        q_test = model.extract_quantum_output(dataset, test_idx)
        X_train = dataset.inputs[train_idx]
        X_test = dataset.inputs[test_idx]

        # Standardise inputs (fairer comparison across surrogate families)
        scaler = StandardScaler()
        Xs_train = scaler.fit_transform(X_train)
        Xs_test = scaler.transform(X_test)

        per_surrogate: List[Dict[str, Any]] = []
        for spec in self.surrogates:
            if verbose:
                print(f"  fitting {spec.name} …")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    surr_models = _fit_per_dim(spec.build(), Xs_train, q_train)
                    preds = _predict_per_dim(surr_models, Xs_test)

                per_dim = [
                    r2_score(q_test[:, d], preds[:, d]) for d in range(q_test.shape[1])
                ]
                ss = _surrogation_score(per_dim)
                per_surrogate.append({
                    "name": spec.name,
                    "family": spec.family,
                    "surrogation_score": ss,
                    "per_dim_r2": [float(r) for r in per_dim],
                    "succeeded": ss < self.thresholds.ss_replaceable,
                })
            except Exception as exc:  # noqa: BLE001 — record but don't crash
                per_surrogate.append({
                    "name": spec.name,
                    "family": spec.family,
                    "surrogation_score": 1.0,
                    "per_dim_r2": [],
                    "succeeded": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })

        # Headline: minimum SS across the battery (best surrogate wins)
        best_idx = int(np.argmin([row["surrogation_score"] for row in per_surrogate]))
        best = per_surrogate[best_idx]
        headline_ss = best["surrogation_score"]

        # Bootstrap CI on the headline SS. Each resample recomputes the
        # SAME statistic the headline reports — `1 - mean_d(max(0, R²_d))`
        # using the best surrogate's predictions on the resampled test set.
        # The pre-T1.2 implementation used a closed-form proxy (MSE divided
        # by mean variance) which mismatched the headline denominator
        # (per-dim variance) — fixed in v0.3.1.
        best_spec = self.surrogates[best_idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_models = _fit_per_dim(best_spec.build(), Xs_train, q_train)
        rng = np.random.default_rng(self.seed)

        def _ss_statistic(resampled_test_idx: np.ndarray) -> float:
            q_t = model.extract_quantum_output(dataset, resampled_test_idx)
            x_t = scaler.transform(dataset.inputs[resampled_test_idx])
            p_t = _predict_per_dim(best_models, x_t)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                per_dim_r2 = [
                    max(0.0, float(r2_score(q_t[:, d], p_t[:, d])))
                    for d in range(q_t.shape[1])
                ]
            return float(min(1.0, max(0.0, 1.0 - float(np.mean(per_dim_r2)))))

        ci_lo, ci_hi = bootstrap_statistic_ci(
            _ss_statistic,
            test_idx,
            n_resamples=self.n_bootstrap,
            rng=rng,
            cluster_ids=dataset.cluster_ids,
        )
        ci_lo = float(max(0.0, min(1.0, ci_lo)))
        ci_hi = float(max(0.0, min(1.0, ci_hi)))

        if headline_ss < self.thresholds.ss_replaceable:
            verdict = "REPLACEABLE"
        else:
            verdict = "NECESSARY"

        # Confidence: how far the CI sits from the threshold (0 if straddles)
        threshold = self.thresholds.ss_replaceable
        if ci_lo > threshold or ci_hi < threshold:
            distance = min(abs(ci_lo - threshold), abs(ci_hi - threshold))
            confidence = float(min(1.0, distance / max(threshold, 0.05) + 0.5))
        else:
            confidence = 0.4

        notes = []
        if any("error" in row for row in per_surrogate):
            failed = [r["name"] for r in per_surrogate if "error" in r]
            notes.append(f"Surrogates that errored: {failed}")

        # ── Optional: permutation p-value ──────────────────────────────
        # Null: q is independent of x (row-shuffled across train+test).
        # Under H0, the best surrogate refit on shuffled q learns nothing
        # generalisable, so SS_perm ≈ 1. Observed SS lower than this null
        # tail ⇒ surrogate succeeds with significance.
        p_value: Optional[float] = None
        if self.calibrate:
            perm_rng = np.random.default_rng(self.seed + 1)
            n_train_local = q_train.shape[0]
            q_all = np.concatenate([q_train, q_test], axis=0)
            perm_stats = np.empty(self.n_permutations)
            for i in range(self.n_permutations):
                perm = perm_rng.permutation(q_all.shape[0])
                q_all_s = q_all[perm]
                q_train_s = q_all_s[:n_train_local]
                q_test_s = q_all_s[n_train_local:]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        perm_models = _fit_per_dim(best_spec.build(), Xs_train, q_train_s)
                        perm_preds = _predict_per_dim(perm_models, Xs_test)
                        per_dim_r2 = [
                            max(0.0, float(r2_score(q_test_s[:, d], perm_preds[:, d])))
                            for d in range(q_test_s.shape[1])
                        ]
                        perm_stats[i] = 1.0 - float(np.mean(per_dim_r2))
                    except Exception:
                        perm_stats[i] = 1.0
            # Lower-tail p-value (Davison-Hinkley): how often does the
            # shuffled-q null produce SS ≤ the observed SS?
            p_value = float(
                (1 + int(np.sum(perm_stats <= headline_ss)))
                / (1 + self.n_permutations)
            )

        return ProbeResult(
            probe_name=self.name,
            primary_score=float(headline_ss),
            primary_score_ci=(ci_lo, ci_hi),
            verdict=verdict,
            confidence=float(confidence),
            details={
                "best_surrogate": best["name"],
                "best_surrogate_family": best["family"],
                "per_surrogate": per_surrogate,
                "n_surrogates_succeeded": sum(1 for r in per_surrogate if r["succeeded"]),
                "n_surrogates": len(per_surrogate),
            },
            config=self.config,
            notes=notes,
            p_value=p_value,
        )
