"""TemporalProbe — measure QCT verdict stability across training checkpoints.

The Sprint 2.5 temporal stability experiment in the thesis re-ran the
HNEP probes at epoch checkpoints 5/10/15/20 of training to test whether
the QCT verdict drifts during training. On FreeSolv, multi-seed runs
revealed bimodality across initialization seeds — the single strongest
empirical argument for multi-seed evaluation in QML to date.

The library cannot train models, so TemporalProbe accepts a list of
adapters (one per checkpoint, ideally same trained model at different
epochs) and runs surrogation at each.

Headline output: does the verdict survive across checkpoints?
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from hnep.adapters.base import Dataset, ModelInterface
from hnep.probes.base import Probe
from hnep.results.probe_result import ProbeResult
from hnep.thresholds import DEFAULT_THRESHOLDS, Thresholds


class TemporalProbe(Probe):
    """Sweep model checkpoints and measure QCT verdict stability.

    Two usage patterns:

    1. **Snapshot list** — pass ``checkpoint_adapters`` (a list of adapters
       wrapping the same model at different training epochs). The probe
       evaluates surrogation at each and reports how the verdict drifts.

    2. **Single adapter** — pass just one adapter; the probe reports its
       surrogation verdict as a single-checkpoint trivial baseline. Useful
       as a sanity check.

    Parameters
    ----------
    checkpoints
        Optional sequence of integer labels (e.g. epoch numbers) matching
        the order of ``checkpoint_adapters`` for reporting. If omitted,
        positional indices ``[0, 1, 2, ...]`` are used.
    thresholds
        QCT thresholds for the per-checkpoint surrogation verdict.
    """

    name = "temporal"
    description = "QCT verdict stability across training checkpoints."

    def __init__(
        self,
        checkpoints: Optional[Sequence[int]] = None,
        thresholds: Thresholds = DEFAULT_THRESHOLDS,
        seed: int = 42,
    ) -> None:
        super().__init__(seed=seed)
        self.checkpoints = list(checkpoints) if checkpoints is not None else None
        self.thresholds = thresholds
        self._config = {
            "seed": seed,
            "checkpoints": self.checkpoints,
            "ss_threshold": thresholds.ss_replaceable,
        }

    def run(
        self,
        model: ModelInterface,
        dataset: Dataset,
        verbose: bool = False,
    ) -> ProbeResult:
        """Standard ``run`` invocation — runs on a single adapter."""
        return self._run_checkpoints([model], dataset, verbose=verbose)

    def run_checkpoints(
        self,
        checkpoint_adapters: Sequence[ModelInterface],
        dataset: Dataset,
        verbose: bool = False,
    ) -> ProbeResult:
        """Run across multiple checkpoint adapters in order."""
        return self._run_checkpoints(
            list(checkpoint_adapters), dataset, verbose=verbose
        )

    # ── Internal ───────────────────────────────────────────────────

    def _run_checkpoints(
        self,
        adapters: List[ModelInterface],
        dataset: Dataset,
        verbose: bool,
    ) -> ProbeResult:
        if len(adapters) == 0:
            raise ValueError("temporal probe requires at least one adapter")

        train_idx = dataset.train_idx
        test_idx = dataset.test_idx
        y_test = dataset.targets[test_idx]

        # Pre-compute scaled inputs once — same across checkpoints
        X_train = dataset.inputs[train_idx]
        X_test = dataset.inputs[test_idx]
        scaler = StandardScaler()
        Xs_train = scaler.fit_transform(X_train)
        Xs_test = scaler.transform(X_test)

        per_checkpoint: list[dict] = []
        for i, adapter in enumerate(adapters):
            cp_label = (self.checkpoints[i] if self.checkpoints is not None
                        and i < len(self.checkpoints) else i)
            if verbose:
                print(f"  [temporal] checkpoint {cp_label}")

            q_train = adapter.extract_quantum_output(dataset, train_idx)
            q_test = adapter.extract_quantum_output(dataset, test_idx)
            y_pred = adapter.predict(dataset, test_idx)

            r2 = float(r2_score(y_test, y_pred))
            ss = _surrogation_score(Xs_train, q_train, Xs_test, q_test)
            verdict = (
                "REPLACEABLE" if ss < self.thresholds.ss_replaceable
                else "NECESSARY"
            )

            per_checkpoint.append({
                "checkpoint": int(cp_label),
                "r2": r2,
                "surrogation_score": float(ss),
                "verdict": verdict,
            })

        # Stability summary
        verdicts = [c["verdict"] for c in per_checkpoint]
        ss_values = [c["surrogation_score"] for c in per_checkpoint]
        unique_verdicts = set(verdicts)

        if len(unique_verdicts) == 1:
            stability = "STABLE"
            confidence = 0.9
        elif len(per_checkpoint) >= 3 and verdicts[0] == verdicts[-1]:
            # Mid-training wobble but end-of-training agrees with start
            stability = "TRANSIENT-DRIFT"
            confidence = 0.6
        else:
            stability = "VERDICT-DRIFTS"
            confidence = 0.4

        ss_swing = (max(ss_values) - min(ss_values)) if ss_values else 0.0

        notes: list[str] = []
        if len(per_checkpoint) == 1:
            notes.append(
                "Only one checkpoint supplied — temporal stability cannot be "
                "characterised. Use run_checkpoints([adapter_epoch_5, ...], "
                "dataset) with multiple snapshots."
            )
        if stability == "VERDICT-DRIFTS":
            notes.append(
                f"Surrogation verdict drifts across {len(per_checkpoint)} "
                "checkpoints — interpret end-of-training metrics with care."
            )

        return ProbeResult(
            probe_name=self.name,
            primary_score=float(ss_swing),
            primary_score_ci=None,
            verdict=stability,
            confidence=float(confidence),
            details={
                "per_checkpoint": per_checkpoint,
                "ss_swing": ss_swing,
                "n_checkpoints": len(per_checkpoint),
                "final_verdict": verdicts[-1],
                "verdict_set": sorted(unique_verdicts),
            },
            config=self.config,
            notes=notes,
        )


def _surrogation_score(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> float:
    """SS = 1 − mean R² across output dims, using Gradient Boosting per dim."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        per_dim_r2 = []
        for d in range(y_train.shape[1]):
            gbr = GradientBoostingRegressor(
                n_estimators=100, max_depth=5,
                learning_rate=0.1, random_state=42,
            )
            gbr.fit(X_train, y_train[:, d])
            preds = gbr.predict(X_test)
            per_dim_r2.append(max(0.0, r2_score(y_test[:, d], preds)))
    return float(min(1.0, max(0.0, 1.0 - float(np.mean(per_dim_r2)))))
