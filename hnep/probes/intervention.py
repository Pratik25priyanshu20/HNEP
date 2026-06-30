"""InterventionProbe — the intervention family.

Implements four counterfactual interventions on the trained model and measures
the relative R² drop in each. The headline score is the relative drop under
the canonical ``zero_quantum`` intervention; the other three (random_noise,
permute, constant) help distinguish *specific-information* dependence from
*placeholder* dependence.

Causal-inference framing
------------------------
These are formal ``do()``-operator interventions on the trained model's
computational graph:

  * ``zero_quantum``  ↔ ``do(quantum = ∅)``
  * ``random_noise``  ↔ ``do(quantum = noise(σ_q))``
  * ``permute``       ↔ ``do(quantum = π(quantum))``
  * ``constant``      ↔ ``do(quantum = mean(quantum))``

Disagreements between them are the most informative output of this probe.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import r2_score

from hnep.adapters.base import Dataset, ModelInterface
from hnep.probes.base import Probe
from hnep.results.probe_result import ProbeResult
from hnep.thresholds import DEFAULT_THRESHOLDS, Thresholds
from hnep.utils.bootstrap import bootstrap_statistic_ci


def _relative_drop(r2_intact: float, r2_intervened: float) -> float:
    """Relative R² drop, signed. Positive ⇒ intervention hurt the model."""
    if abs(r2_intact) < 1e-9:
        return float(r2_intact - r2_intervened)
    return float((r2_intact - r2_intervened) / r2_intact)


class InterventionProbe(Probe):
    """Run the do()-operator family on a trained hybrid model.

    Parameters
    ----------
    interventions
        Which interventions to run. Defaults to all four. The headline
        verdict uses ``zero_quantum``.
    noise_scale
        Standard deviation multiplier for the ``random_noise`` intervention.
        ``1.0`` matches the empirical per-dim std of the quantum output.
    n_bootstrap
        Number of bootstrap resamples for the CI on the headline Δ.
    """

    name = "intervention"
    description = "do()-operator family of structural interventions."

    DEFAULT_INTERVENTIONS = (
        "zero_quantum",
        "random_noise",
        "permute",
        "constant",
    )

    def __init__(
        self,
        interventions: Sequence[str] = DEFAULT_INTERVENTIONS,
        noise_scale: float = 1.0,
        n_bootstrap: int = 500,
        thresholds: Thresholds = DEFAULT_THRESHOLDS,
        seed: int = 42,
        calibrate: bool = False,
        n_permutations: int = 500,
    ) -> None:
        super().__init__(seed=seed)
        self.interventions = list(interventions)
        self.noise_scale = noise_scale
        self.n_bootstrap = n_bootstrap
        self.thresholds = thresholds
        self.calibrate = calibrate
        self.n_permutations = n_permutations
        self._config = {
            "seed": seed,
            "interventions": list(interventions),
            "noise_scale": noise_scale,
            "n_bootstrap": n_bootstrap,
            "load_threshold": thresholds.intervention_load_bearing,
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
        rng = np.random.default_rng(self.seed)
        test_idx = dataset.test_idx
        y_true = dataset.targets[test_idx]

        # Baseline — the intact model's predictions on the test set
        y_intact = model.predict(dataset, test_idx)
        r2_intact = float(r2_score(y_true, y_intact))

        # Quantum output (used by some interventions)
        q_test = model.extract_quantum_output(dataset, test_idx)

        results: Dict[str, Dict[str, Any]] = {}
        for inter_name in self.interventions:
            if verbose:
                print(f"  intervention: {inter_name}")

            override = self._build_override(inter_name, q_test, rng)
            y_inter = model.predict_with_quantum_override(
                dataset, override, indices=test_idx
            )
            r2_inter = float(r2_score(y_true, y_inter))
            delta = _relative_drop(r2_intact, r2_inter)

            results[inter_name] = {
                "r2_intervened": r2_inter,
                "delta_r2": delta,
                "is_load_bearing": delta > self.thresholds.intervention_load_bearing,
            }

        # Headline: zero_quantum is the canonical Δ.
        headline_name = (
            "zero_quantum"
            if "zero_quantum" in results
            else self.interventions[0]
        )
        headline_delta = results[headline_name]["delta_r2"]

        # Bootstrap CI on the headline delta — resample test_idx and recompute
        def _stat(resampled_idx: np.ndarray) -> float:
            yt = dataset.targets[resampled_idx]
            yi = model.predict(dataset, resampled_idx)
            r2_i = float(r2_score(yt, yi))
            # Reuse the same intervention recipe on the resampled set
            q_resampled = model.extract_quantum_output(dataset, resampled_idx)
            ovr = self._build_override(headline_name, q_resampled, rng)
            yv = model.predict_with_quantum_override(
                dataset, ovr, indices=resampled_idx
            )
            r2_v = float(r2_score(yt, yv))
            return _relative_drop(r2_i, r2_v)

        ci_lo, ci_hi = bootstrap_statistic_ci(
            _stat,
            test_idx,
            n_resamples=self.n_bootstrap,
            rng=rng,
            cluster_ids=dataset.cluster_ids,
        )

        if headline_delta > self.thresholds.intervention_load_bearing:
            verdict = "LOAD-BEARING"
        else:
            verdict = "NOT-LOAD-BEARING"

        threshold = self.thresholds.intervention_load_bearing
        if ci_lo > threshold or ci_hi < threshold:
            distance = min(abs(ci_lo - threshold), abs(ci_hi - threshold))
            confidence = float(min(1.0, distance / max(threshold, 0.05) + 0.5))
        else:
            confidence = 0.4

        # Disagreement diagnostic — useful for the IGNORED vs DEAD WEIGHT case
        deltas = [r["delta_r2"] for r in results.values()]
        disagreement = (
            float(np.max(deltas) - np.min(deltas)) if len(deltas) > 1 else 0.0
        )

        notes: List[str] = []
        if disagreement > 0.1:
            notes.append(
                f"Large inter-intervention disagreement (Δ range = {disagreement:.2f}). "
                "Consider inspecting per-intervention results."
            )

        # ── Optional: shuffle-consistency descriptive statistic ────────
        # NOT a p-value. The per-row shuffle of q_test gives "Δ when the
        # model receives wrong-q instead of zero-q" — for any model that
        # genuinely uses q, shuffled-q strictly dominates zero-q in damage
        # (β_q applied to wrong values vs β_q applied to zeros), so
        # Δ_perm ≥ Δ_obs is the rule, not a tail. The null hypothesis
        # required for a true p-value ("β_q = 0") would need refitting
        # the user's decoder, which HNEP cannot do.
        #
        # We expose the mean (Δ_perm − Δ_obs) as a descriptive
        # ``delta_shuffle_consistency``: large positive ⇒ shuffled-q is
        # much worse than zero-q ⇒ model depends on q's *specific values*,
        # not just on q's presence. The QCT classifier does not consume it.
        delta_shuffle_consistency: Optional[float] = None
        if self.calibrate:
            perm_rng = np.random.default_rng(self.seed + 1)
            perm_stats = np.empty(self.n_permutations)
            for i in range(self.n_permutations):
                q_perm = q_test.copy()
                perm_rng.shuffle(q_perm, axis=0)
                y_perm = model.predict_with_quantum_override(
                    dataset, q_perm, indices=test_idx
                )
                r2_perm = float(r2_score(y_true, y_perm))
                y_zero = model.predict_with_quantum_override(
                    dataset, np.zeros_like(q_perm), indices=test_idx
                )
                r2_zero = float(r2_score(y_true, y_zero))
                perm_stats[i] = _relative_drop(r2_perm, r2_zero)
            delta_shuffle_consistency = float(np.mean(perm_stats) - headline_delta)

        details_out: Dict[str, Any] = {
            "headline_intervention": headline_name,
            "r2_intact": r2_intact,
            "per_intervention": results,
            "intervention_disagreement": disagreement,
        }
        if delta_shuffle_consistency is not None:
            details_out["delta_shuffle_consistency"] = delta_shuffle_consistency

        return ProbeResult(
            probe_name=self.name,
            primary_score=float(headline_delta),
            primary_score_ci=(float(ci_lo), float(ci_hi)),
            verdict=verdict,
            confidence=float(confidence),
            details=details_out,
            config=self.config,
            notes=notes,
        )

    # ── Intervention builders ───────────────────────────────────────

    def _build_override(
        self,
        name: str,
        q_test: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if name == "zero_quantum":
            return np.zeros_like(q_test)
        if name == "random_noise":
            scale = q_test.std(axis=0, keepdims=True) * self.noise_scale
            return rng.normal(loc=0.0, scale=np.broadcast_to(scale, q_test.shape).copy(),
                              size=q_test.shape)
        if name == "permute":
            permuted = q_test.copy()
            rng.shuffle(permuted, axis=0)
            return permuted
        if name == "constant":
            mean_q = q_test.mean(axis=0, keepdims=True)
            return np.broadcast_to(mean_q, q_test.shape).copy()
        raise ValueError(f"Unknown intervention type: {name!r}")
