"""NoiseProbe — measure how QCT verdicts survive noise on the quantum output.

The Sprint 2.5 noise analysis in the thesis injected depolarizing noise
directly into the quantum circuit. The library version takes a
framework-agnostic approach: perturb the *already-extracted* quantum outputs
and measure how predictions plus the surrogation verdict behave. This works
for any adapter (JAX, PyTorch, Qiskit, Precomputed).

Three built-in noise models:

* ``gaussian`` — additive Gaussian noise scaled by each output dim's std.
  Most general and lightweight.
* ``depolarizing_approx`` — shrink each quantum output toward zero by
  ``(1 - p)`` plus a small isotropic Gaussian. Mimics depolarizing channel
  effect on expectation values (``⟨Z⟩ → (1-p)·⟨Z⟩``).
* ``bit_flip`` — sign flip per element with probability ``p``. Useful for
  binary-like outputs near ±1.

You can also supply your own ``Callable[[np.ndarray, float, rng], np.ndarray]``
via ``noise_fn``.

The headline verdict is whether the model's R² and the surrogation verdict
*remain stable* up to a chosen noise level — typically ``0.10`` for
NISQ-realistic depolarizing.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional, Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from hnep.adapters.base import Dataset, ModelInterface
from hnep.probes.base import Probe
from hnep.results.probe_result import ProbeResult
from hnep.thresholds import DEFAULT_THRESHOLDS, Thresholds


NoiseFn = Callable[[np.ndarray, float, np.random.Generator], np.ndarray]


# ── Built-in noise models ────────────────────────────────────────────

def gaussian_noise(q_out: np.ndarray, level: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian noise scaled by each dim's std.

    For per-column std σ and level p:
        q_out_noisy = q_out + N(0, p·σ)
    """
    if level <= 0.0:
        return q_out.copy()
    sigma = q_out.std(axis=0, keepdims=True)
    scale = np.broadcast_to(sigma, q_out.shape) * level
    # Use a tiny floor only where the column-std is exactly zero (constant col)
    safe_scale = np.where(scale > 0, scale, 0.0)
    return q_out + rng.normal(0.0, safe_scale, size=q_out.shape) \
        if safe_scale.any() else q_out.copy()


def depolarizing_approx(q_out: np.ndarray, level: float,
                         rng: np.random.Generator) -> np.ndarray:
    """Approximate the effect of depolarizing noise on Pauli expectation values.

    For depolarizing parameter p ∈ [0, 1]:
        ⟨Z⟩_noisy = (1 - p) · ⟨Z⟩_clean + small isotropic perturbation
    """
    if level <= 0.0:
        return q_out.copy()
    shrunk = q_out * (1.0 - level)
    return shrunk + rng.normal(0.0, level * 0.1, size=q_out.shape)


def bit_flip_noise(q_out: np.ndarray, level: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Sign flip per element with probability ``level``."""
    mask = rng.random(q_out.shape) < level
    out = q_out.copy()
    out[mask] = -out[mask]
    return out


BUILTIN_NOISE: dict[str, NoiseFn] = {
    "gaussian": gaussian_noise,
    "depolarizing_approx": depolarizing_approx,
    "bit_flip": bit_flip_noise,
}


# ── Probe ────────────────────────────────────────────────────────────

class NoiseProbe(Probe):
    """Sweep noise levels on the quantum output and measure QCT stability.

    Parameters
    ----------
    noise_levels
        Sequence of noise probabilities/strengths to sweep. ``0.0`` is always
        prepended as the clean baseline.
    noise_type
        One of ``"gaussian"``, ``"depolarizing_approx"``, ``"bit_flip"``, or
        ``"custom"`` (in which case ``noise_fn`` is used).
    noise_fn
        Custom noise callable ``(q_out, level, rng) -> q_out_noisy``. Only
        used if ``noise_type="custom"``.
    n_bootstrap
        Reserved for future per-level bootstrap CIs. Currently unused but kept
        for API symmetry with other probes.
    thresholds
        QCT thresholds for the noise-time surrogation verdict.
    """

    name = "noise"
    description = "Noise sensitivity of predictions and surrogation verdict."

    def __init__(
        self,
        noise_levels: Sequence[float] = (0.01, 0.05, 0.10),
        noise_type: str = "gaussian",
        noise_fn: Optional[NoiseFn] = None,
        n_bootstrap: int = 0,
        thresholds: Thresholds = DEFAULT_THRESHOLDS,
        seed: int = 42,
    ) -> None:
        super().__init__(seed=seed)

        # Always include the clean baseline at the start
        sorted_levels = sorted({0.0, *(float(l) for l in noise_levels)})
        self.noise_levels = sorted_levels

        if noise_type == "custom":
            if noise_fn is None:
                raise ValueError("noise_type='custom' requires noise_fn=callable")
            self._noise_fn = noise_fn
        else:
            if noise_type not in BUILTIN_NOISE:
                raise ValueError(
                    f"noise_type must be one of {list(BUILTIN_NOISE)} or 'custom'; "
                    f"got {noise_type!r}"
                )
            self._noise_fn = BUILTIN_NOISE[noise_type]

        self.noise_type = noise_type
        self.thresholds = thresholds
        self._config = {
            "seed": seed,
            "noise_levels": sorted_levels,
            "noise_type": noise_type,
            "n_bootstrap": n_bootstrap,
            "ss_threshold": thresholds.ss_replaceable,
        }

    # ── Probe API ───────────────────────────────────────────────────

    def run(
        self,
        model: ModelInterface,
        dataset: Dataset,
        verbose: bool = False,
    ) -> ProbeResult:
        rng = np.random.default_rng(self.seed)
        train_idx = dataset.train_idx
        test_idx = dataset.test_idx
        y_test = dataset.targets[test_idx]

        # Clean quantum outputs — these are the source we'll perturb
        q_train_clean = model.extract_quantum_output(dataset, train_idx)
        q_test_clean = model.extract_quantum_output(dataset, test_idx)

        # Baseline predictions (the model's intact behaviour at p=0)
        y_intact = model.predict(dataset, test_idx)
        r2_intact = float(r2_score(y_test, y_intact))

        # Build an inputs proxy for surrogate fitting — defaults to dataset.inputs
        X_train = dataset.inputs[train_idx]
        X_test = dataset.inputs[test_idx]
        scaler = StandardScaler()
        Xs_train = scaler.fit_transform(X_train)
        Xs_test = scaler.transform(X_test)

        # ── Sweep noise levels ──
        per_level: list[dict] = []
        for level in self.noise_levels:
            if verbose:
                print(f"  [noise] level = {level}")

            # Apply noise to test-set quantum outputs
            if level == 0.0:
                q_test_noisy = q_test_clean.copy()
            else:
                q_test_noisy = self._noise_fn(q_test_clean, level, rng)

            # Run model with the noisy quantum output substituted in
            y_noisy = model.predict_with_quantum_override(
                dataset, q_test_noisy, indices=test_idx
            )
            r2_noisy = float(r2_score(y_test, y_noisy))

            # Mean per-sample drift from intact predictions
            drift = float(np.mean(np.abs(y_noisy - y_intact)))

            # Surrogation on the noisy quantum (Gradient Boosting per dim)
            ss = _surrogation_score(
                Xs_train, q_train_clean,                  # surrogate trained on clean
                Xs_test, q_test_noisy,                    # evaluated against noisy outputs
            )
            verdict = (
                "REPLACEABLE" if ss < self.thresholds.ss_replaceable else "NECESSARY"
            )

            per_level.append({
                "level": float(level),
                "r2": r2_noisy,
                "delta_r2_from_intact": r2_intact - r2_noisy,
                "mean_pred_drift": drift,
                "surrogation_score": float(ss),
                "verdict": verdict,
            })

        # ── Stability summary ──
        clean = per_level[0]
        noisiest = per_level[-1]

        max_r2_drop = max(
            entry["delta_r2_from_intact"] for entry in per_level
        )
        verdict_flips = sum(
            1 for entry in per_level if entry["verdict"] != clean["verdict"]
        )

        if verdict_flips == 0 and max_r2_drop < 0.05:
            stability = "STABLE"
            confidence = 0.9
        elif verdict_flips == 0 and max_r2_drop < 0.10:
            stability = "MOSTLY-STABLE"
            confidence = 0.7
        elif verdict_flips == 0:
            stability = "PERFORMANCE-DEGRADES"  # verdict stays but R² drops
            confidence = 0.5
        else:
            stability = "VERDICT-FLIPS"
            confidence = 0.3

        notes: list[str] = []
        if verdict_flips > 0:
            notes.append(
                f"Surrogation verdict flips at {verdict_flips} of "
                f"{len(per_level)} noise levels."
            )
        if max_r2_drop > 0.10:
            notes.append(
                f"R² drops by up to {max_r2_drop:.3f} at the highest tested "
                "noise level — interpret with care."
            )

        # Primary score: maximum R² drop across the noise sweep (higher → worse)
        return ProbeResult(
            probe_name=self.name,
            primary_score=float(max_r2_drop),
            primary_score_ci=None,
            verdict=stability,
            confidence=float(confidence),
            details={
                "r2_intact": r2_intact,
                "per_level": per_level,
                "max_r2_drop": float(max_r2_drop),
                "verdict_flips": int(verdict_flips),
                "noise_type": self.noise_type,
                "noisiest_level": noisiest["level"],
            },
            config=self.config,
            notes=notes,
        )


# ── Helper — same SS formula as SurrogationProbe ─────────────────────

def _surrogation_score(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> float:
    """Compute SS = 1 − mean R² across output dims, clipped to [0, 1].

    Uses Gradient Boosting per dim — the strongest single surrogate from the
    Phase B exhaustive sweep.
    """
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
