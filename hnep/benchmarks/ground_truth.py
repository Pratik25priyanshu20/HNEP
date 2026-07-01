"""Six archetypal synthetic hybrids with known-correct QCT verdicts.

Each archetype is fully specified by a quantum-output recipe and a decoder
coefficient pattern. Targets are constructed *from* the decoder so the
ground-truth coefficients are exact — there is no fitting noise leaking
into the planted role.

The archetypes:

    Archetype       Quantum recipe                       beta_q        Expected
    --------------- ------------------------------------ ------------- ------------
    Genuine         RFF of x (nonlinear, unsurrogateable) large         Genuine
    Regularizer     Wx (linear projection of x)           large         Regularizer
    Ignored         RFF of independent noise              0             Ignored
    Dead Weight     Wx (linear projection of x)           0             Dead Weight
    Inconclusive    alpha*Wx + (1-alpha)*RFF(x),          large         Inconclusive
                    alpha calibrated so SS straddles
    Adversarial     3 linear dims + 1 sin(omega*w^T x),   hard-dim only Genuine
                    omega calibrated so headline SS
                    lands just above the SS threshold

Adapters expose ``get_classical_embedding`` (a fixed nonlinear featurisation
of x) so all six v0.2 probes activate, not only the two consumed by
``QCTClassifier``.

rng ordering — population then sample
-------------------------------------
Every constructor draws **population parameters** (W_c, b_c, W_q, β_q, β_c,
RFF basis, etc.) from rng BEFORE drawing the **sample realisation** (X,
observation noise, splits, target noise). Since the population draws are
fixed-size, the same seed produces the same population regardless of
``n_samples`` — so ``make_*(seed=s, n=400)`` and ``make_*(seed=s, n=2000)``
sample from the same population. This is what makes the per-seed coverage
diagnostic (CI at n=400 vs truth at n=2000) honest.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from hnep import evaluate
from hnep.adapters.base import Dataset
from hnep.adapters.functional import FunctionalAdapter
from hnep.classifiers.qct import QCTVerdict
from hnep.probes.surrogation import default_surrogate_ladder


_D_INPUT = 8
_Q_DIM = 4
_C_DIM = 16
_NOISE_STD = 0.05
_TRAIN_FRAC = 0.75
_BETA_Q_NOISE_SCALE = 0.05
_DEFAULT_CLUSTER_SIZE = 10
_Q_OBSERVATION_NOISE_STD = 0.25
_RFF_N_FREQ = 128
_RFF_FREQ_SCALE = 4.0


# ── Population-parameter draws (n_samples-independent) ───────────────

def _draw_classical_population(
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    W_c = rng.normal(scale=1.0, size=(_D_INPUT, _C_DIM))
    b_c = rng.uniform(-np.pi, np.pi, size=_C_DIM)
    return W_c, b_c


def _draw_linear_q_population(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(scale=0.5, size=(_D_INPUT, _Q_DIM))


def _draw_rff_q_population(
    rng: np.random.Generator, d_input: int = _D_INPUT
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    W = rng.normal(scale=_RFF_FREQ_SCALE, size=(d_input, _RFF_N_FREQ))
    b = rng.uniform(-np.pi, np.pi, size=_RFF_N_FREQ)
    P = rng.normal(size=(_RFF_N_FREQ, _Q_DIM))
    P, _ = np.linalg.qr(P)
    return W, b, P


# ── Population-parameter application (pure, X-dependent) ─────────────

def _apply_classical_features(
    X: np.ndarray, W_c: np.ndarray, b_c: np.ndarray
) -> np.ndarray:
    return np.sin(X @ W_c + b_c)


def _apply_linear_q(X: np.ndarray, W_q: np.ndarray) -> np.ndarray:
    q = X @ W_q
    return q / (q.std(axis=0, keepdims=True) + 1e-9)


def _apply_rff_q(
    X: np.ndarray, W: np.ndarray, b: np.ndarray, P: np.ndarray
) -> np.ndarray:
    feats = np.cos(X @ W + b)
    q = feats @ P
    return q / (q.std(axis=0, keepdims=True) + 1e-9)


# ── Sample-side draws (n_samples-dependent) ─────────────────────────

def _draw_X(rng: np.random.Generator, n_samples: int) -> np.ndarray:
    return rng.normal(size=(n_samples, _D_INPUT))


def _draw_split_and_clusters(
    rng: np.random.Generator, n_samples: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_train = int(_TRAIN_FRAC * n_samples)
    perm = rng.permutation(n_samples)
    train_idx = np.sort(perm[:n_train])
    test_idx = np.sort(perm[n_train:])
    cluster_ids = _assign_cluster_ids(rng, n_samples)
    return train_idx, test_idx, cluster_ids


def _assign_cluster_ids(
    rng: np.random.Generator, n_samples: int, k: int = _DEFAULT_CLUSTER_SIZE
) -> np.ndarray:
    n_clusters = max(1, n_samples // k)
    perm = rng.permutation(n_samples)
    cluster_ids = np.empty(n_samples, dtype=int)
    for c in range(n_clusters):
        start = c * k
        end = (c + 1) * k if c < n_clusters - 1 else n_samples
        cluster_ids[perm[start:end]] = c
    return cluster_ids


# ── Adapter + dataset packing ────────────────────────────────────────

def _build_adapter(
    name: str,
    q_table: np.ndarray,
    c_table: np.ndarray,
    beta_q: np.ndarray,
    beta_c: np.ndarray,
    intercept: float,
) -> FunctionalAdapter:
    n_samples, q_dim = q_table.shape

    def _resolve(indices: Optional[Sequence[int]]) -> np.ndarray:
        if indices is None:
            return np.arange(n_samples)
        return np.asarray(indices)

    def predict_fn(dataset: Dataset, indices: Optional[Sequence[int]]) -> np.ndarray:
        idx = _resolve(indices)
        return q_table[idx] @ beta_q + c_table[idx] @ beta_c + intercept

    def extract_q(dataset: Dataset, indices: Optional[Sequence[int]]) -> np.ndarray:
        return q_table[_resolve(indices)]

    def predict_override(
        dataset: Dataset, q_override: np.ndarray, indices: Optional[Sequence[int]]
    ) -> np.ndarray:
        idx = _resolve(indices)
        return q_override @ beta_q + c_table[idx] @ beta_c + intercept

    def classical_emb(dataset: Dataset, indices: Optional[Sequence[int]]) -> np.ndarray:
        return c_table[_resolve(indices)]

    return FunctionalAdapter(
        name=name,
        predict_fn=predict_fn,
        extract_quantum_fn=extract_q,
        predict_with_override_fn=predict_override,
        quantum_dim=q_dim,
        get_classical_embedding_fn=classical_emb,
    )


def _pack_dataset(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    name: str,
    cluster_ids: Optional[np.ndarray] = None,
) -> Dataset:
    return Dataset(
        inputs=X,
        targets=y,
        train_idx=train_idx,
        val_idx=np.array([], dtype=int),
        test_idx=test_idx,
        metadata={"name": name},
        cluster_ids=cluster_ids,
    )


def _generate_targets(
    q_table: np.ndarray,
    c_table: np.ndarray,
    beta_q: np.ndarray,
    beta_c: np.ndarray,
    intercept: float,
    rng: np.random.Generator,
) -> np.ndarray:
    y_clean = q_table @ beta_q + c_table @ beta_c + intercept
    return y_clean + _NOISE_STD * rng.normal(size=q_table.shape[0])


# ── Calibration helpers ──────────────────────────────────────────────

def _measure_headline_ss_fast(
    q_train: np.ndarray,
    q_test: np.ndarray,
    Xs_train: np.ndarray,
    Xs_test: np.ndarray,
) -> float:
    """Min SS across the 8-surrogate ladder, no bootstrap CI.

    Matches SurrogationProbe's headline rule (best surrogate wins) but skips
    the per-sample-resample step. Used inside calibration loops where we need
    fast headline SS estimates.
    """
    ladder = default_surrogate_ladder()
    best_ss = 1.0
    for spec in ladder:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                per_dim_r2 = []
                for d in range(q_test.shape[1]):
                    m = spec.build()
                    m.fit(Xs_train, q_train[:, d])
                    pred = m.predict(Xs_test)
                    per_dim_r2.append(max(0.0, float(r2_score(q_test[:, d], pred))))
                ss = 1.0 - float(np.mean(per_dim_r2))
                if ss < best_ss:
                    best_ss = ss
        except Exception:
            continue
    return float(max(0.0, min(1.0, best_ss)))


def _calibrate_inconclusive_alpha(
    q_lin_signal: np.ndarray,
    q_rff_signal: np.ndarray,
    obs_noise: np.ndarray,
    X: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    target_ss: Optional[float] = None,
    tol: float = 0.005,
    max_iter: int = 14,
) -> float:
    """Bisect alpha for q = alpha*lin + (1-alpha)*rff + noise so SS≈target.

    ``target_ss=None`` reads ``DEFAULT_THRESHOLDS.ss_replaceable`` at call time
    so the Inconclusive archetype auto-tracks recalibrated thresholds.
    Returns the final alpha (q_table is rebuilt at the call site).
    """
    if target_ss is None:
        from hnep.thresholds import DEFAULT_THRESHOLDS

        target_ss = DEFAULT_THRESHOLDS.ss_replaceable

    scaler = StandardScaler().fit(X[train_idx])
    Xs_train = scaler.transform(X[train_idx])
    Xs_test = scaler.transform(X[test_idx])

    lo, hi = 0.0, 1.0
    alpha = 0.5
    for _ in range(max_iter):
        alpha = 0.5 * (lo + hi)
        q_full = alpha * q_lin_signal + (1.0 - alpha) * q_rff_signal + obs_noise
        ss = _measure_headline_ss_fast(
            q_full[train_idx], q_full[test_idx], Xs_train, Xs_test
        )
        if abs(ss - target_ss) < tol:
            return alpha
        if ss > target_ss:
            lo = alpha
        else:
            hi = alpha
    return alpha


def _calibrate_adversarial_omega(
    lin_dims: np.ndarray,
    proj: np.ndarray,
    obs_noise: np.ndarray,
    X: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    target_ss: Optional[float] = None,
    tol: float = 0.03,
    max_iter: int = 8,
) -> float:
    if target_ss is None:
        from hnep.thresholds import DEFAULT_THRESHOLDS

        target_ss = DEFAULT_THRESHOLDS.ss_replaceable + 0.15

    scaler = StandardScaler().fit(X[train_idx])
    Xs_train = scaler.transform(X[train_idx])
    Xs_test = scaler.transform(X[test_idx])

    lo_omega, hi_omega = 1.0, 40.0
    omega = 20.0
    for _ in range(max_iter):
        omega = 0.5 * (lo_omega + hi_omega)
        hard_dim = np.sin(omega * proj).reshape(-1, 1)
        hard_dim = hard_dim / (hard_dim.std(axis=0, keepdims=True) + 1e-9)
        q_full = np.hstack([lin_dims, hard_dim]) + obs_noise
        ss = _measure_headline_ss_fast(
            q_full[train_idx], q_full[test_idx], Xs_train, Xs_test
        )
        if abs(ss - target_ss) < tol:
            return omega
        if ss > target_ss:
            hi_omega = omega
        else:
            lo_omega = omega
    return omega


# ── Archetype constructors ───────────────────────────────────────────

def make_genuine(
    seed: int = 0, n_samples: int = 400
) -> Tuple[FunctionalAdapter, Dataset, QCTVerdict]:
    """Nonlinear-of-input quantum + decoder uses it heavily → Genuine."""
    rng = np.random.default_rng(seed)

    # Population
    W_c, b_c = _draw_classical_population(rng)
    W_rff, b_rff, P_rff = _draw_rff_q_population(rng)
    beta_q = rng.normal(scale=1.0, size=_Q_DIM)
    beta_c = rng.normal(scale=0.3, size=_C_DIM)

    # Sample
    X = _draw_X(rng, n_samples)
    c_table = _apply_classical_features(X, W_c, b_c)
    q_signal = _apply_rff_q(X, W_rff, b_rff, P_rff)
    obs_noise = rng.normal(scale=_Q_OBSERVATION_NOISE_STD, size=q_signal.shape)
    q_table = q_signal + obs_noise
    train_idx, test_idx, cluster_ids = _draw_split_and_clusters(rng, n_samples)
    y = _generate_targets(q_table, c_table, beta_q, beta_c, 0.0, rng)

    adapter = _build_adapter(
        "benchmark_genuine", q_table, c_table, beta_q, beta_c, 0.0
    )
    dataset = _pack_dataset(
        X, y, train_idx, test_idx, "genuine_bench", cluster_ids=cluster_ids
    )
    return adapter, dataset, QCTVerdict.GENUINE


def make_regularizer(
    seed: int = 0, n_samples: int = 400
) -> Tuple[FunctionalAdapter, Dataset, QCTVerdict]:
    """Linear-of-input quantum (easily mimicked) + decoder uses it heavily →
    Regularizer."""
    rng = np.random.default_rng(seed)

    W_c, b_c = _draw_classical_population(rng)
    W_q = _draw_linear_q_population(rng)
    beta_q = rng.normal(scale=1.0, size=_Q_DIM)
    beta_c = rng.normal(scale=0.3, size=_C_DIM)

    X = _draw_X(rng, n_samples)
    c_table = _apply_classical_features(X, W_c, b_c)
    q_signal = _apply_linear_q(X, W_q)
    obs_noise = rng.normal(scale=_Q_OBSERVATION_NOISE_STD, size=q_signal.shape)
    q_table = q_signal + obs_noise
    train_idx, test_idx, cluster_ids = _draw_split_and_clusters(rng, n_samples)
    y = _generate_targets(q_table, c_table, beta_q, beta_c, 0.0, rng)

    adapter = _build_adapter(
        "benchmark_regularizer", q_table, c_table, beta_q, beta_c, 0.0
    )
    dataset = _pack_dataset(
        X, y, train_idx, test_idx, "regularizer_bench", cluster_ids=cluster_ids
    )
    return adapter, dataset, QCTVerdict.REGULARIZER


def make_ignored(
    seed: int = 0, n_samples: int = 400
) -> Tuple[FunctionalAdapter, Dataset, QCTVerdict]:
    """RFF of independent noise (unsurrogateable from x) + decoder essentially
    ignores it → Ignored.

    β_q is small but non-zero — models the residual gradient flow into a
    structurally ignored branch and gives a non-degenerate Δ distribution
    for T1.2's percentile-based threshold calibration.
    """
    rng = np.random.default_rng(seed)

    W_c, b_c = _draw_classical_population(rng)
    W_rff, b_rff, P_rff = _draw_rff_q_population(rng)
    beta_q = rng.normal(scale=_BETA_Q_NOISE_SCALE, size=_Q_DIM)
    beta_c = rng.normal(scale=0.5, size=_C_DIM)

    X = _draw_X(rng, n_samples)
    c_table = _apply_classical_features(X, W_c, b_c)
    # Independent noise input to the RFF — the "quantum" branch sees its own
    # noise, not x. The noise table is sample-side so n_samples-dependent;
    # the RFF basis is population.
    noise_input = rng.normal(size=(n_samples, _D_INPUT))
    q_signal = _apply_rff_q(noise_input, W_rff, b_rff, P_rff)
    obs_noise = rng.normal(scale=_Q_OBSERVATION_NOISE_STD, size=q_signal.shape)
    q_table = q_signal + obs_noise
    train_idx, test_idx, cluster_ids = _draw_split_and_clusters(rng, n_samples)
    y = _generate_targets(q_table, c_table, beta_q, beta_c, 0.0, rng)

    adapter = _build_adapter(
        "benchmark_ignored", q_table, c_table, beta_q, beta_c, 0.0
    )
    dataset = _pack_dataset(
        X, y, train_idx, test_idx, "ignored_bench", cluster_ids=cluster_ids
    )
    return adapter, dataset, QCTVerdict.IGNORED


def make_dead_weight(
    seed: int = 0, n_samples: int = 400
) -> Tuple[FunctionalAdapter, Dataset, QCTVerdict]:
    """Linear-of-input quantum + decoder essentially ignores it → Dead Weight.

    β_q is small but non-zero (same rationale as ``make_ignored``).
    """
    rng = np.random.default_rng(seed)

    W_c, b_c = _draw_classical_population(rng)
    W_q = _draw_linear_q_population(rng)
    beta_q = rng.normal(scale=_BETA_Q_NOISE_SCALE, size=_Q_DIM)
    beta_c = rng.normal(scale=0.5, size=_C_DIM)

    X = _draw_X(rng, n_samples)
    c_table = _apply_classical_features(X, W_c, b_c)
    q_signal = _apply_linear_q(X, W_q)
    obs_noise = rng.normal(scale=_Q_OBSERVATION_NOISE_STD, size=q_signal.shape)
    q_table = q_signal + obs_noise
    train_idx, test_idx, cluster_ids = _draw_split_and_clusters(rng, n_samples)
    y = _generate_targets(q_table, c_table, beta_q, beta_c, 0.0, rng)

    adapter = _build_adapter(
        "benchmark_dead_weight", q_table, c_table, beta_q, beta_c, 0.0
    )
    dataset = _pack_dataset(
        X, y, train_idx, test_idx, "dead_weight_bench", cluster_ids=cluster_ids
    )
    return adapter, dataset, QCTVerdict.DEAD_WEIGHT


def make_inconclusive(
    seed: int = 0, n_samples: int = 400
) -> Tuple[FunctionalAdapter, Dataset, QCTVerdict]:
    """alpha-calibrated linear/RFF mix → SS bootstrap CI straddles threshold.

    Note: alpha is bisected on the sample at call time, so it varies slightly
    with n_samples. The basis (W_q, W_rff, etc.) is population-fixed.
    """
    rng = np.random.default_rng(seed)

    W_c, b_c = _draw_classical_population(rng)
    W_q_lin = _draw_linear_q_population(rng)
    W_rff, b_rff, P_rff = _draw_rff_q_population(rng)
    beta_q = rng.normal(scale=1.0, size=_Q_DIM)
    beta_c = rng.normal(scale=0.3, size=_C_DIM)

    X = _draw_X(rng, n_samples)
    c_table = _apply_classical_features(X, W_c, b_c)
    q_lin_signal = _apply_linear_q(X, W_q_lin)
    q_rff_signal = _apply_rff_q(X, W_rff, b_rff, P_rff)
    obs_noise = rng.normal(scale=_Q_OBSERVATION_NOISE_STD, size=q_lin_signal.shape)
    train_idx, test_idx, cluster_ids = _draw_split_and_clusters(rng, n_samples)

    alpha = _calibrate_inconclusive_alpha(
        q_lin_signal, q_rff_signal, obs_noise, X, train_idx, test_idx
    )
    q_table = alpha * q_lin_signal + (1.0 - alpha) * q_rff_signal + obs_noise
    y = _generate_targets(q_table, c_table, beta_q, beta_c, 0.0, rng)

    adapter = _build_adapter(
        "benchmark_inconclusive", q_table, c_table, beta_q, beta_c, 0.0
    )
    dataset = _pack_dataset(
        X, y, train_idx, test_idx, "inconclusive_bench", cluster_ids=cluster_ids
    )
    return adapter, dataset, QCTVerdict.INCONCLUSIVE


def make_adversarial_convergent(
    seed: int = 0, n_samples: int = 400
) -> Tuple[FunctionalAdapter, Dataset, QCTVerdict]:
    """Base (SS+Δ) verdict says IGNORED but CKA + MI both say quantum is
    target-aligned — convergent-validity gate emits DISAGREEMENT.

    Construction:
      * Two hidden signals z₁, z₂ = high-frequency sines along two random
        input directions (unsurrogateable from x).
      * q = [z₁, cos(ω·w₁ᵀx), z₂, cos(ω·w₂ᵀx)] — its 4-dim linear span
        covers both z₁ and z₂ completely.
      * y = z₁ + z₂ + noise — target needs both components.
      * Classical embedding c has z₁ injected as its first feature, so
        classical Ridge can recover HALF of y (β_c = [1, tiny, ...]).
      * Decoder β_q is drawn near zero — the trained decoder IGNORES q
        even though q carries the missing half of the target.

    Verdict paths:
      * SS: q is high-frequency sines → surrogates fail → NECESSARY.
      * Δ: β_q ≈ 0 → zeroing q barely changes predictions → NOT-LOAD-BEARING.
        Base = IGNORED.
      * CKA(q, y) ≈ 1 (q's linear span covers z₁+z₂); CKA(c, y) ≈ 0.5
        (only z₁). Quantum-more-aligned.
      * MI(q, y) > MI(c, y) → quantum info share > 0.5.

    Under ``use_convergent_validity=False`` the classifier returns IGNORED
    (wrong — quantum was informative, just wasted). Under
    ``use_convergent_validity=True`` votes split 2 (SS+Δ) vs 2 (CKA+MI)
    and the classifier emits DISAGREEMENT.
    """
    rng = np.random.default_rng(seed)

    W_c, b_c = _draw_classical_population(rng)
    w1 = rng.normal(scale=1.0, size=_D_INPUT)
    w2 = rng.normal(scale=1.0, size=_D_INPUT)
    omega_hard = 15.0
    # β_c = [1, small noise ...] — decoder captures z₁ via c[0] but misses z₂.
    beta_c = np.concatenate([[1.0], rng.normal(scale=0.05, size=_C_DIM - 1)])
    # β_q = 0 exactly (not drawn from the noise scale used by IGNORED/DEAD_WEIGHT).
    # Rationale: AdvConv doesn't feed threshold calibration (Ignored/DW do), so
    # a degenerate Δ = 0 is safe here — and it stops random β_q draws from
    # pushing Δ across the load-bearing threshold with a wide-CI Inconclusive.
    beta_q = np.zeros(_Q_DIM)

    X = _draw_X(rng, n_samples)
    proj1 = X @ w1
    proj2 = X @ w2
    z1 = np.sin(omega_hard * proj1)
    z2 = np.sin(omega_hard * proj2)

    c_table = _apply_classical_features(X, W_c, b_c).copy()
    c_table[:, 0] = z1  # inject z₁ as the first classical feature

    # Each q dim carries a distinct sign-varied combination of z₁ and z₂ so
    # per-dim MI(q_i, y) is high (each is a near-y-clean signal), giving
    # quantum_info_share > 0.5 despite the 16-dim vs 4-dim asymmetry.
    q_signal = np.stack(
        [z1 + z2, z1 - z2, 2.0 * z1 + z2, z1 + 2.0 * z2], axis=1
    )
    q_signal = q_signal / (q_signal.std(axis=0, keepdims=True) + 1e-9)
    obs_noise = rng.normal(scale=_Q_OBSERVATION_NOISE_STD, size=q_signal.shape)
    q_table = q_signal + obs_noise
    train_idx, test_idx, cluster_ids = _draw_split_and_clusters(rng, n_samples)

    y_clean = z1 + z2
    y = y_clean + _NOISE_STD * rng.normal(size=n_samples)

    adapter = _build_adapter(
        "benchmark_adversarial_convergent",
        q_table, c_table, beta_q, beta_c, 0.0,
    )
    dataset = _pack_dataset(
        X, y, train_idx, test_idx, "adversarial_convergent_bench",
        cluster_ids=cluster_ids,
    )
    return adapter, dataset, QCTVerdict.DISAGREEMENT


def make_adversarial(
    seed: int = 0, n_samples: int = 400
) -> Tuple[FunctionalAdapter, Dataset, QCTVerdict]:
    """3 easy linear dims + 1 high-frequency sin dim; decoder reads the hard
    dim heavily. Tests whether the 8-surrogate-min headline + per-dim-mean
    scoring keeps SS above the threshold against a partial-mimicry attack.
    Expected: Genuine.
    """
    rng = np.random.default_rng(seed)

    W_c, b_c = _draw_classical_population(rng)
    W_lin_3 = rng.normal(scale=0.5, size=(_D_INPUT, 3))
    w_hard = rng.normal(scale=1.0, size=_D_INPUT)
    beta_q = np.array([0.1, 0.1, 0.1, 1.5])
    beta_c = rng.normal(scale=0.3, size=_C_DIM)

    X = _draw_X(rng, n_samples)
    c_table = _apply_classical_features(X, W_c, b_c)
    lin_dims = X @ W_lin_3
    lin_dims = lin_dims / (lin_dims.std(axis=0, keepdims=True) + 1e-9)
    proj = X @ w_hard
    obs_noise = rng.normal(scale=_Q_OBSERVATION_NOISE_STD, size=(n_samples, _Q_DIM))
    train_idx, test_idx, cluster_ids = _draw_split_and_clusters(rng, n_samples)

    omega = _calibrate_adversarial_omega(
        lin_dims, proj, obs_noise, X, train_idx, test_idx
    )
    hard_dim = np.sin(omega * proj).reshape(-1, 1)
    hard_dim = hard_dim / (hard_dim.std(axis=0, keepdims=True) + 1e-9)
    q_signal = np.hstack([lin_dims, hard_dim])
    q_table = q_signal + obs_noise
    y = _generate_targets(q_table, c_table, beta_q, beta_c, 0.0, rng)

    adapter = _build_adapter(
        "benchmark_adversarial", q_table, c_table, beta_q, beta_c, 0.0
    )
    dataset = _pack_dataset(
        X, y, train_idx, test_idx, "adversarial_bench", cluster_ids=cluster_ids
    )
    return adapter, dataset, QCTVerdict.GENUINE


# ── Runner + report ──────────────────────────────────────────────────

_ARCHETYPE_FACTORIES: Dict[str, Callable[..., Tuple[FunctionalAdapter, Dataset, QCTVerdict]]] = {
    "genuine": make_genuine,
    "regularizer": make_regularizer,
    "ignored": make_ignored,
    "dead_weight": make_dead_weight,
    "inconclusive": make_inconclusive,
    "adversarial": make_adversarial,
    "adversarial_convergent": make_adversarial_convergent,
}


@dataclass
class BenchmarkReport:
    """Aggregated results from :func:`run_ground_truth_benchmark`.

    Attributes
    ----------
    overall_accuracy
        Fraction of (archetype, seed) runs whose predicted verdict matches the
        archetype's ground truth.
    per_archetype_accuracy
        ``{archetype_name: accuracy}``.
    confusion
        ``{expected_verdict: {predicted_verdict: count}}``. Rows are ground
        truth, columns are HNEP's predictions.
    misclassified
        One record per failing run: ``{archetype, seed, expected, predicted,
        ss, delta}``.
    per_seed
        All runs as a list (same record shape as ``misclassified``) if
        ``return_per_seed=True`` was passed, else ``None``.
    """

    overall_accuracy: float
    per_archetype_accuracy: Dict[str, float]
    confusion: Dict[str, Dict[str, int]]
    misclassified: List[Dict[str, Any]]
    per_seed: Optional[List[Dict[str, Any]]] = None

    def summary(self) -> str:
        verdicts = [v.value for v in QCTVerdict]
        col_w = max(len(v) for v in verdicts) + 1

        lines: List[str] = [
            f"HNEP Ground-Truth Benchmark — overall accuracy: {self.overall_accuracy:.1%}",
            "",
            "Per-archetype accuracy:",
        ]
        for arch in sorted(self.per_archetype_accuracy):
            lines.append(f"  {arch:<14s}  {self.per_archetype_accuracy[arch]:.1%}")

        lines.append("")
        lines.append("Confusion matrix (rows = expected, cols = predicted):")
        header = " " * 16 + "".join(f"{v:>{col_w}s}" for v in verdicts)
        lines.append(header)
        for expected in verdicts:
            row = self.confusion.get(expected, {})
            counts = "".join(f"{row.get(p, 0):>{col_w}d}" for p in verdicts)
            lines.append(f"  {expected:<14s}{counts}")

        if self.misclassified:
            lines.append("")
            lines.append(f"Misclassified ({len(self.misclassified)} cases):")
            for m in self.misclassified:
                lines.append(
                    f"  {m['archetype']:<14s} seed={m['seed']:<3d} "
                    f"expected={m['expected']:<14s} got={m['predicted']:<14s} "
                    f"(SS={m['ss']:.3f}, Δ={m['delta']:.3f})"
                )
        return "\n".join(lines)


def run_ground_truth_benchmark(
    n_seeds: int = 10,
    n_samples: int = 400,
    archetypes: Optional[Sequence[str]] = None,
    return_per_seed: bool = False,
    verbose: bool = False,
    use_convergent_validity: bool = False,
) -> BenchmarkReport:
    """Run the full archetype × seed grid and aggregate verdicts.

    Parameters
    ----------
    n_seeds
        Number of seeds per archetype.
    n_samples
        Samples per synthetic dataset.
    archetypes
        Subset of archetype names to run. ``None`` runs all seven
        (six standard + ``adversarial_convergent``).
    return_per_seed
        If ``True``, the returned report includes every run in
        ``per_seed``. Off by default to keep reports small.
    verbose
        Print per-run progress to stdout.
    use_convergent_validity
        When ``True``, ``evaluate`` runs :class:`RepresentationProbe` and the
        QCT classifier consumes it as two additional votes (CKA + MI). This
        is required for the ``adversarial_convergent`` archetype to classify
        correctly (expected ``DISAGREEMENT`` — unreachable without the gate).
    """
    if archetypes is None:
        archetypes = list(_ARCHETYPE_FACTORIES.keys())
    unknown = [a for a in archetypes if a not in _ARCHETYPE_FACTORIES]
    if unknown:
        raise ValueError(f"Unknown archetype(s): {unknown}")

    verdict_values = [v.value for v in QCTVerdict]
    confusion: Dict[str, Dict[str, int]] = {
        v: {p: 0 for p in verdict_values} for v in verdict_values
    }
    per_seed: List[Dict[str, Any]] = []
    misclassified: List[Dict[str, Any]] = []
    per_arch_correct: Dict[str, int] = {a: 0 for a in archetypes}
    per_arch_total: Dict[str, int] = {a: 0 for a in archetypes}

    for arch in archetypes:
        factory = _ARCHETYPE_FACTORIES[arch]
        for seed in range(n_seeds):
            if verbose:
                print(f"[bench] {arch} seed={seed} …")
            adapter, dataset, expected = factory(seed=seed, n_samples=n_samples)
            result = evaluate(
                adapter,
                dataset,
                verbose=False,
                use_convergent_validity=use_convergent_validity,
            )
            predicted = result.qct_verdict
            sur = result.probes["surrogation"]
            inter = result.probes["intervention"]
            record = {
                "archetype": arch,
                "seed": seed,
                "expected": expected.value,
                "predicted": predicted,
                "ss": float(sur.primary_score),
                "delta": float(inter.primary_score),
            }
            per_seed.append(record)
            confusion[expected.value][predicted] = (
                confusion[expected.value].get(predicted, 0) + 1
            )
            per_arch_total[arch] += 1
            if predicted == expected.value:
                per_arch_correct[arch] += 1
            else:
                misclassified.append(record)

    total = sum(per_arch_total.values())
    correct = sum(per_arch_correct.values())
    per_arch_acc = {
        a: (per_arch_correct[a] / per_arch_total[a]) if per_arch_total[a] else 0.0
        for a in archetypes
    }

    return BenchmarkReport(
        overall_accuracy=correct / total if total else 0.0,
        per_archetype_accuracy=per_arch_acc,
        confusion=confusion,
        misclassified=misclassified,
        per_seed=per_seed if return_per_seed else None,
    )
