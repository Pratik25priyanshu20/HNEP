# Changelog

All notable changes to HNEP are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/) and
this project adheres to [Semantic Versioning](https://semver.org/).

## [0.4.0] — 2026-07-02

Gold-standard API release. Ships the convergent-validity-aware QCT
classifier, adds a seventh benchmark cell (AdversarialConvergent) that
validates it, and reorganises probes into a verdict-driving **core**
(`SurrogationProbe` / `InterventionProbe` / `RepresentationProbe`) and a
supplementary **diagnostic** tier housed under `hnep.diagnostics`. 7-
archetype benchmark under the convergent-validity gate: **100% (70/70)**.
Without the gate: **85.7% (60/70)** — AdversarialConvergent's DISAGREEMENT
verdict is unreachable without CKA + MI. That gap is the empirical case
for the gate.

### Changed — Probes tiered into Core + Diagnostics
- v0.4 tiers the probe API into a validated core
  (`SurrogationProbe`, `InterventionProbe`, `RepresentationProbe`) that
  drives QCT verdicts, and a diagnostic tier
  (`NoiseProbe`, `TemporalProbe`, `ErrorDiversityProbe`) housed under
  `hnep.diagnostics` for supplementary evidence. This reflects a
  considered API design: core probes have been validated against the
  thesis's real-data experiments and the synthetic ground-truth
  benchmark. Diagnostic probes are methodologically sound but returned
  null results (Noise, Temporal) or exposed a readout weakness
  (Error Diversity — Ridge readout unreliable when either branch is
  non-linearly informative) during real-data evaluation. Users may still
  run diagnostics for their own convergent-validity picture, but should
  not use them to override the core QCT verdict.
- `hnep.diagnostics` subpackage created with `NoiseProbe`,
  `TemporalProbe`, `ErrorDiversityProbe`, and the `gaussian_noise` /
  `depolarizing_approx` / `bit_flip_noise` helpers.
- Top-level `from hnep import NoiseProbe` (and the other two) still
  works via a PEP 562 `__getattr__` shim, but emits a
  `DeprecationWarning` pointing at the new location.
- `QCTClassifier` signature stays `(surrogation, intervention,
  representation)` — the classifier cannot accidentally gate on a
  diagnostic probe. The class docstring now explicitly states this.
- Core probe docstrings updated to state "Core probe — verdict-driving";
  diagnostic probe docstrings state "Diagnostic probe — supplementary
  evidence, not verdict-driving" and name their known limitation.

### Added — AdversarialConvergent archetype (7th benchmark cell)
- `hnep.benchmarks.make_adversarial_convergent` constructs a synthetic
  hybrid where SS says NECESSARY and Δ says NOT-LOAD-BEARING (base
  verdict IGNORED) — but CKA(q, y) and MI(q, y) both signal that quantum
  IS target-aligned. Its `expected_verdict = QCTVerdict.DISAGREEMENT`,
  reachable only under `use_convergent_validity=True`.
- Under `use_convergent_validity=False` (default) the archetype scores 0%
  by design — the naïve (SS+Δ) classifier can't emit DISAGREEMENT and
  falls back to the base IGNORED verdict. Under
  `use_convergent_validity=True` all 7 archetypes classify at 100%. That
  contrast is the empirical case for the convergent-validity gate.
- `hnep.evaluate(..., use_convergent_validity=True)` auto-adds
  `RepresentationProbe` to the default probe battery so CKA + MI signals
  are available to the classifier.
- `run_ground_truth_benchmark(use_convergent_validity=True)` threads the
  flag through the runner.

### Added — Convergent-validity gate on the QCT classifier
- `QCTClassifier(use_convergent_validity=True)` counts agreement across
  four signals: SS (surrogation), Δ (intervention), CKA (from the
  representation probe's `quantum_more_aligned_with_target` flag), and MI
  (from `quantum_info_share > 0.5`). The base (SS+Δ) verdict is returned
  when ≥3 of 4 signals agree; otherwise the classifier emits the new
  `QCTVerdict.DISAGREEMENT` verdict.
- `QCTClassifier.classify` accepts an optional `representation: ProbeResult`
  argument. When `use_convergent_validity=False` (the default) or
  `representation is None` or `representation.verdict == "UNAVAILABLE"`,
  the gate is skipped and behavior matches v0.3.0.
- `QCTClassifier.convergent_agreement_min` (default 3) is user-tunable.
- `QCTVerdict.DISAGREEMENT` = "Disagreement" added to the enum.
- `ErrorDiversityProbe` is intentionally NOT a classifier input: see
  the probe's docstring for the readout-failure caveat that keeps it
  diagnostic-only.

### Added — ErrorDiversityProbe diagnostic framing
- Module + class docstring updated to state DIAGNOSTIC ONLY status.
- `details["low_readout_strength"]` flag surfaces when
  `min(R²_q, R²_c) < low_readout_strength_threshold` (default 0.4).
  Below that floor, both Ridge readouts are weak and the residual
  correlation may reflect readout failure rather than true ensemble
  redundancy — any REDUNDANT verdict should be read as Inconclusive.
- New constructor arg `low_readout_strength_threshold`.
- README block explicitly separates ED (diagnostic) from the verdict-
  gating probes (surrogation + intervention).

### Added — Library self-test scripts (NOT thesis reproduction)
- `scripts/locked_v030_results.py` and `scripts/sanity_check_esol.py`
  exercise HNEP end-to-end on the parent thesis repo's precomputed
  extractions (ESOL / FreeSolv / QM9 / Lipo). Both substitute a fresh
  Ridge readout for the trained Flax Hybrid-V1 decoder, so their
  intervention Δ measures "does a from-scratch Ridge benefit from
  quantum features?" — NOT the thesis's "does the trained hybrid depend
  on quantum at inference?". The two questions can flip the sign of Δ,
  and empirically do on 3 of 4 datasets. Report paths renamed to
  `docs/v0.3.0_library_self_test*` and script docstrings carry the
  disclaimer prominently. **Do not read these outputs as thesis
  verdicts.**

## [0.3.0] — 2026-06-30

Methodology hardening release. Replaces inspected thresholds with empirically-
calibrated CI-aware ones, ships a ground-truth synthetic benchmark, adds
cluster-aware block bootstrap with headline-aligned CIs, surfaces per-
evaluation permutation p-values, and documents the SS/Δ asymmetry in
permutation testing. 60-cell verdict accuracy: **100%**; per-seed CI
coverage: **SS 100%, Δ 96%** (Reg/DW/Ig × 15 seeds, n=400 vs n=2000 truth).

### Added — Ground-truth benchmark suite
- `hnep.benchmarks` subpackage with six archetypal synthetic hybrids
  (`make_genuine` / `make_regularizer` / `make_ignored` / `make_dead_weight`
  / `make_inconclusive` / `make_adversarial`) and known QCT verdicts.
- `run_ground_truth_benchmark()` returns a `BenchmarkReport` with overall
  accuracy, per-archetype accuracy, confusion matrix, and misclassified runs.
- Archetype constructors draw all population parameters (W_c, W_q, β_q,
  β_c, RFF basis) BEFORE the sample realisation, so the same `seed`
  produces the same population regardless of `n_samples`. This makes
  per-seed coverage diagnostics — CI@n=400 vs truth@n=2000 — honest.
- Archetypes carry NISQ-style observation noise on the quantum output table
  (σ=0.25) so the SS distribution on linear-q archetypes isn't degenerate;
  Ignored / Dead Weight β_q is small-but-non-zero for the same reason on the
  Δ side. Archetypes ship random cluster IDs (k=10) so the suite exercises
  the block-bootstrap path.

### Added — Empirically-calibrated thresholds
- `Thresholds.from_calibration()` loads thresholds derived from the
  benchmark's known-archetype seeds; `Thresholds.legacy()` keeps the v0.1
  values (SS<0.20, Δ≥0.05) available.
- `DEFAULT_THRESHOLDS` now sources from the calibrated JSON shipped in the
  wheel (`hnep/thresholds_calibration.json`). Current values: SS<0.099,
  Δ≥0.055.
- `scripts/recalibrate_thresholds.py` regenerates the JSON deterministically
  from the **99th percentile of bootstrap-CI upper bounds** on the
  calibration archetypes (Regularizer / Dead Weight for SS, Ignored / Dead
  Weight for Δ). CI-upper-bound calibration is verdict-stable by
  construction: calibration seeds' CIs stay strictly on the
  REPLACEABLE / NOT-LOAD-BEARING side of threshold.

### Added — Cluster (block) bootstrap
- `Dataset.cluster_ids: Optional[np.ndarray]` — per-sample integer cluster
  labels (e.g. molecular scaffold IDs). Defaults to `None` so existing
  adapters keep their v0.2 i.i.d. behaviour.
- `bootstrap_ci()` and `bootstrap_statistic_ci()` accept a new
  `cluster_ids` kwarg. When supplied, the resampling unit is the cluster
  rather than the individual sample — correct CIs on within-cluster-
  correlated data. `cluster_ids=None` is bit-identical to the v0.2
  implementation (same rng → same numerical result).
- `SurrogationProbe` and `InterventionProbe` automatically pass
  `dataset.cluster_ids` through to the bootstrap when available.

### Added — Per-evaluation permutation p-values
- `SurrogationProbe(calibrate=True)` ships a true permutation p-value
  (lower-tail, against the shuffled-q null which makes q independent of
  x → surrogate R² → 0 → SS_perm → 1). Low p ⇒ observed SS is
  significantly below null ⇒ surrogate succeeds with significance.
- `QCTClassifier(use_p_values=True)` gates the REPLACEABLE direction on
  the surrogation p-value: a REGULARIZER / DEAD_WEIGHT verdict requires
  `surrogation.p_value < 0.05`; otherwise the verdict falls back to
  Inconclusive.
- `ProbeResult.p_value: Optional[float]` exposes the value; `None` when
  the probe was constructed without `calibrate=True`.

### Permutation tests — asymmetry between SS and Δ
- `InterventionProbe(calibrate=True)` is deliberately **not** a p-value.
  The per-row-shuffle null doesn't sample "β_q = 0" cleanly — the model's
  trained β_q stays fixed, so a shuffled q gets the decoder to follow
  wrong-but-confident information into wrong predictions, which is
  strictly worse than zero-q for any model that genuinely uses q.
  Δ_perm ≥ Δ_obs reliably and a tail probability would not carry the
  standard reject-H0 semantics. Instead we expose a descriptive
  `delta_shuffle_consistency = mean(Δ_perm) − Δ_obs` in `result.details`
  — positive when shuffled-q is worse than zero-q, i.e. when the model
  depends on q's specific values. The QCT classifier does not consume it.

### Changed — SurrogationProbe bootstrap CI now headline-aligned
- The bootstrap CI now computes the headline statistic
  `1 − mean_d(max(0, R²_d))` per iteration, not the v0.2 closed-form
  proxy `mean(MSE_per_sample) / mean_d(Var(q_d))`. The proxy had a
  per-dim-denominator mismatch with the headline; aligning the two
  removes one bias source for free. Per-seed coverage on n=400 vs n=2000
  truth: SS = 100% / 100% / 100% across Reg / DW / Ig (15 seeds each).

### Chore
- `pyproject.toml` `[tool.pytest.ini_options]` ignores five pre-v0.1
  cruft tests (`test_quantum_circuits.py`, `test_models.py`,
  `test_classical_model.py`, `test_data_loaders.py`, `test_training.py`)
  that import from a parent-project `src.*` module no longer present.
- `scripts/sanity_check_esol.py` exercises HNEP end-to-end on the ESOL
  extraction from the parent thesis repo as a library reproducibility
  self-test. It substitutes a fresh Ridge readout for the trained Flax
  decoder — the resulting intervention Δ is NOT the thesis Δ (opposite
  sign is possible). Script docstring and output carry the disclaimer.

## [0.2.0] — 2026-06-28

Second release. Expands HNEP from two probes to **six**, adds two visualisations
and a molecular-chemistry gallery, ships a verdict explainer, and rounds out the
CLI for shareable summary cards and side-by-side comparison.

### Added — Probes
- `NoiseProbe` — framework-agnostic noise injection on quantum outputs with
  built-in Gaussian, depolarising-approximation, and bit-flip models, plus a
  `noise_fn` hook for custom noise. Verdicts: `STABLE` / `MOSTLY-STABLE` /
  `PERFORMANCE-DEGRADES` / `VERDICT-FLIPS`.
- `TemporalProbe` — single-checkpoint or list-of-checkpoint runs with
  `STABLE` / `TRANSIENT-DRIFT` / `VERDICT-DRIFTS` verdicts; honestly warns
  when only one checkpoint is supplied.
- `ErrorDiversityProbe` — fits Ridge readouts from quantum-only and
  classical-only branches and compares per-molecule errors. Verdicts:
  `DIVERSE` (|r| < 0.5) / `REDUNDANT` (|r| ≥ 0.5) / `UNAVAILABLE`.
- `RepresentationProbe` — CKA + mutual-information bundle. Reports
  quantum↔classical, quantum↔target, classical↔target alignment with
  bootstrap CIs. Verdicts: `QUANTUM-MORE-ALIGNED` /
  `CLASSICAL-MORE-ALIGNED` / `UNAVAILABLE`. Includes `linear_cka()` and
  `cka_with_scalar_target()` as public helpers.

### Added — Visualisations
- `plot_disagreement_heatmap()` — grid of probes × datasets with
  verdict-coded cells, designed to surface convergent-validity disagreements
  (e.g. NECESSARY × REDUNDANT) at a glance. Cell opacity reflects probe
  confidence.
- `plot_activation_atlas()` + `plot_activation_atlas_grid()` — 2-D
  projections of quantum outputs (UMAP → t-SNE → PCA fallback chain) with
  automatic 1-D-collapse warning.

### Added — Molecular Chemistry Gallery
- `hnep.gallery` package with `MoleculeRecord`, `build_gallery()`, and
  `render_gallery_html()`.
- HTML report now renders top-K and bottom-K molecules by QCI as
  RDKit-drawn structures embedded as base64 PNGs. Gracefully degrades
  to SMILES-only when RDKit is not installed.
- `HNEPResult.molecule_records` field for supplying the records.

### Added — HNEPCard + CLI
- `HNEPCard.to_text()` / `to_markdown()` / `to_html()` — single-glance
  summary card per evaluation, suitable for README badges, Slack messages,
  and slide decks.
- `compare_cards_text/markdown/html(results)` — side-by-side multi-model
  comparison tables.
- `load_result_from_json(path)` — round-trip an `HNEPResult` from disk.
- `hnep card <result.json>` — render a card to stdout or a file.
- `hnep compare a.json b.json [...]` — side-by-side comparison with
  `--format {text,markdown,html}` and `-o/--output`.

### Added — Verdict explainer + paper-ready exports
- `explain_result(result)` — deterministic, templated, plain-English
  paragraph that names the evidence behind every QCT verdict. Honestly
  flags low-confidence probes when the verdict is `Inconclusive`.
- `explain_result_html(result)` — same paragraph in a styled HTML block,
  rendered directly under the verdict card in the HTML report.
- `HNEPResult.to_latex()` / `compare_to_latex()` — `booktabs` tables ready
  to drop into a paper as `\input{hnep_table.tex}`. Properly escapes
  `_`/`&`/`%`/`#` in model and dataset names.
- `HNEPResult.to_markdown()` / `compare_to_markdown()` — first-class
  Markdown reports (verdict + explainer blockquote + probe table + notes)
  suitable for README sections, GitHub issues, and model cards.

### Tests
- **170 tests passing** (up from 77 in 0.1.0): +13 Day-1, +13 Day-2,
  +15 Day-3, +17 Day-4, +18 Day-6, +27 Day-7.

### Notes
- All v0.2 additions are additive — the v0.1.0 surface is unchanged.
- Probe thresholds and the cost-utility model remain the v0.1 values; a
  permutation-derived threshold is on the v0.3 roadmap.

## [0.1.0] — 2026-06-21

First public release. The library is feature-complete for the QCT workflow
and is published as an **alpha** — APIs may shift in 0.x as users discover
edge cases. Semver guarantees begin at 1.0.

### Added — Core
- `hnep.evaluate(model, dataset)` — top-level one-line evaluation.
- `HNEPResult` aggregating QCT verdict + per-probe results + manifest, with
  `.to_html()`, `.to_json()`, `.to_csv()` exports.
- `Dataset`, `ModelInterface` — the framework-agnostic adapter contract.

### Added — Probes
- `SurrogationProbe` — 8-surrogate ladder (Ridge, KNN, RF, GB, four MLPs)
  with bootstrap confidence intervals and the most-conservative headline SS.
- `InterventionProbe` — `do()`-operator family: `zero_quantum`,
  `random_noise`, `permute`, `constant`. Headline = zero-quantum drop.
- Both probes report per-sample data, per-sub-probe scores, and a notes
  field for caveats (e.g. surrogate failure modes).

### Added — Classifier
- `QCTClassifier` mapping (surrogation, intervention) → one of
  `Genuine` / `Regularizer` / `Ignored` / `Dead Weight` / `Inconclusive`.
- Inconclusive is emitted when a bootstrap CI straddles a threshold.
- Tunable thresholds via `Thresholds(ss_replaceable, intervention_load_bearing)`.

### Added — Adapters
- `FunctionalAdapter` — wrap any model with three callables.
- `PrecomputedAdapter` — wrap cached arrays + a decoder function.
- `JaxFlaxAdapter` — for Flax models conforming to the thesis surface
  (`forward_with_quantum`, `extract_quantum_outputs`).
- `PyTorchAdapter` — skeleton template subclassed by the user.
- Heavier adapters lazy-import their framework (`hnep[jax]` / `hnep[pytorch]`).

### Added — Cost-utility
- `compute_qus()` returning a `QuantumUtilityScore` (R² gain per compute
  multiplier) with a categorical verdict.
- `pareto_frontier()` filtering dominated points in (accuracy, cost) space.
- `estimate_hardware_cost()` projecting NISQ cost across IBM / IonQ /
  Rigetti / IQM (indicative, user-overridable).

### Added — Visualisations
- `plot_qct_plane()` — the HNEP logo. Four shaded quadrants, threshold
  lines, confidence rectangles per model.
- `plot_convergent_validity_radar()` — N-axis radar (one per probe),
  overlay multiple models.
- `plot_pareto_with_hardware_cost()` — log-x R² vs compute with optional
  NISQ price annotations.

### Added — Reports
- `render_html_report()` — self-contained HTML report with embedded base64
  figures, verdict card colour-coded by QCT class, probe table with
  confidence bars, manifest section.
- `HNEPResult.to_html(path, other_results=...)` for A/B comparisons.

### Added — Tooling
- `hnep` CLI with `--version` and stubs for `evaluate` / `replay`.
- Manifest files (`hnep_version`, `timestamp`, `platform`, `thresholds`,
  probes run, runtime) attached to every result.
- Sphinx documentation site (Furo theme).
- GitHub Actions CI: tests across Python 3.9 – 3.12, lint, build, docs.
- Type-hint marker (`py.typed`).
- pyproject extras: `[jax]`, `[pytorch]`, `[pennylane]`, `[qiskit]`,
  `[molecular]`, `[reports]`, `[all]`, `[dev]`.

### Tests
- 77 tests passing — smoke, probes, classifier, cost-utility, visuals,
  reports, adapters.

### Known limitations
- Probe thresholds are picked by inspection (SS < 0.2, Δ ≥ 0.05). A
  permutation-derived threshold is on the v0.2 roadmap.
- Visualisation styling is not yet customisable.
- `hnep evaluate` and `hnep replay` CLI subcommands are stubs (full
  wire-up lands in v0.2).
- Molecular utilities (RDKit gallery, atom-level QCI) are scheduled for v0.2.
