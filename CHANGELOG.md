# Changelog

All notable changes to HNEP are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/) and
this project adheres to [Semantic Versioning](https://semver.org/).

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
