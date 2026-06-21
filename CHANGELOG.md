# Changelog

All notable changes to HNEP are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/) and
this project adheres to [Semantic Versioning](https://semver.org/).

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
