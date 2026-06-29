# HNEP v0.2.0 — Release Notes

*Released 28 June 2026*

Second release. HNEP now ships **six probes**, two new visualisations, a
molecular chemistry gallery, a verdict explainer, shareable summary cards,
and paper-ready LaTeX/Markdown exports.

> Install / upgrade: `pip install --upgrade hnep`

---

## Highlights

* **Six probes** — up from two in v0.1. Each independently answers one
  question about your hybrid model; the QCT classifier still consumes the
  surrogation + intervention pair.

  | Probe | Question |
  |---|---|
  | `SurrogationProbe` | Can a classical surrogate reproduce the quantum output? |
  | `InterventionProbe` | Does removing the quantum branch hurt performance? |
  | **`NoiseProbe`** *(new)* | Is the verdict stable under realistic quantum noise? |
  | **`TemporalProbe`** *(new)* | Does the verdict change across training checkpoints? |
  | **`ErrorDiversityProbe`** *(new)* | Do the quantum and classical branches err on the same molecules? |
  | **`RepresentationProbe`** *(new)* | CKA + mutual information — which embedding is more target-aligned? |

* **Verdict explainer** — `result.explain()` returns a deterministic
  plain-English paragraph naming the evidence behind every verdict. The
  HTML report renders it directly under the verdict card.

* **Molecular Chemistry Gallery** — RDKit-rendered top-K and bottom-K
  molecules by QCI embedded in the HTML report (`hnep[molecular]`).

* **HNEPCard + CLI** — a single-glance summary card for slides, READMEs,
  and Slack messages, plus `hnep card` and `hnep compare` commands.

* **Paper-ready exports** — `result.to_latex()` produces a `booktabs`
  table; `result.to_markdown()` produces a README-ready report. Never
  retype a probe score into your paper again.

* **New visualisations** — `plot_disagreement_heatmap()` surfaces
  convergent-validity disagreements at a glance; `plot_activation_atlas()`
  projects quantum outputs to 2-D (UMAP → t-SNE → PCA fallback).

## Backward compatibility

All v0.1 APIs are **unchanged**. Everything in v0.2 is additive.

## Tests

* **176 passing** (up from 77 in v0.1).
* `tests/test_v02_surface.py` pins the public API contract.

## What's next (v0.3)

* Permutation-derived thresholds
* Atom-level QCI
* Integrations: Weights & Biases, Hugging Face, GitHub Action

## Install

```bash
pip install --upgrade hnep
pip install "hnep[molecular]"   # for the Molecular Chemistry Gallery
pip install "hnep[all]"         # for everything
```

## CLI

```bash
hnep card result.json --format markdown
hnep compare model_a.json model_b.json -o compare.html --format html
```

## Acknowledgements

Built as part of the Hybrid Network Evaluation Protocol thesis work at
Imperial College London. See the [paper-grade
storyline](docs/) for the underlying scientific narrative.
