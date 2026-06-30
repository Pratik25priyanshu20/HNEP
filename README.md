# HNEP — Hybrid Network Evaluation Protocol

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.2.0-orange.svg)](#)
[![Tests](https://img.shields.io/badge/tests-170%20passing-brightgreen.svg)](#)
[![Style](https://img.shields.io/badge/style-ruff-orange.svg)](#)

> **Does the quantum component in your hybrid model actually contribute meaningful computation, or could a classical surrogate replace it?**

Most QML benchmarks report end-task accuracy and call it a day.
**HNEP applies multiple independent probes** to your trained hybrid model — surrogation, structural intervention, representation analysis, plus diagnostic probes (error diversity, noise, temporal) — and returns a verdict you can defend.

---

## Install

```bash
pip install hnep
```

Optional framework extras:

```bash
pip install "hnep[jax]"          # JAX / Flax models
pip install "hnep[pytorch]"      # PyTorch models
pip install "hnep[pennylane]"    # PennyLane quantum backend
pip install "hnep[molecular]"    # RDKit-based molecular utilities
pip install "hnep[reports]"      # interactive HTML reports (jinja2, plotly)
pip install "hnep[all]"          # everything
```

## 30-second quickstart

```python
import hnep

# Wrap your trained model in any of HNEP's adapters
adapter = hnep.FunctionalAdapter(
    name="my_model",
    predict_fn=my_predict,
    extract_quantum_fn=my_extract_quantum,
    predict_with_override_fn=my_predict_override,
    quantum_dim=4,
)

# Run the full HNEP protocol
result = hnep.evaluate(adapter, my_dataset)

print(result.qct_verdict)           # → "Regularizer"
print(result.qct_confidence)        # → 0.94
result.to_html("report.html")       # full HTML report with figures
```

The report opens in your browser as a self-contained HTML file with the **QCT plane**, the **convergent-validity radar**, and a manifest your run can be replayed from.

## What HNEP returns

```text
HNEP Evaluation Report
============================================================
Model:       Hybrid-V1
Dataset:     ESOL (n=1128, scaffold split)

QCT Verdict: Regularizer  (confidence: 0.94)
------------------------------------------------------------
  surrogation    score=0.077  [0.06, 0.10]   → REPLACEABLE
  intervention   score=0.29   [0.22, 0.36]   → LOAD-BEARING
------------------------------------------------------------

Notes:
  • Surrogate swap is safe at inference. Quantum's role appears
    confined to the training phase.
```

## The QCT taxonomy

HNEP classifies your model into one of four roles plus an honest fallback:

|                | LOAD-BEARING        | NOT LOAD-BEARING   |
|----------------|---------------------|--------------------|
| **NECESSARY**  | Genuine Contributor | Unique-but-Ignored |
| **REPLACEABLE**| Architectural Regularizer | Dead Weight  |

When the bootstrap confidence intervals straddle a threshold, HNEP returns `Inconclusive` rather than guessing.

## Adapters

Wrap your model in one line:

| Adapter | Use when |
|---------|----------|
| `FunctionalAdapter` | You can write three Python callbacks. **Most users start here.** |
| `PrecomputedAdapter` | You already cached the test-set extractions to disk. |
| `JaxFlaxAdapter`    | Your model is a Flax module with our standard conventions. |
| `PyTorchAdapter`    | Your model is a `torch.nn.Module` — subclass + override 2 hooks. |

See [`hnep/examples/`](hnep/examples) for runnable examples of each.

## What's in v0.2.0

**Six probes** — each independently answers one question about your hybrid:

| Probe | Question |
|---|---|
| `SurrogationProbe` | Can a classical surrogate reproduce the quantum output? |
| `InterventionProbe` | Does removing the quantum branch hurt performance? |
| `NoiseProbe` | Is the verdict stable under realistic quantum noise? |
| `TemporalProbe` | Does the verdict change across training checkpoints? |
| `ErrorDiversityProbe` *(diagnostic only — see note below)* | Do quantum and classical branches err on the same molecules? |
| `RepresentationProbe` | CKA + mutual information — which embedding is more target-aligned? |

> **`ErrorDiversityProbe` is diagnostic-only.** It does not gate the QCT
> verdict; `QCTClassifier` consumes only `SurrogationProbe` + `InterventionProbe`.
> Both ED readouts are Ridge linear models, so on hybrids whose quantum branch
> encodes non-linear information the residuals can correlate spuriously and
> produce a misleading REDUNDANT verdict. The probe surfaces a
> `low_readout_strength` flag in `details` (fires when `min(R²_q, R²_c) < 0.4`)
> — treat any REDUNDANT verdict under that flag as Inconclusive. A
> non-linear readout ladder is on the roadmap.

**Killer outputs**

* `plot_qct_plane`, `plot_convergent_validity_radar`, `plot_pareto_with_hardware_cost`
* `plot_disagreement_heatmap` — probes × datasets verdict grid
* `plot_activation_atlas` — UMAP/t-SNE/PCA projection of quantum outputs
* **Molecular Chemistry Gallery** — RDKit-rendered top-K / bottom-K molecules by QCI, embedded in the HTML report
* **HNEPCard** — single-glance summary card (text / Markdown / HTML)
* **Verdict explainer** — `result.explain()` returns a deterministic plain-English paragraph naming the evidence behind the verdict
* **Paper-ready exports** — `result.to_latex()` (`booktabs` tables) and `result.to_markdown()` (README-ready)
* HTML / JSON / CSV reports + manifest files for reproducibility

**CLI**

```bash
hnep card result.json --format markdown
hnep compare model_a.json model_b.json model_c.json --format html -o compare.html
```

**Adapters** — Functional, Precomputed, JAX/Flax, PyTorch skeleton (heavy deps lazy-imported).

## Roadmap

See [`docs/HNEP_LIBRARY_ROADMAP.md`](docs/HNEP_LIBRARY_ROADMAP.md) for the version-by-version plan.

* **v0.3** — permutation-derived thresholds, atom-level QCI, Weights & Biases / Hugging Face / GitHub Action integrations
* **v0.4** — `hnep.dev` web sandbox
* **v0.5** — HNEP Doctor (AI-powered recommendations)
* **v1.0** — frozen API, HNEP Arena leaderboard, governance committee

## Documentation

Locally:

```bash
pip install "hnep[dev]"
cd docs_hnep && make html
open _build/html/index.html
```

Or skim the [Sphinx source files](docs_hnep/) directly.

## Citing HNEP

```bibtex
@misc{priyanshu2026hnep,
  title  = {HNEP: Hybrid Network Evaluation Protocol for Quantum Machine Learning},
  author = {Priyanshu, Pratik},
  year   = {2026},
  url    = {https://github.com/Pratik25priyanshu20/HNEP},
}
```

## License

MIT. See [LICENSE](LICENSE).
