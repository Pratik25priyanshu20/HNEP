# HNEP — Hybrid Network Evaluation Protocol

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)](#)
[![Tests](https://img.shields.io/badge/tests-77%20passing-brightgreen.svg)](#)
[![Style](https://img.shields.io/badge/style-ruff-orange.svg)](#)

> **Does the quantum component in your hybrid model actually contribute meaningful computation, or could a classical surrogate replace it?**

Most QML benchmarks report end-task accuracy and call it a day.
**HNEP applies multiple independent probes** to your trained hybrid model — surrogation, structural intervention, error diversity, representation analysis — and returns a verdict you can defend.

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

## What's in v0.1.0

* Surrogation probe — 8-surrogate ladder (linear, KNN, trees, MLPs)
* Intervention family — zero, random-noise, permutation, constant
* QCT classifier with bootstrap confidence intervals
* Cost-utility analysis: Pareto frontier + QUS + NISQ hardware cost projection
* Killer figures: QCT plane, convergent validity radar, Pareto plot
* HTML / JSON / CSV reports
* Manifest files for reproducibility
* CLI: `hnep evaluate`, `hnep replay`
* Four adapters (Functional, Precomputed, JAX/Flax, PyTorch skeleton)

## Roadmap

See [`docs/HNEP_LIBRARY_ROADMAP.md`](docs/HNEP_LIBRARY_ROADMAP.md) for the version-by-version plan.

* **v0.2** — molecular chemistry gallery, atom-level QCI, more probes
* **v0.3** — Weights & Biases / Hugging Face / GitHub Action integrations
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
