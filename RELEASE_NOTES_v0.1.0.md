# HNEP v0.1.0 — Foundation Release

First public release of HNEP — the Hybrid Network Evaluation Protocol.

```bash
pip install hnep
```

## What is HNEP?

> *Does the quantum component in your hybrid model actually contribute
> meaningful computation, or could a classical surrogate replace it?*

Most QML benchmarks report end-task accuracy and call it a day. HNEP
applies **multiple independent probes** to your trained hybrid model —
surrogation and structural intervention in v0.1.0 — and returns a QCT
verdict (Genuine / Regularizer / Ignored / Dead Weight) with bootstrap
confidence intervals.

## 30-second quickstart

```python
import hnep

adapter = hnep.FunctionalAdapter(
    name="my_model",
    predict_fn=my_predict,
    extract_quantum_fn=my_extract_quantum,
    predict_with_override_fn=my_predict_override,
    quantum_dim=4,
)

result = hnep.evaluate(adapter, my_dataset)
print(result.qct_verdict)        # → "Regularizer"
result.to_html("report.html")    # full report with figures
```

## What's in v0.1.0

| Layer        | What ships |
|--------------|-----------|
| Probes       | Surrogation (8-surrogate ladder) + Intervention family (zero / random / permute / constant) |
| Classifier   | QCT 2×2 with bootstrap confidence intervals + `Inconclusive` fallback |
| Adapters     | `FunctionalAdapter`, `PrecomputedAdapter`, `JaxFlaxAdapter`, `PyTorchAdapter` skeleton |
| Cost-utility | QUS, Pareto frontier, NISQ hardware-cost projection |
| Reports      | HTML (with embedded figures), JSON, CSV |
| Visuals      | QCT plane, convergent-validity radar, Pareto plot |
| CLI          | `hnep` entrypoint |
| Tooling      | Manifest files, py.typed marker, Sphinx docs, GitHub Actions CI |

77 tests passing across Python 3.9 – 3.12.

## Known limitations (will be addressed in v0.2)

- Thresholds are picked by inspection (SS < 0.2, Δ ≥ 0.05) rather than
  derived from a permutation null. A permutation-based threshold lands
  in v0.2.
- The molecular chemistry gallery and atom-level QCI visualisation are
  scheduled for v0.2.
- `hnep evaluate` and `hnep replay` CLI subcommands are stubs.

## Roadmap

See [`docs/HNEP_LIBRARY_ROADMAP.md`](docs/HNEP_LIBRARY_ROADMAP.md):

* **v0.2** — molecular chemistry gallery, atom-level QCI, more probes
* **v0.3** — Weights & Biases / Hugging Face / GitHub Action integrations
* **v0.4** — `hnep.dev` web sandbox
* **v0.5** — HNEP Doctor (AI-powered recommendations)
* **v1.0** — frozen API, leaderboard, governance

## Acknowledgements

HNEP grew out of my Master's thesis on multi-method evaluation for
hybrid quantum-classical models for molecular property prediction. The
methodology paper is in preparation.

If you find HNEP useful please cite it:

```bibtex
@misc{priyanshu2026hnep,
  title  = {HNEP: Hybrid Network Evaluation Protocol for Quantum Machine Learning},
  author = {Priyanshu, Pratik},
  year   = {2026},
  url    = {https://github.com/Pratik25priyanshu20/HNEP},
}
```

— Pratik
