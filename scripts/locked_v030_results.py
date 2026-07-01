"""Library self-test on the four parent-thesis extractions — NOT thesis reproduction.

============================================================================
WARNING — read before quoting any number this script produces
============================================================================
This script does NOT reproduce the thesis analysis. It measures a different
quantity and returns different verdicts, sometimes with opposite-sign Δ.

- Thesis: interventional Δ on the trained Hybrid-V1 Flax decoder — asks
  "does the *trained* hybrid depend on quantum at inference?"
- This script: interventional Δ on a *fresh* Ridge readout re-fitted on
  ``(quantum, gnn) → Flax_predictions`` — asks "does a from-scratch Ridge
  benefit from quantum features?"

Those two questions have different answers. The Ridge readout is a
convenience surrogate for the Flax decoder (whose weights live in the
parent repo and require its model code to load). It lets us exercise the
HNEP pipeline end-to-end on real molecular data as a library
reproducibility self-test, but its verdicts are NOT canonical for any
scientific claim about the hybrid model. For canonical verdicts on ESOL /
FreeSolv / QM9 / Lipo, run the thesis analysis with the trained decoder.

Output labelled ``docs/v0.3.0_library_self_test*`` for exactly this
reason. Do not rename to imply thesis reproduction.
============================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


_PARENT_EXTRACTIONS = Path(
    "/Users/futurediary/Desktop/vs-Stack/Projects/"
    "Quantum_ML_for _Drug_Discovery/experiments/phase_b/extractions"
)

_DATASETS = [
    ("ESOL", "esol_extraction.npz"),
    ("FreeSolv", "freesol_extraction.npz"),
    ("QM9", "qm9_extraction.npz"),
    ("Lipo", "lipophilicity_extraction.npz"),
]


def _build_adapter_and_dataset(extraction_path: Path, name: str):
    """Build a HNEP adapter + Dataset around the precomputed extraction.

    Substitutes the trained Flax decoder with a Ridge readout fit on
    (quantum, gnn) → predictions. Returns (adapter, dataset, ridge_r2_test)
    so the report can record how faithfully the Ridge approximation tracks
    the original network on the test split.
    """
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from hnep import FunctionalAdapter  # noqa: E402
    from hnep.adapters.base import Dataset  # noqa: E402

    e = np.load(extraction_path)
    q = e["quantum_outputs"].astype(np.float64)
    c = e["gnn_embeddings"].astype(np.float64)
    y = e["targets"].astype(np.float64)
    y_orig = e["predictions"].astype(np.float64)
    train_idx = e["train_idx"].astype(int)
    val_idx = e["val_idx"].astype(int)
    test_idx = e["test_idx"].astype(int)
    descriptors = e["mol_descriptors"].astype(np.float64)
    n = q.shape[0]

    feats = np.hstack([q, c])
    scaler = StandardScaler().fit(feats[train_idx])
    feats_s = scaler.transform(feats)
    ridge = Ridge(alpha=1.0).fit(feats_s[train_idx], y_orig[train_idx])
    y_ridge = ridge.predict(feats_s)
    ridge_r2_test = float(r2_score(y_orig[test_idx], y_ridge[test_idx]))

    coef = ridge.coef_
    intercept = ridge.intercept_

    def _predict_from(q_in, c_in):
        return scaler.transform(np.hstack([q_in, c_in])) @ coef + intercept

    def _resolve(indices):
        return np.arange(n) if indices is None else np.asarray(indices)

    def predict_fn(ds, indices):
        idx = _resolve(indices)
        return _predict_from(q[idx], c[idx])

    def extract_q_fn(ds, indices):
        return q[_resolve(indices)]

    def predict_override_fn(ds, q_override, indices):
        idx = _resolve(indices)
        return _predict_from(q_override, c[idx])

    def classical_emb_fn(ds, indices):
        return c[_resolve(indices)]

    adapter = FunctionalAdapter(
        name=f"{name.lower()}_ridge_surrogate_v030",
        predict_fn=predict_fn,
        extract_quantum_fn=extract_q_fn,
        predict_with_override_fn=predict_override_fn,
        quantum_dim=q.shape[1],
        get_classical_embedding_fn=classical_emb_fn,
    )
    dataset = Dataset(
        inputs=descriptors,
        targets=y,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        metadata={"name": name, "n": n, "q_dim": q.shape[1]},
    )
    return adapter, dataset, ridge_r2_test


def _format_ci(ci, fmt: str = ".4f") -> str:
    if ci is None:
        return "—"
    lo, hi = ci
    return f"[{lo:{fmt}}, {hi:{fmt}}]"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extractions", type=Path, default=_PARENT_EXTRACTIONS)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/v0.3.0_library_self_test.md"),
    )
    parser.add_argument("--results-json", type=Path,
                        default=Path("docs/v0.3.0_library_self_test.json"))
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import hnep  # noqa: E402
    from hnep import evaluate  # noqa: E402
    from hnep.probes.error_diversity import ErrorDiversityProbe  # noqa: E402
    from hnep.probes.intervention import InterventionProbe  # noqa: E402
    from hnep.probes.surrogation import SurrogationProbe  # noqa: E402

    thresholds = hnep.Thresholds.from_calibration()
    rows = []
    json_rows = []

    for label, fname in _DATASETS:
        path = args.extractions / fname
        if not path.is_file():
            print(f"[locked] SKIP {label}: extraction not found at {path}")
            continue
        print(f"[locked] {label}: building adapter from {fname}")
        adapter, dataset, ridge_r2 = _build_adapter_and_dataset(path, label)

        sur = SurrogationProbe(thresholds=thresholds)
        inter = InterventionProbe(thresholds=thresholds)
        ed = ErrorDiversityProbe()
        result = evaluate(adapter, dataset, probes=[sur, inter, ed],
                          thresholds=thresholds)
        s = result.probes["surrogation"]
        i = result.probes["intervention"]
        d = result.probes["error_diversity"]

        rows.append({
            "dataset": label,
            "verdict": result.qct_verdict,
            "ss": s.primary_score,
            "ss_ci": s.primary_score_ci,
            "delta": i.primary_score,
            "delta_ci": i.primary_score_ci,
            "ed_r": d.primary_score,
            "ed_verdict": d.verdict,
            "ed_low_readout": d.details.get("low_readout_strength", False),
            "ed_r2_q": d.details.get("quantum_only_r2"),
            "ed_r2_c": d.details.get("classical_only_r2"),
            "ridge_decoder_r2_test": ridge_r2,
            "n_samples": dataset.targets.shape[0],
            "best_surrogate": s.details.get("best_surrogate"),
        })
        json_rows.append({
            "dataset": label,
            "qct_verdict": result.qct_verdict,
            "surrogation": s.as_dict(),
            "intervention": i.as_dict(),
            "error_diversity": d.as_dict(),
            "ridge_decoder_r2_test": ridge_r2,
        })
        print(
            f"[locked] {label}: verdict={result.qct_verdict}  "
            f"SS={s.primary_score:.4f}  Δ={i.primary_score:.4f}  "
            f"ED r={d.primary_score:.3f} ({d.verdict})"
        )

    # ── Markdown report ──
    lines: list = [
        f"# HNEP v{hnep.__version__} — Library Self-Test (NOT Thesis Reproduction)",
        "",
        f"_Generated {datetime.now(timezone.utc).isoformat()}_",
        "",
        "> **These verdicts are NOT the thesis results.** The script that",
        "> produced them substitutes a fresh Ridge readout for the trained",
        "> Flax hybrid decoder, so the intervention Δ measures a different",
        "> quantity than the thesis storyline (opposite sign on 3 of 4",
        "> datasets). See ``scripts/locked_v030_results.py`` docstring for",
        "> the full disclaimer. Use this report only as a library",
        "> reproducibility self-test.",
        "",
        f"**Thresholds (T1.2 + v0.3.0 CI-aware recalibration):** "
        f"{thresholds.describe()}",
        "",
        "## QCT verdicts (Ridge-surrogate decoder)",
        "",
        "| Dataset | n | QCT verdict | SS | SS CI | Δ | Δ CI |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['n_samples']} | **{r['verdict']}** | "
            f"{r['ss']:.4f} | {_format_ci(r['ss_ci'])} | "
            f"{r['delta']:.4f} | {_format_ci(r['delta_ci'])} |"
        )

    lines += [
        "",
        "## ErrorDiversityProbe (diagnostic only)",
        "",
        "ErrorDiversityProbe is included for context and does NOT gate the QCT "
        "verdict. Both readouts are Ridge linear; the `low_readout_strength` "
        "flag fires when `min(R²_q, R²_c) < 0.4`. Treat any REDUNDANT verdict "
        "under that flag as Inconclusive.",
        "",
        "| Dataset | ED verdict | Pearson r | R²_q (Ridge) | R²_c (Ridge) | low_readout_strength |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        flag = "⚠️ True" if r["ed_low_readout"] else "False"
        r2q = (
            f"{r['ed_r2_q']:.3f}"
            if isinstance(r["ed_r2_q"], float) else str(r["ed_r2_q"])
        )
        r2c = (
            f"{r['ed_r2_c']:.3f}"
            if isinstance(r["ed_r2_c"], float) else str(r["ed_r2_c"])
        )
        lines.append(
            f"| {r['dataset']} | {r['ed_verdict']} | "
            f"{r['ed_r']:.3f} | {r2q} | {r2c} | {flag} |"
        )

    lines += [
        "",
        "## Best surrogate per dataset (from SurrogationProbe)",
        "",
        "| Dataset | Best surrogate |",
        "|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['best_surrogate']} |")

    lines += [
        "",
        "## Caveat — Ridge surrogate decoder",
        "",
        "The trained Flax decoders for the four hybrid models live in the "
        "parent thesis repo's checkpoints and require the parent's model code "
        "to load. This script substitutes the decoder with a Ridge readout fit "
        "on `(quantum, gnn) → predictions` from each precomputed extraction. "
        "Test-set R² of the Ridge against the original Flax predictions:",
        "",
        "| Dataset | Ridge ≈ Flax R² (test) |",
        "|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['ridge_decoder_r2_test']:.4f} |")

    lines += [
        "",
        "The SurrogationProbe SS verdict is **exact** for all four datasets — "
        "it depends only on `(quantum, gnn)`, which we have at full fidelity "
        "from the extractions. The InterventionProbe Δ is the part where the "
        "Ridge approximation matters; borderline Δ verdicts could shift by a "
        "few percent under a full Flax re-evaluation.",
        "",
        "## Reproducing",
        "",
        "```bash",
        "python scripts/locked_v030_results.py",
        "```",
        "",
        f"Raw JSON: `{args.results_json}`",
    ]

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(lines), encoding="utf-8")
    args.results_json.write_text(json.dumps({
        "hnep_version": hnep.__version__,
        "thresholds": {
            "ss_replaceable": thresholds.ss_replaceable,
            "intervention_load_bearing": thresholds.intervention_load_bearing,
        },
        "results": json_rows,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")

    print(f"[locked] report → {args.report_path}")
    print(f"[locked] json   → {args.results_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
