"""HNEPResult — the top-level result of a full HNEP evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from hnep.results.probe_result import ProbeResult


@dataclass
class HNEPResult:
    """Top-level container for a full HNEP evaluation.

    Returned by :func:`hnep.evaluate`. Aggregates the QCT verdict, every
    underlying :class:`ProbeResult`, the cost-utility analysis, and the
    manifest that allows the run to be replayed.
    """

    #: Display name of the model evaluated.
    model_name: str = ""

    #: Display name of the dataset.
    dataset_name: str = ""

    #: Final QCT verdict — one of ``"Genuine"``, ``"Regularizer"``,
    #: ``"Ignored"``, ``"Dead Weight"``, or ``"Inconclusive"``.
    qct_verdict: str = "Inconclusive"

    #: Confidence in the verdict, ``[0, 1]``.
    qct_confidence: float = 0.0

    #: All probe results keyed by probe name.
    probes: Dict[str, ProbeResult] = field(default_factory=dict)

    #: Cost-utility analysis output (populated by Phase 3 code).
    cost_utility: Optional[Dict[str, Any]] = None

    #: Free-form mapping with seeds, library versions, hardware, etc.
    manifest: Dict[str, Any] = field(default_factory=dict)

    #: Human-readable notes (warnings, caveats).
    notes: List[str] = field(default_factory=list)

    #: Optional molecular gallery — list of :class:`MoleculeRecord` (or
    #: equivalent dicts). When present, the HTML report renders top-K and
    #: bottom-K molecules by QCI as RDKit-drawn structures.
    molecule_records: List[Any] = field(default_factory=list)

    # ── Display helpers ─────────────────────────────────────────────

    def summary(self) -> str:
        """Plain-text one-screen summary."""
        lines = [
            "HNEP Evaluation Report",
            "=" * 60,
            f"Model:       {self.model_name}",
            f"Dataset:     {self.dataset_name}",
            "",
            f"QCT Verdict: {self.qct_verdict}  (confidence: {self.qct_confidence:.2f})",
            "-" * 60,
        ]
        for name, result in self.probes.items():
            ci = ""
            if result.primary_score_ci:
                lo, hi = result.primary_score_ci
                ci = f"  [{lo:.3f}, {hi:.3f}]"
            lines.append(
                f"  {name:<14s} score={result.primary_score:.4f}{ci}"
                f"   → {result.verdict}"
            )
        if self.notes:
            lines.append("")
            lines.append("Notes:")
            for n in self.notes:
                lines.append(f"  • {n}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"HNEPResult(model={self.model_name!r}, "
            f"dataset={self.dataset_name!r}, "
            f"verdict={self.qct_verdict!r}, "
            f"confidence={self.qct_confidence:.2f}, "
            f"n_probes={len(self.probes)})"
        )

    # ── Export methods ──────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Fully JSON-serialisable dict representation."""
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "qct_verdict": self.qct_verdict,
            "qct_confidence": float(self.qct_confidence),
            "probes": {k: v.as_dict() for k, v in self.probes.items()},
            "cost_utility": self.cost_utility,
            "manifest": self.manifest,
            "notes": list(self.notes),
        }

    def to_json(self, path: Optional[Union[str, Path]] = None, indent: int = 2) -> str:
        """Serialise to JSON. If ``path`` is given, also write to that file."""
        text = json.dumps(self.to_dict(), indent=indent, default=str)
        if path is not None:
            Path(path).write_text(text)
        return text

    def to_csv(self, path: Union[str, Path]) -> None:
        """Write a flat CSV table of per-probe scores."""
        import csv

        path = Path(path)
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "probe", "primary_score", "ci_lower", "ci_upper",
                "verdict", "confidence",
            ])
            for name, result in self.probes.items():
                ci_lo = ci_hi = ""
                if result.primary_score_ci:
                    ci_lo, ci_hi = result.primary_score_ci
                writer.writerow([
                    name,
                    f"{result.primary_score:.6f}",
                    f"{ci_lo:.6f}" if ci_lo != "" else "",
                    f"{ci_hi:.6f}" if ci_hi != "" else "",
                    result.verdict,
                    f"{result.confidence:.4f}",
                ])

    def to_html(self, path: Union[str, Path], other_results=None) -> str:
        """Render and write the HTML report (returns the HTML string)."""
        from hnep.reports.html import render_html_report
        return render_html_report(self, path=path, other_results=other_results)

    # ── Day-7 exports ───────────────────────────────────────────────

    def to_latex(self, **kwargs) -> str:
        """Render as a self-contained LaTeX ``booktabs`` table.

        See :func:`hnep.exports.to_latex` for keyword options
        (``caption``, ``label``, ``include_caption``).
        """
        from hnep.exports import to_latex
        return to_latex(self, **kwargs)

    def to_markdown(self, include_explanation: bool = True) -> str:
        """Render as a self-contained Markdown report (README-ready)."""
        from hnep.exports import to_markdown
        return to_markdown(self, include_explanation=include_explanation)

    def explain(self) -> str:
        """One-paragraph plain-English explanation of the verdict."""
        from hnep.explain import explain_result
        return explain_result(self)
