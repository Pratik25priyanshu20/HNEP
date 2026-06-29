"""LaTeX and Markdown exports for HNEPResult.

Both formats are designed for one purpose: never re-type a probe score by
hand into a thesis chapter, paper appendix, README, or model card. Each
function returns a self-contained snippet that can be dropped into the host
document with ``\\input`` (LaTeX) or fenced into a Markdown block.

Design choices
--------------

* LaTeX output uses ``booktabs`` (``\\toprule`` / ``\\midrule``) — the
  prevailing convention for ML papers — but the only required package is
  ``booktabs``. If a user prefers plain ``tabular``, they can post-process.
* All numeric formatting matches the precision used in the rest of HNEP
  (3 decimals for scores, 2 for confidence) for visual consistency.
* The Markdown export reuses the same table structure as the HNEPCard so
  reports and cards do not drift.
"""

from __future__ import annotations

from typing import Sequence

from hnep.results.hnep_result import HNEPResult


# ── small helpers ─────────────────────────────────────────────────────

def _latex_escape(s: str) -> str:
    """Escape the few characters that bite LaTeX. Probe names and verdicts
    use plain identifiers, but model and dataset names can contain
    underscores or ampersands."""
    return (
        s.replace("\\", "\\textbackslash{}")
         .replace("&", r"\&")
         .replace("%", r"\%")
         .replace("$", r"\$")
         .replace("#", r"\#")
         .replace("_", r"\_")
         .replace("{", r"\{")
         .replace("}", r"\}")
         .replace("^", r"\^{}")
         .replace("~", r"\~{}")
    )


def _fmt_ci(probe) -> str:
    if probe.primary_score_ci is None:
        return "--"
    lo, hi = probe.primary_score_ci
    return f"[{lo:+.3f}, {hi:+.3f}]"


# ── LaTeX ─────────────────────────────────────────────────────────────

def to_latex(result: HNEPResult, caption: str | None = None,
             label: str | None = None,
             include_caption: bool = True) -> str:
    """Render a single :class:`HNEPResult` as a LaTeX ``booktabs`` table.

    Drop the output into a paper as ``\\input{hnep_table.tex}`` or wrap it
    with your own ``\\begin{table}`` environment by passing
    ``include_caption=False``.
    """
    model = _latex_escape(result.model_name)
    dataset = _latex_escape(result.dataset_name)
    verdict = _latex_escape(result.qct_verdict)

    rows = []
    for name, p in result.probes.items():
        rows.append(
            f"  {_latex_escape(name)} & "
            f"{p.primary_score:+.3f} & "
            f"{_fmt_ci(p)} & "
            f"{_latex_escape(p.verdict)} & "
            f"{p.confidence:.2f} \\\\"
        )
    rows_str = "\n".join(rows) if rows else "  -- & -- & -- & -- & -- \\\\"

    cap_text = (
        caption
        or f"HNEP evaluation of {model} on {dataset}. "
           f"QCT verdict: \\textbf{{{verdict}}} "
           f"(confidence {result.qct_confidence:.2f})."
    )
    label_str = f"\\label{{{label}}}" if label else ""

    body = (
        "\\begin{tabular}{lrrrr}\n"
        "\\toprule\n"
        "Probe & Score & 95\\% CI & Verdict & Confidence \\\\\n"
        "\\midrule\n"
        f"{rows_str}\n"
        "\\bottomrule\n"
        "\\end{tabular}"
    )

    if not include_caption:
        return body

    return (
        "\\begin{table}[h]\n"
        "  \\centering\n"
        f"  {body}\n"
        f"  \\caption{{{cap_text}}}{label_str}\n"
        "\\end{table}"
    )


def compare_to_latex(results: Sequence[HNEPResult],
                      caption: str | None = None,
                      label: str | None = None,
                      include_caption: bool = True) -> str:
    """Side-by-side LaTeX comparison table for several models.

    Columns: probe name + one column per model. Includes a QCT verdict row
    at the top as a header summary."""
    if not results:
        return "% no results to compare"

    # Union of probe names (in order of first appearance)
    seen: list[str] = []
    for r in results:
        for name in r.probes.keys():
            if name not in seen:
                seen.append(name)

    headers = (
        "Probe & "
        + " & ".join(
            f"{_latex_escape(r.model_name)}"
            for r in results
        )
        + " \\\\"
    )

    verdict_row = (
        "\\textbf{QCT verdict} & "
        + " & ".join(
            f"\\textbf{{{_latex_escape(r.qct_verdict)}}} "
            f"({r.qct_confidence:.2f})"
            for r in results
        )
        + " \\\\"
    )

    rows = []
    for probe in seen:
        cells = [probe]
        for r in results:
            p = r.probes.get(probe)
            if p is None:
                cells.append("--")
            else:
                cells.append(
                    f"{_latex_escape(p.verdict)} "
                    f"({p.primary_score:+.3f})"
                )
        rows.append("  " + " & ".join(_latex_escape(c) if i == 0 else c
                                       for i, c in enumerate(cells)) + " \\\\")

    col_spec = "l" + "l" * len(results)
    body = (
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n"
        f"{headers}\n"
        "\\midrule\n"
        f"{verdict_row}\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}"
    )
    if not include_caption:
        return body

    cap_text = (
        caption
        or f"HNEP comparison across {len(results)} models. "
           "Each cell reports the probe verdict and headline score."
    )
    label_str = f"\\label{{{label}}}" if label else ""
    return (
        "\\begin{table}[h]\n"
        "  \\centering\n"
        f"  {body}\n"
        f"  \\caption{{{cap_text}}}{label_str}\n"
        "\\end{table}"
    )


# ── Markdown ──────────────────────────────────────────────────────────

def to_markdown(result: HNEPResult, include_explanation: bool = True) -> str:
    """Render a single result as a self-contained Markdown report.

    Includes the QCT verdict, the optional verdict explainer, the probe
    table, and any notes. Suitable for README sections, GitHub issue
    comments, and model-card panels.
    """
    out = [
        f"### HNEP Evaluation — `{result.model_name}` on `{result.dataset_name}`",
        "",
        f"**QCT verdict:** **{result.qct_verdict}** "
        f"(confidence {result.qct_confidence:.2f})",
    ]
    if include_explanation:
        from hnep.explain import explain_result
        out.append("")
        out.append("> " + explain_result(result).replace("\n", "\n> "))

    out.extend([
        "",
        "| Probe | Score | 95% CI | Verdict | Confidence |",
        "|---|---|---|---|---|",
    ])
    for name, p in result.probes.items():
        out.append(
            f"| `{name}` | {p.primary_score:+.4f} | {_fmt_ci(p)} | "
            f"{p.verdict} | {p.confidence:.2f} |"
        )
    if result.notes:
        out.append("")
        out.append("**Notes:** " + " · ".join(result.notes))
    return "\n".join(out)


def compare_to_markdown(results: Sequence[HNEPResult]) -> str:
    """Side-by-side Markdown comparison of several results."""
    if not results:
        return "_no results to compare_"

    seen: list[str] = []
    for r in results:
        for name in r.probes.keys():
            if name not in seen:
                seen.append(name)

    header_cells = ["Probe"] + [
        f"`{r.model_name}`<br><sub>{r.dataset_name}</sub>" for r in results
    ]
    lines = [
        "| " + " | ".join(header_cells) + " |",
        "|" + "|".join("---" for _ in header_cells) + "|",
        "| **QCT verdict** | "
        + " | ".join(
            f"**{r.qct_verdict}** ({r.qct_confidence:.2f})"
            for r in results
        )
        + " |",
    ]
    for probe in seen:
        cells = [f"`{probe}`"]
        for r in results:
            p = r.probes.get(probe)
            cells.append("—" if p is None
                         else f"{p.verdict} ({p.primary_score:+.3f})")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
