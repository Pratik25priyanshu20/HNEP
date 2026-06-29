"""Day-7 tests: LaTeX + Markdown exports + verdict explainer."""

from __future__ import annotations

import pytest

import hnep
from hnep.explain import explain_result, explain_result_html
from hnep.exports import (
    compare_to_latex,
    compare_to_markdown,
    to_latex,
    to_markdown,
)
from hnep.results.hnep_result import HNEPResult
from hnep.results.probe_result import ProbeResult


# ── fixtures ──────────────────────────────────────────────────────────

def _result(
    *, verdict="Regularizer",
    ss=0.12, ss_ci=(0.08, 0.16),
    delta=0.08, delta_ci=None,
    extras: dict | None = None,
    model="HybridV1", dataset="ESOL",
    confidence=0.78,
) -> HNEPResult:
    probes = {
        "surrogation": ProbeResult(
            probe_name="surrogation",
            primary_score=ss, primary_score_ci=ss_ci,
            verdict=("REPLACEABLE" if ss < 0.2 else "NECESSARY"),
            confidence=0.85,
        ),
        "intervention": ProbeResult(
            probe_name="intervention",
            primary_score=delta, primary_score_ci=delta_ci,
            verdict=("LOAD-BEARING" if delta >= 0.05 else "NOT-LOAD-BEARING"),
            confidence=0.7,
        ),
    }
    if extras:
        probes.update(extras)
    return HNEPResult(
        model_name=model, dataset_name=dataset,
        qct_verdict=verdict, qct_confidence=confidence,
        probes=probes,
        notes=["session-2 run"],
    )


# ── Verdict explainer ────────────────────────────────────────────────

def test_explain_includes_verdict_and_confidence():
    out = explain_result(_result(verdict="Genuine"))
    assert "**Verdict: Genuine.**" in out
    assert "confidence" in out


def test_explain_mentions_surrogation_and_intervention_with_numbers():
    out = explain_result(_result(ss=0.12, delta=0.08))
    assert "Surrogation" in out
    assert "0.120" in out
    assert "Intervention" in out
    # Δ is formatted with a sign
    assert "+0.080" in out


def test_explain_chooses_replaceable_branch_when_below_threshold():
    out = explain_result(_result(ss=0.05))
    assert "reproducible" in out
    assert "below" in out


def test_explain_chooses_unique_branch_when_above_threshold():
    out = explain_result(_result(ss=0.4))
    assert "cannot be reproduced" in out
    assert "above" in out


def test_explain_chooses_load_bearing_branch_for_large_delta():
    out = explain_result(_result(delta=0.20))
    assert "load-bearing" in out
    assert "ablating" in out.lower()


def test_explain_chooses_not_load_bearing_branch_for_small_delta():
    out = explain_result(_result(delta=0.01))
    assert "not load-bearing" in out
    assert "under" in out


def test_explain_adds_secondary_probes_when_present():
    extras = {
        "error_diversity": ProbeResult(
            probe_name="error_diversity",
            primary_score=0.20, verdict="DIVERSE", confidence=0.6,
        ),
        "representation": ProbeResult(
            probe_name="representation",
            primary_score=0.35, verdict="QUANTUM-MORE-ALIGNED", confidence=0.6,
        ),
        "noise": ProbeResult(
            probe_name="noise", primary_score=0.95,
            verdict="STABLE", confidence=0.9,
        ),
    }
    out = explain_result(_result(extras=extras))
    assert "Error diversity" in out
    assert "Representation" in out
    assert "Noise" in out


def test_explain_flags_inconclusive_with_weak_probes():
    weak = _result(verdict="Inconclusive")
    # Pull confidence below 0.5 for the weakest probe.
    weak.probes["intervention"].confidence = 0.3
    out = explain_result(weak)
    assert "Inconclusive" in out
    assert "intervention" in out


def test_explain_html_strips_bold_markers():
    html = explain_result_html(_result())
    assert "**" not in html
    assert "<strong>" in html
    assert "</strong>" in html


def test_explain_html_renders_self_contained_block():
    html = explain_result_html(_result())
    assert html.strip().startswith("<div")
    assert html.strip().endswith("</div>")


def test_result_explain_method_works():
    r = _result()
    assert "Verdict" in r.explain()


# ── LaTeX export ─────────────────────────────────────────────────────

def test_to_latex_produces_booktabs_table():
    tex = to_latex(_result())
    assert "\\toprule" in tex
    assert "\\midrule" in tex
    assert "\\bottomrule" in tex
    assert "\\begin{tabular}" in tex
    assert "\\end{tabular}" in tex


def test_to_latex_escapes_underscores_in_names():
    r = _result(model="Hybrid_V1", dataset="qm9_alpha")
    tex = to_latex(r)
    assert "Hybrid\\_V1" in tex
    assert "qm9\\_alpha" in tex


def test_to_latex_includes_caption_and_label():
    tex = to_latex(_result(), caption="Hello", label="tab:hnep")
    assert "\\caption{Hello}" in tex
    assert "\\label{tab:hnep}" in tex
    assert "\\begin{table}" in tex


def test_to_latex_can_skip_caption_wrapper():
    tex = to_latex(_result(), include_caption=False)
    assert "\\begin{table}" not in tex
    assert "\\begin{tabular}" in tex


def test_compare_to_latex_handles_multiple_models():
    r1 = _result(model="ModelA", verdict="Regularizer")
    r2 = _result(model="ModelB", verdict="Genuine")
    tex = compare_to_latex([r1, r2])
    assert "ModelA" in tex
    assert "ModelB" in tex
    assert "Regularizer" in tex
    assert "Genuine" in tex
    # one column per model + probe column
    assert "{lll}" in tex


def test_compare_to_latex_empty_returns_marker():
    assert "no results" in compare_to_latex([])


def test_result_to_latex_method_works():
    tex = _result().to_latex(label="tab:test")
    assert "\\label{tab:test}" in tex


# ── Markdown export ──────────────────────────────────────────────────

def test_to_markdown_includes_verdict_and_table():
    md = to_markdown(_result())
    assert "HNEP Evaluation" in md
    assert "**QCT verdict:**" in md
    assert "**Regularizer**" in md
    assert "| Probe | Score | 95% CI | Verdict | Confidence |" in md


def test_to_markdown_includes_explanation_by_default():
    md = to_markdown(_result())
    # Explanation rendered as a blockquote
    assert "> **Verdict:" in md


def test_to_markdown_can_skip_explanation():
    md = to_markdown(_result(), include_explanation=False)
    assert "> **Verdict:" not in md


def test_to_markdown_pipes_balanced():
    md = to_markdown(_result())
    pipe_rows = [l for l in md.splitlines() if l.startswith("|")]
    for r in pipe_rows:
        assert r.count("|") == 6   # 5 cols → 6 pipes


def test_compare_to_markdown_balances_columns():
    r1 = _result(model="A")
    r2 = _result(model="B")
    md = compare_to_markdown([r1, r2])
    pipe_rows = [l for l in md.splitlines() if l.startswith("|")]
    counts = {r.count("|") for r in pipe_rows}
    assert len(counts) == 1   # all rows agree


def test_compare_to_markdown_empty_returns_marker():
    assert "no results" in compare_to_markdown([])


def test_result_to_markdown_method_works():
    md = _result().to_markdown()
    assert "HNEP Evaluation" in md


# ── HTML report integrates explainer ────────────────────────────────

def test_html_report_includes_explainer_block():
    r = _result()
    html = hnep.render_html_report(r)
    assert "<strong>Verdict:" in html
    # The card verdict already shows the label; the explainer adds the
    # follow-up sentence — check for vocabulary that only appears in the
    # explainer:
    assert "Surrogation" in html
    assert "Intervention" in html


# ── top-level exports ───────────────────────────────────────────────

def test_day7_top_level_exports():
    for name in (
        "explain_result", "explain_result_html",
        "to_latex", "compare_to_latex",
        "to_markdown", "compare_to_markdown",
    ):
        assert hasattr(hnep, name), name
