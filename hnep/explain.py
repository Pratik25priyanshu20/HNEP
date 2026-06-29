"""Verdict explainer — turns a QCT verdict into plain-English reasoning.

The explainer reads the same probe outputs the classifier used and emits one
paragraph that names the evidence that drove the verdict. It is a
deterministic template (no LLM, no hidden state), so the explanation can be
audited, version-controlled, and cited alongside the numbers it describes.

Usage::

    explanation = explain_result(result)
    print(explanation)

Or via the high-level convenience method::

    print(result.explain())

The explainer reasons about, in order of priority:

  1. Surrogation (replaceable vs necessary)
  2. Intervention (load-bearing vs not)
  3. Convergent / divergent evidence from secondary probes
     (noise, temporal, error_diversity, representation)

Each piece adds a sentence only when its probe was actually run, so the
length of the output scales with how many probes the user ran.
"""

from __future__ import annotations

from typing import List, Optional

from hnep.results.hnep_result import HNEPResult
from hnep.results.probe_result import ProbeResult
from hnep.thresholds import DEFAULT_THRESHOLDS, Thresholds


_VERDICT_LEAD = {
    "Genuine": (
        "The quantum component does **something the classical branch "
        "cannot reproduce**, and removing it materially hurts the model."
    ),
    "Regularizer": (
        "The quantum output is **reproducible from classical features**, "
        "yet the model still relies on it — the quantum branch is "
        "functioning as a regulariser rather than a distinct computation."
    ),
    "Ignored": (
        "The quantum branch produces a **unique signal**, but the rest of "
        "the model **does not use it** — removing the quantum component "
        "leaves performance essentially unchanged."
    ),
    "Dead Weight": (
        "The quantum output is **reproducible classically** *and* "
        "**ablating it does not hurt performance** — the quantum component "
        "is doing no useful work in this hybrid."
    ),
    "Inconclusive": (
        "The probes did not give a confident verdict — at least one "
        "confidence interval straddles a decision threshold. "
        "More seeds, more data, or a tighter bootstrap are needed before "
        "a QCT verdict can be assigned."
    ),
}


def _fmt_ci(probe: Optional[ProbeResult]) -> str:
    if probe is None or probe.primary_score_ci is None:
        return ""
    lo, hi = probe.primary_score_ci
    return f" (95% CI [{lo:+.3f}, {hi:+.3f}])"


def _surrogation_sentence(probe: Optional[ProbeResult],
                           thresholds: Thresholds) -> str:
    if probe is None:
        return ""
    s = probe.primary_score
    th = thresholds.ss_replaceable
    ci = _fmt_ci(probe)
    if s < th:
        return (
            f"**Surrogation** found the quantum output is reproducible — "
            f"a classical surrogate matches it with SS = {s:.3f}{ci}, "
            f"below the {th:.2f} REPLACEABLE threshold."
        )
    return (
        f"**Surrogation** found the quantum output cannot be reproduced "
        f"by a classical surrogate (SS = {s:.3f}{ci}, above the "
        f"{th:.2f} REPLACEABLE threshold)."
    )


def _intervention_sentence(probe: Optional[ProbeResult],
                            thresholds: Thresholds) -> str:
    if probe is None:
        return ""
    delta = probe.primary_score
    th = thresholds.intervention_load_bearing
    ci = _fmt_ci(probe)
    if delta >= th:
        return (
            f"**Intervention** confirms the quantum branch is load-bearing: "
            f"ablating it drops R² by Δ = {delta:+.3f}{ci} "
            f"(threshold {th:+.2f})."
        )
    return (
        f"**Intervention** shows the quantum branch is not load-bearing: "
        f"ablating it changes R² by only Δ = {delta:+.3f}{ci}, "
        f"under the {th:+.2f} threshold."
    )


def _secondary_sentences(result: HNEPResult) -> List[str]:
    """Add corroborating or contradicting evidence from secondary probes."""
    out: List[str] = []

    err = result.probes.get("error_diversity")
    if err is not None:
        if err.verdict == "DIVERSE":
            out.append(
                "**Error diversity** corroborates: the quantum branch and the "
                "classical branch make decorrelated mistakes "
                f"(|r| = {abs(err.primary_score):.2f}), so each is providing "
                "information the other does not."
            )
        elif err.verdict == "REDUNDANT":
            out.append(
                "**Error diversity** is in tension: the quantum and classical "
                "branches make highly correlated errors "
                f"(|r| = {abs(err.primary_score):.2f}), suggesting they are "
                "computing similar functions."
            )

    rep = result.probes.get("representation")
    if rep is not None:
        if rep.verdict == "QUANTUM-MORE-ALIGNED":
            out.append(
                "**Representation analysis** finds the quantum embedding is "
                "more target-aligned than the classical one (CKA-Q vs target "
                "exceeds CKA-C vs target)."
            )
        elif rep.verdict == "CLASSICAL-MORE-ALIGNED":
            out.append(
                "**Representation analysis** finds the classical embedding "
                "is more target-aligned than the quantum one — the quantum "
                "branch carries less direct signal."
            )

    noise = result.probes.get("noise")
    if noise is not None:
        if noise.verdict == "STABLE":
            out.append(
                "**Noise injection** finds the verdict is robust to small "
                "perturbations of the quantum output."
            )
        elif noise.verdict == "VERDICT-FLIPS":
            out.append(
                "**Noise injection** is a warning sign: realistic noise "
                "levels flip the verdict, which weakens any deployment claim "
                "based on these results."
            )

    temp = result.probes.get("temporal")
    if temp is not None and temp.verdict == "VERDICT-DRIFTS":
        out.append(
            "**Temporal probe** finds the verdict changes across training "
            "checkpoints — the QCT label depends on which checkpoint you "
            "evaluate."
        )

    return out


def explain_result(
    result: HNEPResult,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> str:
    """Return a single-paragraph plain-English explanation of the verdict.

    The text uses Markdown-style ``**bold**`` markers for the verdict, probe
    names, and any quantitative claim, so it renders cleanly in Markdown,
    Slack, and the HTML report (which strips or interprets the markers).
    """
    parts: List[str] = []

    parts.append(f"**Verdict: {result.qct_verdict}.** "
                 f"(confidence {result.qct_confidence:.2f})")

    lead = _VERDICT_LEAD.get(result.qct_verdict)
    if lead:
        parts.append(lead)

    s = _surrogation_sentence(result.probes.get("surrogation"), thresholds)
    if s:
        parts.append(s)

    i = _intervention_sentence(result.probes.get("intervention"), thresholds)
    if i:
        parts.append(i)

    parts.extend(_secondary_sentences(result))

    if result.qct_verdict == "Inconclusive":
        bad = [name for name, p in result.probes.items() if p.confidence < 0.5]
        if bad:
            parts.append(
                "Probes with the weakest evidence: "
                + ", ".join(f"`{b}`" for b in bad) + "."
            )

    return " ".join(parts)


def explain_result_html(
    result: HNEPResult,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> str:
    """Render the explanation as a self-contained HTML snippet.

    Translates the ``**bold**`` markers to ``<strong>...</strong>`` and wraps
    the paragraph in a styled card.
    """
    text = explain_result(result, thresholds=thresholds)
    # Lightweight bold conversion — explainer never produces nested markers.
    html_text = ""
    buf = ""
    bold = False
    i = 0
    while i < len(text):
        if text[i:i + 2] == "**":
            html_text += buf
            buf = ""
            html_text += "</strong>" if bold else "<strong>"
            bold = not bold
            i += 2
        else:
            buf += text[i]
            i += 1
    html_text += buf
    if bold:
        # Defensive: unbalanced ** — close the tag so HTML doesn't break.
        html_text += "</strong>"

    return (
        '<div style="background:#f9fafc; border-left:4px solid #1f77b4; '
        'padding:0.8em 1.1em; margin:0.8em 0; border-radius:4px; '
        'line-height:1.55; font-size:0.95em;">'
        f"{html_text}"
        "</div>"
    )
