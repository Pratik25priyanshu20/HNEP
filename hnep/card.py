"""HNEPCard — compact single-glance summary of an evaluation.

A card is what you show *outside* a deep technical context: a slide, a
README badge, a Slack message, a model registry entry. It is one screen of
information per model+dataset pair, with no scrolling.

Three output forms:

* :meth:`HNEPCard.to_text`   — ASCII card for terminals and READMEs.
* :meth:`HNEPCard.to_markdown` — compact Markdown table for PRs / issues.
* :meth:`HNEPCard.to_html`   — self-contained HTML snippet for embedding.

For side-by-side multi-model comparison, use :func:`compare_cards`.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Iterable, Sequence

from hnep.results.hnep_result import HNEPResult


# Compact symbols (ASCII for terminal-portability — no emoji unless asked).
_VERDICT_BADGE = {
    "Genuine":      "[G ]",
    "Regularizer":  "[R ]",
    "Ignored":      "[I ]",
    "Dead Weight":  "[D ]",
    "Inconclusive": "[? ]",
}

_PROBE_SHORT_VERDICT = {
    # Surrogation
    "REPLACEABLE": "repl",
    "NECESSARY":   "nec ",
    # Intervention
    "LOAD-BEARING":     "load",
    "NOT-LOAD-BEARING": "free",
    # Error diversity
    "DIVERSE":   "div ",
    "REDUNDANT": "red ",
    # Representation
    "QUANTUM-MORE-ALIGNED":   "qali",
    "CLASSICAL-MORE-ALIGNED": "cali",
    # Noise
    "STABLE":              "stbl",
    "MOSTLY-STABLE":       "m-st",
    "PERFORMANCE-DEGRADES":"deg ",
    "VERDICT-FLIPS":       "flip",
    # Temporal
    "TRANSIENT-DRIFT": "trdf",
    "VERDICT-DRIFTS":  "vdrf",
    # Fallbacks
    "UNAVAILABLE":  "n/a ",
    "INCONCLUSIVE": "?   ",
}


def _short(verdict: str) -> str:
    return _PROBE_SHORT_VERDICT.get(verdict, verdict[:4].lower().ljust(4))


@dataclass
class HNEPCard:
    """Single-model summary card built from an :class:`HNEPResult`."""

    result: HNEPResult

    # ── Text ─────────────────────────────────────────────────────────

    def to_text(self, width: int = 64) -> str:
        r = self.result
        badge = _VERDICT_BADGE.get(r.qct_verdict, "[? ]")
        bar = "─" * (width - 2)

        lines = [
            f"┌{bar}┐",
            self._line(f" HNEP Card  ·  {r.model_name}  on  {r.dataset_name}", width),
            self._line(f" {bar[:-1]}", width),
            self._line(
                f" QCT: {badge} {r.qct_verdict}"
                f"   confidence: {r.qct_confidence:.2f}",
                width,
            ),
            self._line("", width),
            self._line(" Probes:", width),
        ]

        for name, probe in r.probes.items():
            ci = ""
            if probe.primary_score_ci:
                lo, hi = probe.primary_score_ci
                ci = f" [{lo:+.2f},{hi:+.2f}]"
            row = (
                f"   {name:<18s} {probe.primary_score:+.3f}{ci}"
                f"  → {_short(probe.verdict)}"
                f"  conf {probe.confidence:.2f}"
            )
            lines.append(self._line(row, width))

        if r.notes:
            lines.append(self._line("", width))
            lines.append(self._line(" Notes:", width))
            for n in r.notes[:3]:
                # wrap notes that exceed the inner width.
                for chunk in self._wrap(n, width - 4):
                    lines.append(self._line(f"   • {chunk}", width))

        lines.append(f"└{bar}┘")
        return "\n".join(lines)

    @staticmethod
    def _line(content: str, width: int) -> str:
        inner = width - 2
        if len(content) > inner:
            content = content[: inner - 1] + "…"
        return f"│{content:<{inner}}│"

    @staticmethod
    def _wrap(text: str, width: int) -> list[str]:
        words = text.split()
        out: list[str] = []
        cur: list[str] = []
        ln = 0
        for w in words:
            if ln + len(w) + (1 if cur else 0) > width:
                out.append(" ".join(cur))
                cur = [w]
                ln = len(w)
            else:
                cur.append(w)
                ln += len(w) + (1 if len(cur) > 1 else 0)
        if cur:
            out.append(" ".join(cur))
        return out or [""]

    # ── Markdown ─────────────────────────────────────────────────────

    def to_markdown(self) -> str:
        r = self.result
        out = [
            f"### HNEP Card — `{r.model_name}` on `{r.dataset_name}`",
            "",
            f"**QCT verdict:** **{r.qct_verdict}** "
            f"(confidence {r.qct_confidence:.2f})",
            "",
            "| Probe | Score | 95% CI | Verdict | Confidence |",
            "|---|---|---|---|---|",
        ]
        for name, p in r.probes.items():
            ci = "—"
            if p.primary_score_ci:
                lo, hi = p.primary_score_ci
                ci = f"[{lo:+.3f}, {hi:+.3f}]"
            out.append(
                f"| `{name}` | {p.primary_score:+.4f} | {ci} | "
                f"{p.verdict} | {p.confidence:.2f} |"
            )
        if r.notes:
            out.append("")
            out.append("**Notes:** " + " · ".join(r.notes))
        return "\n".join(out)

    # ── HTML ─────────────────────────────────────────────────────────

    HTML_CSS = """
    .hnep-card {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      max-width: 460px; border: 1px solid #ccc; border-radius: 6px;
      padding: 1em 1.2em; margin: 0.8em 0; background: #fff;
    }
    .hnep-card .hc-title { font-weight: 600; font-size: 0.95em;
                           color: #444; margin-bottom: 0.3em; }
    .hnep-card .hc-verdict { font-size: 1.3em; font-weight: 700;
                             margin: 0.2em 0 0.3em 0; }
    .hnep-card .hc-conf { color: #777; font-size: 0.82em;
                          margin-bottom: 0.8em; }
    .hnep-card table { width: 100%; border-collapse: collapse;
                        font-size: 0.85em; }
    .hnep-card th, .hnep-card td { text-align: left; padding: 3px 6px;
                                     border-bottom: 1px solid #eee; }
    .hnep-card th { color: #888; font-weight: 500; font-size: 0.78em;
                     text-transform: uppercase; letter-spacing: 0.04em; }
    .hnep-card .hc-footer { color: #aaa; font-size: 0.72em;
                             margin-top: 0.7em; text-align: right; }
    .hc-v-genuine    { color: #2ca02c; }
    .hc-v-regularizer { color: #1f77b4; }
    .hc-v-ignored    { color: #ff7f0e; }
    .hc-v-dead-weight { color: #d62728; }
    .hc-v-inconclusive { color: #888; }
    """

    def to_html(self, include_css: bool = True) -> str:
        r = self.result
        verdict_class = "hc-v-" + r.qct_verdict.lower().replace(" ", "-")
        css_block = f"<style>{self.HTML_CSS}</style>" if include_css else ""

        rows = []
        for name, p in r.probes.items():
            ci = "—"
            if p.primary_score_ci:
                lo, hi = p.primary_score_ci
                ci = f"[{lo:+.3f}, {hi:+.3f}]"
            rows.append(
                f"<tr><td><code>{escape(name)}</code></td>"
                f"<td>{p.primary_score:+.4f}</td>"
                f"<td>{ci}</td>"
                f"<td>{escape(p.verdict)}</td>"
                f"<td>{p.confidence:.2f}</td></tr>"
            )
        table = (
            "<table><thead><tr><th>Probe</th><th>Score</th><th>CI</th>"
            "<th>Verdict</th><th>Conf</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

        return f"""{css_block}
<div class="hnep-card">
  <div class="hc-title">HNEP Card · <code>{escape(r.model_name)}</code> on
       <code>{escape(r.dataset_name)}</code></div>
  <div class="hc-verdict {verdict_class}">{escape(r.qct_verdict)}</div>
  <div class="hc-conf">QCT verdict — confidence {r.qct_confidence:.2f}</div>
  {table}
  <div class="hc-footer">generated by hnep</div>
</div>
"""


# ── Multi-model comparison ────────────────────────────────────────────

def _all_probe_names(results: Sequence[HNEPResult]) -> list[str]:
    seen: list[str] = []
    for r in results:
        for n in r.probes.keys():
            if n not in seen:
                seen.append(n)
    return seen


def compare_cards_text(
    results: Sequence[HNEPResult],
    column_width: int = 16,
) -> str:
    """Side-by-side ASCII comparison table for several models."""
    if not results:
        return "(no results to compare)"

    probes = _all_probe_names(results)
    headers = ["Probe"] + [r.model_name for r in results]

    def fmt_cell(s: str) -> str:
        s = s if s is not None else ""
        if len(s) > column_width:
            s = s[: column_width - 1] + "…"
        return s.ljust(column_width)

    sep = "+" + "+".join("-" * (column_width + 2) for _ in headers) + "+"
    lines = [sep, "| " + " | ".join(fmt_cell(h) for h in headers) + " |", sep]

    # QCT verdict row
    verdict_row = ["QCT VERDICT"] + [r.qct_verdict for r in results]
    lines.append("| " + " | ".join(fmt_cell(v) for v in verdict_row) + " |")
    conf_row = ["confidence"] + [f"{r.qct_confidence:.2f}" for r in results]
    lines.append("| " + " | ".join(fmt_cell(v) for v in conf_row) + " |")
    lines.append(sep)

    # Probe rows
    for probe in probes:
        row = [probe]
        for r in results:
            p = r.probes.get(probe)
            if p is None:
                row.append("—")
            else:
                row.append(f"{p.verdict} ({p.primary_score:+.3f})")
        lines.append("| " + " | ".join(fmt_cell(v) for v in row) + " |")
    lines.append(sep)
    return "\n".join(lines)


def compare_cards_markdown(results: Sequence[HNEPResult]) -> str:
    if not results:
        return "_no results to compare_"
    probes = _all_probe_names(results)
    headers = ["Probe"] + [f"`{r.model_name}` on `{r.dataset_name}`" for r in results]

    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|",
           "| **QCT verdict** | " + " | ".join(
               f"**{r.qct_verdict}** ({r.qct_confidence:.2f})" for r in results
           ) + " |"]
    for probe in probes:
        cells = [probe]
        for r in results:
            p = r.probes.get(probe)
            cells.append("—" if p is None
                         else f"{p.verdict} ({p.primary_score:+.3f})")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def compare_cards_html(results: Sequence[HNEPResult],
                        include_css: bool = True) -> str:
    if not results:
        return "<em>no results to compare</em>"
    probes = _all_probe_names(results)
    css = f"<style>{HNEPCard.HTML_CSS}</style>" if include_css else ""
    headers = (
        "<tr><th>Probe</th>"
        + "".join(
            f"<th><code>{escape(r.model_name)}</code><br>"
            f"<small>{escape(r.dataset_name)}</small></th>"
            for r in results
        )
        + "</tr>"
    )
    verdict_row = (
        '<tr><td><strong>QCT verdict</strong></td>'
        + "".join(
            f'<td class="hc-v-{r.qct_verdict.lower().replace(" ", "-")}">'
            f'<strong>{escape(r.qct_verdict)}</strong><br>'
            f'<small>conf {r.qct_confidence:.2f}</small></td>'
            for r in results
        )
        + "</tr>"
    )
    rows = [verdict_row]
    for probe in probes:
        cells = [f"<td><code>{escape(probe)}</code></td>"]
        for r in results:
            p = r.probes.get(probe)
            if p is None:
                cells.append("<td>—</td>")
            else:
                cells.append(
                    f"<td>{escape(p.verdict)}"
                    f"<br><small>{p.primary_score:+.3f}</small></td>"
                )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f"{css}<div class='hnep-card' style='max-width:none;'>"
        f"<div class='hc-title'>HNEP comparison "
        f"({len(results)} models)</div>"
        f"<table><thead>{headers}</thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        f"</div>"
    )


def load_result_from_json(path) -> HNEPResult:
    """Reconstruct an :class:`HNEPResult` from JSON written by
    :meth:`HNEPResult.to_json`."""
    import json
    from pathlib import Path

    from hnep.results.probe_result import ProbeResult

    data = json.loads(Path(path).read_text())

    probes = {}
    for name, p in (data.get("probes") or {}).items():
        ci = p.get("primary_score_ci")
        probes[name] = ProbeResult(
            probe_name=p.get("probe_name", name),
            primary_score=float(p.get("primary_score", 0.0)),
            primary_score_ci=tuple(ci) if ci else None,
            verdict=p.get("verdict", "unknown"),
            confidence=float(p.get("confidence", 0.0)),
            details=p.get("details", {}) or {},
            config=p.get("config", {}) or {},
            notes=list(p.get("notes", []) or []),
        )

    return HNEPResult(
        model_name=data.get("model_name", ""),
        dataset_name=data.get("dataset_name", ""),
        qct_verdict=data.get("qct_verdict", "Inconclusive"),
        qct_confidence=float(data.get("qct_confidence", 0.0)),
        probes=probes,
        cost_utility=data.get("cost_utility"),
        manifest=data.get("manifest", {}) or {},
        notes=list(data.get("notes", []) or []),
    )
