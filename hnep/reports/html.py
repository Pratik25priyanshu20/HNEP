"""HTML report renderer — embeds the QCT plane and convergent-validity
radar inline as base64 PNGs so the result is a single self-contained file.
"""

from __future__ import annotations

import base64
import io
from html import escape
from pathlib import Path
from typing import Optional, Sequence

from hnep.results.hnep_result import HNEPResult
from hnep.visualizations.qct_plane import plot_qct_plane
from hnep.visualizations.radar import plot_convergent_validity_radar
from hnep.gallery.molecular import (
    GALLERY_CSS,
    MoleculeRecord,
    build_gallery,
    render_gallery_html,
)
from hnep.explain import explain_result_html


CSS = """
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  max-width: 1080px; margin: 2em auto; padding: 0 1.5em;
  color: #222; line-height: 1.6;
}
h1 { font-size: 1.9em; margin: 0 0 0.2em 0; }
h2 { font-size: 1.3em; margin: 1.8em 0 0.6em 0;
     border-bottom: 1px solid #ddd; padding-bottom: 4px; }
h3 { font-size: 1.05em; margin: 1.2em 0 0.4em 0; color: #444; }
.subtitle { color: #777; font-style: italic; margin-bottom: 1em; }
.verdict-card {
  background: #f5f8fc; border-left: 5px solid #1f77b4;
  padding: 1em 1.4em; margin: 1em 0; border-radius: 4px;
}
.verdict-card .verdict { font-size: 1.6em; font-weight: bold; color: #1f77b4; }
.verdict-card .conf { color: #555; font-size: 0.9em; margin-top: 4px; }
.verdict-genuine    { border-color: #2ca02c; }
.verdict-genuine .verdict { color: #2ca02c; }
.verdict-regularizer { border-color: #1f77b4; }
.verdict-regularizer .verdict { color: #1f77b4; }
.verdict-ignored    { border-color: #ff7f0e; }
.verdict-ignored .verdict { color: #ff7f0e; }
.verdict-dead-weight { border-color: #d62728; }
.verdict-dead-weight .verdict { color: #d62728; }
.verdict-inconclusive { border-color: #888; }
.verdict-inconclusive .verdict { color: #888; }

table { border-collapse: collapse; width: 100%; margin: 0.8em 0; font-size: 0.92em; }
th, td { padding: 7px 10px; text-align: left; border: 1px solid #ddd; }
th { background: #eef2f5; }
tr:nth-child(even) td { background: #fafafa; }
.confidence-bar {
  display: inline-block; height: 8px; width: 60px;
  background: #eee; border-radius: 4px; position: relative; vertical-align: middle;
}
.confidence-fill { display: block; height: 100%; border-radius: 4px;
                    background: #4caf50; }
.confidence-fill.med { background: #ffaa33; }
.confidence-fill.low { background: #d62728; }
.figure { text-align: center; margin: 1.5em 0; }
.figure img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
.figure .caption { font-size: 0.85em; color: #666; margin-top: 4px; }
.footer { font-size: 0.8em; color: #888; margin-top: 3em;
           border-top: 1px solid #ddd; padding-top: 1em; }
.note { background: #fff8e6; border-left: 4px solid #e0c060;
        padding: 0.6em 1em; margin: 1em 0; font-size: 0.9em; }
code { background: #f0f0f0; padding: 1px 5px; border-radius: 3px;
       font-size: 0.92em; }
""" + GALLERY_CSS


def _confidence_class(c: float) -> str:
    if c >= 0.7:
        return ""
    if c >= 0.4:
        return "med"
    return "low"


def _verdict_class(verdict: str) -> str:
    return "verdict-" + verdict.lower().replace(" ", "-")


def _fig_to_base64(fig) -> str:
    """Render a matplotlib figure to a base64-encoded PNG data URI."""
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{data}"


def render_html_report(
    result: HNEPResult,
    path: Optional[str | Path] = None,
    other_results: Optional[Sequence[HNEPResult]] = None,
) -> str:
    """Render a full HTML report.

    Parameters
    ----------
    result
        The primary :class:`HNEPResult` to report on.
    path
        Optional output file. If given, the HTML is also written there.
    other_results
        Optional additional results to overlay on the QCT plane and radar for
        A/B comparison.

    Returns
    -------
    The rendered HTML string.
    """
    results_for_viz = [result] + (list(other_results) if other_results else [])

    qct_fig = plot_qct_plane(results_for_viz)
    qct_img = _fig_to_base64(qct_fig)

    radar_fig = plot_convergent_validity_radar(results_for_viz)
    radar_img = _fig_to_base64(radar_fig)

    # Probe table rows
    probe_rows = []
    for name, probe in result.probes.items():
        ci_lo, ci_hi = ("", "")
        if probe.primary_score_ci:
            ci_lo, ci_hi = probe.primary_score_ci
            ci_str = f"[{ci_lo:.3f}, {ci_hi:.3f}]"
        else:
            ci_str = "—"
        conf_class = _confidence_class(probe.confidence)
        probe_rows.append(f"""
          <tr>
            <td><strong>{escape(name)}</strong></td>
            <td>{probe.primary_score:.4f}</td>
            <td>{ci_str}</td>
            <td>{escape(probe.verdict)}</td>
            <td>
              <span class="confidence-bar">
                <span class="confidence-fill {conf_class}" style="width: {probe.confidence*100:.0f}%"></span>
              </span>
              <small>{probe.confidence:.2f}</small>
            </td>
          </tr>""")
    probe_table = "\n".join(probe_rows) if probe_rows else "<tr><td colspan='5'>No probes were run.</td></tr>"

    # Notes
    notes_html = ""
    for n in result.notes:
        notes_html += f'<div class="note">{escape(n)}</div>'
    for probe in result.probes.values():
        for n in probe.notes:
            notes_html += f'<div class="note"><strong>{escape(probe.probe_name)}:</strong> {escape(n)}</div>'

    # Manifest
    manifest_lines = "\n".join(
        f"  <li><code>{escape(str(k))}</code> = <code>{escape(str(v))}</code></li>"
        for k, v in result.manifest.items()
    )

    verdict_class = _verdict_class(result.qct_verdict)

    # ── Molecular Chemistry Gallery (optional) ─────────────────────────
    gallery_html = ""
    if result.molecule_records:
        records = []
        for r in result.molecule_records:
            if isinstance(r, MoleculeRecord):
                records.append(r)
            elif isinstance(r, dict):
                records.append(MoleculeRecord(**r))
        if records:
            gallery = build_gallery(records, top_k=6, bottom_k=6)
            gallery_html = render_gallery_html(
                gallery,
                title=f"Molecular Chemistry Gallery — {escape(result.dataset_name)}",
                include_css=False,   # CSS already injected in <head>
            )

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>HNEP Report — {escape(result.model_name)}</title>
<style>{CSS}</style>
</head><body>

<h1>HNEP Evaluation Report</h1>
<div class="subtitle">
  <strong>Model:</strong> {escape(result.model_name)} &nbsp;·&nbsp;
  <strong>Dataset:</strong> {escape(result.dataset_name)}
</div>

<div class="verdict-card {verdict_class}">
  <div class="verdict">{escape(result.qct_verdict)}</div>
  <div class="conf">QCT verdict — confidence {result.qct_confidence:.2f}</div>
</div>

{explain_result_html(result)}

{notes_html}

<h2>The QCT Plane</h2>
<div class="figure">
  <img src="{qct_img}" alt="QCT plane">
  <div class="caption">
    Surrogation Score (x) vs Intervention Δ (y). Quadrants colour-coded by
    QCT verdict. Rectangles show 95% confidence intervals.
  </div>
</div>

<h2>Convergent Validity Radar</h2>
<div class="figure">
  <img src="{radar_img}" alt="Convergent validity radar">
  <div class="caption">
    Each probe contributes one axis. Jaggedness reveals where probes agree
    or disagree on the underlying quantum contribution.
  </div>
</div>

<h2>Probe Results</h2>
<table>
  <thead><tr>
    <th>Probe</th><th>Score</th><th>95% CI</th><th>Verdict</th><th>Confidence</th>
  </tr></thead>
  <tbody>{probe_table}</tbody>
</table>

{gallery_html}

<h2>Run Manifest</h2>
<ul>{manifest_lines}</ul>

<div class="footer">
  Generated by <strong>HNEP</strong> · Hybrid Network Evaluation Protocol ·
  v0.1.0 alpha · <code>pip install hnep</code>
</div>

</body></html>
"""

    if path is not None:
        Path(path).write_text(html, encoding="utf-8")
    return html
