"""Molecular Chemistry Gallery.

Renders the top-K and bottom-K molecules (by Quantum Contribution Importance,
absolute prediction, or any user-supplied score) as a grid of structures.

Designed for the HTML report and for inclusion in slide decks. RDKit is
required to actually render structures; without it the gallery degrades to a
SMILES-only fallback so users still see *what* the model is highlighting,
even if not its 2-D structure.

The chemistry view is the bridge between abstract probe verdicts and what a
medicinal chemist would actually recognise: "ah — the quantum branch is
attending to the amide group across the high-QCI molecules."
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from html import escape
from typing import Iterable, Optional, Sequence


@dataclass
class MoleculeRecord:
    """One row in the gallery.

    Attributes
    ----------
    smiles
        SMILES string of the molecule.
    qci
        Quantum Contribution Importance score (or any other ranking scalar).
    prediction
        Model's prediction for the molecule (optional).
    target
        Ground-truth target value (optional).
    label
        Optional override for the caption.
    extras
        Any extra fields to display under the caption (e.g. dataset, atom
        highlights). Renders as ``key=value`` pairs.
    """

    smiles: str
    qci: float
    prediction: Optional[float] = None
    target: Optional[float] = None
    label: Optional[str] = None
    extras: dict = field(default_factory=dict)


def _have_rdkit() -> bool:
    try:
        import rdkit  # noqa: F401
        return True
    except Exception:
        return False


def _render_molecule_png(smiles: str, size: int = 260,
                          highlight_atoms: Optional[Sequence[int]] = None,
                          ) -> Optional[bytes]:
    """Render a SMILES to a PNG. Returns ``None`` if RDKit unavailable or
    the SMILES fails to parse."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
    except Exception:
        return None

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
    drawer.drawOptions().addAtomIndices = False
    if highlight_atoms:
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer, mol, highlightAtoms=list(highlight_atoms),
        )
    else:
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def build_gallery(
    records: Iterable[MoleculeRecord],
    top_k: int = 6,
    bottom_k: int = 6,
) -> dict:
    """Pick the top-K and bottom-K molecules by ``qci``.

    Returns a dict ``{"top": [...], "bottom": [...], "rdkit_available": bool}``.
    Each list is sorted (top descending, bottom ascending).
    """
    records = list(records)
    if not records:
        return {"top": [], "bottom": [], "rdkit_available": _have_rdkit()}

    by_score = sorted(records, key=lambda r: r.qci)
    bottom = by_score[: max(0, bottom_k)]
    top = list(reversed(by_score[-max(0, top_k):])) if top_k > 0 else []
    return {
        "top": top,
        "bottom": bottom,
        "rdkit_available": _have_rdkit(),
    }


def _png_to_data_uri(png: bytes) -> str:
    b64 = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _record_card_html(rec: MoleculeRecord, size: int = 240) -> str:
    """One molecule card."""
    png = _render_molecule_png(rec.smiles, size=size)
    if png is not None:
        img_html = (
            f'<img src="{_png_to_data_uri(png)}" '
            f'alt="{escape(rec.smiles)}" '
            'style="width:100%; max-width:'
            f'{size}px; height:auto;">'
        )
    else:
        img_html = (
            '<div style="border:1px dashed #ccc; padding:1.5em; '
            'text-align:center; color:#999;">'
            'no structure render<br><small>(RDKit unavailable or '
            'invalid SMILES)</small></div>'
        )

    label = rec.label if rec.label else rec.smiles
    extra_lines = []
    if rec.prediction is not None:
        extra_lines.append(f"pred = {rec.prediction:.3f}")
    if rec.target is not None:
        extra_lines.append(f"target = {rec.target:.3f}")
    for k, v in rec.extras.items():
        extra_lines.append(f"{k} = {v}")

    extras_html = ""
    if extra_lines:
        extras_html = (
            '<div style="font-size:0.78em; color:#666; '
            'margin-top:2px;">' + " · ".join(escape(s) for s in extra_lines)
            + "</div>"
        )

    return f"""
    <div class="mol-card">
      {img_html}
      <div class="mol-caption">
        <code style="font-size:0.8em;">{escape(label)}</code>
        <div style="font-size:0.85em; color:#444; margin-top:2px;">
          QCI = <strong>{rec.qci:.3f}</strong>
        </div>
        {extras_html}
      </div>
    </div>
    """


GALLERY_CSS = """
.mol-gallery-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 0.9em;
  margin: 0.8em 0 1.4em 0;
}
.mol-card {
  border: 1px solid #e0e0e0;
  border-radius: 6px;
  padding: 0.6em;
  background: #fff;
  text-align: center;
}
.mol-card .mol-caption {
  margin-top: 0.4em;
  text-align: left;
  word-break: break-all;
}
.gallery-section-title {
  font-size: 1.0em;
  font-weight: 600;
  color: #444;
  margin: 1.1em 0 0.3em 0;
}
.gallery-empty {
  color: #888;
  font-style: italic;
  padding: 0.6em 0;
}
"""


def render_gallery_html(
    gallery: dict,
    title: str = "Molecular Chemistry Gallery",
    top_label: str = "Top-K by QCI (quantum attends most)",
    bottom_label: str = "Bottom-K by QCI (quantum attends least)",
    include_css: bool = True,
    card_size: int = 240,
) -> str:
    """Render a gallery dict (from :func:`build_gallery`) as a self-contained
    HTML snippet."""
    css_block = f"<style>{GALLERY_CSS}</style>" if include_css else ""

    def section(label, records):
        if not records:
            return (
                f'<div class="gallery-section-title">{escape(label)}</div>'
                '<div class="gallery-empty">No molecules.</div>'
            )
        cards = "\n".join(_record_card_html(r, size=card_size) for r in records)
        return (
            f'<div class="gallery-section-title">{escape(label)}</div>'
            f'<div class="mol-gallery-grid">{cards}</div>'
        )

    top_html = section(top_label, gallery.get("top", []))
    bottom_html = section(bottom_label, gallery.get("bottom", []))

    warning = ""
    if not gallery.get("rdkit_available", True):
        warning = (
            '<div style="background:#fff8e6; border-left:4px solid #e0c060; '
            'padding:0.5em 0.9em; font-size:0.85em; margin-bottom:0.6em;">'
            'RDKit is not installed — molecules are shown as SMILES strings '
            'only. <code>pip install rdkit</code> to render structures.'
            '</div>'
        )

    return f"""
{css_block}
<h2>{escape(title)}</h2>
{warning}
{top_html}
{bottom_html}
"""
