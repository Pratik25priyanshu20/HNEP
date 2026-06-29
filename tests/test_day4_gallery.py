"""Day-4 tests: Molecular Chemistry Gallery."""

from __future__ import annotations

import pytest

import hnep
from hnep.gallery.molecular import (
    GALLERY_CSS,
    MoleculeRecord,
    _have_rdkit,
    _png_to_data_uri,
    _record_card_html,
    _render_molecule_png,
    build_gallery,
    render_gallery_html,
)
from hnep.results.hnep_result import HNEPResult
from hnep.results.probe_result import ProbeResult


# Real ESOL-like SMILES for end-to-end realism.
EXAMPLE_RECORDS = [
    MoleculeRecord(smiles="CCO", qci=0.10, prediction=-0.3, target=-0.5),
    MoleculeRecord(smiles="CC(=O)O", qci=0.95, prediction=0.2, target=0.1),
    MoleculeRecord(smiles="c1ccccc1", qci=0.42),
    MoleculeRecord(smiles="CCN(CC)CC", qci=0.05),
    MoleculeRecord(smiles="CC(=O)Nc1ccc(O)cc1", qci=0.88,
                   prediction=-1.2, target=-1.1,
                   extras={"dataset": "ESOL"}),
    MoleculeRecord(smiles="O=C1NC(=O)C(=O)N1", qci=0.71),
]


# ── build_gallery ─────────────────────────────────────────────────────

def test_build_gallery_picks_top_and_bottom_by_qci():
    g = build_gallery(EXAMPLE_RECORDS, top_k=2, bottom_k=2)
    top_smiles = [r.smiles for r in g["top"]]
    bottom_smiles = [r.smiles for r in g["bottom"]]
    # Top-2 by QCI: 0.95, 0.88
    assert top_smiles == ["CC(=O)O", "CC(=O)Nc1ccc(O)cc1"]
    # Bottom-2 by QCI: 0.05, 0.10
    assert bottom_smiles == ["CCN(CC)CC", "CCO"]
    assert "rdkit_available" in g


def test_build_gallery_handles_empty_list():
    g = build_gallery([], top_k=4, bottom_k=4)
    assert g["top"] == []
    assert g["bottom"] == []


def test_build_gallery_handles_k_larger_than_records():
    g = build_gallery(EXAMPLE_RECORDS[:2], top_k=10, bottom_k=10)
    # Returns at most all records on each side (overlap allowed by design).
    assert len(g["top"]) == 2
    assert len(g["bottom"]) == 2


def test_build_gallery_zero_k():
    g = build_gallery(EXAMPLE_RECORDS, top_k=0, bottom_k=0)
    assert g["top"] == []
    assert g["bottom"] == []


# ── PNG rendering ─────────────────────────────────────────────────────

def test_render_molecule_png_with_rdkit():
    if not _have_rdkit():
        pytest.skip("RDKit not installed")
    png = _render_molecule_png("CCO", size=200)
    assert isinstance(png, (bytes, bytearray))
    # PNG magic bytes
    assert bytes(png).startswith(b"\x89PNG")


def test_render_molecule_png_returns_none_for_bad_smiles():
    if not _have_rdkit():
        pytest.skip("RDKit not installed")
    png = _render_molecule_png("not_a_smiles_!!!", size=200)
    assert png is None


def test_png_to_data_uri_round_trip():
    uri = _png_to_data_uri(b"hello-png")
    assert uri.startswith("data:image/png;base64,")


# ── HTML rendering ────────────────────────────────────────────────────

def test_record_card_html_contains_qci_and_smiles():
    html = _record_card_html(EXAMPLE_RECORDS[0])
    assert "QCI" in html
    assert "0.100" in html  # QCI formatting
    assert "CCO" in html


def test_record_card_html_renders_extras():
    rec = MoleculeRecord(smiles="CC", qci=0.5, prediction=1.5, target=2.0,
                         extras={"dataset": "ESOL"})
    html = _record_card_html(rec)
    assert "pred" in html
    assert "1.500" in html
    assert "target" in html
    assert "2.000" in html
    assert "dataset" in html
    assert "ESOL" in html


def test_render_gallery_html_full():
    g = build_gallery(EXAMPLE_RECORDS, top_k=3, bottom_k=3)
    html = render_gallery_html(g, title="Test Gallery")
    assert "Test Gallery" in html
    assert "mol-gallery-grid" in html
    assert "Top-K" in html
    assert "Bottom-K" in html
    # CSS included by default
    assert "mol-card" in html


def test_render_gallery_html_can_skip_css():
    g = build_gallery(EXAMPLE_RECORDS, top_k=2, bottom_k=2)
    html = render_gallery_html(g, include_css=False)
    # Defining the CSS rule should be absent when include_css=False
    assert ".mol-gallery-grid" not in html
    # But the class attributes should still be there for the grid layout
    assert "mol-gallery-grid" in html


def test_render_gallery_html_empty():
    g = build_gallery([], top_k=4, bottom_k=4)
    html = render_gallery_html(g)
    assert "No molecules" in html


def test_gallery_css_constant_is_non_trivial():
    assert isinstance(GALLERY_CSS, str)
    assert ".mol-card" in GALLERY_CSS
    assert ".mol-gallery-grid" in GALLERY_CSS


# ── HNEPResult integration ────────────────────────────────────────────

def _toy_result(with_gallery: bool) -> HNEPResult:
    result = HNEPResult(
        model_name="HybridV1",
        dataset_name="ESOL",
        qct_verdict="Regularizer",
        qct_confidence=0.78,
        probes={
            "surrogation": ProbeResult(
                probe_name="surrogation", primary_score=0.12,
                verdict="REPLACEABLE", confidence=0.85,
            ),
        },
        manifest={"seed": 0},
    )
    if with_gallery:
        result.molecule_records = list(EXAMPLE_RECORDS)
    return result


def test_html_report_omits_gallery_when_no_records():
    result = _toy_result(with_gallery=False)
    html = hnep.render_html_report(result)
    assert "Molecular Chemistry Gallery" not in html


def test_html_report_includes_gallery_when_records_present():
    result = _toy_result(with_gallery=True)
    html = hnep.render_html_report(result)
    assert "Molecular Chemistry Gallery" in html
    # Section labels appear
    assert "Top-K" in html
    assert "Bottom-K" in html
    # SMILES strings are echoed somewhere in the gallery
    assert "CC(=O)O" in html


def test_html_report_accepts_dict_molecule_records():
    """Users should be able to pass plain dicts as well as MoleculeRecord."""
    result = _toy_result(with_gallery=False)
    result.molecule_records = [
        {"smiles": "CCO", "qci": 0.4, "prediction": -1.0, "target": -0.9},
        {"smiles": "CC(=O)O", "qci": 0.9},
    ]
    html = hnep.render_html_report(result)
    assert "Molecular Chemistry Gallery" in html
    assert "0.900" in html  # QCI of the top molecule


# ── top-level exports ─────────────────────────────────────────────────

def test_day4_exports_at_top_level():
    assert hasattr(hnep, "MoleculeRecord")
    assert hasattr(hnep, "build_gallery")
    assert hasattr(hnep, "render_gallery_html")
