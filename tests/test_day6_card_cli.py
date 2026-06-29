"""Day-6 tests: HNEPCard + `hnep card`/`hnep compare` CLI."""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

import pytest

import hnep
from hnep.card import (
    HNEPCard,
    compare_cards_html,
    compare_cards_markdown,
    compare_cards_text,
    load_result_from_json,
)
from hnep.cli import build_parser, main
from hnep.results.hnep_result import HNEPResult
from hnep.results.probe_result import ProbeResult


# ── fixtures ──────────────────────────────────────────────────────────

def make_result(model="HybridV1", dataset="ESOL", verdict="Regularizer"):
    return HNEPResult(
        model_name=model,
        dataset_name=dataset,
        qct_verdict=verdict,
        qct_confidence=0.78,
        probes={
            "surrogation": ProbeResult(
                probe_name="surrogation",
                primary_score=0.12,
                primary_score_ci=(0.08, 0.16),
                verdict="REPLACEABLE",
                confidence=0.85,
            ),
            "intervention": ProbeResult(
                probe_name="intervention",
                primary_score=-0.06,
                verdict="NOT-LOAD-BEARING",
                confidence=0.7,
            ),
            "error_diversity": ProbeResult(
                probe_name="error_diversity",
                primary_score=0.28,
                verdict="DIVERSE",
                confidence=0.62,
            ),
        },
        manifest={"seed": 0},
        notes=["this is a test note"],
    )


# ── HNEPCard text/markdown/html ──────────────────────────────────────

def test_card_to_text_contains_model_dataset_and_verdict():
    card = HNEPCard(make_result())
    text = card.to_text()
    assert "HybridV1" in text
    assert "ESOL" in text
    assert "Regularizer" in text
    # All probes are listed
    assert "surrogation" in text
    assert "intervention" in text
    assert "error_diversity" in text
    # Notes shown
    assert "test note" in text


def test_card_to_text_box_drawing_balanced():
    """Top and bottom borders should match the configured width."""
    card = HNEPCard(make_result())
    text = card.to_text(width=70)
    lines = text.splitlines()
    # Box border characters used
    assert lines[0].startswith("┌") and lines[0].endswith("┐")
    assert lines[-1].startswith("└") and lines[-1].endswith("┘")
    assert len(lines[0]) == 70
    assert len(lines[-1]) == 70


def test_card_to_markdown_is_valid_markdown_table():
    md = HNEPCard(make_result()).to_markdown()
    assert "### HNEP Card" in md
    # Table header
    assert "| Probe | Score | 95% CI | Verdict | Confidence |" in md
    # Pipes count per row should be 6 (5 columns → 6 separators)
    rows = [l for l in md.splitlines() if l.startswith("|")]
    for r in rows:
        assert r.count("|") == 6


def test_card_to_html_contains_verdict_class_and_probes():
    html = HNEPCard(make_result(verdict="Genuine")).to_html()
    assert "hc-v-genuine" in html
    assert "<table>" in html
    assert "surrogation" in html
    # CSS injected by default
    assert ".hnep-card" in html


def test_card_to_html_can_skip_css():
    html = HNEPCard(make_result()).to_html(include_css=False)
    assert ".hnep-card" not in html        # rule
    assert 'class="hnep-card"' in html     # element class still present


# ── compare ──────────────────────────────────────────────────────────

def test_compare_text_has_all_models_and_probes():
    a = make_result(model="A", verdict="Regularizer")
    b = make_result(model="B", verdict="Genuine")
    text = compare_cards_text([a, b])
    assert "A" in text and "B" in text
    assert "Regularizer" in text
    assert "Genuine" in text
    assert "surrogation" in text


def test_compare_markdown_pipes_are_balanced():
    a = make_result(model="A")
    b = make_result(model="B")
    md = compare_cards_markdown([a, b])
    rows = [l for l in md.splitlines() if l.startswith("|")]
    # Header has 3 columns → 4 pipes; every row should match.
    for r in rows:
        assert r.count("|") == rows[0].count("|")


def test_compare_html_includes_each_model():
    a = make_result(model="ModelA")
    b = make_result(model="ModelB")
    html = compare_cards_html([a, b])
    assert "ModelA" in html
    assert "ModelB" in html


def test_compare_handles_missing_probes_in_some_models():
    a = make_result(model="A")
    b = make_result(model="B")
    # Remove a probe from B
    b.probes.pop("error_diversity")
    text = compare_cards_text([a, b])
    # error_diversity row should still exist with an em-dash for B
    assert "error_diversity" in text
    # at least one em-dash present
    assert "—" in text


def test_compare_empty_returns_marker():
    assert "(no" in compare_cards_text([])
    assert "_no" in compare_cards_markdown([])
    assert "no results" in compare_cards_html([])


# ── load_result_from_json round-trip ─────────────────────────────────

def test_load_result_from_json_round_trip(tmp_path: Path):
    r = make_result()
    p = tmp_path / "result.json"
    r.to_json(p)

    r2 = load_result_from_json(p)
    assert r2.model_name == r.model_name
    assert r2.dataset_name == r.dataset_name
    assert r2.qct_verdict == r.qct_verdict
    assert pytest.approx(r2.qct_confidence) == r.qct_confidence
    assert set(r2.probes.keys()) == set(r.probes.keys())
    # CI must round-trip as a tuple (not a list) because consumers assume
    # `if probe.primary_score_ci:` plus indexable access.
    assert isinstance(r2.probes["surrogation"].primary_score_ci, tuple)
    assert r2.probes["surrogation"].primary_score_ci == (0.08, 0.16)


# ── CLI ──────────────────────────────────────────────────────────────

def _capture_stdout(fn) -> str:
    buf = StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        fn()
    finally:
        sys.stdout = old
    return buf.getvalue()


def test_cli_help_runs():
    parser = build_parser()
    # smoke — argparse raises SystemExit on --help, but build_parser shouldn't
    assert parser.prog == "hnep"


def test_cli_card_to_stdout(tmp_path: Path):
    r = make_result()
    p = tmp_path / "result.json"
    r.to_json(p)

    out = _capture_stdout(lambda: main(["card", str(p)]))
    assert "HybridV1" in out
    assert "Regularizer" in out


def test_cli_card_writes_to_output_path(tmp_path: Path):
    r = make_result()
    p = tmp_path / "result.json"
    r.to_json(p)
    out_html = tmp_path / "card.html"
    rc = main(["card", str(p), "--format", "html", "-o", str(out_html)])
    assert rc == 0
    assert out_html.exists()
    text = out_html.read_text()
    assert 'class="hnep-card"' in text


def test_cli_compare_requires_two_files(tmp_path: Path):
    r = make_result()
    p = tmp_path / "result.json"
    r.to_json(p)
    rc = main(["compare", str(p)])
    assert rc == 2


def test_cli_compare_text(tmp_path: Path):
    r_a = make_result(model="A")
    r_b = make_result(model="B", verdict="Genuine")
    pa = tmp_path / "a.json"; r_a.to_json(pa)
    pb = tmp_path / "b.json"; r_b.to_json(pb)

    out = _capture_stdout(lambda: main(["compare", str(pa), str(pb)]))
    assert "A" in out and "B" in out
    assert "Genuine" in out


def test_cli_card_missing_file_returns_error_code(tmp_path: Path):
    rc = main(["card", str(tmp_path / "nope.json")])
    assert rc == 2


def test_cli_top_level_exports():
    assert hasattr(hnep, "HNEPCard")
    assert hasattr(hnep, "compare_cards_text")
    assert hasattr(hnep, "load_result_from_json")
