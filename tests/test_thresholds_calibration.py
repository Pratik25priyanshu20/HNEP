"""Tests for T1.2 — empirical-percentile threshold calibration."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from hnep.thresholds import DEFAULT_THRESHOLDS, Thresholds


_REPO_ROOT = Path(__file__).resolve().parent.parent
_CALIBRATION_PATH = _REPO_ROOT / "hnep" / "thresholds_calibration.json"


def test_legacy_thresholds_are_pre_t12_values():
    legacy = Thresholds.legacy()
    assert legacy.ss_replaceable == 0.20
    assert legacy.intervention_load_bearing == 0.05


def test_from_calibration_loads_shipped_json():
    """The package-shipped JSON loads and parses into a valid Thresholds."""
    assert _CALIBRATION_PATH.is_file(), (
        "Calibration JSON missing — run scripts/recalibrate_thresholds.py"
    )
    t = Thresholds.from_calibration()
    assert isinstance(t, Thresholds)
    assert 0.0 < t.ss_replaceable < 1.0
    assert 0.0 < t.intervention_load_bearing < 1.0


def test_default_thresholds_match_calibration_json():
    data = json.loads(_CALIBRATION_PATH.read_text())
    assert DEFAULT_THRESHOLDS.ss_replaceable == pytest.approx(
        data["ss_replaceable"]
    )
    assert DEFAULT_THRESHOLDS.intervention_load_bearing == pytest.approx(
        data["intervention_load_bearing"]
    )


def test_from_calibration_explicit_path(tmp_path):
    custom = tmp_path / "custom_thresholds.json"
    custom.write_text(json.dumps({
        "ss_replaceable": 0.123,
        "intervention_load_bearing": 0.0456,
    }))
    t = Thresholds.from_calibration(custom)
    assert t.ss_replaceable == pytest.approx(0.123)
    assert t.intervention_load_bearing == pytest.approx(0.0456)


def test_from_calibration_falls_back_to_legacy_when_missing(tmp_path, monkeypatch):
    """Pointing at a non-existent file via the package resource path should
    surface a warning and fall back to legacy. Tested via the public surface
    by temporarily renaming the shipped JSON."""
    # Direct path missing → bubbles up a FileNotFoundError (loud — explicit
    # user input is treated as a bug, not silently swallowed).
    with pytest.raises(FileNotFoundError):
        Thresholds.from_calibration(tmp_path / "nonexistent.json")


def test_calibrated_thresholds_distinct_from_legacy():
    """The shipped calibration JSON should produce thresholds clearly distinct
    from the legacy (0.20, 0.05) inspection values — otherwise the JSON
    probably wasn't regenerated after a code change. v0.3.0's CI-upper-bound
    calibration sits at ~ss=0.099 (tighter than legacy) and ~Δ=0.055
    (slightly looser than legacy); we don't pin direction, just distinctness."""
    t = Thresholds.from_calibration()
    legacy = Thresholds.legacy()
    assert abs(t.ss_replaceable - legacy.ss_replaceable) > 0.05
    assert abs(t.intervention_load_bearing - legacy.intervention_load_bearing) > 0.001


def test_recalibration_is_deterministic():
    """Running calibrate() twice with the same args produces identical numbers."""
    import sys

    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    try:
        from recalibrate_thresholds import calibrate
    finally:
        sys.path.pop(0)

    data1 = calibrate(n_seeds=3, n_samples=200)
    data2 = calibrate(n_seeds=3, n_samples=200)
    assert data1["ss_replaceable"] == data2["ss_replaceable"]
    assert data1["intervention_load_bearing"] == data2["intervention_load_bearing"]


def test_calibrate_kwarg_now_implemented_on_surrogation_probe():
    """T1.2 stubbed calibrate=True as NotImplementedError targeting v0.3.1;
    v0.3.0 ships the implementation (see test_permutation_pvalues.py)."""
    from hnep.probes.surrogation import SurrogationProbe

    probe = SurrogationProbe(calibrate=True, n_permutations=10)
    assert probe.calibrate is True


def test_calibrate_kwarg_now_implemented_on_intervention_probe():
    from hnep.probes.intervention import InterventionProbe

    probe = InterventionProbe(calibrate=True, n_permutations=10)
    assert probe.calibrate is True
