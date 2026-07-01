"""v0.4 Core + Diagnostics tiering — namespace + deprecation shim + QCT insulation."""

from __future__ import annotations

import inspect
import warnings

import pytest

import hnep


_DIAGNOSTIC_NAMES = ("NoiseProbe", "TemporalProbe", "ErrorDiversityProbe")


def test_diagnostics_subpackage_exports_all_three_probes():
    import hnep.diagnostics as diag

    for name in _DIAGNOSTIC_NAMES:
        assert hasattr(diag, name), f"hnep.diagnostics.{name} missing"
        assert name in diag.__all__


def test_diagnostics_subpackage_exports_noise_helpers():
    import hnep.diagnostics as diag

    for helper in ("gaussian_noise", "depolarizing_approx", "bit_flip_noise"):
        assert hasattr(diag, helper), (
            f"hnep.diagnostics.{helper} noise helper missing"
        )


def test_diagnostics_namespace_imports_do_not_warn():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from hnep.diagnostics import NoiseProbe, TemporalProbe  # noqa: F401
        from hnep.diagnostics import ErrorDiversityProbe  # noqa: F401
    diag_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert not diag_warnings, (
        f"Unexpected DeprecationWarning from hnep.diagnostics: "
        f"{[str(x.message) for x in diag_warnings]}"
    )


@pytest.mark.parametrize("name", _DIAGNOSTIC_NAMES)
def test_top_level_import_emits_deprecation_warning(name):
    """Backward-compat shim — hnep.NoiseProbe still resolves, but warns."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cls = getattr(hnep, name)
    assert cls is not None
    dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert dep, f"hnep.{name} did not emit a DeprecationWarning"
    msg = str(dep[0].message)
    assert f"hnep.diagnostics.{name}" in msg
    assert "supplementary evidence" in msg


def test_top_level_import_resolves_to_the_same_class_as_diagnostics_namespace():
    import hnep.diagnostics as diag

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert hnep.NoiseProbe is diag.NoiseProbe
        assert hnep.TemporalProbe is diag.TemporalProbe
        assert hnep.ErrorDiversityProbe is diag.ErrorDiversityProbe


def test_hnep_top_level_getattr_still_raises_on_unknown_name():
    with pytest.raises(AttributeError):
        hnep.DefinitelyNotAProbe


def test_qct_classifier_signature_excludes_diagnostic_probes():
    """The classifier signature must be exactly (surrogation, intervention,
    representation). Any diagnostic probe name in the signature would allow
    silent gating by a null-result probe."""
    from hnep.classifiers.qct import QCTClassifier

    sig = inspect.signature(QCTClassifier.classify)
    param_names = set(sig.parameters.keys())
    for diag_kwarg in ("noise", "temporal", "error_diversity"):
        assert diag_kwarg not in param_names, (
            f"QCTClassifier.classify accepts {diag_kwarg} — it must not."
        )
    assert param_names == {"self", "surrogation", "intervention", "representation"}


def test_evaluate_verdict_unaffected_by_diagnostic_probes_in_probe_list():
    """Adding diagnostic probes to the evaluate() probe list must not
    change qct_verdict. This is the runtime version of the insulation
    invariant — the QCT verdict is a function of core probes only."""
    from hnep.benchmarks import make_genuine
    from hnep.diagnostics import NoiseProbe
    from hnep.probes.intervention import InterventionProbe
    from hnep.probes.surrogation import SurrogationProbe

    adapter, dataset, _ = make_genuine(seed=0, n_samples=300)
    sur = SurrogationProbe(n_bootstrap=100)
    inter = InterventionProbe(n_bootstrap=100)
    noise = NoiseProbe(noise_levels=(0.01, 0.05), noise_type="gaussian")

    without_diag = hnep.evaluate(adapter, dataset, probes=[sur, inter])
    with_diag = hnep.evaluate(adapter, dataset, probes=[sur, inter, noise])
    assert without_diag.qct_verdict == with_diag.qct_verdict
    assert "noise" in with_diag.probes
