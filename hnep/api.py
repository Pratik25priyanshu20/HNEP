"""HNEP public API — the top-level :func:`evaluate` function."""

from __future__ import annotations

import platform
import sys
import time
from datetime import datetime, timezone
from typing import List, Optional, Sequence

from hnep import __version__
from hnep.adapters.base import Dataset, ModelInterface
from hnep.classifiers.qct import QCTClassifier, QCTVerdict
from hnep.probes.base import Probe
from hnep.probes.intervention import InterventionProbe
from hnep.probes.surrogation import SurrogationProbe
from hnep.results.hnep_result import HNEPResult
from hnep.thresholds import DEFAULT_THRESHOLDS, Thresholds


def evaluate(
    model: ModelInterface,
    dataset: Dataset,
    probes: Optional[Sequence[Probe]] = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    verbose: bool = False,
) -> HNEPResult:
    """Run a full HNEP evaluation on ``model``.

    Parameters
    ----------
    model
        A model wrapped in :class:`ModelInterface`.
    dataset
        The evaluation data, packaged as :class:`Dataset`.
    probes
        Optional explicit list of probes. If ``None``, the v0.1.0 default
        battery (surrogation + intervention) is used.
    thresholds
        QCT classification thresholds.
    verbose
        Print probe-level progress to stdout.
    """
    if probes is None:
        probes = [
            SurrogationProbe(thresholds=thresholds),
            InterventionProbe(thresholds=thresholds),
        ]

    result = HNEPResult(
        model_name=model.name,
        dataset_name=dataset.metadata.get("name", "unnamed_dataset"),
        qct_verdict=QCTVerdict.INCONCLUSIVE.value,
        qct_confidence=0.0,
    )

    t0 = time.perf_counter()
    for probe in probes:
        if verbose:
            print(f"[hnep] running {probe.name} …")
        probe_t0 = time.perf_counter()
        probe_result = probe.run(model, dataset, verbose=verbose)
        probe_t1 = time.perf_counter()
        probe_result.details["elapsed_seconds"] = round(probe_t1 - probe_t0, 3)
        result.probes[probe.name] = probe_result

    # If we have both probes, run the QCT classifier
    sur = result.probes.get("surrogation")
    inter = result.probes.get("intervention")
    if sur is not None and inter is not None:
        verdict = QCTClassifier(thresholds=thresholds).classify(sur, inter)
        result.qct_verdict = verdict.value
        # Aggregate confidence as the minimum of the two probes' confidences,
        # which is conservative — verdict is only as confident as its weakest leg.
        result.qct_confidence = float(min(sur.confidence, inter.confidence))

    # Manifest for reproducibility
    result.manifest = {
        "hnep_version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "thresholds": {
            "ss_replaceable": thresholds.ss_replaceable,
            "intervention_load_bearing": thresholds.intervention_load_bearing,
        },
        "probes_run": [p.name for p in probes],
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
    }

    return result
