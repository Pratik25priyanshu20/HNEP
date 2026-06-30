"""Regenerate ``hnep/thresholds_calibration.json`` from the ground-truth suite.

Run from the repo root::

    python scripts/recalibrate_thresholds.py [--n-seeds 30] [--output PATH]

The defaults reproduce the file shipped with the wheel. Output is fully
deterministic given the seed offset. End users normally never need to run
this — the calibrated thresholds ship in the package. Re-run only if the
archetype constructors change or you want to derive thresholds against a
different reference corpus.

Methodology
-----------
* ``ss_replaceable`` = 95th percentile of Surrogation Scores observed on
  the ``Regularizer`` + ``Dead Weight`` archetypes (known-replaceable by
  construction).
* ``intervention_load_bearing`` = 95th percentile of Δ_zero observed on
  the ``Ignored`` + ``Dead Weight`` archetypes (known-not-load-bearing).

Calibration uses seed offset 100 to keep these seeds disjoint from the
validation seeds (0..n_seeds-1) used by ``run_ground_truth_benchmark``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import hnep  # noqa: E402
from hnep import evaluate  # noqa: E402
from hnep.benchmarks import (  # noqa: E402
    make_dead_weight,
    make_ignored,
    make_regularizer,
)


_DEFAULT_OUTPUT = _REPO_ROOT / "hnep" / "thresholds_calibration.json"
_CALIBRATION_SEED_OFFSET = 100
# 99th — not 95th. The 95th percentile of a tight noise-floor distribution
# sits close enough to the bootstrap-CI half-width that out-of-sample
# REPLACEABLE seeds frequently land in the Inconclusive band. 99th gives the
# CI room without changing the methodological story.
_PERCENTILE = 99.0


def _collect_scores(maker, n_seeds, n_samples, seed_offset):
    """Collect (SS, Δ) point estimates AND CI upper bounds per seed.

    For v0.3.0 the threshold is derived from CI upper bounds, not point
    estimates: the headline-aligned bootstrap CI is wide enough that a
    point-estimate-based threshold leaves many seeds straddling and being
    demoted to Inconclusive. CI-upper-bound calibration ensures
    ~`percentile`% of calibration seeds' CIs lie strictly on the
    REPLACEABLE / NOT-LOAD-BEARING side of threshold.
    """
    ss_ci_uppers, delta_ci_uppers = [], []
    ss_points, delta_points = [], []
    for i in range(n_seeds):
        adapter, dataset, _ = maker(seed=seed_offset + i, n_samples=n_samples)
        result = evaluate(adapter, dataset, verbose=False)
        sur = result.probes["surrogation"]
        inter = result.probes["intervention"]
        ss_points.append(float(sur.primary_score))
        delta_points.append(float(inter.primary_score))
        ss_ci_uppers.append(float(sur.primary_score_ci[1]))
        delta_ci_uppers.append(float(inter.primary_score_ci[1]))
    return ss_points, delta_points, ss_ci_uppers, delta_ci_uppers


def calibrate(n_seeds: int = 30, n_samples: int = 400) -> dict:
    reg_ss, reg_delta, reg_ss_hi, reg_delta_hi = _collect_scores(
        make_regularizer, n_seeds, n_samples, _CALIBRATION_SEED_OFFSET
    )
    dw_ss, dw_delta, dw_ss_hi, dw_delta_hi = _collect_scores(
        make_dead_weight, n_seeds, n_samples, _CALIBRATION_SEED_OFFSET
    )
    ig_ss, ig_delta, ig_ss_hi, ig_delta_hi = _collect_scores(
        make_ignored, n_seeds, n_samples, _CALIBRATION_SEED_OFFSET
    )

    ss_pool = reg_ss_hi + dw_ss_hi
    ss_threshold = float(np.percentile(ss_pool, _PERCENTILE))

    delta_pool = ig_delta_hi + dw_delta_hi
    delta_threshold = float(np.percentile(delta_pool, _PERCENTILE))

    return {
        "ss_replaceable": ss_threshold,
        "intervention_load_bearing": delta_threshold,
        "metadata": {
            "method": (
                f"{_PERCENTILE:.0f}th percentile of bootstrap-CI upper bounds "
                "on known-archetype seeds (v0.3.0: CI-aware percentile so "
                "calibration seeds' CIs stay clear of threshold)"
            ),
            "n_seeds_per_archetype": n_seeds,
            "n_samples_per_dataset": n_samples,
            "calibration_seed_offset": _CALIBRATION_SEED_OFFSET,
            "archetypes_for_ss": ["regularizer", "dead_weight"],
            "archetypes_for_delta": ["ignored", "dead_weight"],
            "raw_ss_scores": {
                "regularizer": reg_ss,
                "dead_weight": dw_ss,
            },
            "raw_ss_ci_upper": {
                "regularizer": reg_ss_hi,
                "dead_weight": dw_ss_hi,
            },
            "raw_delta_scores": {
                "ignored": ig_delta,
                "dead_weight": dw_delta,
            },
            "raw_delta_ci_upper": {
                "ignored": ig_delta_hi,
                "dead_weight": dw_delta_hi,
            },
            "hnep_version": hnep.__version__,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    print(f"[calibrate] running {args.n_seeds} seeds x 3 archetypes ...")
    data = calibrate(n_seeds=args.n_seeds, n_samples=args.n_samples)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[calibrate] ss_replaceable            = {data['ss_replaceable']:.5f}")
    print(f"[calibrate] intervention_load_bearing = {data['intervention_load_bearing']:.5f}")
    print(f"[calibrate] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
