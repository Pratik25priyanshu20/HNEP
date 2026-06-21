"""ProbeResult — uniform output container for any HNEP probe."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ProbeResult:
    """Standardised return type for any :class:`hnep.Probe`.

    Attributes
    ----------
    probe_name
        Identifier of the probe that produced this result (e.g. ``"surrogation"``).
    primary_score
        Headline scalar score (e.g. Surrogation Score, intervention Δ).
    primary_score_ci
        ``(lower, upper)`` 95% confidence interval for ``primary_score``.
        ``None`` if the probe could not estimate one.
    verdict
        Short label produced by the probe ("REPLACEABLE", "LOAD-BEARING",
        "INCONCLUSIVE", ...). Probes report inconclusive verdicts honestly.
    confidence
        Float in ``[0, 1]`` expressing how robust the verdict is — typically
        derived from the bootstrap distribution or seed variance.
    details
        Free-form mapping holding per-sample arrays, per-sub-probe scores,
        intermediate computations, and anything else the probe wants to
        expose for downstream analysis or visualisation.
    config
        The configuration that produced this result, copied from
        :attr:`Probe.config`. Goes into the manifest for replay.
    notes
        Human-readable notes the probe wants to surface (warnings, caveats).
    """

    probe_name: str
    primary_score: float
    primary_score_ci: Optional[Tuple[float, float]] = None
    verdict: str = "unknown"
    confidence: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        ci = ""
        if self.primary_score_ci is not None:
            lo, hi = self.primary_score_ci
            ci = f" CI=[{lo:.3f}, {hi:.3f}]"
        return (
            f"ProbeResult({self.probe_name!r}: "
            f"score={self.primary_score:.3f}{ci} "
            f"verdict={self.verdict!r} confidence={self.confidence:.2f})"
        )

    def as_dict(self) -> Dict[str, Any]:
        """JSON-serialisable summary (drops numpy arrays in ``details``)."""
        from hnep.utils.serialise import simplify
        return {
            "probe_name": self.probe_name,
            "primary_score": float(self.primary_score),
            "primary_score_ci": (
                list(self.primary_score_ci) if self.primary_score_ci else None
            ),
            "verdict": self.verdict,
            "confidence": float(self.confidence),
            "details": simplify(self.details),
            "config": simplify(self.config),
            "notes": list(self.notes),
        }
