"""Probe ABC — base class every HNEP probe inherits from."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from hnep.adapters.base import Dataset, ModelInterface
from hnep.results.probe_result import ProbeResult


class Probe(ABC):
    """Abstract base class for evaluation probes.

    A probe takes a :class:`ModelInterface` plus a :class:`Dataset` and returns
    a :class:`ProbeResult` containing a primary score, optional sub-scores,
    confidence intervals, and any visualisations the probe produced.

    Probes are designed to be:

    * **Composable** — usable standalone via ``probe.run(model, dataset)`` or
      collected together inside :func:`hnep.evaluate`.
    * **Reproducible** — every probe accepts a ``seed`` and emits a manifest
      fragment so its run can be replayed.
    * **Honest** — every probe reports confidence intervals and surfaces an
      ``"inconclusive"`` verdict when the signal is too weak to characterise.
    """

    #: Short identifier used in reports and manifests.
    name: str = "probe"

    #: Human-readable description.
    description: str = ""

    def __init__(self, seed: int = 42, **kwargs: Any) -> None:
        self.seed = seed
        # Subclasses may store hyperparameters in self._config for the manifest.
        self._config: dict = {"seed": seed, **kwargs}

    @abstractmethod
    def run(
        self,
        model: ModelInterface,
        dataset: Dataset,
        verbose: bool = False,
    ) -> ProbeResult:
        """Run the probe and return the result.

        Parameters
        ----------
        model
            The wrapped model to evaluate.
        dataset
            The data to evaluate on. The probe will use the test split.
        verbose
            If ``True``, print progress to stdout.
        """

    @property
    def config(self) -> dict:
        """The probe's configuration — included in manifests for replay."""
        return dict(self._config)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._config})"
