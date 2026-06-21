"""ModelInterface — the contract every model adapter must satisfy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np


@dataclass
class Dataset:
    """A bundle of numpy arrays describing the data HNEP probes will see.

    Attributes
    ----------
    inputs
        Per-sample classical features used as auxiliary input to the surrogate
        models. Shape ``(N, d_input)``. For molecular tasks this is typically
        RDKit descriptors; for other domains it is whatever inputs the user
        considers "classical context".
    targets
        Ground-truth target values, shape ``(N,)`` for regression or
        ``(N, k)`` for multi-output. Always in the original (un-normalised)
        scale — adapters that train in a normalised space should denormalise
        before exposing predictions.
    train_idx, val_idx, test_idx
        Index arrays describing the train/validation/test split. Probes operate
        on the test set; surrogates are fit on the train set.
    metadata
        Free-form mapping for anything the adapter wants to attach — sample
        identifiers (e.g. SMILES), per-sample descriptors, etc. HNEP never
        reads required information from here; it is for the user's benefit and
        for nicer reports.
    """

    inputs: np.ndarray
    targets: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelInterface(ABC):
    """Abstract base class every model adapter implements.

    The goal is to be framework-agnostic: HNEP probes call methods on
    ``ModelInterface`` and never touch JAX, PyTorch, or Qiskit directly. To
    plug HNEP into a new framework, subclass this and implement the abstract
    methods.

    The required methods cover three things HNEP needs:

    1. **Prediction with quantum intact** — to baseline performance.
    2. **Quantum output extraction** — to surrogate it.
    3. **Counterfactual prediction** — to measure structural interventions.

    Subclasses may additionally implement optional hooks like
    :meth:`get_classical_embedding` so richer probes (CKA, MI) can work.
    """

    # ── Required methods ─────────────────────────────────────────────

    @abstractmethod
    def predict(self, dataset: Dataset, indices: Optional[Sequence[int]] = None) -> np.ndarray:
        """Run the full trained model on (a subset of) the dataset.

        Parameters
        ----------
        dataset
            The data we are evaluating on.
        indices
            Optional subset of indices into ``dataset``. If ``None``, predict
            on the entire dataset.

        Returns
        -------
        Predictions in the **original** target scale. Shape ``(len(indices),)``
        for single-output or ``(len(indices), k)`` for multi-output.
        """

    @abstractmethod
    def extract_quantum_output(
        self, dataset: Dataset, indices: Optional[Sequence[int]] = None
    ) -> np.ndarray:
        """Extract the quantum branch's per-sample output vector.

        Returns
        -------
        Array of shape ``(len(indices), q_dim)`` where ``q_dim`` is the
        dimensionality of the quantum output (e.g. 4 for a 4-qubit Z-basis
        measurement).
        """

    @abstractmethod
    def predict_with_quantum_override(
        self,
        dataset: Dataset,
        quantum_override: np.ndarray,
        indices: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Run the model substituting a custom quantum output vector.

        Used by the intervention family of probes:

        - ``quantum_override = np.zeros(...)`` → zero-quantum intervention
        - ``quantum_override = noise``         → random-noise intervention
        - ``quantum_override = permuted_q``    → permutation intervention
        - ``quantum_override = constant_q``    → constant-quantum intervention

        Parameters
        ----------
        quantum_override
            Array of shape ``(len(indices), q_dim)`` substituted for the
            adapter's own quantum output at inference.

        Returns
        -------
        Predictions in the original target scale.
        """

    # ── Optional hooks (used by richer probes) ────────────────────────

    def get_classical_embedding(
        self, dataset: Dataset, indices: Optional[Sequence[int]] = None
    ) -> Optional[np.ndarray]:
        """Return the classical branch's embedding for each sample.

        Used by CKA / MI / error-diversity probes. Default returns ``None``;
        the corresponding probes will be skipped with a clear message.

        Returns
        -------
        Array of shape ``(len(indices), c_dim)`` or ``None`` if not supported.
        """
        return None

    def get_quantum_input(
        self, dataset: Dataset, indices: Optional[Sequence[int]] = None
    ) -> Optional[np.ndarray]:
        """Return the compressed quantum input vector for each sample.

        Used by the quantum-input gating analysis. Default returns ``None``.
        """
        return None

    # ── Identity / metadata ──────────────────────────────────────────

    @property
    def name(self) -> str:
        """Display name for this model — shows up in reports and figures."""
        return self.__class__.__name__

    @property
    def quantum_dim(self) -> int:
        """Number of quantum output dimensions. Default infers by inspection."""
        return getattr(self, "_quantum_dim", 0) or 0
