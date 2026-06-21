"""PrecomputedAdapter — for users who already extracted everything.

Many HNEP users already have:

* the trained model's predictions on the test set,
* the quantum-branch outputs per sample,
* the classical embeddings,
* and a callable that maps ``(gnn_emb, quantum_output) -> prediction``.

PrecomputedAdapter wraps those numpy arrays + a single decoder function so
HNEP can run probes without ever touching the original framework. This is
especially convenient when the user trained on a GPU server and now wants to
analyse the cached extractions locally on a laptop.

Example
-------

>>> import hnep, numpy as np
>>> adapter = hnep.PrecomputedAdapter(
...     name="MyModel",
...     gnn_embeddings=gnn_emb,           # shape (N, c_dim)
...     quantum_outputs=q_out,            # shape (N, q_dim)
...     intact_predictions=preds,         # shape (N,)
...     decoder_fn=my_decoder,            # callable(gnn_emb, q_out) -> preds
... )
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from hnep.adapters.base import Dataset, ModelInterface


DecoderFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


class PrecomputedAdapter(ModelInterface):
    """Adapter built from cached arrays + a decoder callable.

    Parameters
    ----------
    name
        Display name.
    gnn_embeddings
        Classical embedding per sample, shape ``(N, c_dim)``.
    quantum_outputs
        Quantum branch output per sample, shape ``(N, q_dim)``.
    intact_predictions
        Predictions from the full intact model in the original target scale,
        shape ``(N,)``.
    decoder_fn
        ``(gnn_emb_batch, q_batch) -> predictions``. Used by the intervention
        probe to compute counterfactual predictions with overridden quantum.
    quantum_inputs
        Optional ``(N, q_input_dim)`` — used by the quantum-input gating
        analysis (Phase 2+ extensions).
    """

    def __init__(
        self,
        name: str,
        gnn_embeddings: np.ndarray,
        quantum_outputs: np.ndarray,
        intact_predictions: np.ndarray,
        decoder_fn: DecoderFn,
        quantum_inputs: Optional[np.ndarray] = None,
    ) -> None:
        gnn_embeddings = np.asarray(gnn_embeddings)
        quantum_outputs = np.asarray(quantum_outputs)
        intact_predictions = np.asarray(intact_predictions).reshape(-1)
        n = intact_predictions.shape[0]
        if gnn_embeddings.shape[0] != n:
            raise ValueError(
                f"gnn_embeddings rows ({gnn_embeddings.shape[0]}) "
                f"do not match intact_predictions ({n})."
            )
        if quantum_outputs.shape[0] != n:
            raise ValueError(
                f"quantum_outputs rows ({quantum_outputs.shape[0]}) "
                f"do not match intact_predictions ({n})."
            )

        self._name = name
        self.gnn_embeddings = gnn_embeddings
        self.quantum_outputs = quantum_outputs
        self.intact_predictions = intact_predictions
        self.decoder_fn = decoder_fn
        self.quantum_inputs = quantum_inputs
        self._quantum_dim = quantum_outputs.shape[1]

    @property
    def name(self) -> str:
        return self._name

    @property
    def quantum_dim(self) -> int:
        return self._quantum_dim

    # ── ModelInterface ──────────────────────────────────────────────

    def predict(self, dataset: Dataset, indices: Optional[Sequence[int]] = None):
        idx = self._resolve(indices, dataset)
        return self.intact_predictions[idx]

    def extract_quantum_output(self, dataset, indices=None):
        idx = self._resolve(indices, dataset)
        return self.quantum_outputs[idx]

    def predict_with_quantum_override(self, dataset, quantum_override, indices=None):
        idx = self._resolve(indices, dataset)
        gnn = self.gnn_embeddings[idx]
        return np.asarray(self.decoder_fn(gnn, np.asarray(quantum_override)))

    def get_classical_embedding(self, dataset, indices=None):
        idx = self._resolve(indices, dataset)
        return self.gnn_embeddings[idx]

    def get_quantum_input(self, dataset, indices=None):
        if self.quantum_inputs is None:
            return None
        idx = self._resolve(indices, dataset)
        return self.quantum_inputs[idx]

    @staticmethod
    def _resolve(indices, dataset):
        if indices is None:
            return np.arange(len(dataset.targets))
        return np.asarray(indices)
