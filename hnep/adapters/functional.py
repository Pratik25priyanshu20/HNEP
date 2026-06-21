"""FunctionalAdapter — the most general adapter.

A user wraps any model by supplying three callables, regardless of framework.
This is what we recommend for first-time users and for one-off evaluations
where you don't want to invest in a framework-specific adapter class.

Example
-------

>>> def predict(dataset, indices):
...     return my_model.predict(dataset.inputs[indices])
>>> def extract_quantum(dataset, indices):
...     return my_model.quantum_output(dataset.inputs[indices])
>>> def predict_override(dataset, q_override, indices):
...     return my_model.predict_with_quantum(dataset.inputs[indices], q_override)
>>> adapter = hnep.FunctionalAdapter(
...     name="my_model",
...     predict_fn=predict,
...     extract_quantum_fn=extract_quantum,
...     predict_with_override_fn=predict_override,
...     quantum_dim=4,
... )
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from hnep.adapters.base import Dataset, ModelInterface


PredictFn = Callable[[Dataset, Optional[Sequence[int]]], np.ndarray]
ExtractFn = Callable[[Dataset, Optional[Sequence[int]]], np.ndarray]
OverrideFn = Callable[
    [Dataset, np.ndarray, Optional[Sequence[int]]], np.ndarray
]


class FunctionalAdapter(ModelInterface):
    """Adapter that delegates to user-supplied callables.

    Parameters
    ----------
    name
        Display name for reports.
    predict_fn
        ``(dataset, indices) -> predictions`` in the original target scale.
    extract_quantum_fn
        ``(dataset, indices) -> quantum_outputs`` of shape ``(N, q_dim)``.
    predict_with_override_fn
        ``(dataset, q_override, indices) -> predictions`` with the supplied
        quantum override substituted at inference.
    quantum_dim
        Quantum output dimensionality.
    get_classical_embedding_fn
        Optional — used by CKA/MI probes.
    get_quantum_input_fn
        Optional — used by quantum-input analysis.
    """

    def __init__(
        self,
        name: str,
        predict_fn: PredictFn,
        extract_quantum_fn: ExtractFn,
        predict_with_override_fn: OverrideFn,
        quantum_dim: int,
        get_classical_embedding_fn: Optional[ExtractFn] = None,
        get_quantum_input_fn: Optional[ExtractFn] = None,
    ) -> None:
        self._name = name
        self._predict = predict_fn
        self._extract = extract_quantum_fn
        self._override = predict_with_override_fn
        self._classical_emb = get_classical_embedding_fn
        self._q_input = get_quantum_input_fn
        self._quantum_dim = quantum_dim

    @property
    def name(self) -> str:
        return self._name

    @property
    def quantum_dim(self) -> int:
        return self._quantum_dim

    def predict(self, dataset, indices=None):
        return np.asarray(self._predict(dataset, indices))

    def extract_quantum_output(self, dataset, indices=None):
        return np.asarray(self._extract(dataset, indices))

    def predict_with_quantum_override(self, dataset, quantum_override, indices=None):
        return np.asarray(self._override(dataset, quantum_override, indices))

    def get_classical_embedding(self, dataset, indices=None):
        if self._classical_emb is None:
            return None
        return np.asarray(self._classical_emb(dataset, indices))

    def get_quantum_input(self, dataset, indices=None):
        if self._q_input is None:
            return None
        return np.asarray(self._q_input(dataset, indices))
