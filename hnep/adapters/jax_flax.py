"""JaxFlaxAdapter — for Flax models with a quantum branch.

This adapter expects a Flax-style model object exposing:

  * ``model.predict(params, x)`` → intact prediction (scalar or array)
  * ``model.forward_with_quantum(params, x)`` → ``(prediction, info_dict)``
    where ``info_dict`` contains ``"gnn_embedding"`` and ``"quantum_output"``
  * ``model.extract_quantum_outputs(params, batch)`` → ``(N, q_dim)``

These are exactly the conventions used by HNEP's thesis Hybrid-V1
implementation (``src/models/hybrid/hybrid_model.py``). If your model has a
different surface, write a thin wrapper that conforms — or use
:class:`FunctionalAdapter` / :class:`PrecomputedAdapter` instead.

Optional dependency
-------------------
Loading this module does not import JAX or Flax — they are only required if
you actually instantiate :class:`JaxFlaxAdapter`. Install the extras with::

    pip install "hnep[jax]"
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np

from hnep.adapters.base import Dataset, ModelInterface


GraphFn = Callable[[Dataset, np.ndarray], Sequence[Any]]
"""``(dataset, indices) -> list[graph]`` — user-supplied to map dataset rows
to the per-sample graph dict your Flax model consumes."""


class JaxFlaxAdapter(ModelInterface):
    """Adapter for JAX/Flax hybrid models conforming to the thesis surface.

    Parameters
    ----------
    name
        Display name for reports.
    model
        The trained Flax module (e.g. an instance of ``HybridRegressor``).
    params
        Trained parameters (Flax pytree).
    graph_fn
        Callable ``(dataset, indices) -> list[graph]`` translating dataset rows
        into the per-sample graph objects your model expects.
    quantum_dim
        Output dimensionality of the quantum branch.
    normalize_mean, normalize_std
        Optional target normaliser values — the model's intrinsic predictions
        are de-normalised by ``y * std + mean`` before being returned. Pass
        both as ``None`` if your model already predicts in original scale.
    decoder_override_fn
        Optional callable
        ``(model, params, info, q_override) -> prediction`` implementing the
        post-quantum decoder pass given an override. If omitted, the adapter
        derives it from a default heuristic that runs the model's decoder
        modules manually (see source).
    """

    def __init__(
        self,
        name: str,
        model: Any,
        params: Any,
        graph_fn: GraphFn,
        quantum_dim: int,
        normalize_mean: Optional[float] = None,
        normalize_std: Optional[float] = None,
        decoder_override_fn: Optional[
            Callable[[Any, Any, dict, np.ndarray], np.ndarray]
        ] = None,
    ) -> None:
        self._name = name
        self.model = model
        self.params = params
        self.graph_fn = graph_fn
        self._quantum_dim = quantum_dim
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.decoder_override_fn = decoder_override_fn

    @property
    def name(self) -> str:
        return self._name

    @property
    def quantum_dim(self) -> int:
        return self._quantum_dim

    # ── ModelInterface implementations ──────────────────────────────

    def predict(self, dataset, indices=None):
        idx = self._resolve(indices, dataset)
        graphs = self.graph_fn(dataset, idx)
        preds = np.asarray([
            float(self.model.predict(self.params, g)) for g in graphs
        ])
        return self._denormalise(preds)

    def extract_quantum_output(self, dataset, indices=None):
        idx = self._resolve(indices, dataset)
        graphs = self.graph_fn(dataset, idx)
        rows = []
        for g in graphs:
            _, info = self.model.forward_with_quantum(self.params, g)
            rows.append(np.asarray(info["quantum_output"]).reshape(-1))
        return np.stack(rows, axis=0)

    def predict_with_quantum_override(self, dataset, quantum_override, indices=None):
        idx = self._resolve(indices, dataset)
        graphs = self.graph_fn(dataset, idx)
        if self.decoder_override_fn is not None:
            preds = []
            for g, q in zip(graphs, quantum_override):
                _, info = self.model.forward_with_quantum(self.params, g)
                preds.append(float(self.decoder_override_fn(
                    self.model, self.params, info, np.asarray(q))))
            preds = np.asarray(preds)
        else:
            preds = np.asarray([
                self._default_override(g, np.asarray(q))
                for g, q in zip(graphs, quantum_override)
            ])
        return self._denormalise(preds)

    def get_classical_embedding(self, dataset, indices=None):
        idx = self._resolve(indices, dataset)
        graphs = self.graph_fn(dataset, idx)
        rows = []
        for g in graphs:
            _, info = self.model.forward_with_quantum(self.params, g)
            rows.append(np.asarray(info["gnn_embedding"]).reshape(-1))
        return np.stack(rows, axis=0)

    # ── Default override implementation ─────────────────────────────

    def _default_override(self, graph: Any, q_override: np.ndarray) -> float:
        """Run the model's decoder manually using ``q_override``.

        Falls back to a structural match against typical Flax decoder modules
        (``Dense_X`` names) — replicating the implementation used in
        ``notebooks/14_sprint25_temporal_noise.ipynb``. Override
        ``decoder_override_fn`` if your model has a different surface.
        """
        import jax  # local import — only required when this branch is hit
        import jax.numpy as jnp

        _, info = self.model.forward_with_quantum(self.params, graph)
        gnn_emb = np.asarray(info["gnn_embedding"]).reshape(-1)
        feats = jnp.concatenate(
            [jnp.asarray(gnn_emb), jnp.asarray(q_override)]
        )

        params_dict = self.params["params"] if "params" in self.params else self.params
        dense_keys = sorted([k for k in params_dict.keys() if "Dense" in k])
        # Pick the first Dense layer whose kernel matches our concat shape.
        decoder_keys = None
        for i, k in enumerate(dense_keys):
            kernel = params_dict[k]["kernel"]
            if kernel.shape[0] == feats.shape[0]:
                decoder_keys = dense_keys[i:]
                break
        if decoder_keys is None:
            decoder_keys = dense_keys[-3:]

        x = feats
        for j, key in enumerate(decoder_keys):
            w = params_dict[key]["kernel"]
            b = params_dict[key]["bias"]
            x = x @ w + b
            if j < len(decoder_keys) - 1:
                x = jax.nn.gelu(x)
        return float(jnp.asarray(x).squeeze())

    # ── Helpers ─────────────────────────────────────────────────────

    def _denormalise(self, preds: np.ndarray) -> np.ndarray:
        if self.normalize_mean is None or self.normalize_std is None:
            return preds
        return preds * self.normalize_std + self.normalize_mean

    @staticmethod
    def _resolve(indices, dataset):
        if indices is None:
            return np.arange(len(dataset.targets))
        return np.asarray(indices)
