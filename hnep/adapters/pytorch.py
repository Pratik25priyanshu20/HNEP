"""PyTorchAdapter — skeleton + minimum-working example for PyTorch models.

The PyTorch ecosystem doesn't yet have a single "hybrid model" convention as
clean as the JAX/Flax thesis stack, so this adapter is intentionally a
template that users subclass for their specific architecture.

Two ways to use it
------------------

**Option A — Subclass and override hooks.** Inherit from
:class:`PyTorchAdapter` and implement:

  * ``_forward_intact(self, batch)`` → tensor of predictions
  * ``_forward_with_override(self, batch, q_override)`` → predictions using
    a substituted quantum output
  * ``_extract_quantum_output(self, batch)`` → tensor of quantum outputs
  * (optional) ``_extract_classical_embedding(self, batch)``

The base class handles tensor↔numpy conversion, indexing, and normalisation.

**Option B — Wrap your model in** :class:`FunctionalAdapter` **instead.**
For many users that's simpler than subclassing.

Optional dependency
-------------------
This module imports ``torch`` lazily — only when :class:`PyTorchAdapter` is
instantiated. Install with::

    pip install "hnep[pytorch]"
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from hnep.adapters.base import Dataset, ModelInterface


class PyTorchAdapter(ModelInterface):
    """Skeleton adapter for PyTorch-style models.

    Subclasses override the protected ``_forward_*`` and ``_extract_*`` hooks.

    Parameters
    ----------
    name
        Display name.
    model
        The trained ``torch.nn.Module``. Will be put in ``.eval()`` mode.
    quantum_dim
        Quantum output dimensionality.
    device
        Optional device string ("cpu", "cuda", "mps", ...). If ``None``, the
        model's current device is used.
    normalize_mean, normalize_std
        Optional inverse-normalisation for predictions.
    """

    def __init__(
        self,
        name: str,
        model: Any,
        quantum_dim: int,
        device: Optional[str] = None,
        normalize_mean: Optional[float] = None,
        normalize_std: Optional[float] = None,
    ) -> None:
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PyTorchAdapter requires torch. Install with `pip install \"hnep[pytorch]\"`."
            ) from exc

        self._torch = torch
        self._name = name
        self.model = model.eval()
        self._quantum_dim = quantum_dim
        self.device = device or next(model.parameters()).device
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

    @property
    def name(self) -> str:
        return self._name

    @property
    def quantum_dim(self) -> int:
        return self._quantum_dim

    # ── Public ModelInterface ───────────────────────────────────────

    def predict(self, dataset, indices=None):
        idx = self._resolve(indices, dataset)
        with self._torch.no_grad():
            batch = self._build_batch(dataset, idx)
            preds = self._forward_intact(batch)
        return self._post_predict(preds)

    def extract_quantum_output(self, dataset, indices=None):
        idx = self._resolve(indices, dataset)
        with self._torch.no_grad():
            batch = self._build_batch(dataset, idx)
            qout = self._extract_quantum_output(batch)
        return self._to_numpy(qout)

    def predict_with_quantum_override(self, dataset, quantum_override, indices=None):
        idx = self._resolve(indices, dataset)
        with self._torch.no_grad():
            batch = self._build_batch(dataset, idx)
            qovr = self._torch.as_tensor(
                quantum_override, dtype=self._torch.float32, device=self.device
            )
            preds = self._forward_with_override(batch, qovr)
        return self._post_predict(preds)

    def get_classical_embedding(self, dataset, indices=None):
        idx = self._resolve(indices, dataset)
        with self._torch.no_grad():
            batch = self._build_batch(dataset, idx)
            emb = self._extract_classical_embedding(batch)
        return None if emb is None else self._to_numpy(emb)

    # ── Hooks to override ────────────────────────────────────────────

    def _build_batch(self, dataset: Dataset, indices: np.ndarray) -> Any:
        """Default: return a torch.Tensor of ``dataset.inputs[indices]``.
        Override to assemble a more complex batch object (e.g. PyG Data)."""
        return self._torch.as_tensor(
            dataset.inputs[indices], dtype=self._torch.float32, device=self.device
        )

    def _forward_intact(self, batch: Any):
        """Default: ``self.model(batch)``."""
        return self.model(batch)

    def _forward_with_override(self, batch: Any, q_override):
        """Subclasses must implement this — the base method raises."""
        raise NotImplementedError(
            "Override _forward_with_override(self, batch, q_override) in your "
            "PyTorchAdapter subclass — HNEP cannot know your model's structure."
        )

    def _extract_quantum_output(self, batch: Any):
        """Subclasses must implement this — the base method raises."""
        raise NotImplementedError(
            "Override _extract_quantum_output(self, batch) in your "
            "PyTorchAdapter subclass — HNEP cannot know where your quantum "
            "branch's output lives in the graph."
        )

    def _extract_classical_embedding(self, batch: Any):
        """Optional override — return ``None`` (default) to skip CKA/MI."""
        return None

    # ── Helpers ──────────────────────────────────────────────────────

    def _post_predict(self, preds_tensor) -> np.ndarray:
        preds = self._to_numpy(preds_tensor).reshape(-1)
        if self.normalize_mean is None or self.normalize_std is None:
            return preds
        return preds * self.normalize_std + self.normalize_mean

    def _to_numpy(self, tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    @staticmethod
    def _resolve(indices, dataset):
        if indices is None:
            return np.arange(len(dataset.targets))
        return np.asarray(indices)
