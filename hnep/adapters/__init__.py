"""Adapters — framework-agnostic model interfaces.

A user wraps their trained hybrid model in an adapter so HNEP can probe it
without caring whether the underlying framework is JAX, PyTorch, Qiskit, etc.

The base class is :class:`ModelInterface`. Pre-built adapters:

* :class:`FunctionalAdapter` — most general; user supplies three callables.
* :class:`PrecomputedAdapter` — wraps cached arrays + a decoder function.
* :class:`JaxFlaxAdapter` — Flax models conforming to the thesis surface.
* :class:`PyTorchAdapter` — skeleton template for PyTorch users.
"""

from hnep.adapters.base import Dataset, ModelInterface
from hnep.adapters.functional import FunctionalAdapter
from hnep.adapters.precomputed import PrecomputedAdapter

# Heavier adapters import only when accessed — keep optional deps optional.
__all__ = [
    "Dataset",
    "ModelInterface",
    "FunctionalAdapter",
    "PrecomputedAdapter",
    "JaxFlaxAdapter",
    "PyTorchAdapter",
]


def __getattr__(name: str):
    if name == "JaxFlaxAdapter":
        from hnep.adapters.jax_flax import JaxFlaxAdapter
        return JaxFlaxAdapter
    if name == "PyTorchAdapter":
        from hnep.adapters.pytorch import PyTorchAdapter
        return PyTorchAdapter
    raise AttributeError(f"module 'hnep.adapters' has no attribute {name!r}")
