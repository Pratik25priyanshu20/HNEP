"""Seed management and reproducibility helpers."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python's ``random``, ``numpy``, and ``PYTHONHASHSEED``.

    Frameworks (JAX, PyTorch) handle their own RNGs; the adapter layer is
    responsible for seeding them. This function covers the shared substrate.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
