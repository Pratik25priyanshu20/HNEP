"""Internal helpers to convert numpy / pandas objects to JSON-safe forms."""

from __future__ import annotations

from typing import Any

import numpy as np


def simplify(obj: Any) -> Any:
    """Recursively convert numpy / pandas types into JSON-safe ones.

    * numpy scalars → Python scalars
    * numpy arrays  → nested lists (truncated if very large)
    * dicts / lists → simplified element-wise
    * anything else → returned unchanged
    """
    if isinstance(obj, dict):
        return {k: simplify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [simplify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        # Inline arrays up to 200 elements; otherwise truncate.
        if obj.size <= 200:
            return obj.tolist()
        head = obj.flatten()[:100].tolist()
        return {"__truncated_array__": True, "shape": list(obj.shape),
                "first_100": head}
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
