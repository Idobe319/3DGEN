"""Minimal testing utilities shim used by TRELLIS flexicubes import.
Only provides `check_tensor` with the same signature used in TRELLIS.
"""
from typing import Any


def check_tensor(x: Any) -> bool:
    """Basic sanity check for tensor-like objects.
    Returns True if object looks like a numeric sequence / array. This is
    intentionally permissive so it doesn't block TRELLIS from importing.
    """
    try:
        # Numpy arrays and similar
        import numpy as _np
        if isinstance(x, _np.ndarray):
            return True
    except Exception:
        pass

    # duck-typing: has shape or .dtype or is iterable of numbers
    if hasattr(x, "shape") or hasattr(x, "dtype"):
        return True
    try:
        iter(x)
        return True
    except Exception:
        return False
