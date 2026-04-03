from __future__ import annotations

import numpy as np

def sliding_window_embedding(signal, dimension: int = 3, delay: int = 1):
    """Simple Takens-style embedding placeholder."""
    x = np.asarray(signal, dtype=float)
    if dimension <= 0 or delay <= 0:
        raise ValueError("dimension and delay must be > 0")
    last = len(x) - (dimension - 1) * delay
    if last <= 0:
        return np.empty((0, dimension), dtype=float)
    return np.asarray([x[i:i+dimension*delay:delay] for i in range(last)], dtype=float)
