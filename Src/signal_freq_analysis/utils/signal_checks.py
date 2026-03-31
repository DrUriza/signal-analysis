from __future__ import annotations

import numpy as np

def ensure_1d_signal(signal):
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1:
        raise ValueError("signal must be 1D")
    if x.size == 0:
        raise ValueError("signal cannot be empty")
    return x
