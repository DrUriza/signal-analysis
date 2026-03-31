from __future__ import annotations

import numpy as np

def compute_md_proxy_features(signal):
    """Minimal proxy features for micro-doppler style variation."""
    x = np.asarray(signal, dtype=float)
    if x.size == 0:
        raise ValueError("signal cannot be empty")
    dx = np.diff(x, prepend=x[0])
    return {
        "mean_abs_delta": float(np.mean(np.abs(dx))),
        "std_signal": float(np.std(x)),
        "energy": float(np.sum(x*x)),
    }
