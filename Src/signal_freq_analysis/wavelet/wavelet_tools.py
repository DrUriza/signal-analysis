from __future__ import annotations

import numpy as np

def moving_window_energy(signal, window_size: int = 8):
    """Placeholder feature that mimics localized time-scale energy behavior."""
    x = np.asarray(signal, dtype=float)
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if len(x) < window_size:
        return np.array([], dtype=float)
    energies = []
    for i in range(len(x) - window_size + 1):
        w = x[i:i+window_size]
        energies.append(float(np.sum(w*w)))
    return np.asarray(energies, dtype=float)
