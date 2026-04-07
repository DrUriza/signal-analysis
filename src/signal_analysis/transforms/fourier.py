from __future__ import annotations

import numpy as np
import pandas as pd


def compute_fft_magnitude(series, sample_rate_hz: float = 1.0):
    """
    Compute one-sided FFT magnitude spectrum.
    """
    s = pd.Series(series, dtype=float).to_numpy()
    if len(s) == 0:
        return np.array([]), np.array([])
    fft_vals = np.fft.rfft(s)
    freqs = np.fft.rfftfreq(len(s), d=1.0 / sample_rate_hz)
    mags = np.abs(fft_vals) / len(s)
    return freqs, mags
