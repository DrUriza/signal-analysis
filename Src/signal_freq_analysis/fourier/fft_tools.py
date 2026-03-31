from __future__ import annotations

import numpy as np

def compute_fft_magnitude(signal, sample_rate_hz: float):
    """Return positive FFT frequencies and normalized magnitude."""
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1:
        raise ValueError("signal must be 1D")
    n = len(x)
    if n == 0:
        raise ValueError("signal cannot be empty")
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    spectrum = np.fft.rfft(x)
    magnitude = np.abs(spectrum) / n
    return freqs, magnitude
