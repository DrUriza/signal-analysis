from __future__ import annotations

import numpy as np
import pandas as pd

from signal_analysis.utils.helpers import to_series


def compute_fft_magnitude(
    series,
    sample_rate_hz: float = 1.0,
    detrend: bool = False,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute one-sided FFT magnitude spectrum.

    Parameters
    ----------
    series : array-like
        Input 1D signal.
    sample_rate_hz : float
        Sampling rate in Hz.
    detrend : bool
        If True, remove the mean before FFT.
    normalize : bool
        If True, divide magnitude by signal length.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (frequencies, magnitudes)
    """
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")

    s = to_series(series, name="signal").to_numpy(dtype=float)
    if len(s) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    if detrend:
        s = s - np.mean(s)

    fft_vals = np.fft.rfft(s)
    freqs = np.fft.rfftfreq(len(s), d=1.0 / sample_rate_hz)
    mags = np.abs(fft_vals)

    if normalize:
        mags = mags / len(s)

    return freqs, mags


def compute_fft_power_spectrum(
    series,
    sample_rate_hz: float = 1.0,
    detrend: bool = False,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute one-sided FFT power spectrum.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (frequencies, power)
    """
    freqs, mags = compute_fft_magnitude(
        series,
        sample_rate_hz=sample_rate_hz,
        detrend=detrend,
        normalize=normalize,
    )
    power = mags ** 2
    return freqs, power


def dominant_frequency(
    series,
    sample_rate_hz: float = 1.0,
    detrend: bool = False,
    ignore_dc: bool = True,
) -> float:
    """
    Estimate dominant frequency from the magnitude spectrum.

    Parameters
    ----------
    series : array-like
        Input 1D signal.
    sample_rate_hz : float
        Sampling rate in Hz.
    detrend : bool
        If True, remove mean before FFT.
    ignore_dc : bool
        If True, ignore the zero-frequency component.

    Returns
    -------
    float
        Dominant frequency in Hz.
    """
    freqs, mags = compute_fft_magnitude(
        series,
        sample_rate_hz=sample_rate_hz,
        detrend=detrend,
        normalize=True,
    )

    if len(freqs) == 0:
        return 0.0

    start = 1 if ignore_dc and len(freqs) > 1 else 0
    idx = int(np.argmax(mags[start:])) + start
    return float(freqs[idx])


def spectral_energy(series, sample_rate_hz: float = 1.0, detrend: bool = False) -> float:
    """
    Compute total spectral energy from the one-sided FFT magnitude spectrum.
    """
    _, power = compute_fft_power_spectrum(series, sample_rate_hz=sample_rate_hz, detrend=detrend, normalize=True)
    return float(np.sum(power))


def spectral_centroid(series, sample_rate_hz: float = 1.0, detrend: bool = False) -> float:
    """
    Compute the spectral centroid.

    Returns
    -------
    float
        Spectral centroid in Hz.
    """
    freqs, mags = compute_fft_magnitude(series, sample_rate_hz=sample_rate_hz, detrend=detrend, normalize=True)
    denom = float(np.sum(mags))
    if len(freqs) == 0 or denom == 0.0:
        return 0.0
    return float(np.sum(freqs * mags) / denom)


def fft_feature_summary(series, sample_rate_hz: float = 1.0, detrend: bool = False) -> pd.Series:
    """
    Compute a compact FFT-based feature summary.

    Returns
    -------
    pd.Series
        Summary with:
        - dominant_frequency
        - spectral_energy
        - spectral_centroid
    """
    return pd.Series(
        {
            "dominant_frequency": dominant_frequency(
                series,
                sample_rate_hz=sample_rate_hz,
                detrend=detrend,
            ),
            "spectral_energy": spectral_energy(
                series,
                sample_rate_hz=sample_rate_hz,
                detrend=detrend,
            ),
            "spectral_centroid": spectral_centroid(
                series,
                sample_rate_hz=sample_rate_hz,
                detrend=detrend,
            ),
        },
        name="fft_summary",
    )


__all__ = ["compute_fft_magnitude", "compute_fft_power_spectrum", "dominant_frequency", "spectral_energy", "spectral_centroid",
           "fft_feature_summary"]

