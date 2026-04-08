from __future__ import annotations

import math
import numpy  as np
import pandas as pd

from signal_analysis.utils.helpers import to_series, validate_window


def moving_window_energy(series, window_size: int = 4, min_periods: int | None = None, normalize: bool = False) -> pd.Series:
    """
    Compute rolling window energy of a 1D series.

    Energy is defined as the rolling sum of squared amplitudes.

    Parameters
    ----------
    series : pandas.Series | list | numpy.ndarray
        Input scalar time series.
    window_size : int
        Rolling window size.
    min_periods : int | None
        Minimum required observations in the rolling window.
        If None, defaults to window_size.
    normalize : bool
        If True, divide the energy by the window size.

    Returns
    -------
    pandas.Series
        Rolling energy series.
    """
    validate_window(window_size, "window_size")
    s = to_series(series, name="signal")
    mp = window_size if min_periods is None else min_periods

    energy = (s ** 2).rolling(window=window_size, min_periods=mp).sum()
    if normalize:
        energy = energy / float(window_size)

    energy.name = f"window_energy_{window_size}"
    return energy


def _haar_step(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    One-level Haar decomposition step.

    If the number of samples is odd, the last sample is dropped.
    """
    n = len(values)
    if n < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    if n % 2 != 0:
        values = values[:-1]

    approx = (values[0::2] + values[1::2]) / math.sqrt(2.0)
    detail = (values[0::2] - values[1::2]) / math.sqrt(2.0)
    return approx, detail


def compute_haar_wavelet_decomposition(
    series,
    levels: int = 1,
) -> dict[str, pd.Series]:
    """
    Compute a simple multi-level Haar wavelet decomposition.

    Parameters
    ----------
    series : pandas.Series | list | numpy.ndarray
        Input scalar time series.
    levels : int
        Number of decomposition levels.

    Returns
    -------
    dict[str, pandas.Series]
        Dictionary with:
        - approximation
        - detail_level_1
        - detail_level_2
        - ...
    """
    validate_window(levels, "levels")
    s = to_series(series, name="signal")
    current = s.to_numpy(dtype=float)

    details: dict[str, pd.Series] = {}

    for level in range(1, levels + 1):
        approx, detail = _haar_step(current)
        details[f"detail_level_{level}"] = pd.Series(
            detail,
            name=f"detail_level_{level}",
            dtype=float,
        )
        current = approx
        if len(current) < 2:
            break

    result: dict[str, pd.Series] = {
        "approximation": pd.Series(current, name="approximation", dtype=float)
    }
    result.update(details)
    return result


def compute_wavelet_energy(
    series,
    levels: int = 1,
    normalize: bool = False,
) -> pd.Series:
    """
    Compute Haar-wavelet energy summary over decomposition levels.

    Parameters
    ----------
    series : pandas.Series | list | numpy.ndarray
        Input scalar time series.
    levels : int
        Number of Haar decomposition levels.
    normalize : bool
        If True, normalize energies by total energy.

    Returns
    -------
    pandas.Series
        Energy summary with fields:
        - approximation_energy
        - detail_energy_level_1
        - detail_energy_level_2
        - ...
    """
    validate_window(levels, "levels")
    coeffs = compute_haar_wavelet_decomposition(series, levels=levels)

    energies: dict[str, float] = {}
    total_energy = 0.0

    approx = coeffs["approximation"].dropna().to_numpy(dtype=float)
    approx_energy = float(np.sum(approx ** 2))
    energies["approximation_energy"] = approx_energy
    total_energy += approx_energy

    detail_keys = sorted(k for k in coeffs.keys() if k.startswith("detail_level_"))
    for key in detail_keys:
        values = coeffs[key].dropna().to_numpy(dtype=float)
        energy = float(np.sum(values ** 2))
        energies[f"{key}_energy"] = energy
        total_energy += energy

    if normalize and total_energy > 0:
        energies = {k: v / total_energy for k, v in energies.items()}

    return pd.Series(energies, name="wavelet_energy")


def wavelet_energy_placeholder(
    series,
    levels: int = 1,
) -> pd.Series:
    """
    Backward-compatible wrapper around compute_wavelet_energy().
    """
    return compute_wavelet_energy(series, levels=levels, normalize=False)


__all__ = [
    "moving_window_energy",
    "compute_haar_wavelet_decomposition",
    "compute_wavelet_energy",
    "wavelet_energy_placeholder",
]
