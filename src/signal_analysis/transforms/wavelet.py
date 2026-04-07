from __future__ import annotations

import pandas as pd


def wavelet_energy_placeholder(series) -> pd.Series:
    """
    Placeholder for future wavelet-based energy features.
    """
    s = pd.Series(series, dtype=float)
    return pd.Series([float((s ** 2).sum())], name="wavelet_energy_placeholder")
