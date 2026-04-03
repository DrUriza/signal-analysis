from __future__ import annotations

import numpy as np
import pandas as pd

from signal_analysis.indicators.momentum import compute_rsi
from signal_analysis.utils.helpers import to_series, validate_window


def compute_stochastic(
    high,
    low,
    close,
    window: int = 14,
    smooth_window: int = 3,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    Compute Stochastic Oscillator (%K and %D).
    """
    validate_window(window, "window")
    validate_window(smooth_window, "smooth_window")

    h = to_series(high, name="high")
    l = to_series(low, name="low")
    c = to_series(close, name="close")

    mp = window if min_periods is None else min_periods

    low_min = l.rolling(window=window, min_periods=mp).min()
    high_max = h.rolling(window=window, min_periods=mp).max()

    denom = (high_max - low_min).replace(0.0, np.nan)
    stoch_k = 100.0 * (c - low_min) / denom
    stoch_d = stoch_k.rolling(window=smooth_window, min_periods=smooth_window).mean()

    return pd.DataFrame(
        {
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
        },
        index=c.index,
    )


def compute_stochastic_signal(
    high,
    low,
    close,
    window: int = 14,
    smooth_window: int = 3,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Compute Stochastic signal line (%D).
    """
    return compute_stochastic(
        high=high,
        low=low,
        close=close,
        window=window,
        smooth_window=smooth_window,
        min_periods=min_periods,
    )["stoch_d"]


def compute_williams_r(
    high,
    low,
    close,
    window: int = 14,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Compute Williams %R.
    """
    validate_window(window, "window")

    h = to_series(high, name="high")
    l = to_series(low, name="low")
    c = to_series(close, name="close")

    mp = window if min_periods is None else min_periods

    highest_high = h.rolling(window=window, min_periods=mp).max()
    lowest_low = l.rolling(window=window, min_periods=mp).min()

    denom = (highest_high - lowest_low).replace(0.0, np.nan)
    wr = -100.0 * (highest_high - c) / denom
    wr.name = f"williams_r_{window}"
    return wr


def compute_stochrsi(
    series,
    window: int = 14,
    smooth1: int = 3,
    smooth2: int = 3,
    fillna: bool = False,
) -> pd.DataFrame:
    """
    Compute Stochastic RSI, %K, and %D.
    """
    validate_window(window, "window")
    validate_window(smooth1, "smooth1")
    validate_window(smooth2, "smooth2")

    s = to_series(series)
    rsi = compute_rsi(s, window=window, fillna=fillna)

    lowest_rsi = rsi.rolling(window=window, min_periods=window).min()
    highest_rsi = rsi.rolling(window=window, min_periods=window).max()

    denom = (highest_rsi - lowest_rsi).replace(0.0, np.nan)
    stochrsi = (rsi - lowest_rsi) / denom
    stochrsi_k = stochrsi.rolling(window=smooth1, min_periods=smooth1).mean()
    stochrsi_d = stochrsi_k.rolling(window=smooth2, min_periods=smooth2).mean()

    result = pd.DataFrame(
        {
            "stochrsi": stochrsi,
            "stochrsi_k": stochrsi_k,
            "stochrsi_d": stochrsi_d,
        },
        index=s.index,
    )

    if fillna:
        result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return result


def compute_stochrsi_k(
    series,
    window: int = 14,
    smooth1: int = 3,
    smooth2: int = 3,
    fillna: bool = False,
) -> pd.Series:
    """
    Compute StochRSI %K.
    """
    return compute_stochrsi(
        series=series,
        window=window,
        smooth1=smooth1,
        smooth2=smooth2,
        fillna=fillna,
    )["stochrsi_k"]


def compute_stochrsi_d(
    series,
    window: int = 14,
    smooth1: int = 3,
    smooth2: int = 3,
    fillna: bool = False,
) -> pd.Series:
    """
    Compute StochRSI %D.
    """
    return compute_stochrsi(
        series=series,
        window=window,
        smooth1=smooth1,
        smooth2=smooth2,
        fillna=fillna,
    )["stochrsi_d"]


__all__ = [
    "compute_stochastic",
    "compute_stochastic_signal",
    "compute_williams_r",
    "compute_stochrsi",
    "compute_stochrsi_k",
    "compute_stochrsi_d",
]
