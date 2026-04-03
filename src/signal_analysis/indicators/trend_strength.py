from __future__ import annotations

import numpy  as np
import pandas as pd

from signal_analysis.utils.helpers import ema_series, to_series, validate_window
from signal_analysis.utils.ohlc    import true_range

def compute_macd_components(series, window_slow: int = 26, window_fast: int = 12, window_signal: int = 9, min_periods: int | None = None) -> pd.DataFrame:
    """
    Compute MACD line, signal line, and histogram from a single series.
    """
    validate_window(window_slow, "window_slow")
    validate_window(window_fast, "window_fast")
    validate_window(window_signal, "window_signal")

    if window_fast >= window_slow:
        raise ValueError("window_fast should be smaller than window_slow.")

    s  = to_series(series)
    mp = None if min_periods is None else min_periods

    ema_fast = ema_series(s, window_fast, adjust=False, min_periods=mp)
    ema_slow = ema_series(s, window_slow, adjust=False, min_periods=mp)

    macd_line   = ema_fast - ema_slow
    signal_line = ema_series(macd_line, window_signal, adjust=False, min_periods=mp)
    hist        = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist}, index=s.index)

def compute_macd(series, window_slow: int = 26, window_fast: int = 12, window_signal: int = 9, min_periods: int | None = None) -> pd.Series:
    """
    Compute MACD line.
    """
    return compute_macd_components(series, window_slow=window_slow, window_fast=window_fast, window_signal=window_signal, min_periods=min_periods)["macd"]

def compute_macd_signal(series, window_slow: int = 26, window_fast: int = 12, window_signal: int = 9, min_periods: int | None = None) -> pd.Series:
    """
    Compute MACD signal line.
    """
    return compute_macd_components(series, window_slow=window_slow, window_fast=window_fast, window_signal=window_signal, min_periods=min_periods)["macd_signal"]

def compute_macd_hist(series, window_slow: int = 26, window_fast: int = 12, window_signal: int = 9, min_periods: int | None = None) -> pd.Series:
    """
    Compute MACD histogram.
    """
    return compute_macd_components(series, window_slow=window_slow, window_fast=window_fast, window_signal=window_signal, min_periods=min_periods)["macd_hist"]

def compute_plus_di(high, low, close, window: int = 14) -> pd.Series:
    """
    Compute the positive directional indicator (+DI).
    """
    result = compute_directional_indicators(high, low, close, window=window)
    return result["plus_di"]

def compute_minus_di(high, low, close, window: int = 14) -> pd.Series:
    """
    Compute the negative directional indicator (-DI).
    """
    result = compute_directional_indicators(high, low, close, window=window)
    return result["minus_di"]

def compute_adx(high, low, close, window: int = 14) -> pd.Series:
    """
    Compute the Average Directional Index (ADX).
    """
    result = compute_directional_indicators(high, low, close, window=window)
    return result["adx"]

def compute_directional_indicators(high, low, close, window: int = 14) -> pd.DataFrame:
    """
    Compute +DI, -DI, DX, and ADX.
    """
    validate_window(window, "window")

    h = to_series(high, name="high")
    l = to_series(low, name="low")
    c = to_series(close, name="close")

    up_move   = h.diff()
    down_move = -l.diff()

    plus_dm  = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=h.index, name="plus_dm")
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=h.index, name="minus_dm")

    tr = true_range(h, l, c)

    atr = tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    plus_di  = 100.0 * (plus_dm_smooth / atr)
    minus_di = 100.0 * (minus_dm_smooth / atr)

    di_sum = plus_di + minus_di
    dx     = 100.0 * ((plus_di - minus_di).abs() / di_sum)
    dx     = dx.replace([np.inf, -np.inf], np.nan)
    adx    = dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    return pd.DataFrame({"plus_di": plus_di,
                         "minus_di": minus_di,
                         "dx": dx,
                         "adx": adx}, index=h.index)

__all__ = ["compute_macd_components", "compute_macd", "compute_macd_signal", "compute_macd_hist",
           "compute_plus_di", "compute_minus_di", "compute_adx", "compute_directional_indicators"]
