from __future__ import annotations
import numpy    as np
import pandas   as pd

from signal_analysis.utils.helpers import ema_series, to_series, validate_window

def compute_rsi(series, window: int = 14, fillna: bool = False) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).
    """
    validate_window(window, "window")
    s = to_series(series)

    diff = s.diff(1)
    up_direction   = diff.where(diff > 0, 0.0)
    down_direction = -diff.where(diff < 0, 0.0)
    min_periods = 0 if fillna else window
    ema_up      = up_direction.ewm(alpha=1 / window, min_periods=min_periods, adjust=False).mean()
    ema_down    = down_direction.ewm(alpha=1 / window, min_periods=min_periods, adjust=False).mean()

    relative_strength = ema_up / ema_down
    rsi = pd.Series(np.where(ema_down == 0, 100.0, 100.0 - (100.0 / (1.0 + relative_strength))), index=s.index, name=f"rsi_{window}")
    if fillna:
        rsi = rsi.fillna(50.0)
    return rsi


def compute_tsi(series, window_slow: int = 25, window_fast: int = 13, fillna: bool = False) -> pd.Series:
    """
    Compute True Strength Index (TSI).
    """
    validate_window(window_slow, "window_slow")
    validate_window(window_fast, "window_fast")
    s = to_series(series)

    diff = s.diff(1)
    min_periods_slow = 0 if fillna else window_slow
    min_periods_fast = 0 if fillna else window_fast

    smoothed = ema_series(ema_series(diff, window_slow, adjust=False, min_periods=min_periods_slow),
                          window_fast,
                          adjust=False,
                          min_periods=min_periods_fast)
    smoothed_abs = ema_series(ema_series(diff.abs(), window_slow, adjust=False, min_periods=min_periods_slow),
                              window_fast,
                              adjust=False,
                              min_periods=min_periods_fast)
    tsi = 100.0 * (smoothed / smoothed_abs)
    tsi = pd.Series(tsi, index=s.index, name=f"tsi_{window_slow}_{window_fast}")
    if fillna:
        tsi = tsi.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return tsi

def compute_roc(series, window: int = 12, fillna: bool = False) -> pd.Series:
    """
    Compute Rate of Change (ROC).
    """
    validate_window(window, "window")
    s = to_series(series)
    shifted = s.shift(window)
    roc = ((s - shifted) / shifted) * 100.0
    roc = pd.Series(roc, index=s.index, name=f"roc_{window}")
    if fillna:
        roc = roc.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return roc


__all__ = ["compute_rsi", "compute_tsi", "compute_roc"]
