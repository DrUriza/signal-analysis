from   __future__ import annotations
import numpy      as np
import pandas     as pd

from signal_analysis.utils.helpers import ema_series, to_series, validate_window

def compute_sma(series, window: int = 14, min_periods: int | None = None) -> pd.Series:
    """
    Compute Simple Moving Average (SMA).
    """
    validate_window(window, "window")
    s = to_series(series)
    mp = window if min_periods is None else min_periods
    sma = s.rolling(window=window, min_periods=mp).mean()
    sma.name = f"sma_{window}"
    return sma

def compute_ema(series, window: int = 14, adjust: bool = False, min_periods: int | None = None) -> pd.Series:
    """
    Compute Exponential Moving Average (EMA).
    """
    s = to_series(series)
    ema = ema_series(s, window=window, adjust=adjust, min_periods=min_periods)
    ema.name = f"ema_{window}"
    return ema

def compute_wma(series, window: int = 9, min_periods: int | None = None) -> pd.Series:
    """
    Compute Weighted Moving Average (WMA) with linearly increasing weights.
    """
    validate_window(window, "window")
    s  = to_series(series)
    mp = window if min_periods is None else min_periods

    weights = np.arange(1, window + 1, dtype=float)
    weights = weights / weights.sum()

    def _weighted_avg(values: np.ndarray) -> float:
        return float(np.dot(values, weights))

    wma = s.rolling(window=window, min_periods=mp).apply(_weighted_avg, raw=True)
    wma.name = f"wma_{window}"
    return wma

def compute_kama(series, er_window: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """
    Compute Kaufman's Adaptive Moving Average (KAMA).
    """
    validate_window(er_window, "er_window")
    validate_window(fast, "fast")
    validate_window(slow, "slow")

    s = to_series(series)
    values = s.to_numpy(dtype=float)

    change = np.abs(s - s.shift(er_window))
    volatility = s.diff().abs().rolling(er_window, min_periods=er_window).sum()

    efficiency_ratio = change / volatility
    efficiency_ratio = efficiency_ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    smoothing_constant = (efficiency_ratio * (fast_sc - slow_sc) + slow_sc) ** 2

    kama = np.full(len(values), np.nan, dtype=float)
    first_valid_idx = s.first_valid_index()
    if first_valid_idx is None:
        return pd.Series(kama, index=s.index, name=f"kama_{er_window}_{fast}_{slow}")

    first_pos = s.index.get_loc(first_valid_idx)
    kama[first_pos] = values[first_pos]

    for i in range(first_pos + 1, len(values)):
        if np.isnan(values[i]):
            kama[i] = np.nan
            continue

        prev_kama = kama[i - 1]
        if np.isnan(prev_kama):
            prev_kama = values[i - 1]
        sc = (float(smoothing_constant.iloc[i]) if not np.isnan(smoothing_constant.iloc[i]) else 0.0)
        kama[i] = prev_kama + sc * (values[i] - prev_kama)
    kama_series = pd.Series(kama, index=s.index, name=f"kama_{er_window}_{fast}_{slow}")
    return kama_series


__all__ = ["compute_sma", "compute_ema", "compute_wma", "compute_kama"]
