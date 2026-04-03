from   __future__ import annotations
import numpy      as np
import pandas     as pd

from signal_analysis.utils.helpers import to_series, validate_window
from signal_analysis.utils.ohlc    import true_range


def compute_bollinger_bands(series, window: int = 20, n_std: float = 2.0, min_periods: int | None = None) -> pd.DataFrame:
    """
    Compute Bollinger Bands from a single series.
    """
    validate_window(window, "window")
    s = to_series(series)
    mp = window if min_periods is None else min_periods

    middle = s.rolling(window=window, min_periods=mp).mean()
    rolling_std = s.rolling(window=window, min_periods=mp).std(ddof=0)

    upper = middle + n_std * rolling_std
    lower = middle - n_std * rolling_std
    bandwidth = upper - lower

    percent_b = (s - lower) / (upper - lower)
    percent_b = percent_b.replace([np.inf, -np.inf], np.nan)

    result = pd.DataFrame({"bb_middle": middle,
                           "bb_upper": upper,
                           "bb_lower": lower,
                           "bb_bandwidth": bandwidth,
                           "bb_percent_b": percent_b}, index=s.index)
    return result

def compute_tr(high, low, close) -> pd.Series:
    """
    Thin wrapper around utils.true_range for indicator namespace consistency.
    """
    tr = true_range(high, low, close)
    tr.name = "true_range"
    return tr

def compute_atr(high, low, close, window: int = 14, min_periods: int | None = None) -> pd.Series:
    """
    Compute Average True Range (ATR) using Wilder-style EMA smoothing.
    """
    validate_window(window, "window")
    h = to_series(high, name="high")
    l = to_series(low, name="low")
    c = to_series(close, name="close")

    tr  = true_range(h, l, c)
    mp  = window if min_periods is None else min_periods
    atr = tr.ewm(alpha=1 / window, adjust=False, min_periods=mp).mean()
    atr.name = f"atr_{window}"
    return atr

__all__ = ["compute_bollinger_bands", "compute_true_range", "compute_atr"]
