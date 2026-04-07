from __future__                    import annotations
import pandas                      as pd
from signal_analysis.utils.helpers import to_series, validate_window


def series_to_ohlc_windows(series, window: int, step: int | None = None) -> pd.DataFrame:
    """
    Convert a 1D scalar series into windowed OHLC summaries.

    Parameters
    ----------
    series : pandas.Series | list | numpy.ndarray
        Input scalar time series.
    window : int
        Number of samples per window.
    step : int | None
        Step between windows. If None, defaults to `window`.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - open
        - high
        - low
        - close
        - start_idx
        - end_idx
    """
    validate_window(window, "window")
    s = to_series(series)

    if step is None:
        step = window
    validate_window(step, "step")

    rows = []
    for start in range(0, len(s) - window + 1, step):
        chunk = s.iloc[start : start + window]
        rows.append({"open": float(chunk.iloc[0]),
                     "high": float(chunk.max()),
                     "low": float(chunk.min()),
                     "close": float(chunk.iloc[-1]),
                     "start_idx": int(start),
                     "end_idx": int(start + window - 1)})
    return pd.DataFrame(rows)

def series_to_ohlc_dataframe(series, window: int, step: int | None = None, index_mode: str = "end") -> pd.DataFrame:
    """
    Convert a scalar series into an OHLC DataFrame with a reusable index.
    Parameters
    ----------
    series : pandas.Series | list | numpy.ndarray
        Input scalar time series.
    window : int
        Number of samples per window.
    step : int | None
        Step between windows. If None, defaults to `window`.
    index_mode : str
        How to index the resulting rows:
        - "end": use end_idx
        - "start": use start_idx
        - "range": keep default RangeIndex

    Returns
    -------
    pandas.DataFrame
        OHLC DataFrame.
    """
    ohlc = series_to_ohlc_windows(series=series, window=window, step=step)

    if index_mode == "end":
        ohlc = ohlc.set_index("end_idx", drop=False)
    elif index_mode == "start":
        ohlc = ohlc.set_index("start_idx", drop=False)
    elif index_mode == "range":
        pass
    else:
        raise ValueError("index_mode must be one of: 'end', 'start', 'range'.")
    return ohlc

__all__ = ["series_to_ohlc_windows", "series_to_ohlc_dataframe"]

