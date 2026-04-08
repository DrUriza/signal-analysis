import pandas                      as pd
from signal_analysis.adapters.ohlc import (series_to_ohlc_dataframe, series_to_ohlc_windows)


def test_series_to_ohlc_windows_returns_dataframe():
    s = [1, 2, 3, 2, 5, 6, 4, 7]
    out = series_to_ohlc_windows(s, window=4, step=2)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["open", "high", "low", "close", "start_idx", "end_idx"]


def test_series_to_ohlc_dataframe_with_end_index():
    s = [1, 2, 3, 2, 5, 6, 4, 7]
    out = series_to_ohlc_dataframe(s, window=4, step=2, index_mode="end")
    assert isinstance(out, pd.DataFrame)
    assert "end_idx" in out.columns
