import pandas                      as pd
from signal_analysis.adapters.ohlc import (ohlc_to_series, series_to_ohlc_dataframe, series_to_ohlc_windows)


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

def test_ohlc_to_series_weighted_values():
    ohlc = pd.DataFrame(
        {
            "open": [10.0, 11.0],
            "high": [12.0, 13.0],
            "low": [9.0, 10.0],
            "close": [11.0, 12.0],
        }
    )
    out = ohlc_to_series(ohlc, mode="weighted")
    assert out.name == "weighted_close"
    assert out.tolist() == [10.75, 11.75]
