import pandas as pd

from signal_analysis.indicators import (
    compute_adx,
    compute_directional_indicators,
    compute_macd,
    compute_macd_components,
    compute_macd_hist,
    compute_macd_signal,
    compute_minus_di,
    compute_plus_di,
)


def test_compute_macd_components_returns_dataframe():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9])
    out = compute_macd_components(s, window_slow=6, window_fast=3, window_signal=2)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["macd", "macd_signal", "macd_hist"]


def test_compute_macd_series_exports():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9])
    macd = compute_macd(s, window_slow=6, window_fast=3, window_signal=2)
    sig = compute_macd_signal(s, window_slow=6, window_fast=3, window_signal=2)
    hist = compute_macd_hist(s, window_slow=6, window_fast=3, window_signal=2)
    assert isinstance(macd, pd.Series)
    assert isinstance(sig, pd.Series)
    assert isinstance(hist, pd.Series)


def test_compute_directional_indicators_returns_dataframe():
    high = pd.Series([2, 3, 4, 5, 6, 7, 8])
    low = pd.Series([1, 2, 3, 4, 5, 6, 7])
    close = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])

    out = compute_directional_indicators(high, low, close, window=3)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["plus_di", "minus_di", "dx", "adx"]


def test_adx_and_di_wrappers_return_series():
    high = pd.Series([2, 3, 4, 5, 6, 7, 8])
    low = pd.Series([1, 2, 3, 4, 5, 6, 7])
    close = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])

    plus_di = compute_plus_di(high, low, close, window=3)
    minus_di = compute_minus_di(high, low, close, window=3)
    adx = compute_adx(high, low, close, window=3)

    assert isinstance(plus_di, pd.Series)
    assert isinstance(minus_di, pd.Series)
    assert isinstance(adx, pd.Series)
