import pandas as pd

from signal_analysis.indicators import (
    compute_stochastic,
    compute_stochastic_signal,
    compute_stochrsi,
    compute_stochrsi_d,
    compute_stochrsi_k,
    compute_williams_r,
)


def test_compute_stochastic_returns_dataframe():
    high = pd.Series([2, 3, 4, 5, 6, 7])
    low = pd.Series([1, 2, 3, 4, 5, 6])
    close = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])

    out = compute_stochastic(high, low, close, window=3, smooth_window=2)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["stoch_k", "stoch_d"]


def test_compute_stochastic_signal_returns_series():
    high = pd.Series([2, 3, 4, 5, 6, 7])
    low = pd.Series([1, 2, 3, 4, 5, 6])
    close = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])

    out = compute_stochastic_signal(high, low, close, window=3, smooth_window=2)
    assert isinstance(out, pd.Series)


def test_compute_williams_r_returns_series():
    high = pd.Series([2, 3, 4, 5, 6, 7])
    low = pd.Series([1, 2, 3, 4, 5, 6])
    close = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])

    out = compute_williams_r(high, low, close, window=3)
    assert isinstance(out, pd.Series)
    assert out.name == "williams_r_3"


def test_compute_stochrsi_returns_dataframe():
    s = pd.Series([1, 2, 3, 2, 4, 5, 4, 6, 7, 8])
    out = compute_stochrsi(s, window=4, smooth1=2, smooth2=2, fillna=True)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["stochrsi", "stochrsi_k", "stochrsi_d"]


def test_compute_stochrsi_wrappers_return_series():
    s = pd.Series([1, 2, 3, 2, 4, 5, 4, 6, 7, 8])
    k = compute_stochrsi_k(s, window=4, smooth1=2, smooth2=2, fillna=True)
    d = compute_stochrsi_d(s, window=4, smooth1=2, smooth2=2, fillna=True)
    assert isinstance(k, pd.Series)
    assert isinstance(d, pd.Series)
