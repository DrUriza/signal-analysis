import pandas as pd

from signal_analysis.indicators import (
    compute_ema,
    compute_kama,
    compute_sma,
    compute_wma,
)


def test_compute_sma_returns_series_with_expected_name():
    s = pd.Series([1, 2, 3, 4, 5])
    out = compute_sma(s, window=3)
    assert isinstance(out, pd.Series)
    assert out.name == "sma_3"
    assert out.iloc[-1] == 4.0


def test_compute_ema_returns_series_with_expected_name():
    s = pd.Series([1, 2, 3, 4, 5])
    out = compute_ema(s, window=3)
    assert isinstance(out, pd.Series)
    assert out.name == "ema_3"
    assert out.notna().sum() >= 1


def test_compute_wma_returns_series_with_expected_name():
    s = pd.Series([1, 2, 3, 4, 5, 6])
    out = compute_wma(s, window=3)
    assert isinstance(out, pd.Series)
    assert out.name == "wma_3"
    assert out.iloc[-1] > out.iloc[-2]


def test_compute_kama_returns_series():
    s = pd.Series([1, 2, 3, 2, 4, 5, 4, 6, 7, 8, 7, 9])
    out = compute_kama(s, er_window=3, fast=2, slow=5)
    assert isinstance(out, pd.Series)
    assert out.name == "kama_3_2_5"
    assert len(out) == len(s)
