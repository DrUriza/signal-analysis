import pandas as pd

from signal_analysis.indicators import (
    compute_roc,
    compute_rsi,
    compute_tsi,
)


def test_compute_rsi_returns_series():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    out = compute_rsi(s, window=5)
    assert isinstance(out, pd.Series)
    assert out.name == "rsi_5"
    assert len(out) == len(s)


def test_compute_tsi_returns_series():
    s = pd.Series([1, 2, 3, 2, 4, 5, 4, 6, 7, 8])
    out = compute_tsi(s, window_slow=4, window_fast=2)
    assert isinstance(out, pd.Series)
    assert out.name == "tsi_4_2"
    assert len(out) == len(s)


def test_compute_roc_returns_series():
    s = pd.Series([10, 11, 12, 13, 14, 15])
    out = compute_roc(s, window=2)
    assert isinstance(out, pd.Series)
    assert out.name == "roc_2"
    assert len(out) == len(s)
