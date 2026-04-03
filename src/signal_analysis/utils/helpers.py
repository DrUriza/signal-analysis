from __future__ import annotations

import pandas as pd


def to_series(data, name: str = "series") -> pd.Series:
    """
    Convert input data to a pandas Series of floats.
    """
    if isinstance(data, pd.Series):
        return data.astype(float).copy()
    return pd.Series(data, dtype=float, name=name)


def validate_window(window: int, name: str = "window") -> None:
    """
    Validate that a window-like parameter is a positive integer.
    """
    if not isinstance(window, int) or window <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def ema_series(
    series: pd.Series,
    window: int,
    adjust: bool = False,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Reusable EMA helper.
    """
    validate_window(window, "window")
    mp = window if min_periods is None else min_periods
    return series.ewm(span=window, adjust=adjust, min_periods=mp).mean()
