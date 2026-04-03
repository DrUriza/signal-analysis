from __future__ import annotations

import pandas as pd

from signal_analysis.utils.helpers import to_series


def cross_above(a, b) -> pd.Series:
    """
    True where series a crosses above series b.
    """
    sa = to_series(a, name="a")
    sb = to_series(b, name="b")
    return (sa > sb) & (sa.shift(1) <= sb.shift(1))


def cross_below(a, b) -> pd.Series:
    """
    True where series a crosses below series b.
    """
    sa = to_series(a, name="a")
    sb = to_series(b, name="b")
    return (sa < sb) & (sa.shift(1) >= sb.shift(1))


def cross_level_up(series, level: float) -> pd.Series:
    """
    True where series crosses upward through a scalar level.
    """
    s = to_series(series)
    return (s > level) & (s.shift(1) <= level)


def cross_level_down(series, level: float) -> pd.Series:
    """
    True where series crosses downward through a scalar level.
    """
    s = to_series(series)
    return (s < level) & (s.shift(1) >= level)
