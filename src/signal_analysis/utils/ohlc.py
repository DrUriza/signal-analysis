from __future__ import annotations

import pandas as pd

from signal_analysis.utils.helpers import to_series


def true_range(high, low, close) -> pd.Series:
    """
    Compute True Range from OHLC inputs.
    """
    h = to_series(high, name="high")
    l = to_series(low, name="low")
    c = to_series(close, name="close")

    prev_close = c.shift(1)

    tr_components = pd.concat(
        [
            (h - l).abs(),
            (h - prev_close).abs(),
            (l - prev_close).abs(),
        ],
        axis=1,
    )

    tr = tr_components.max(axis=1)
    tr.name = "true_range"
    return tr
