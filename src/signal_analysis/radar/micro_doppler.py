from __future__ import annotations

import pandas as pd


def micro_doppler_features_placeholder(series) -> pd.DataFrame:
    """
    Placeholder for future micro-Doppler feature extraction.
    """
    s = pd.Series(series, dtype=float)
    return pd.DataFrame(
        {
            "length": [int(len(s))],
            "energy": [float((s ** 2).sum())],
        }
    )
