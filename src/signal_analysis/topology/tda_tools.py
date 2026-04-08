from __future__ import annotations

import numpy as np
import pandas as pd


def sliding_window_embedding(
    signal,
    dimension: int = 3,
    delay: int = 1,
) -> np.ndarray:
    """
    Create a simple Takens-style sliding-window embedding.

    Parameters
    ----------
    signal : array-like
        Input 1D signal.
    dimension : int
        Embedding dimension.
    delay : int
        Delay between coordinates.

    Returns
    -------
    np.ndarray
        Embedded point cloud of shape (n_windows, dimension).
    """
    x = np.asarray(signal, dtype=float)

    if dimension <= 0 or delay <= 0:
        raise ValueError("dimension and delay must be > 0")

    last = len(x) - (dimension - 1) * delay
    if last <= 0:
        return np.empty((0, dimension), dtype=float)

    return np.asarray(
        [x[i : i + dimension * delay : delay] for i in range(last)],
        dtype=float,
    )


def pairwise_distance_matrix(points: np.ndarray) -> np.ndarray:
    """
    Compute a dense Euclidean pairwise distance matrix.

    Parameters
    ----------
    points : np.ndarray
        Point cloud of shape (n_points, n_features).

    Returns
    -------
    np.ndarray
        Distance matrix of shape (n_points, n_points).
    """
    if points.ndim != 2:
        raise ValueError("points must be a 2D array")

    if len(points) == 0:
        return np.empty((0, 0), dtype=float)

    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def wasserstein_like_distance(
    a,
    b,
    p: int = 1,
) -> float:
    """
    Lightweight Wasserstein-like distance for 1D samples.

    This is not a persistence-diagram Wasserstein distance.
    It is a practical transport-style distance between two 1D sequences,
    based on sorted samples.

    Parameters
    ----------
    a, b : array-like
        Input 1D samples.
    p : int
        Power for the p-Wasserstein-like distance.

    Returns
    -------
    float
        Distance value.
    """
    xa = np.asarray(a, dtype=float).ravel()
    xb = np.asarray(b, dtype=float).ravel()

    if len(xa) == 0 or len(xb) == 0:
        return 0.0

    n = min(len(xa), len(xb))
    xa = np.sort(xa)[:n]
    xb = np.sort(xb)[:n]

    if p <= 0:
        raise ValueError("p must be > 0")

    if p == 1:
        return float(np.mean(np.abs(xa - xb)))

    return float(np.mean(np.abs(xa - xb) ** p) ** (1.0 / p))


def rolling_wasserstein_profile(
    series,
    window: int = 20,
    p: int = 1,
    normalize_to_100: bool = True,
) -> pd.Series:
    """
    Compute a rolling Wasserstein-like profile between adjacent windows.

    For each position i, compare:
    - series[i : i + window]
    - series[i + window : i + 2*window]

    Parameters
    ----------
    series : array-like
        Input 1D series.
    window : int
        Window size.
    p : int
        Power for the p-Wasserstein-like distance.
    normalize_to_100 : bool
        If True, normalize the resulting profile to [0, 100].

    Returns
    -------
    pd.Series
        Rolling distance profile aligned to the second window end.
    """
    x = pd.Series(series, dtype=float)
    n = len(x)

    if window <= 0:
        raise ValueError("window must be > 0")

    out = pd.Series(np.nan, index=x.index, name=f"wd_profile_{window}")

    if n < 2 * window:
        return out.fillna(0.0)

    vals = []
    idxs = []

    for i in range(0, n - 2 * window + 1):
        a = x.iloc[i : i + window].to_numpy(dtype=float)
        b = x.iloc[i + window : i + 2 * window].to_numpy(dtype=float)
        dist = wasserstein_like_distance(a, b, p=p)
        vals.append(dist)
        idxs.append(x.index[i + 2 * window - 1])

    vals = np.asarray(vals, dtype=float)

    if normalize_to_100 and len(vals) > 0:
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        vals = 100.0 * (vals - vmin) / (vmax - vmin + 1e-12)

    out.loc[idxs] = vals
    return out.ffill().fillna(0.0)


def wasserstein_regime_signal(
    distance_profile,
    high_on: float = 95.0,
    high_off: float = 80.0,
    low_on: float = 20.0,
    low_off: float = 35.0,
) -> pd.Series:
    """
    Convert a normalized Wasserstein-like profile into a simple regime signal.

    Output values:
    -  1.0 : high-regime / strong-change region
    - -1.0 : low-regime / stable region
    -  0.0 : neutral region

    Parameters
    ----------
    distance_profile : array-like
        Typically the output of rolling_wasserstein_profile().
    high_on, high_off, low_on, low_off : float
        Hysteresis thresholds.

    Returns
    -------
    pd.Series
        Regime signal.
    """
    wd = pd.Series(distance_profile, dtype=float)
    signal = pd.Series(0.0, index=wd.index, name="wd_regime_signal")

    signal[wd >= high_on] = 1.0
    signal[wd <= low_on] = -1.0
    signal[(wd < high_off) & (wd > low_off)] = 0.0

    return signal


def wasserstein_distances(
    data,
    window: int = 20,
    column: str = "close",
    high_on: float = 95.0,
    high_off: float = 80.0,
    low_on: float = 20.0,
    low_off: float = 35.0,
) -> tuple[pd.Series, pd.Series]:
    """
    Convenience wrapper inspired by the previous topology analyzer.

    Parameters
    ----------
    data : pandas.DataFrame | array-like
        Input data. If DataFrame, `column` is used.
    window : int
        Adjacent window size.
    column : str
        Column to use when `data` is a DataFrame.
    high_on, high_off, low_on, low_off : float
        Hysteresis thresholds.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (distance_profile, regime_signal)
    """
    if isinstance(data, pd.DataFrame):
        if column not in data.columns:
            raise ValueError(f"column '{column}' not found in DataFrame")
        series = data[column]
    else:
        series = data

    wd = rolling_wasserstein_profile(series, window=window, p=1, normalize_to_100=True)
    pos = wasserstein_regime_signal(
        wd,
        high_on=high_on,
        high_off=high_off,
        low_on=low_on,
        low_off=low_off,
    )
    return wd, pos


__all__ = [
    "sliding_window_embedding",
    "pairwise_distance_matrix",
    "wasserstein_like_distance",
    "rolling_wasserstein_profile",
    "wasserstein_regime_signal",
    "wasserstein_distances",
]
