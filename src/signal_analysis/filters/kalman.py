from __future__ import annotations

import numpy as np
import pandas as pd

from signal_analysis.utils.helpers import to_series


def _validate_positive_float(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def kalman_filter_1d(
    series,
    process_var: float = 1e-5,
    measurement_var: float = 1e-1,
    initial_cov: float = 1e-1,
    dt: float = 1.0,
    return_velocity: bool = False,
) -> pd.Series | pd.DataFrame:
    """
    Causal 1D Kalman filter with constant-velocity state model.

    State vector:
        x = [position, velocity]

    Parameters
    ----------
    series : pandas.Series | list | numpy.ndarray
        Input scalar series.
    process_var : float
        Process noise variance.
    measurement_var : float
        Measurement noise variance.
    initial_cov : float
        Initial covariance scaling.
    dt : float
        Time step.
    return_velocity : bool
        If True, return a DataFrame with estimate and velocity.

    Returns
    -------
    pandas.Series | pandas.DataFrame
        Estimated series, or DataFrame with estimate and velocity.
    """
    _validate_positive_float(process_var, "process_var")
    _validate_positive_float(measurement_var, "measurement_var")
    _validate_positive_float(initial_cov, "initial_cov")
    _validate_positive_float(dt, "dt")

    s = to_series(series, name="signal")
    idx = s.index
    z_values = s.to_numpy(dtype=float)

    F = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
    H = np.array([[1.0, 0.0]], dtype=float)
    Q = process_var * np.eye(2, dtype=float)
    R = np.array([[measurement_var]], dtype=float)

    x = np.array([z_values[0], 0.0], dtype=float)
    P = initial_cov * np.eye(2, dtype=float)

    estimate = np.full(len(s), np.nan, dtype=float)
    velocity = np.full(len(s), np.nan, dtype=float)

    for i, z in enumerate(z_values):
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        y = np.array([z], dtype=float) - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x = x_pred + K @ y
        P = (np.eye(2, dtype=float) - K @ H) @ P_pred

        estimate[i] = x[0]
        velocity[i] = x[1]

    if return_velocity:
        return pd.DataFrame(
            {
                "estimate": estimate,
                "velocity": velocity,
            },
            index=idx,
        )

    return pd.Series(estimate, index=idx, name="kalman_1d")


def kalman_filter_multivariate(
    data: pd.DataFrame,
    columns: list[str] | None = None,
    process_var: float = 1e-5,
    measurement_var: float = 1e-1,
    initial_cov: float = 1e-1,
    dt: float = 1.0,
    return_velocity: bool = False,
) -> pd.DataFrame:
    """
    Causal multivariate Kalman filter with constant-velocity state model.

    For n observed variables, the state is:
        x = [values(0..n-1), velocities(0..n-1)]

    Parameters
    ----------
    data : pandas.DataFrame
        Input observed data.
    columns : list[str] | None
        Columns to filter. If None, use all numeric columns.
    process_var : float
        Process noise variance.
    measurement_var : float
        Measurement noise variance.
    initial_cov : float
        Initial covariance scaling.
    dt : float
        Time step.
    return_velocity : bool
        If True, include estimated velocity columns.

    Returns
    -------
    pandas.DataFrame
        Estimated values, optionally with velocity columns.
    """
    _validate_positive_float(process_var, "process_var")
    _validate_positive_float(measurement_var, "measurement_var")
    _validate_positive_float(initial_cov, "initial_cov")
    _validate_positive_float(dt, "dt")

    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()

    if not columns:
        raise ValueError("No numeric columns available to filter.")

    obs = data[columns].astype(float).copy()
    idx = obs.index

    prices = obs.to_numpy(dtype=float)
    n_steps, n_vars = prices.shape

    # State transition: [values, velocities]
    F = np.block(
        [
            [np.eye(n_vars), dt * np.eye(n_vars)],
            [np.zeros((n_vars, n_vars)), np.eye(n_vars)],
        ]
    )

    # Observation model: only values are measured
    H = np.block(
        [
            np.eye(n_vars),
            np.zeros((n_vars, n_vars)),
        ]
    )

    Q = process_var * np.eye(2 * n_vars)
    R = measurement_var * np.eye(n_vars)

    x = np.hstack([prices[0], np.zeros(n_vars)])
    P = initial_cov * np.eye(2 * n_vars)

    estimates = np.full((n_steps, n_vars), np.nan, dtype=float)
    velocities = np.full((n_steps, n_vars), np.nan, dtype=float)

    for i in range(n_steps):
        z = prices[i]

        # Prediction
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # Update
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x = x_pred + K @ y
        P = (np.eye(2 * n_vars) - K @ H) @ P_pred

        estimates[i, :] = x[:n_vars]
        velocities[i, :] = x[n_vars:]

    result = pd.DataFrame(estimates, index=idx, columns=columns)

    if return_velocity:
        vel_cols = [f"{col}_velocity" for col in columns]
        vel_df = pd.DataFrame(velocities, index=idx, columns=vel_cols)
        result = pd.concat([result, vel_df], axis=1)

    return result


def kalman_filter_ohlc(
    data: pd.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    process_var: float = 1e-5,
    measurement_var: float = 1e-1,
    initial_cov: float = 1e-1,
    dt: float = 1.0,
    return_velocity: bool = False,
) -> pd.DataFrame:
    """
    Convenience wrapper for filtering OHLC data.

    Returns a DataFrame with filtered OHLC columns in the same canonical order:
        open, high, low, close

    Notes
    -----
    This keeps the filter causal:
    - no future leakage
    - no symmetric smoothing
    - output length equals input length
    """
    columns = [open_col, high_col, low_col, close_col]
    result = kalman_filter_multivariate(
        data=data,
        columns=columns,
        process_var=process_var,
        measurement_var=measurement_var,
        initial_cov=initial_cov,
        dt=dt,
        return_velocity=return_velocity,
    )

    if return_velocity:
        ordered_cols = columns + [f"{col}_velocity" for col in columns]
        return result[ordered_cols]

    return result[columns]


__all__ = [
    "kalman_filter_1d",
    "kalman_filter_multivariate",
    "kalman_filter_ohlc",
]
