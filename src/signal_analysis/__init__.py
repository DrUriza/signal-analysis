"""
signal_analysis package.

Reusable signal-analysis toolkit for indicators, transforms, topology,
filters, adapters, and radar-oriented feature extraction.

Current stable public API:
- indicators
- utils
- adapters.ohlc
- transforms.fourier

Some submodules are still placeholders and can be expanded incrementally.
"""
from signal_analysis.filters.kalman import (kalman_filter_1d, kalman_filter_multivariate, kalman_filter_ohlc) 

from signal_analysis.indicators import (
    compute_adx,
    compute_atr,
    compute_bollinger_bands,
    compute_directional_indicators,
    compute_ema,
    compute_kama,
    compute_macd,
    compute_macd_components,
    compute_macd_hist,
    compute_macd_signal,
    compute_minus_di,
    compute_plus_di,
    compute_roc,
    compute_rsi,
    compute_sma,
    compute_stochastic,
    compute_stochastic_signal,
    compute_stochrsi,
    compute_stochrsi_d,
    compute_stochrsi_k,
    compute_tsi,
    compute_tr,
    compute_williams_r,
    compute_wma,
)
from signal_analysis.utils import (
    cross_above,
    cross_below,
    cross_level_down,
    cross_level_up,
    ema_series,
    to_series,
    true_range,
    validate_window,
)
from signal_analysis.adapters.ohlc import series_to_ohlc_windows
from signal_analysis.transforms.fourier import compute_fft_magnitude

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "compute_sma",
    "compute_ema",
    "compute_wma",
    "compute_kama",
    "compute_rsi",
    "compute_tsi",
    "compute_roc",
    "compute_bollinger_bands",
    "compute_tr",
    "compute_atr",
    "compute_macd_components",
    "compute_macd",
    "compute_macd_signal",
    "compute_macd_hist",
    "compute_plus_di",
    "compute_minus_di",
    "compute_adx",
    "compute_directional_indicators",
    "compute_stochastic",
    "compute_stochastic_signal",
    "compute_williams_r",
    "compute_stochrsi",
    "compute_stochrsi_k",
    "compute_stochrsi_d",
    "to_series",
    "validate_window",
    "ema_series",
    "true_range",
    "cross_above",
    "cross_below",
    "cross_level_up",
    "cross_level_down",
    "series_to_ohlc_windows",
    "compute_fft_magnitude",
    "kalman_filter_1d",
    "kalman_filter_multivariate",
    "kalman_filter_ohlc"]
