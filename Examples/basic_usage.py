from signal_analysis.adapters.ohlc import series_to_ohlc_windows
from signal_analysis.indicators import (
    compute_atr,
    compute_bollinger_bands,
    compute_macd_components,
    compute_rsi,
    compute_sma,
    compute_stochastic,
)
from signal_analysis.radar.micro_doppler import micro_doppler_features_placeholder
from signal_analysis.transforms.fourier import compute_fft_magnitude
from signal_analysis.transforms.wavelet import wavelet_energy_placeholder

signal = [0.0, 1.0, 0.0, -1.0, 0.5, 0.0, -0.5, 0.0, 1.0, 0.5, -0.2, 0.3]

print("=== RAW SIGNAL ===")
print(signal)

print("\n=== MOVING AVERAGES / MOMENTUM ===")
print("SMA:")
print(compute_sma(signal, window=3))

print("\nRSI:")
print(compute_rsi(signal, window=5))

print("\n=== BOLLINGER BANDS ===")
bb = compute_bollinger_bands(signal, window=4, n_std=2.0)
print(bb)

print("\n=== FFT ===")
freqs, mag = compute_fft_magnitude(signal, sample_rate_hz=8.0)
print("FFT bins:", freqs)
print("FFT mag:", mag)

print("\n=== WAVELET PLACEHOLDER ===")
energy = wavelet_energy_placeholder(signal)
print(energy)

print("\n=== OHLC ADAPTER ===")
ohlc = series_to_ohlc_windows(signal, window=4, step=2)
print(ohlc)

print("\n=== STOCHASTIC OVER ADAPTED OHLC ===")
stoch = compute_stochastic(
    high=ohlc["high"],
    low=ohlc["low"],
    close=ohlc["close"],
    window=2,
    smooth_window=2,
)
print(stoch)

print("\n=== ATR OVER ADAPTED OHLC ===")
atr = compute_atr(
    high=ohlc["high"],
    low=ohlc["low"],
    close=ohlc["close"],
    window=2,
)
print(atr)

print("\n=== MACD ===")
macd = compute_macd_components(signal, window_slow=6, window_fast=3, window_signal=2)
print(macd)

print("\n=== MICRO-DOPPLER PLACEHOLDER ===")
md = micro_doppler_features_placeholder(signal)
print(md)
