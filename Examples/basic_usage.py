from signal_freq_analysis.fourier import compute_fft_magnitude
from signal_freq_analysis.wavelet import moving_window_energy
from signal_freq_analysis.tda import sliding_window_embedding
from signal_freq_analysis.micro_doppler import compute_md_proxy_features

signal = [0.0, 1.0, 0.0, -1.0, 0.5, 0.0, -0.5, 0.0]

freqs, mag = compute_fft_magnitude(signal, sample_rate_hz=8.0)
energy = moving_window_energy(signal, window_size=4)
embedding = sliding_window_embedding(signal, dimension=3, delay=1)
md = compute_md_proxy_features(signal)

print("FFT bins:", freqs)
print("FFT mag:", mag)
print("Wavelet-like energy:", energy)
print("Embedding shape:", embedding.shape)
print("MD features:", md)
