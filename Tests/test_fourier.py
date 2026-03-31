from signal_freq_analysis.fourier import compute_fft_magnitude

def test_fft_shapes():
    freqs, mag = compute_fft_magnitude([0, 1, 0, -1], sample_rate_hz=4.0)
    assert len(freqs) == len(mag)
