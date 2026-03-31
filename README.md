# signal_freq_analysis

Reusable signal and frequency analysis library for Fourier, Wavelet, TDA, and Micro-Doppler workflows.

## Install (editable)
```bash
pip install -e .
```

## Modules
- `signal_freq_analysis.fourier`
- `signal_freq_analysis.wavelet`
- `signal_freq_analysis.tda`
- `signal_freq_analysis.micro_doppler`

## Example
```python
from signal_freq_analysis.fourier import compute_fft_magnitude

freqs, mag = compute_fft_magnitude([0, 1, 0, -1], sample_rate_hz=4.0)
print(freqs)
print(mag)
```
