from collections.abc import Sequence

import numpy as np
from scipy.signal.windows import hann


def compute_fft(data_matrix: np.ndarray) -> np.ndarray:
    n_samples = data_matrix.shape[0]
    window = hann(n_samples).reshape(-1, 1)
    windowed_data = data_matrix * window
    return np.fft.rfft(windowed_data, axis=0)


def compute_banded_delays(data_matrix: np.ndarray, fs: float, bands: Sequence) -> dict:
    """
    Computes Lead-Lag matrices for specific frequency bands.
    Positive result = Column J lags Column I (I happens first).
    """
    n_samples, n_series = data_matrix.shape

    # 1. FFT
    fft_matrix = compute_fft(data_matrix)
    freqs = np.fft.rfftfreq(n_samples, d=1 / fs)

    # Remove DC
    fft_matrix = fft_matrix[1:, :]
    freqs = freqs[1:]

    # 2. Cross Spectrum Cube
    # S_cube[k, i, j] = X_i * conj(X_j)
    S_cube = fft_matrix[:, :, None] * np.conj(fft_matrix[:, None, :])

    # 3. Phase and Delay
    phase_cube = np.angle(S_cube)
    omega = (2 * np.pi * freqs)[:, None, None]

    delay_cube = phase_cube / omega
    weights_cube = np.abs(S_cube) ** 2

    auto_spectra = np.abs(fft_matrix) ** 2  # shape (n_freqs, n_series)

    results = []
    coherences = []
    for f_min, f_max in bands:
        freq_mask = (freqs >= f_min) & (freqs <= f_max)
        freqs_in_band = np.sum(freq_mask)
        if freqs_in_band == 0:
            results.append(np.full((n_series, n_series), np.nan))
            continue
        band_delays = delay_cube[freq_mask, :, :]
        band_weights = weights_cube[freq_mask, :, :]
        # Weighted Average
        numerator = np.sum(band_delays * band_weights, axis=0)
        denominator = np.sum(band_weights, axis=0)

        avg_delay_matrix = numerator / (denominator + 1e-10)
        results.append(avg_delay_matrix)

        # Coherence: |mean(S_xy)|² / (mean(S_xx) * mean(S_yy))
        band_cross = S_cube[freq_mask, :, :]
        mean_cross = np.mean(band_cross, axis=0)  # complex, shape (n_series, n_series)

        band_auto = auto_spectra[freq_mask, :]  # real, shape (n_bins, n_series)
        mean_auto = np.mean(band_auto, axis=0)  # shape (n_series,)

        coh = np.abs(mean_cross) ** 2 / (mean_auto[:, None] * mean_auto[None, :] + 1e-10)
        coherences.append(coh)

    return results, coherences


