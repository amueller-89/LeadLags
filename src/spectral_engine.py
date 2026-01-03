import numpy as np
from scipy.signal.windows import hann


def compute_fft(data_matrix: np.ndarray) -> np.ndarray:
    n_samples = data_matrix.shape[0]
    window = hann(n_samples).reshape(-1, 1)
    windowed_data = data_matrix * window
    return np.fft.rfft(windowed_data, axis=0)


def compute_banded_delays(data_matrix: np.ndarray, fs: float, bands: dict) -> dict:
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

    # FORMULA: delay = phase / omega
    delay_cube = phase_cube / omega

    # Weights (Magnitude Squared = Power)
    weights_cube = np.abs(S_cube) ** 2

    results = {}

    for band_name, (f_min, f_max) in bands.items():
        freq_mask = (freqs >= f_min) & (freqs <= f_max)

        if np.sum(freq_mask) == 0:
            results[band_name] = np.full((n_series, n_series), np.nan)
            continue

        band_delays = delay_cube[freq_mask, :, :]
        band_weights = weights_cube[freq_mask, :, :]

        # Weighted Average
        numerator = np.sum(band_delays * band_weights, axis=0)
        denominator = np.sum(band_weights, axis=0)

        avg_delay_matrix = numerator / (denominator + 1e-10)
        results[band_name] = avg_delay_matrix

    return results


def estimate_delay_pair(X_fft: np.ndarray, Y_fft: np.ndarray, freqs: np.ndarray) -> float:
    """
    Estimates delay between two pre-computed FFT vectors.
    Low-level function for specific pair analysis.
    """
    # 1. Compute Cross-Spectrum: X * conj(Y)
    # This is the reused intermediate result!
    S_xy = X_fft * np.conj(Y_fft)

    # 2. Compute Coherence (Simplified for single segment)
    # Note: In a real rolling window, you might smooth this S_xy
    # over adjacent frequencies or use Welch's method.
    P_xx = np.real(X_fft * np.conj(X_fft))
    P_yy = np.real(Y_fft * np.conj(Y_fft))
    coherence = (np.abs(S_xy) ** 2) / (P_xx * P_yy + 1e-10)

    # 3. Filter by Coherence (Masking)
    # We only trust phases where correlation is high (> 0.5)
    valid_mask = coherence > 0.5

    if np.sum(valid_mask) < 5:
        return np.nan  # Not enough reliable data points

    # 4. Phase and Delay Calculation
    # phase = angle(S_xy)
    phase = np.angle(S_xy)

    # delay(w) = -phase(w) / w
    # We ignore the DC component (freq=0) to avoid division by zero
    w = 2 * np.pi * freqs
    delays = -phase[valid_mask] / w[valid_mask]

    # 5. Aggregate Delays (Robust Median)
    # Median is safer than Mean for noisy crypto data
    avg_delay = np.median(delays)

    return avg_delay


def compute_all_vs_all_delays(data_matrix: np.ndarray, fs: float = 1.0) -> np.ndarray:
    """
    THE NUCLEAR OPTION: Computes N*N delay matrix instantly.

    Args:
        data_matrix: (n_samples, n_series) - e.g. (30, 100)
        fs: Sampling frequency

    Returns:
        delay_matrix: (n_series, n_series) where [i,j] is lag of i relative to j
    """
    n_samples, n_series = data_matrix.shape

    # 1. Pre-compute ALL FFTs (O(N) operation)
    # Result shape: (n_freqs, n_series)
    fft_matrix = compute_fft(data_matrix)
    freqs = np.fft.rfftfreq(n_samples, d=1 / fs)

    # Avoid DC component (index 0) for division stability later
    fft_matrix = fft_matrix[1:, :]
    freqs = freqs[1:]

    # 2. Vectorized Cross-Spectrum (O(N^2) but strictly matrix math)
    # We use broadcasting to compute the outer product of the series.
    # Shape transformation:
    # A: (n_freqs, n_series, 1)
    # B: (n_freqs, 1, n_series)
    # Product: (n_freqs, n_series, n_series) -> The Cross Spectrum Cube

    # This computes S_xy for EVERY pair at EVERY frequency at once.
    S_cube = fft_matrix[:, :, None] * np.conj(fft_matrix[:, None, :])

    # 3. Compute Phase Cube
    phase_cube = np.angle(S_cube)  # Shape: (n_freqs, n_series, n_series)

    # 4. Compute Weighted Delay
    # Instead of simple median, we can do a coherence-weighted average.
    # Magnitude of S_xy roughly correlates with reliability (simplified)
    weights = np.abs(S_cube)

    # delay = -phase / omega
    # Broadcast omega across the series dimensions
    omega = (2 * np.pi * freqs)[:, None, None]

    # Calculate raw delays for the whole cube
    delay_cube = -phase_cube / omega

    # 5. Collapse to 2D Matrix (Weighted Average across frequencies)
    # This squeezes the 'frequency' dimension, leaving just Series x Series
    numerator = np.sum(delay_cube * weights, axis=0)
    denominator = np.sum(weights, axis=0)

    delay_matrix = numerator / (denominator + 1e-10)

    return delay_matrix
