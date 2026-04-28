import numpy as np


def generate_two_shifted_timeseries(
    Fs=100, T=10, f1=0.5, lag_time_1=2.0, f2=3.0, lag_time_2=-0.5, seed=42
):
    """
    Generates synthetic signal pair (X, Y) where Y is a linear superposition
    of frequency-shifted components of X.

    """
    t = np.arange(0, T, 1 / Fs)
    n_samples = len(t)

    np.random.seed(seed)
    noise_x = np.random.normal(0, 0.5, n_samples)
    x = 1.0 * np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t) + noise_x

    noise_y = np.random.normal(0, 0.5, n_samples)
    y = (
        1.0 * np.sin(2 * np.pi * f1 * (t - lag_time_1))
        + 0.5 * np.sin(2 * np.pi * f2 * (t - lag_time_2))
        + noise_y
    )

    return t, x, y


def generate_shifted_timeseries(
    Fs=1000,
    T=10,
    frequencies=None,
    amplitudes=None,
    offsets=None,
    noise_std=0.5,
    seed=42,
):
    """
    Generates N synthetic timeseries where each has frequency-specific
    time offsets relative to a common reference signal.

    Parameters
    ----------
    Fs : float
        Sampling frequency in Hz
    T : float
        Duration in seconds
    frequencies : list of float
        Frequencies of the sinusoidal components
    amplitudes : list of float
        Amplitude of each frequency component (same across all series)
    offsets : list of list of float
        offsets[i][f] is the time offset of series i at frequency f.
        Positive = lag, negative = lead. First row is typically zeros (reference).
    noise_std : float
        Standard deviation of additive Gaussian noise
    seed : int
        Random seed

    Returns
    -------
    t : ndarray
        Time vector
    signals : list of ndarray
        List of N timeseries
    """
    if offsets is None:
        offsets = [[0.0, 0.0, 0.0], [0.5, -0.05, 0.2], [0.5, 0.05, -0.1]]
    if amplitudes is None:
        amplitudes = [1.0, 0.5, 0.3]
    if frequencies is None:
        frequencies = [0.5, 3.0, 1.0]
    assert len(frequencies) == len(amplitudes), "Need one amplitude per frequency"
    assert all(len(row) == len(frequencies) for row in offsets), (
        "Each offset row must match number of frequencies"
    )

    assert np.all(np.abs(np.array(offsets)) < 1 / (2.0 * np.array(frequencies))), (
        "at frequency f, only shifts of magnitude 1/(2f) can be detected"
    )

    t = np.arange(0, T, 1 / Fs)
    n_samples = len(t)
    np.random.seed(seed)

    signals = []
    for series_offsets in offsets:
        signal = np.zeros(n_samples)
        for freq, amp, offset in zip(frequencies, amplitudes, series_offsets):
            signal += amp * np.sin(2 * np.pi * freq * (t - offset))
        signal += np.random.normal(0, noise_std, n_samples)
        signals.append(signal)

    offsets = np.array(offsets)
    lags = offsets[None, :, :] - offsets[:, None, :]
    return t, signals, lags, frequencies
