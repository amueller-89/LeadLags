import numpy as np
import pytest

from spectral_engine import compute_banded_delays


def generate_shifted_timeseries(
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


def test_spectral_lag_estimation_accuracy():
    """
    Unit test asserting that spectral lag estimation falls within
    15% relative error of the ground truth for distinct frequency bands.
    """
    # 1. Configuration
    fs = 200.0  # noqa: N806
    t = 10.0  # noqa: N806
    tolerance = 0.15  # 15% allowed deviation  # noqa: N806

    # Target parameters
    f_low, lag_low_target = 0.3, 1.0
    f_high, lag_high_target = 3.0, -0.1

    _, x, y = generate_shifted_timeseries(
        Fs=fs, T=t, f1=f_low, lag_time_1=lag_low_target, f2=f_high, lag_time_2=lag_high_target
    )

    data_matrix = np.column_stack((x, y))

    bands = {
        "low": (f_low * 0.9, f_low * 1.1),
        "high": (f_high * 0.9, f_high * 1.1),
    }

    results = compute_banded_delays(data_matrix, fs=fs, bands=bands)

    calc_low = results["low"][0, 1]
    assert calc_low == pytest.approx(lag_low_target, rel=tolerance), (
        f"Low freq lag diverged. Expected {lag_low_target}, got {calc_low}"
    )

    calc_high = results["high"][0, 1]
    assert calc_high == pytest.approx(lag_high_target, rel=tolerance), (
        f"High freq lag diverged. Expected {lag_high_target}, got {calc_high}"
    )
