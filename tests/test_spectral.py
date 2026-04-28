import numpy as np
import pytest

from spectral_engine import compute_banded_delays

from .util import generate_shifted_timeseries, generate_two_shifted_timeseries


def test_pairwise_spectral_lag():
    """
    Unit test asserting that spectral lag estimation falls within
    15% relative error of the ground truth for distinct frequency bands.
    """
    # Configuration
    fs = 200.0
    t = 10.0
    tolerance = 0.15

    # Target parameters
    f_low, lag_low_target = 0.3, 1.0
    f_high, lag_high_target = 3.0, -0.1

    _, x, y = generate_two_shifted_timeseries(
        Fs=fs, T=t, f1=f_low, lag_time_1=lag_low_target, f2=f_high, lag_time_2=lag_high_target
    )

    data_matrix = np.column_stack((x, y))

    bands = [(f_low * 0.9, f_low * 1.1), (f_high * 0.9, f_high * 1.1)]

    results, coherences = compute_banded_delays(data_matrix, fs=fs, bands=bands)

    calc_low = results[0][0, 1]
    assert calc_low == pytest.approx(lag_low_target, rel=tolerance), (
        f"Low freq lag diverged. Expected {lag_low_target}, got {calc_low}"
    )

    calc_high = results[1][0, 1]
    assert calc_high == pytest.approx(lag_high_target, rel=tolerance), (
        f"High freq lag diverged. Expected {lag_high_target}, got {calc_high}"
    )

    # Coherence should be high (close to 1) for clean synthetic signals
    assert coherences[0][0, 1] > 0.8, f"Low band coherence too low: {coherences[0][0, 1]}"
    assert coherences[1][0, 1] > 0.8, f"High band coherence too low: {coherences[1][0, 1]}"

    # Diagonal should be 1 (signal is perfectly coherent with itself)
    np.testing.assert_allclose(np.diag(coherences[0]), 1.0, atol=0.01)
    np.testing.assert_allclose(np.diag(coherences[1]), 1.0, atol=0.01)


def test_nfold_spectral_lag():
    _, signals, lags, freqs = generate_shifted_timeseries()
    data_matrix = np.column_stack(signals)
    bands = [(f * 0.9, f * 1.1) for f in freqs]
    results, coherences = compute_banded_delays(data_matrix, fs=1000, bands=bands)
    for f_idx, comp_matrix in enumerate(results):
        np.testing.assert_allclose(comp_matrix, lags[:, :, f_idx], rtol=0.1, atol=0.01)

    for f_idx, coh_matrix in enumerate(coherences):
        # Off-diagonal coherence should be high for clean signals
        mask = ~np.eye(coh_matrix.shape[0], dtype=bool)
        assert np.all(coh_matrix[mask] > 0.7), f"Band {f_idx} coherence too low"
        # Diagonal should be 1
        np.testing.assert_allclose(np.diag(coh_matrix), 1.0, atol=0.01)


def test_antisymmetry():
    _, signals, _, freqs = generate_shifted_timeseries()
    data_matrix = np.column_stack(signals)
    bands = [(f * 0.9, f * 1.1) for f in freqs]
    results, coherences = compute_banded_delays(data_matrix, fs=1000, bands=bands)
    for mat in results:
        np.testing.assert_allclose(mat, -mat.T, atol=1e-10)


def test_more_series():
    """Test with 5 timeseries"""
    frequencies = [0.5, 2.0]
    amplitudes = [1.0, 0.5]
    offsets = [
        [0.0, 0.0],
        [0.3, 0.1],
        [-0.2, 0.05],
        [0.5, -0.1],
        [-0.4, 0.15],
    ]

    _, signals, lags, freqs = generate_shifted_timeseries(
        frequencies=frequencies,
        amplitudes=amplitudes,
        offsets=offsets,
    )
    data_matrix = np.column_stack(signals)
    bands = [(f * 0.9, f * 1.1) for f in freqs]
    results, coherences = compute_banded_delays(data_matrix, fs=1000, bands=bands)

    for f_idx, comp_matrix in enumerate(results):
        np.testing.assert_allclose(comp_matrix, lags[:, :, f_idx], rtol=0.1, atol=0.01)


def test_different_sampling_rates():
    """Test that fs=500 and fs=2000 both work"""
    for fs in [500, 2000]:
        _, signals, lags, freqs = generate_shifted_timeseries(Fs=fs)
        data_matrix = np.column_stack(signals)
        bands = [(f * 0.9, f * 1.1) for f in freqs]
        results, coherences = compute_banded_delays(data_matrix, fs=fs, bands=bands)

        for f_idx, comp_matrix in enumerate(results):
            np.testing.assert_allclose(comp_matrix, lags[:, :, f_idx], rtol=0.1, atol=0.01)


def test_zero_diagonal():
    """Verify diagonal entries are ~0"""
    _, signals, _, freqs = generate_shifted_timeseries()
    data_matrix = np.column_stack(signals)
    bands = [(f * 0.9, f * 1.1) for f in freqs]
    results, coherences = compute_banded_delays(data_matrix, fs=1000, bands=bands)
    for mat in results:
        np.testing.assert_allclose(np.diag(mat), 0, atol=1e-10)


def test_coherence_under_noise():
    _, signals, _, freqs = generate_shifted_timeseries(noise_std=5.0)
    data_matrix = np.column_stack(signals)
    bands = [(10.0, 11.0)]  # no signal here, just noise
    results, coherences = compute_banded_delays(data_matrix, fs=1000, bands=bands)
    print(coherences[0])


def test_high_noise():
    """Test robustness to noise - tolerances should degrade gracefully"""
    _, signals, lags, freqs = generate_shifted_timeseries(noise_std=5.0)
    data_matrix = np.column_stack(signals)
    bands = [(f * 0.9, f * 1.1) for f in freqs]
    results, coherences = compute_banded_delays(data_matrix, fs=1000, bands=bands)
    print(coherences[0])

    # Looser tolerances for noisy data
    for f_idx, comp_matrix in enumerate(results):
        np.testing.assert_allclose(comp_matrix, lags[:, :, f_idx], rtol=0.3, atol=0.05)


def test_empty_band():
    """Band with no signal energy should return NaN or not explode"""
    _, signals, _, _ = generate_shifted_timeseries()
    data_matrix = np.column_stack(signals)

    # Band where no signal exists
    bands = [(50.0, 60.0)]
    results, coherences = compute_banded_delays(data_matrix, fs=1000, bands=bands)

    # Should either be NaN or very small (just noise)
    mat = results[0]
    assert np.all(np.isnan(mat)) or np.all(np.abs(mat) < 1.0), (
        f"Empty band should give NaN or noise-level results, got {mat}"
    )

    #  Empty band should have low coherence (just noise)
    coh = coherences[0]
    mask = ~np.eye(coh.shape[0], dtype=bool)
    assert np.all(coh[mask] < 0.5), f"Empty band coherence should be low, got {coh}"


def test_frequency_at_band_edge():
    """Signal frequency at edge of band still detected"""
    frequencies = [0.95]  # Right at edge of (0.9, 1.1) band
    amplitudes = [1.0]
    offsets = [[0.0], [0.2]]

    _, signals, lags, freqs = generate_shifted_timeseries(
        frequencies=frequencies,
        amplitudes=amplitudes,
        offsets=offsets,
    )
    data_matrix = np.column_stack(signals)
    bands = [(0.9, 1.1)]
    results, coherences = compute_banded_delays(data_matrix, fs=1000, bands=bands)

    np.testing.assert_allclose(results[0], lags[:, :, 0], rtol=0.15, atol=0.02)


def test_single_bin_band():
    """Very narrow band catching only one FFT bin"""
    _, signals, lags, freqs = generate_shifted_timeseries()
    data_matrix = np.column_stack(signals)

    # With fs=1000, T=10, freq resolution is 0.1 Hz
    # Band of width 0.05 should catch at most one bin
    bands = [(0.48, 0.52)]
    results, coherences = compute_banded_delays(data_matrix, fs=1000, bands=bands)

    # Should still work, just noisier
    assert results[0].shape == (3, 3)
    assert not np.any(np.isnan(results[0])), "Single bin shouldn't produce NaN"


def test_coherence_symmetry():
    """Coherence matrix should be symmetric (unlike lag which is antisymmetric)"""
    _, signals, _, freqs = generate_shifted_timeseries()
    data_matrix = np.column_stack(signals)
    bands = [(f * 0.9, f * 1.1) for f in freqs]
    results, coherences = compute_banded_delays(data_matrix, fs=1000, bands=bands)

    for coh in coherences:
        np.testing.assert_allclose(coh, coh.T, atol=1e-10)


def test_coherence_bounds():
    """Coherence should always be in [0, 1]"""
    _, signals, _, freqs = generate_shifted_timeseries()
    data_matrix = np.column_stack(signals)
    bands = [(f * 0.9, f * 1.1) for f in freqs]
    results, coherences = compute_banded_delays(data_matrix, fs=1000, bands=bands)

    for coh in coherences:
        assert np.all(coh >= 0), f"Coherence below 0: {coh.min()}"
        assert np.all(coh <= 1 + 1e-10), f"Coherence above 1: {coh.max()}"
