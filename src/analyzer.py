"""
Lead-lag analyzer using spectral methods.

Wrapper around spectral_engine to provide high-level analysis functions.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from spectral_engine import compute_banded_delays


@dataclass
class FrequencyBand:
    """Definition of a frequency band for analysis."""

    name: str
    f_min: float  # Minimum frequency in Hz
    f_max: float  # Maximum frequency in Hz

    @property
    def period_range(self) -> tuple[float, float]:
        """Return the period range in seconds."""
        return (1 / self.f_max, 1 / self.f_min)

    def __str__(self) -> str:
        p_min, p_max = self.period_range
        return f"{self.name}: {p_min:.1f}s - {p_max:.1f}s periods"


class LeadLagAnalyzer:
    """Analyze lead-lag relationships in time series data."""

    def __init__(self, sampling_frequency: float):
        """
        Initialize analyzer.

        Args:
            sampling_frequency: Sampling frequency in Hz (e.g., 0.2 Hz for 5-second bars)
        """
        self.fs = sampling_frequency
        self.sampling_period = 1.0 / sampling_frequency

    def analyze(
        self,
        returns: pd.DataFrame,
        bands: list[FrequencyBand]
    ) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Perform lead-lag analysis across multiple frequency bands.

        Args:
            returns: DataFrame with returns (columns = assets, index = timestamps)
            bands: List of frequency bands to analyze

        Returns:
            Dictionary mapping band name to (delays_df, coherence_df) tuple
        """
        # Convert returns to numpy array
        returns_array = returns.values
        asset_names = list(returns.columns)

        # Create band tuples for compute_banded_delays
        band_tuples = [(band.f_min, band.f_max) for band in bands]

        # Run spectral analysis
        delays_list, coherences_list = compute_banded_delays(
            returns_array,
            fs=self.fs,
            bands=band_tuples
        )

        # Convert to DataFrames and package results
        results = {}
        for band, delays, coherences in zip(bands, delays_list, coherences_list):
            delays_df = pd.DataFrame(
                delays,
                index=asset_names,
                columns=asset_names
            )
            coherence_df = pd.DataFrame(
                coherences,
                index=asset_names,
                columns=asset_names
            )
            results[band.name] = (delays_df, coherence_df)

        return results

    def rank_lead_lag_relationships(
        self,
        delays_df: pd.DataFrame,
        coherence_df: pd.DataFrame,
        min_coherence: float = 0.3
    ) -> pd.DataFrame:
        """
        Rank lead-lag relationships by strength.

        Args:
            delays_df: Delay matrix (seconds)
            coherence_df: Coherence matrix (0-1)
            min_coherence: Minimum coherence threshold

        Returns:
            DataFrame with columns: leader, lagger, delay_sec, coherence
            Sorted by coherence (strongest first)
        """
        relationships = []
        n = len(delays_df)

        for i in range(n):
            for j in range(i + 1, n):  # Upper triangle only
                coh = coherence_df.iloc[i, j]

                if coh >= min_coherence:
                    delay = delays_df.iloc[i, j]

                    if delay > 0:
                        leader = delays_df.index[i]
                        lagger = delays_df.columns[j]
                    else:
                        leader = delays_df.columns[j]
                        lagger = delays_df.index[i]
                        delay = abs(delay)

                    relationships.append({
                        'leader': leader,
                        'lagger': lagger,
                        'delay_sec': delay,
                        'coherence': coh
                    })

        # Convert to DataFrame and sort
        df = pd.DataFrame(relationships)

        if not df.empty:
            df = df.sort_values('coherence', ascending=False)

        return df

    def get_asset_leadership_score(
        self,
        delays_df: pd.DataFrame,
        coherence_df: pd.DataFrame,
        min_coherence: float = 0.3
    ) -> pd.Series:
        """
        Calculate leadership score for each asset.

        Leadership score = weighted average of lead times, where weights are coherences.
        Positive score = tends to lead, negative score = tends to lag.

        Args:
            delays_df: Delay matrix (seconds)
            coherence_df: Coherence matrix (0-1)
            min_coherence: Minimum coherence threshold

        Returns:
            Series with leadership scores for each asset
        """
        scores = {}
        assets = delays_df.index

        for asset in assets:
            # Get delays where this asset is involved
            delays_as_row = delays_df.loc[asset, :]  # asset vs others
            coherences = coherence_df.loc[asset, :]

            # Filter by coherence threshold
            mask = (coherences >= min_coherence) & (coherences.index != asset)

            if mask.sum() == 0:
                scores[asset] = 0.0
                continue

            # Weighted average of delays (coherence as weights)
            weighted_delay = (delays_as_row[mask] * coherences[mask]).sum() / coherences[mask].sum()
            scores[asset] = weighted_delay

        return pd.Series(scores).sort_values(ascending=False)


def create_crypto_frequency_bands(sampling_freq: float) -> list[FrequencyBand]:
    """
    Create appropriate frequency bands for cryptocurrency analysis.

    Automatically adjusts bands based on sampling frequency to ensure all bands
    are within Nyquist limits. Different sampling rates get different band definitions.

    Args:
        sampling_freq: Sampling frequency in Hz
            - 0.2 Hz (5s bars) from tick data: ultra-fast to medium
            - 0.0167 Hz (1min OHLCV): short to long term
            - Lower frequencies: intraday to daily

    Returns:
        List of FrequencyBand objects valid for the given sampling rate

    Examples:
        >>> # For 1-minute OHLCV data
        >>> bands = create_crypto_frequency_bands(1/60)  # 0.0167 Hz
        >>> # Returns: short_term (2.5-10min), medium_term (10-60min), long_term (1-6hr)

        >>> # For 5-second tick data
        >>> bands = create_crypto_frequency_bands(1/5)  # 0.2 Hz
        >>> # Returns: ultra_fast (15s-1min), fast (1-10min), medium (10-60min)
    """
    nyquist = sampling_freq / 2
    min_period = 2.0 / sampling_freq  # Nyquist limit (minimum detectable period)

    print(f"\n{'='*60}")
    print(f"Frequency Band Generation")
    print(f"{'='*60}")
    print(f"Sampling frequency: {sampling_freq:.6f} Hz ({1/sampling_freq:.1f}s period)")
    print(f"Nyquist frequency:  {nyquist:.6f} Hz ({1/nyquist:.1f}s period)")
    print(f"Min detectable period: {min_period:.1f}s")
    print()

    # Choose bands based on sampling frequency
    # High frequency: 5-10 second bars (from tick data)
    if sampling_freq >= 0.15:
        bands = [
            FrequencyBand(
                name="ultra_fast",
                f_min=1/60,    # 1 minute period
                f_max=1/15     # 15 seconds period
            ),
            FrequencyBand(
                name="fast",
                f_min=1/600,   # 10 minutes period
                f_max=1/60     # 1 minute period
            ),
            FrequencyBand(
                name="medium",
                f_min=1/3600,  # 1 hour period
                f_max=1/600    # 10 minutes period
            ),
        ]
        print("Detected: High-frequency data (tick-derived 5-10s bars)")

    # Medium frequency: 1-minute OHLCV bars
    elif sampling_freq >= 0.015:
        bands = [
            FrequencyBand(
                name="short_term",
                f_min=1/600,    # 10 minutes period
                f_max=1/150,    # 2.5 minutes period (well below Nyquist limit)
            ),
            FrequencyBand(
                name="medium_term",
                f_min=1/3600,   # 1 hour period
                f_max=1/600     # 10 minutes period
            ),
            FrequencyBand(
                name="long_term",
                f_min=1/21600,  # 6 hours period
                f_max=1/3600    # 1 hour period
            ),
        ]
        print("Detected: 1-minute OHLCV data (Binance native resolution)")

    # Low frequency: 5-minute or hourly bars
    elif sampling_freq >= 0.002:
        bands = [
            FrequencyBand(
                name="short_term",
                f_min=1/3600,   # 1 hour period
                f_max=1/600     # 10 minutes period
            ),
            FrequencyBand(
                name="medium_term",
                f_min=1/14400,  # 4 hours period
                f_max=1/3600    # 1 hour period
            ),
            FrequencyBand(
                name="long_term",
                f_min=1/86400,  # 1 day period
                f_max=1/14400   # 4 hours period
            ),
        ]
        print("Detected: 5-15 minute bars")

    # Very low frequency: hourly or daily bars
    else:
        bands = [
            FrequencyBand(
                name="intraday",
                f_min=1/86400,  # 1 day period
                f_max=1/7200    # 2 hours period
            ),
            FrequencyBand(
                name="daily",
                f_min=1/604800, # 1 week period
                f_max=1/86400   # 1 day period
            ),
        ]
        print("Detected: Hourly or daily bars")

    # Validate and filter bands
    valid_bands = []
    print("\nBand Validation:")
    print("-" * 60)

    for band in bands:
        p_min, p_max = band.period_range
        if band.f_max <= nyquist:
            valid_bands.append(band)
            print(f"  ✓ {band.name:15s} : {p_min:>6.0f}s - {p_max:>8.0f}s periods")
        else:
            print(f"  ✗ {band.name:15s} : {p_min:>6.0f}s - {p_max:>8.0f}s (EXCEEDS NYQUIST LIMIT)")

    if not valid_bands:
        raise ValueError(
            f"\nNo valid frequency bands for sampling frequency {sampling_freq:.6f} Hz.\n"
            f"Nyquist limit is {nyquist:.6f} Hz (minimum period {min_period:.1f}s).\n"
            f"All requested bands exceed the Nyquist frequency."
        )

    print("=" * 60)
    return valid_bands


if __name__ == '__main__':
    # Example usage
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    from data_loader import BinanceDataLoader, get_default_crypto_symbols
    from preprocessor import CryptoPreprocessor, PreprocessConfig

    # Load and preprocess data
    loader = BinanceDataLoader()
    symbols = get_default_crypto_symbols()

    data = loader.fetch_multiple_symbols(
        symbols=symbols,
        timeframe='1m',
        days_back=1,
        save=True
    )

    config = PreprocessConfig(resample_freq='1min')
    preprocessor = CryptoPreprocessor(config)
    _, returns = preprocessor.process_pipeline(data)

    # Analyze
    sampling_freq = 1.0 / 60.0  # 1-minute bars = 0.0167 Hz
    analyzer = LeadLagAnalyzer(sampling_freq)

    bands = create_crypto_frequency_bands(sampling_freq)
    results = analyzer.analyze(returns, bands)

    # Display results
    for band_name, (delays_df, coherence_df) in results.items():
        print(f"\n{'='*60}")
        print(f"Band: {band_name}")
        print(f"{'='*60}")

        # Rank relationships
        ranked = analyzer.rank_lead_lag_relationships(
            delays_df,
            coherence_df,
            min_coherence=0.3
        )

        if not ranked.empty:
            print("\nTop Lead-Lag Relationships:")
            print(ranked.head(10).to_string(index=False))

        # Leadership scores
        scores = analyzer.get_asset_leadership_score(
            delays_df,
            coherence_df,
            min_coherence=0.3
        )

        print("\nLeadership Scores (positive = leads, negative = lags):")
        print(scores.to_string())
