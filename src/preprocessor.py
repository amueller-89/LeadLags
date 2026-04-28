"""
Data preprocessing for lead-lag analysis.

Handles timestamp alignment, resampling, returns calculation, and data quality.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing pipeline."""

    resample_freq: str = '1min'  # Native OHLCV resolution (1min). Use '5min', '15min', '1h' for downsampling.
    fill_method: str = 'ffill'  # Forward fill for crypto (trades continue between bars)
    min_valid_ratio: float = 0.95  # Minimum fraction of valid data required
    remove_outliers: bool = True
    outlier_std: float = 10.0  # Remove returns beyond this many std devs
    price_column: str = 'close'  # Which price to use for returns


class CryptoPreprocessor:
    """Preprocesses cryptocurrency data for lead-lag analysis."""

    def __init__(self, config: Optional[PreprocessConfig] = None):
        """
        Initialize preprocessor.

        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessConfig()

    def _parse_freq_to_seconds(self, freq_str: str) -> float:
        """
        Parse frequency string to seconds.

        Args:
            freq_str: Frequency string like '5s', '1min', '5min', '1h', '1d'

        Returns:
            Period in seconds

        Examples:
            '5s' -> 5.0
            '1min' -> 60.0
            '5min' -> 300.0
            '1h' -> 3600.0
        """
        freq_str = freq_str.lower().strip()

        if freq_str.endswith('s'):
            num_str = freq_str[:-1]
            # Handle pandas inferred 's' (which means '1s')
            if num_str == '' or num_str == 'T':
                return 1.0
            return float(num_str)
        elif freq_str.endswith('min') or freq_str.endswith('t'):
            mins = freq_str[:-3] if freq_str.endswith('min') else freq_str[:-1]
            # Handle pandas inferred 'min' (which means '1min')
            if mins == '' or mins == 'T':
                return 60.0
            return float(mins) * 60
        elif freq_str.endswith('h'):
            num_str = freq_str[:-1]
            # Handle pandas inferred 'h' (which means '1h')
            if num_str == '':
                return 3600.0
            return float(num_str) * 3600
        elif freq_str.endswith('d'):
            num_str = freq_str[:-1]
            # Handle pandas inferred 'd' (which means '1d')
            if num_str == '':
                return 86400.0
            return float(num_str) * 86400
        else:
            raise ValueError(
                f"Cannot parse frequency string: '{freq_str}'. "
                f"Expected format: <number><unit> where unit is 's', 'min', 'h', or 'd'. "
                f"Examples: '5s', '1min', '5min', '1h', '1d'"
            )

    def align_timestamps(
        self,
        data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Align timestamps across all symbols to a common time grid.

        Args:
            data_dict: Dictionary mapping symbol to OHLCV DataFrame

        Returns:
            Dictionary with aligned DataFrames
        """
        if not data_dict:
            raise ValueError("Empty data dictionary")

        # First pass: Remove duplicate timestamps from all symbols
        for symbol, df in data_dict.items():
            if df.index.duplicated().any():
                n_duplicates = df.index.duplicated().sum()
                print(f"Warning: Found {n_duplicates} duplicate timestamps in {symbol}")
                print("Removing duplicates (keeping first occurrence)...")
                data_dict[symbol] = df[~df.index.duplicated(keep='first')]

            # Check if we have enough data points after cleaning
            if len(data_dict[symbol]) < 3:
                raise ValueError(
                    f"Insufficient data for {symbol}: only {len(data_dict[symbol])} unique timestamp(s) after removing duplicates. "
                    "Need at least 3 data points for analysis."
                )

        # Find common time range
        start_times = [df.index.min() for df in data_dict.values()]
        end_times = [df.index.max() for df in data_dict.values()]

        common_start = max(start_times)
        common_end = min(end_times)

        print(f"Aligning timestamps from {common_start} to {common_end}")

        # Create common time index
        # Get the frequency from the first dataset
        first_df = next(iter(data_dict.values()))
        first_symbol = next(iter(data_dict.keys()))

        inferred_freq = pd.infer_freq(first_df.index[:100])  # Infer from first 100 rows
        print(f"Inferred frequency from {first_symbol}: {inferred_freq}")

        if inferred_freq is None:
            # If frequency can't be inferred, use the median time diff
            time_diffs = first_df.index[1:] - first_df.index[:-1]
            median_diff = time_diffs.median()
            print(f"Could not infer frequency. Calculated median time diff: {median_diff}")
            print(f"  Min diff: {time_diffs.min()}, Max diff: {time_diffs.max()}")

            # Check for zero or invalid frequency
            if median_diff <= pd.Timedelta(0):
                raise ValueError(
                    f"Invalid median time difference: {median_diff}. "
                    "Data may have duplicate timestamps or invalid time ordering. "
                    "Check your input data for issues."
                )

            # Check if median_diff is too small (less than 1 nanosecond)
            if median_diff < pd.Timedelta(nanoseconds=1):
                raise ValueError(
                    f"Median time difference too small: {median_diff}. "
                    "Cannot create valid date range. Check data frequency."
                )

            try:
                common_index = pd.date_range(
                    start=common_start,
                    end=common_end,
                    freq=median_diff
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to create date range with freq={median_diff}: {e}. "
                    f"Data frequency may be irregular. Consider resampling first."
                )
        else:
            common_index = pd.date_range(
                start=common_start,
                end=common_end,
                freq=inferred_freq
            )

        # Reindex all dataframes to common index
        aligned_data = {}
        for symbol, df in data_dict.items():
            # Filter to common range first
            df_filtered = df[(df.index >= common_start) & (df.index <= common_end)]

            # Reindex to common grid
            df_aligned = df_filtered.reindex(common_index, method=self.config.fill_method)

            # Forward fill any remaining NaNs (from gaps)
            df_aligned = df_aligned.ffill()

            # Check data quality
            valid_ratio = df_aligned.notna().all(axis=1).sum() / len(df_aligned)
            if valid_ratio < self.config.min_valid_ratio:
                print(f"Warning: {symbol} has only {valid_ratio:.1%} valid data")

            aligned_data[symbol] = df_aligned

        print(f"Aligned {len(aligned_data)} symbols to {len(common_index)} timestamps")

        return aligned_data

    def resample_ohlcv(
        self,
        data_dict: Dict[str, pd.DataFrame],
        freq: Optional[str] = None,
        source_native_freq: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Resample OHLCV/tick data to a different frequency.

        Args:
            data_dict: Dictionary mapping symbol to OHLCV DataFrame
            freq: Target frequency (e.g., '1s', '5s', '1min', '5min', '15min', '1h'). Uses config if None.
            source_native_freq: Native resolution of source data. Auto-detected if None.

        Returns:
            Dictionary with resampled DataFrames

        Raises:
            ValueError: If attempting to upsample OHLCV data (create interpolated fake data)
        """
        freq = freq or self.config.resample_freq

        # Auto-detect source frequency if not provided
        if source_native_freq is None:
            first_df = next(iter(data_dict.values()))
            inferred = pd.infer_freq(first_df.index[:min(100, len(first_df))])

            if inferred:
                source_native_freq = inferred
                print(f"Detected source frequency: {source_native_freq}")
            else:
                # Calculate from median time diff
                if len(first_df) > 1:
                    time_diffs = first_df.index[1:] - first_df.index[:-1]
                    median_seconds = time_diffs.median().total_seconds()

                    # Convert to frequency string
                    if median_seconds < 60:
                        source_native_freq = f"{int(median_seconds)}s"
                    elif median_seconds < 3600:
                        source_native_freq = f"{int(median_seconds/60)}min"
                    else:
                        source_native_freq = f"{int(median_seconds/3600)}h"

                    print(f"Inferred source frequency from median: {source_native_freq} ({median_seconds}s)")
                else:
                    # Default to 1min if we can't determine
                    source_native_freq = '1min'
                    print(f"Warning: Could not infer frequency, assuming {source_native_freq}")

        # Validate resampling direction
        source_seconds = self._parse_freq_to_seconds(source_native_freq)
        target_seconds = self._parse_freq_to_seconds(freq)

        # Only block upsampling for OHLCV data (>= 1 minute resolution)
        # Allow any resampling for sub-minute tick data
        if target_seconds < source_seconds and source_seconds >= 60:
            # This is upsampling OHLCV data - not allowed
            num_fake_bars = int(source_seconds / target_seconds)
            raise ValueError(
                f"\nCannot resample OHLCV data from {source_native_freq} to {freq}.\n\n"
                f"This would create {num_fake_bars} interpolated fake bars between each real bar.\n"
                f"Spectral analysis on fake data produces meaningless results.\n\n"
                f"Solutions:\n"
                f"  1. Use native resolution: --resample {source_native_freq}\n"
                f"  2. Downsample only: --resample 5min, --resample 15min, --resample 1h\n"
                f"  3. For sub-second/sub-minute analysis, load tick data from cache\n\n"
                f"Current data: {source_native_freq} OHLCV bars from Binance.\n"
                f"Cannot create real {freq} bars without actual trade data."
            )

        if target_seconds == source_seconds:
            print(f"Using native resolution: {freq} bars (no resampling needed)")
        elif target_seconds < source_seconds:
            # This is upsampling tick data - allowed since we have the underlying resolution
            print(f"Resampling tick data from {source_native_freq} to {freq} bars")
        else:
            print(f"Downsampling from {source_native_freq} to {freq} bars (aggregating {source_seconds/target_seconds:.1f}:1)")

        resampled_data = {}
        for symbol, df in data_dict.items():
            # Standard OHLCV resampling rules
            resampled = df.resample(freq).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })

            # Forward fill any NaN values (from periods with no trades)
            resampled = resampled.ffill()

            # Drop any remaining NaNs at the start
            resampled = resampled.dropna()

            resampled_data[symbol] = resampled

        print(f"Resampled {len(resampled_data)} symbols")

        return resampled_data

    def calculate_returns(
        self,
        data_dict: Dict[str, pd.DataFrame],
        log_returns: bool = True
    ) -> pd.DataFrame:
        """
        Calculate returns for each symbol.

        Args:
            data_dict: Dictionary mapping symbol to OHLCV DataFrame
            log_returns: If True, use log returns; otherwise simple returns

        Returns:
            DataFrame with returns for each symbol (columns = symbols)
        """
        price_col = self.config.price_column
        returns_dict = {}

        for symbol, df in data_dict.items():
            if price_col not in df.columns:
                raise ValueError(f"{price_col} column not found in {symbol}")

            prices = df[price_col]

            if log_returns:
                returns = np.log(prices / prices.shift(1))
            else:
                returns = prices.pct_change()

            # Clean symbol name for column (remove /)
            clean_symbol = symbol.replace('/', '_').replace('USDT', '').strip('_')
            returns_dict[clean_symbol] = returns

        # Combine into single DataFrame
        returns_df = pd.DataFrame(returns_dict)

        # Drop NaNs (first row will be NaN)
        returns_df = returns_df.dropna()

        print(f"Calculated returns: {returns_df.shape[0]} rows x {returns_df.shape[1]} assets")

        return returns_df

    def remove_outliers(
        self,
        returns_df: pd.DataFrame,
        n_std: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Remove outlier returns beyond n standard deviations.

        Args:
            returns_df: DataFrame of returns
            n_std: Number of standard deviations. Uses config if None.

        Returns:
            DataFrame with outliers clipped
        """
        n_std = n_std or self.config.outlier_std

        print(f"Removing outliers beyond {n_std} standard deviations")

        cleaned = returns_df.copy()

        for col in cleaned.columns:
            mean = cleaned[col].mean()
            std = cleaned[col].std()

            # Clip values beyond n_std
            lower_bound = mean - n_std * std
            upper_bound = mean + n_std * std

            n_outliers = ((cleaned[col] < lower_bound) | (cleaned[col] > upper_bound)).sum()
            if n_outliers > 0:
                print(f"  {col}: clipped {n_outliers} outliers")

            cleaned[col] = cleaned[col].clip(lower_bound, upper_bound)

        return cleaned

    def get_data_quality_report(
        self,
        data_dict: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Generate data quality report.

        Args:
            data_dict: Dictionary mapping symbol to DataFrame

        Returns:
            DataFrame with quality metrics for each symbol
        """
        quality_metrics = []

        for symbol, df in data_dict.items():
            metrics = {
                'symbol': symbol,
                'n_rows': len(df),
                'start_time': df.index.min(),
                'end_time': df.index.max(),
                'duration_hours': (df.index.max() - df.index.min()).total_seconds() / 3600,
                'missing_values': df.isna().sum().sum(),
                'missing_pct': df.isna().sum().sum() / (len(df) * len(df.columns)) * 100,
            }

            # Check for gaps in timestamps
            time_diffs = df.index[1:] - df.index[:-1]
            median_diff = time_diffs.median()
            large_gaps = (time_diffs > median_diff * 2).sum()
            metrics['large_gaps'] = large_gaps

            quality_metrics.append(metrics)

        return pd.DataFrame(quality_metrics)

    def process_pipeline(
        self,
        data_dict: Dict[str, pd.DataFrame],
        align: bool = True,
        resample: bool = True,
        calc_returns: bool = True,
        remove_outliers: bool = True
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Run full preprocessing pipeline.

        Args:
            data_dict: Dictionary mapping symbol to OHLCV DataFrame
            align: Whether to align timestamps
            resample: Whether to resample to target frequency
            calc_returns: Whether to calculate returns
            remove_outliers: Whether to remove outlier returns

        Returns:
            Tuple of (processed OHLCV dict, returns DataFrame)
        """
        print("=" * 60)
        print("Starting preprocessing pipeline")
        print("=" * 60)

        # Step 1: Data quality report
        quality_report = self.get_data_quality_report(data_dict)
        print("\nData Quality Report:")
        print(quality_report.to_string(index=False))

        # Step 2: Align timestamps
        if align:
            print("\n" + "=" * 60)
            data_dict = self.align_timestamps(data_dict)

        # Step 3: Resample
        if resample:
            print("\n" + "=" * 60)
            data_dict = self.resample_ohlcv(data_dict)

        # Step 4: Calculate returns
        returns_df = None
        if calc_returns:
            print("\n" + "=" * 60)
            returns_df = self.calculate_returns(data_dict)

            # Step 5: Remove outliers
            if remove_outliers and self.config.remove_outliers:
                print("\n" + "=" * 60)
                returns_df = self.remove_outliers(returns_df)

            # Print final statistics
            print("\n" + "=" * 60)
            print("Returns Statistics:")
            print(returns_df.describe())

        print("\n" + "=" * 60)
        print("Preprocessing complete!")
        print("=" * 60)

        return data_dict, returns_df


if __name__ == '__main__':
    # Example usage
    from data_loader import BinanceDataLoader, get_default_crypto_symbols

    # Load data
    loader = BinanceDataLoader()
    symbols = get_default_crypto_symbols()

    # Try to load existing data
    print("Loading data...")
    data = loader.fetch_multiple_symbols(
        symbols=symbols,
        timeframe='1m',
        days_back=1,  # Just 1 day for testing
        save=True
    )

    # Preprocess
    config = PreprocessConfig(
        resample_freq='1min',  # Native resolution (Binance OHLCV)
        remove_outliers=True,
        outlier_std=10.0
    )

    preprocessor = CryptoPreprocessor(config)
    processed_ohlcv, returns = preprocessor.process_pipeline(data)

    print(f"\nFinal returns shape: {returns.shape}")
    print(f"Columns: {list(returns.columns)}")
