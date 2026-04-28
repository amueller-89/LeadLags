"""
Test script to validate the full pipeline on real Binance data.

This script:
1. Fetches recent crypto data from Binance
2. Preprocesses the data (align, resample, calculate returns)
3. Runs spectral analysis to detect lead-lag relationships
4. Displays results
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import pandas as pd
from data_loader import BinanceDataLoader, get_default_crypto_symbols
from preprocessor import CryptoPreprocessor, PreprocessConfig
from spectral_engine import compute_banded_delays


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_delay_matrix(delays, coherence, symbols, band_name):
    """Print a formatted delay matrix with coherence."""
    print(f"\n{band_name}:")
    print("\nDelay Matrix (seconds) - Positive = Row leads Column:")
    print("-" * 60)

    # Create formatted table
    header = "        " + "".join(f"{s:>10}" for s in symbols)
    print(header)

    for i, row_symbol in enumerate(symbols):
        row_values = "".join(f"{delays[i,j]:>10.3f}" for j in range(len(symbols)))
        print(f"{row_symbol:>8}{row_values}")

    print("\nCoherence Matrix (0-1, higher = stronger relationship):")
    print("-" * 60)
    print(header)

    for i, row_symbol in enumerate(symbols):
        row_values = "".join(f"{coherence[i,j]:>10.3f}" for j in range(len(symbols)))
        print(f"{row_symbol:>8}{row_values}")


def analyze_lead_lag_relationships(delays, coherence, symbols, band_name, threshold=0.3):
    """Analyze and report lead-lag relationships."""
    print(f"\n{band_name} - Key Relationships (coherence > {threshold}):")
    print("-" * 60)

    relationships = []

    n = len(symbols)
    for i in range(n):
        for j in range(i + 1, n):  # Only upper triangle
            coh = coherence[i, j]
            if coh > threshold:
                delay = delays[i, j]
                if delay > 0:
                    leader = symbols[i]
                    lagger = symbols[j]
                else:
                    leader = symbols[j]
                    lagger = symbols[i]
                    delay = -delay

                relationships.append({
                    'leader': leader,
                    'lagger': lagger,
                    'delay_sec': delay,
                    'coherence': coh
                })

    # Sort by coherence (strongest relationships first)
    relationships.sort(key=lambda x: x['coherence'], reverse=True)

    if relationships:
        for rel in relationships:
            print(f"  {rel['leader']:>8} leads {rel['lagger']:>8} by "
                  f"{rel['delay_sec']:>6.2f}s (coherence={rel['coherence']:.3f})")
    else:
        print(f"  No strong relationships found (coherence > {threshold})")


def main():
    """Run the full test pipeline."""

    print_section("CRYPTO LEAD-LAG ANALYSIS - REAL DATA TEST")

    # Configuration
    symbols = get_default_crypto_symbols()
    days_back = 2  # Fetch 2 days of data (limited for testing)
    resample_freq = '1min'  # Native Binance OHLCV resolution (1-minute bars)

    print(f"\nConfiguration:")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Period: {days_back} days")
    print(f"  Resample frequency: {resample_freq}")

    # Step 1: Fetch data
    print_section("Step 1: Fetching Data from Binance")

    loader = BinanceDataLoader()

    try:
        data = loader.fetch_multiple_symbols(
            symbols=symbols,
            timeframe='1m',  # Start with 1-minute data
            days_back=days_back,
            save=True
        )
    except Exception as e:
        print(f"\nError fetching data: {e}")
        print("\nTrying to load from cached files...")
        available = loader.list_available_data()
        if available:
            print(f"Available cached files: {available[:5]}")
            print("Please run the data_loader.py script manually first to cache data")
        else:
            print("No cached data found. Please check your internet connection.")
        return

    if not data:
        print("No data fetched. Exiting.")
        return

    # Step 2: Preprocess
    print_section("Step 2: Preprocessing Data")

    config = PreprocessConfig(
        resample_freq=resample_freq,
        remove_outliers=True,
        outlier_std=10.0,
        price_column='close'
    )

    preprocessor = CryptoPreprocessor(config)

    try:
        processed_ohlcv, returns = preprocessor.process_pipeline(
            data,
            align=True,
            resample=True,
            calc_returns=True,
            remove_outliers=True
        )
    except Exception as e:
        print(f"\nError during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return

    if returns is None or returns.empty:
        print("No returns calculated. Exiting.")
        return

    # Step 3: Spectral Analysis
    print_section("Step 3: Spectral Lead-Lag Analysis")

    # Calculate sampling frequency based on resample period
    # For 1-minute bars: 1/60 Hz = 0.0167 Hz
    sampling_freq = 1.0 / 60.0  # 1-minute bars

    # Use auto-detected frequency bands from analyzer
    from analyzer import create_crypto_frequency_bands
    bands_obj = create_crypto_frequency_bands(sampling_freq)

    # Convert to tuple format for backwards compatibility
    bands = [(band.f_min, band.f_max) for band in bands_obj]
    band_names = [band.name for band in bands_obj]

    print(f"\nNumber of samples: {len(returns)}")
    print(f"Duration: {len(returns) * 60 / 60:.1f} minutes ({len(returns) * 60 / 3600:.1f} hours)")
    print(f"\nAnalyzing {len(bands)} frequency bands (auto-detected for {resample_freq} resolution):")
    for band_obj in bands_obj:
        print(f"  - {band_obj}")

    # Convert returns DataFrame to numpy array
    returns_array = returns.values
    symbols_clean = list(returns.columns)

    print(f"\nRunning spectral analysis...")
    try:
        delays_list, coherences_list = compute_banded_delays(
            returns_array,
            fs=sampling_freq,
            bands=bands
        )
    except Exception as e:
        print(f"\nError during spectral analysis: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 4: Display Results
    print_section("Step 4: Results")

    for i, (delays, coherence, band_name) in enumerate(
        zip(delays_list, coherences_list, band_names)
    ):
        print_delay_matrix(delays, coherence, symbols_clean, band_name)
        analyze_lead_lag_relationships(
            delays,
            coherence,
            symbols_clean,
            band_name,
            threshold=0.3
        )

    # Summary
    print_section("Summary")

    print("\nDoes BTC lead other cryptocurrencies?")
    print("-" * 60)

    btc_idx = symbols_clean.index('BTC') if 'BTC' in symbols_clean else None

    if btc_idx is not None:
        for i, (delays, coherence, band_name) in enumerate(
            zip(delays_list, coherences_list, band_names)
        ):
            print(f"\n{band_name}:")

            # Check BTC vs all others
            for j, symbol in enumerate(symbols_clean):
                if j != btc_idx and coherence[btc_idx, j] > 0.3:
                    delay = delays[btc_idx, j]
                    if delay > 0:
                        print(f"  BTC leads {symbol} by {delay:.2f}s "
                              f"(coherence={coherence[btc_idx, j]:.3f})")
                    else:
                        print(f"  BTC lags {symbol} by {-delay:.2f}s "
                              f"(coherence={coherence[btc_idx, j]:.3f})")

    print_section("Test Complete!")
    print("\nNext steps:")
    print("  1. Try different frequency bands")
    print("  2. Fetch more data (longer periods)")
    print("  3. Implement visualization")
    print("  4. Compare with ML methods (GAT, TGAT)")


if __name__ == '__main__':
    main()
