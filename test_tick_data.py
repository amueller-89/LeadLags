#!/usr/bin/env python3
"""
Test script for tick data fetching and resampling functionality.

Tests:
1. Fetch tick data from Binance
2. Resample to OHLCV
3. Verify file naming and parsing
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import BinanceDataLoader
from backend_utils import list_cached_data

def test_tick_data_fetching():
    """Test fetching and resampling tick data."""
    print("=" * 70)
    print("  TICK DATA FETCHING TEST")
    print("=" * 70)

    # Initialize loader
    loader = BinanceDataLoader()

    # Test 1: Fetch and resample tick data
    print("\nTest 1: Fetching tick data for BTC/USDT (1 hour, resample to 30s)")
    print("-" * 70)

    try:
        symbol = "BTC/USDT"
        resample_to = "30s"
        hours_back = 1.0

        print(f"Fetching tick data for {symbol}...")
        ohlcv, raw = loader.fetch_and_resample_trades(
            symbol=symbol,
            resample_to=resample_to,
            hours_back=hours_back,
            save_raw=False,
            save_resampled=True
        )

        print(f"\n✓ Successfully fetched and resampled tick data")
        print(f"  - Resampled OHLCV shape: {ohlcv.shape}")
        print(f"  - Timeframe: {resample_to}")
        print(f"  - Time range: {ohlcv.index.min()} to {ohlcv.index.max()}")
        print(f"\nFirst few bars:")
        print(ohlcv.head())

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Verify file was saved
    print("\n" + "=" * 70)
    print("Test 2: Verify file naming and parsing")
    print("-" * 70)

    try:
        cached_files = list_cached_data()
        tick_files = [f for f in cached_files if 'tick' in f['timeframe'].lower()]

        if tick_files:
            print(f"\n✓ Found {len(tick_files)} tick data file(s):")
            for f in tick_files:
                print(f"  - {f['symbol']} {f['timeframe']}: {f['start_date']} to {f['end_date']} ({f['file_size']})")
        else:
            print("\n⚠ No tick files found in cache")

    except Exception as e:
        print(f"\n✗ Error listing cached files: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Test raw tick data fetching
    print("\n" + "=" * 70)
    print("Test 3: Fetch raw tick data (without saving)")
    print("-" * 70)

    try:
        print("Fetching raw tick data for ETH/USDT (30 minutes)...")
        tick_df = loader.fetch_trades(
            symbol="ETH/USDT",
            hours_back=0.5,  # 30 minutes
            save=False
        )

        print(f"\n✓ Successfully fetched raw tick data")
        print(f"  - Shape: {tick_df.shape}")
        print(f"  - Columns: {list(tick_df.columns)}")
        print(f"  - Time range: {tick_df.index.min()} to {tick_df.index.max()}")
        print(f"\nFirst few trades:")
        print(tick_df.head())

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Test resampling separately
    print("\n" + "=" * 70)
    print("Test 4: Test resampling function")
    print("-" * 70)

    try:
        print("Resampling tick data to 1-minute bars...")
        ohlcv_1min = loader.resample_tick_to_ohlcv(tick_df, timeframe='1min')

        print(f"\n✓ Successfully resampled")
        print(f"  - Original ticks: {len(tick_df)}")
        print(f"  - Resampled bars: {len(ohlcv_1min)}")
        print(f"  - Columns: {list(ohlcv_1min.columns)}")
        print(f"\nFirst few 1-minute bars:")
        print(ohlcv_1min.head())

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 70)
    print("  ALL TESTS PASSED ✓")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = test_tick_data_fetching()
    sys.exit(0 if success else 1)
