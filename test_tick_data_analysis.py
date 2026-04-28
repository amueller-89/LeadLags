#!/usr/bin/env python3
"""
Test tick data analysis flow - verifying that tick data can be analyzed properly.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from preprocessor import CryptoPreprocessor, PreprocessConfig

print("=" * 70)
print("  TICK DATA ANALYSIS FLOW TEST")
print("=" * 70)

# Test 1: Create mock tick data at 1s resolution
print("\nTest 1: Creating mock 1-second tick data...")
dates_1s = pd.date_range('2024-01-01', periods=3600, freq='1s')  # 1 hour at 1s
df_tick_1s = pd.DataFrame({
    'open': 100 + np.random.randn(3600) * 0.1,
    'high': 100 + np.random.randn(3600) * 0.1 + 0.1,
    'low': 100 + np.random.randn(3600) * 0.1 - 0.1,
    'close': 100 + np.random.randn(3600) * 0.1,
    'volume': np.random.rand(3600) * 1000
}, index=dates_1s)

data_dict_1s = {'BTC/USDT': df_tick_1s}
print(f"✓ Created 1s tick data: {len(df_tick_1s)} samples")

# Test 2: Resample 1s tick data to 5s (should be allowed)
print("\nTest 2: Resampling 1s tick data to 5s (upsampling tick data)...")
try:
    config = PreprocessConfig(resample_freq='5s')
    preprocessor = CryptoPreprocessor(config)

    resampled_5s = preprocessor.resample_ohlcv(data_dict_1s)

    print(f"✓ Successfully resampled 1s → 5s")
    print(f"  Original samples: {len(df_tick_1s)}")
    print(f"  Resampled samples: {len(resampled_5s['BTC/USDT'])}")
    print(f"  Ratio: {len(df_tick_1s) / len(resampled_5s['BTC/USDT']):.1f}:1")

except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Resample 1s tick data to 1min (downsampling - should work)
print("\nTest 3: Resampling 1s tick data to 1min (downsampling)...")
try:
    config_1min = PreprocessConfig(resample_freq='1min')
    preprocessor_1min = CryptoPreprocessor(config_1min)

    resampled_1min = preprocessor_1min.resample_ohlcv(data_dict_1s)

    print(f"✓ Successfully resampled 1s → 1min")
    print(f"  Original samples: {len(df_tick_1s)}")
    print(f"  Resampled samples: {len(resampled_1min['BTC/USDT'])}")
    print(f"  Ratio: {len(df_tick_1s) / len(resampled_1min['BTC/USDT']):.1f}:1")

except Exception as e:
    print(f"✗ Unexpected error: {e}")
    sys.exit(1)

# Test 4: Try to upsample OHLCV 1min data to 1s (should fail)
print("\nTest 4: Trying to upsample 1min OHLCV to 1s (should fail)...")
dates_1min = pd.date_range('2024-01-01', periods=60, freq='1min')
df_ohlcv_1min = pd.DataFrame({
    'open': 100 + np.random.randn(60) * 0.5,
    'high': 100 + np.random.randn(60) * 0.5 + 0.5,
    'low': 100 + np.random.randn(60) * 0.5 - 0.5,
    'close': 100 + np.random.randn(60) * 0.5,
    'volume': np.random.rand(60) * 10000
}, index=dates_1min)

data_dict_1min = {'BTC/USDT': df_ohlcv_1min}

try:
    config_upsample = PreprocessConfig(resample_freq='1s')
    preprocessor_upsample = CryptoPreprocessor(config_upsample)

    # This should fail
    resampled_bad = preprocessor_upsample.resample_ohlcv(data_dict_1min)

    print("✗ Should have failed - upsampling OHLCV creates fake data!")
    sys.exit(1)

except ValueError as e:
    if "Cannot resample OHLCV data" in str(e):
        print("✓ Correctly blocked upsampling of OHLCV data")
        print(f"  Error message preview: {str(e)[:100]}...")
    else:
        print(f"✗ Wrong error: {e}")
        sys.exit(1)

# Test 5: Downsample OHLCV 1min to 5min (should work)
print("\nTest 5: Downsampling 1min OHLCV to 5min (should work)...")
try:
    config_downsample = PreprocessConfig(resample_freq='5min')
    preprocessor_downsample = CryptoPreprocessor(config_downsample)

    resampled_down = preprocessor_downsample.resample_ohlcv(data_dict_1min)

    print(f"✓ Successfully downsampled 1min → 5min")
    print(f"  Original samples: {len(df_ohlcv_1min)}")
    print(f"  Resampled samples: {len(resampled_down['BTC/USDT'])}")
    print(f"  Ratio: {len(df_ohlcv_1min) / len(resampled_down['BTC/USDT']):.1f}:1")

except Exception as e:
    print(f"✗ Unexpected error: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("  ALL TESTS PASSED ✓")
print("=" * 70)
print("\nSummary of fixes:")
print("  ✓ Auto-detect source data frequency")
print("  ✓ Allow resampling of sub-minute tick data to any frequency")
print("  ✓ Block upsampling of OHLCV data (>= 1min) to prevent fake bars")
print("  ✓ Allow downsampling of all data types")
print("\nTick data workflow now works correctly:")
print("  1. Fetch tick data (saves raw + resampled)")
print("  2. Only resampled tick files shown in cache browser")
print("  3. Load resampled tick data (e.g., 1s bars)")
print("  4. Analyze with any bar size (1s, 5s, 30s, 1min, etc.)")
