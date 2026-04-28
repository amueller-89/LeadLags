#!/usr/bin/env python3
"""
Test preprocessor duplicate timestamp handling.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from preprocessor import CryptoPreprocessor, PreprocessConfig

print("=" * 70)
print("  PREPROCESSOR ERROR HANDLING TEST")
print("=" * 70)

# Test 1: Create data with duplicate timestamps
print("\nTest 1: Testing duplicate timestamp detection and removal...")

dates = pd.date_range('2024-01-01', periods=100, freq='1s')
# Create data first
values = np.random.randn(100, 5)

df1_clean = pd.DataFrame(values, columns=['open', 'high', 'low', 'close', 'volume'], index=dates)

# Add some duplicate rows
duplicate_rows = df1_clean.iloc[10:15].copy()
df1 = pd.concat([df1_clean, duplicate_rows]).sort_index()

df2 = df1.copy()

data_dict = {
    'SYMBOL1': df1,
    'SYMBOL2': df2
}

print(f"Created test data with {df1.index.duplicated().sum()} duplicate timestamps")

try:
    config = PreprocessConfig(resample_freq='1min')
    preprocessor = CryptoPreprocessor(config)

    # This should detect and remove duplicates
    aligned_data = preprocessor.align_timestamps(data_dict)

    # Check that duplicates were removed
    for symbol, df in aligned_data.items():
        if df.index.duplicated().any():
            print(f"✗ {symbol} still has duplicates")
            sys.exit(1)

    print("✓ Duplicates successfully detected and removed")
    print(f"  Original length: {len(df1)}")
    print(f"  After removal: {len(aligned_data['SYMBOL1'])}")

except Exception as e:
    print(f"✗ Error during alignment: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Test error handling for invalid median_diff
print("\nTest 2: Testing error handling for invalid frequency...")

# Create data with all same timestamps (will cause zero median_diff)
bad_dates = pd.DatetimeIndex([pd.Timestamp('2024-01-01')] * 10)
df_bad = pd.DataFrame({
    'open': np.random.randn(10),
    'high': np.random.randn(10),
    'low': np.random.randn(10),
    'close': np.random.randn(10),
    'volume': np.random.randn(10)
}, index=bad_dates)

bad_data_dict = {'BAD_SYMBOL': df_bad}

try:
    aligned_bad = preprocessor.align_timestamps(bad_data_dict)
    print("✗ Should have raised an error for insufficient data")
    sys.exit(1)
except ValueError as e:
    if "Insufficient data" in str(e) or "Invalid median time difference" in str(e):
        print(f"✓ Correctly caught data quality error")
        print(f"  Error message: {str(e)[:100]}...")
    else:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\n" + "=" * 70)
print("  ALL TESTS PASSED ✓")
print("=" * 70)
print("\nFixes implemented:")
print("  ✓ Duplicate timestamp detection for all symbols")
print("  ✓ Automatic duplicate removal (keeps first occurrence)")
print("  ✓ Better error messages for invalid frequencies")
print("  ✓ Debugging output showing inferred/calculated frequencies")
print("  ✓ UI warning when resampling is disabled")
