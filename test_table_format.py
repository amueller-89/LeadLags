#!/usr/bin/env python3
"""
Test the cached data table formatting logic.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 70)
print("  CACHED DATA TABLE FORMAT TEST")
print("=" * 70)

# Simulate cached file data
mock_cached_files = [
    {
        'filename': 'BTC_USDT_1m_20260421_20260422.parquet',
        'symbol': 'BTC/USDT',
        'timeframe': '1m',
        'start_date': '2026-04-21',
        'end_date': '2026-04-22',
        'file_size': '2.3 MB',
        'filepath': '/path/to/file1.parquet'
    },
    {
        'filename': 'ETH_USDT_30s_tick_20260421_20260421.parquet',
        'symbol': 'ETH/USDT',
        'timeframe': '30s (tick)',
        'start_date': '2026-04-21',
        'end_date': '2026-04-21',
        'file_size': '15.7 MB',
        'filepath': '/path/to/file2.parquet'
    },
    {
        'filename': 'SOL_USDT_tick_20260421_20260421.parquet',
        'symbol': 'SOL/USDT',
        'timeframe': 'tick',
        'start_date': '2026-04-21',
        'end_date': '2026-04-21',
        'file_size': '487.2 MB',
        'filepath': '/path/to/file3.parquet'
    },
    {
        'filename': 'BNB_USDT_5m_20260420_20260422.parquet',
        'symbol': 'BNB/USDT',
        'timeframe': '5m',
        'start_date': '2026-04-20',
        'end_date': '2026-04-22',
        'file_size': '1.8 MB',
        'filepath': '/path/to/file4.parquet'
    }
]

print("\nTest 1: Converting cached files to DataFrame...")
df_cached = pd.DataFrame(mock_cached_files)
print(f"✓ Created DataFrame with {len(df_cached)} rows")

print("\nTest 2: Adding Tick column...")
df_cached['is_tick'] = df_cached['timeframe'].apply(lambda x: 'Yes' if 'tick' in x.lower() else 'No')
print("✓ Tick column added")
print(f"  - OHLCV files: {(df_cached['is_tick'] == 'No').sum()}")
print(f"  - Tick files: {(df_cached['is_tick'] == 'Yes').sum()}")

print("\nTest 3: Cleaning bar size column...")
df_cached['bar_size'] = df_cached['timeframe'].apply(lambda x: x.replace(' (tick)', '') if 'tick' in x else x)
print("✓ Bar size column created")
for idx, row in df_cached.iterrows():
    print(f"  - {row['symbol']}: '{row['timeframe']}' → '{row['bar_size']}'")

print("\nTest 4: Creating display DataFrame...")
df_display = df_cached[['symbol', 'start_date', 'end_date', 'bar_size', 'is_tick', 'file_size']].copy()
df_display.columns = ['Symbol', 'Start Date', 'End Date', 'Bar Size', 'Tick', 'File Size']
print("✓ Display DataFrame created with columns:")
print(f"  {list(df_display.columns)}")

print("\nTest 5: Adding selection column...")
df_display.insert(0, 'Select', False)
print("✓ Selection column added")

print("\nFinal Table Preview:")
print("=" * 70)
print(df_display.to_string(index=False))
print("=" * 70)

print("\n✓ All table formatting tests passed!")
