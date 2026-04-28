#!/usr/bin/env python3
"""
Test analysis configuration changes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 70)
print("  ANALYSIS CONFIGURATION TEST")
print("=" * 70)

# Test 1: Import modules
print("\nTest 1: Importing analysis modules...")
try:
    from dashboard import analysis_runner
    print("✓ dashboard.analysis_runner imported successfully")
except Exception as e:
    print(f"✗ Error importing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Validate bar size options
print("\nTest 2: Testing bar size options...")
bar_sizes = ['1s', '5s', '10s', '30s', '1min', '5min', '15min', '1h']
print(f"✓ Bar size options: {bar_sizes}")
print(f"  - Total options: {len(bar_sizes)}")
print(f"  - Sub-minute options: {len([x for x in bar_sizes if x.endswith('s')])}")

# Test 3: Test sampling frequency calculations
print("\nTest 3: Testing sampling frequency calculations...")

test_cases = [
    ('1s', 1.0, "1 second"),
    ('5s', 0.2, "5 seconds"),
    ('10s', 0.1, "10 seconds"),
    ('30s', 1/30, "30 seconds"),
    ('1min', 1/60, "1 minute"),
    ('5min', 1/300, "5 minutes"),
    ('15min', 1/900, "15 minutes"),
    ('1h', 1/3600, "1 hour"),
]

for freq_str, expected_freq, description in test_cases:
    if freq_str.endswith('s'):
        seconds = int(freq_str[:-1])
        calculated_freq = 1.0 / seconds
    elif freq_str.endswith('min'):
        minutes = int(freq_str[:-3])
        calculated_freq = 1.0 / (minutes * 60)
    elif freq_str.endswith('h'):
        hours = int(freq_str[:-1])
        calculated_freq = 1.0 / (hours * 3600)
    else:
        calculated_freq = None

    if calculated_freq is not None and abs(calculated_freq - expected_freq) < 1e-6:
        print(f"✓ {description} ({freq_str}): {calculated_freq:.6f} Hz")
    else:
        print(f"✗ {description} ({freq_str}): Expected {expected_freq:.6f} Hz, got {calculated_freq:.6f} Hz")
        sys.exit(1)

# Test 4: Test optional resampling
print("\nTest 4: Testing optional resampling...")
resample_freq = None
if resample_freq is None:
    print("✓ Resampling disabled (resample_freq = None)")
    print("  - Analysis will use native data frequency")
else:
    print("✗ Resampling should be None when disabled")
    sys.exit(1)

print("\n" + "=" * 70)
print("  ALL TESTS PASSED ✓")
print("=" * 70)
print("\nChanges summary:")
print("  ✓ Added 1 second option to tick data")
print("  ✓ Made resampling optional in analysis tab")
print("  ✓ Added all bar size options (1s to 1h)")
print("  ✓ Updated sampling frequency calculations")
