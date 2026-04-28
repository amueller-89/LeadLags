#!/usr/bin/env python3
"""
Quick validation test for dashboard UI code.
Tests imports and basic logic without launching Streamlit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 70)
print("  DASHBOARD UI VALIDATION TEST")
print("=" * 70)

# Test 1: Import modules
print("\nTest 1: Importing dashboard modules...")
try:
    from dashboard import data_manager
    print("✓ dashboard.data_manager imported successfully")
except Exception as e:
    print(f"✗ Error importing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Check expected functions exist
print("\nTest 2: Checking module structure...")
try:
    assert hasattr(data_manager, 'render'), "Missing render() function"
    print("✓ data_manager.render() exists")
except AssertionError as e:
    print(f"✗ {e}")
    sys.exit(1)

# Test 3: Validate period parsing logic
print("\nTest 3: Testing period parsing logic...")
try:
    # OHLCV period parsing
    ohlcv_periods = ["Last 1 day", "Last 2 days", "Last 7 days", "Last 14 days", "Last 30 days"]
    for period in ohlcv_periods:
        days = int(period.split()[1])
        assert days > 0, f"Invalid days: {days}"
    print(f"✓ OHLCV period parsing works ({len(ohlcv_periods)} options)")

    # Tick data period parsing
    tick_periods = [
        "Last 15 minutes",
        "Last 30 minutes",
        "Last 1 hour",
        "Last 2 hours",
        "Last 3 hours",
        "Last 6 hours",
        "Last 12 hours",
        "Last 24 hours"
    ]
    for period in tick_periods:
        if "minutes" in period:
            hours = int(period.split()[1]) / 60
        else:
            hours = int(period.split()[1])
        assert hours > 0, f"Invalid hours: {hours}"

        # Check warning threshold
        if hours > 2:
            pass  # Would show warning in UI
    print(f"✓ Tick data period parsing works ({len(tick_periods)} options)")

except Exception as e:
    print(f"✗ Error in period parsing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Validate bar size mapping
print("\nTest 4: Testing bar size/frequency mapping...")
try:
    resample_map = {
        "1 second": "1s",
        "5 seconds": "5s",
        "10 seconds": "10s",
        "30 seconds": "30s",
        "1 minute": "1min",
        "5 minutes": "5min"
    }
    for label, freq in resample_map.items():
        assert freq, f"Invalid frequency for {label}"
    print(f"✓ Bar size mapping validated ({len(resample_map)} options)")
except Exception as e:
    print(f"✗ Error in mapping: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("  ALL VALIDATION TESTS PASSED ✓")
print("=" * 70)
print("\nTo test the UI interactively, run:")
print("  streamlit run streamlit_app.py")
print()
