#!/usr/bin/env python3
"""
Streamlit dashboard entry point for crypto lead-lag analysis.

Usage:
    streamlit run streamlit_app.py

This is the main entry point for the Streamlit dashboard. It imports and runs
the main dashboard application from src/dashboard/app.py.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dashboard.app import main

if __name__ == "__main__":
    main()
