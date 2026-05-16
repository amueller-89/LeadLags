"""
Session state management for Streamlit dashboard.
"""

import sys
from pathlib import Path

import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessor import PreprocessConfig


def initialize_session_state():
    """Initialize all session state variables."""

    # Data Management
    if 'loaded_data' not in st.session_state:
        st.session_state.loaded_data = None

    if 'data_source' not in st.session_state:
        st.session_state.data_source = None  # 'fetched' or 'cached'

    # Preprocessing Configuration
    if 'preprocessing_config' not in st.session_state:
        st.session_state.preprocessing_config = PreprocessConfig(
            resample_freq='1min',
            remove_outliers=True,
            outlier_std=10.0,
            price_column='close'
        )

    # Analysis Configuration
    if 'min_coherence' not in st.session_state:
        st.session_state.min_coherence = 0.3

    # Analysis Results
    if 'current_analysis_run' not in st.session_state:
        st.session_state.current_analysis_run = None

    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

    # UI State
    if 'selected_cached_files' not in st.session_state:
        st.session_state.selected_cached_files = []

    if 'selected_run_dir' not in st.session_state:
        st.session_state.selected_run_dir = None

    # TGAT Results
    if 'tgat_results' not in st.session_state:
        st.session_state.tgat_results = None

    if 'tgat_trainer' not in st.session_state:
        st.session_state.tgat_trainer = None

    if 'tgat_inference_ctx' not in st.session_state:
        st.session_state.tgat_inference_ctx = None


def reset_loaded_data():
    """Reset loaded data and related state."""
    st.session_state.loaded_data = None
    st.session_state.data_source = None


def reset_analysis_results():
    """Reset analysis results."""
    st.session_state.current_analysis_run = None
    st.session_state.analysis_results = None


def get_loaded_data_summary() -> str:
    """
    Get a summary of currently loaded data.

    Returns:
        String summary of loaded data, or None if no data loaded
    """
    import pandas as pd

    if st.session_state.loaded_data is None:
        return None

    data = st.session_state.loaded_data
    symbols = list(data.keys())
    n_symbols = len(symbols)

    if n_symbols == 0:
        return "No data loaded"

    # Get date range from first symbol
    first_symbol = symbols[0]
    df = data[first_symbol]
    start_date = df.index.min().strftime('%Y-%m-%d %H:%M:%S')
    end_date = df.index.max().strftime('%Y-%m-%d %H:%M:%S')
    n_samples = len(df)

    # Try to infer frequency
    inferred_freq = pd.infer_freq(df.index[:min(100, len(df))])

    if inferred_freq:
        freq_display = inferred_freq
    else:
        # Calculate median time diff if frequency can't be inferred
        if len(df) > 1:
            time_diffs = df.index[1:] - df.index[:-1]
            median_diff = time_diffs.median()
            freq_display = f"~{median_diff} (median)"
        else:
            freq_display = "unknown"

    summary = f"**{n_symbols} symbol(s):** {', '.join(symbols)}"
    summary += f"\n**Date range:** {start_date} to {end_date}"
    summary += f"\n**Samples:** {n_samples:,} per symbol"
    summary += f"\n**Frequency:** {freq_display}"

    # Determine data type based on frequency
    if inferred_freq:
        if inferred_freq.endswith('s') or inferred_freq in ['1s', '5s', '10s', '30s']:
            data_type = "Tick data (resampled)"
        elif inferred_freq in ['min', '1min', '5min', '15min', '1h']:
            data_type = "OHLCV candles"
        else:
            data_type = "Unknown"
        summary += f"\n**Type:** {data_type}"

    if st.session_state.data_source:
        source_label = {
            'fetched': 'Freshly fetched from Binance',
            'cached': 'Loaded from cache',
            'tick': 'Tick data (fetched and resampled)'
        }.get(st.session_state.data_source, st.session_state.data_source)
        summary += f"\n**Source:** {source_label}"

    return summary
