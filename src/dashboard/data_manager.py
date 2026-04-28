"""
Data Management UI component for Streamlit dashboard.

Handles data fetching, cached data browsing, and data deletion.
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import BinanceDataLoader, get_default_crypto_symbols
from backend_utils import list_cached_data, load_cached_data, delete_cached_data, get_data_quality_summary
from dashboard.state import reset_loaded_data, get_loaded_data_summary


def render():
    """Render the Data Management tab."""

    st.header("Data Management")

    # Show currently loaded data status
    current_data_summary = get_loaded_data_summary()
    if current_data_summary:
        st.success("**Currently Loaded Data:**\n\n" + current_data_summary)
    else:
        st.info("No data currently loaded. Fetch new data or load from cache below.")

    st.markdown("---")

    # Section 1: Fetch New Data
    st.subheader("1. Fetch New Data from Binance")

    # Data type selection (OUTSIDE form so conditionals update immediately)
    data_type = st.radio(
        "Data Type",
        options=["OHLCV Candles", "Tick Data (Trades)"],
        index=0,
        horizontal=True,
        help="OHLCV: Aggregated candles. Tick: Individual trades (last ~24h only, much larger files)"
    )

    with st.form("fetch_data_form"):
        col1, col2 = st.columns(2)

        default_symbols = get_default_crypto_symbols()

        with col1:
            # Symbol selection (different defaults for tick vs OHLCV)
            if data_type == "Tick Data (Trades)":
                symbols = st.multiselect(
                    "Select Crypto Symbols",
                    options=default_symbols,
                    default=[default_symbols[0]],  # Only BTC for tick data by default
                    help="⚠️ Tick data is large - start with 1-2 symbols"
                )
            else:
                symbols = st.multiselect(
                    "Select Crypto Symbols",
                    options=default_symbols,
                    default=default_symbols[:3],  # Default to first 3
                    help="Select one or more trading pairs to fetch"
                )

            # Historical period selection (conditional)
            if data_type == "OHLCV Candles":
                period = st.selectbox(
                    "Historical Period",
                    options=["Last 1 day", "Last 2 days", "Last 7 days", "Last 14 days", "Last 30 days"],
                    index=1,  # Default to "Last 2 days"
                    help="How far back to fetch data (always fetches complete days)"
                )
                # Extract days from selection
                days = int(period.split()[1])
            else:  # Tick Data
                period = st.selectbox(
                    "Historical Period",
                    options=[
                        "Last 15 minutes",
                        "Last 30 minutes",
                        "Last 1 hour",
                        "Last 2 hours",
                        "Last 3 hours",
                        "Last 6 hours",
                        "Last 12 hours",
                        "Last 24 hours"
                    ],
                    index=2,  # Default to "Last 1 hour"
                    help="How far back to fetch data (⚠️ API limited to ~24 hours)"
                )

                # Parse period to hours
                if "minutes" in period:
                    hours = int(period.split()[1]) / 60
                else:
                    hours = int(period.split()[1])

                # Warning for large fetches
                if hours > 2:
                    st.warning(
                        f"⚠️ **Large data fetch**: Fetching {hours:.0f} hours of tick data may take "
                        f"several minutes and produce large files (~{hours * 5:.0f} MB per symbol)."
                    )

        with col2:
            # Bar size selection (conditional)
            if data_type == "OHLCV Candles":
                timeframe = st.selectbox(
                    "Bar Size",
                    options=['1m', '5m', '15m', '1h'],
                    index=0,
                    help="Candle bar size (1m is native Binance resolution)"
                )
            else:  # Tick Data
                resample_to = st.selectbox(
                    "Bar Size",
                    options=["1 second", "5 seconds", "10 seconds", "30 seconds", "1 minute", "5 minutes"],
                    index=0,  # Default to 1 second
                    help="Tick data will be aggregated into bars of this size"
                )

                # Storage size estimation for tick data (always visible)
                if symbols:
                    st.markdown("")  # Spacing
                    # Raw tick data is ~100x larger than resampled
                    resampled_mb = len(symbols) * hours * 5  # Rough estimate: 5MB/hr/symbol
                    raw_mb = resampled_mb * 100  # Raw is much larger

                    if resampled_mb < 1:
                        resampled_str = f"~{resampled_mb*1000:.0f} KB"
                    else:
                        resampled_str = f"~{resampled_mb:.0f} MB"

                    if raw_mb < 1024:
                        raw_str = f"~{raw_mb:.0f} MB"
                    else:
                        raw_str = f"~{raw_mb/1024:.1f} GB"

                    st.caption(f"**Estimated storage:**")
                    st.caption(f"• Resampled: {resampled_str}")
                    st.caption(f"• Raw tick data: {raw_str}")
                    st.caption("_Both will be saved._")

            st.markdown("")  # Spacing
            fetch_button = st.form_submit_button("Fetch Data", type="primary", use_container_width=True)

        # Informational message for tick data
        if data_type == "Tick Data (Trades)":
            st.info(
                "ℹ️ **Tick Data Info:**\n"
                "- API provides last ~24 hours only\n"
                "- For historical data: [Binance Public Data](https://data.binance.vision/)\n"
                "- Individual trades are 100-1000x larger than OHLCV"
            )

    if fetch_button:
        if not symbols:
            st.error("Please select at least one symbol")
        else:
            # Route to appropriate fetching method based on data type
            if data_type == "OHLCV Candles":
                # OHLCV data fetching
                with st.spinner(f"Fetching {len(symbols)} symbol(s) from Binance..."):
                    try:
                        loader = BinanceDataLoader()

                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        data = {}
                        for i, symbol in enumerate(symbols):
                            status_text.text(f"Fetching {symbol}... ({i+1}/{len(symbols)})")
                            symbol_data = loader.fetch_ohlcv(symbol, timeframe, days_back=days, save=True)
                            data[symbol] = symbol_data
                            progress_bar.progress((i + 1) / len(symbols))

                        # Store in session state
                        st.session_state.loaded_data = data
                        st.session_state.data_source = 'fetched'

                        status_text.empty()
                        progress_bar.empty()
                        st.success(f"Successfully fetched {len(symbols)} symbol(s)!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error fetching data: {str(e)}")

            else:  # Tick Data (Trades)
                # Tick data fetching and resampling
                with st.spinner(f"Fetching tick data for {len(symbols)} symbol(s)..."):
                    try:
                        loader = BinanceDataLoader()

                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # Convert bar size selection to frequency string
                        resample_map = {
                            "1 second": "1s",
                            "5 seconds": "5s",
                            "10 seconds": "10s",
                            "30 seconds": "30s",
                            "1 minute": "1min",
                            "5 minutes": "5min"
                        }
                        freq = resample_map[resample_to]

                        data = {}
                        for i, symbol in enumerate(symbols):
                            status_text.text(f"Fetching tick data for {symbol}... ({i+1}/{len(symbols)})")
                            ohlcv, raw = loader.fetch_and_resample_trades(
                                symbol,
                                resample_to=freq,
                                hours_back=hours,
                                save_raw=True,  # Always save raw tick data
                                save_resampled=True
                            )
                            data[symbol] = ohlcv
                            progress_bar.progress((i + 1) / len(symbols))

                        # Store in session state
                        st.session_state.loaded_data = data
                        st.session_state.data_source = 'tick'

                        status_text.empty()
                        progress_bar.empty()
                        st.success(f"Successfully fetched and resampled tick data for {len(symbols)} symbol(s)!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error fetching tick data: {str(e)}")

    st.markdown("---")

    # Section 2: Browse & Load Cached Data
    st.subheader("2. Browse & Load Cached Data")

    cached_files = list_cached_data()

    if not cached_files:
        st.warning("No cached data files found in the data/ directory.")
    else:
        st.write(f"Found {len(cached_files)} cached file(s):")

        # Convert to DataFrame and prepare for table display
        df_cached = pd.DataFrame(cached_files)

        # Add "Tick" column (Yes/No based on whether it's tick data)
        df_cached['is_tick'] = df_cached['timeframe'].apply(lambda x: 'Yes' if 'tick' in x.lower() else 'No')

        # Clean up bar size column (remove " (tick)" suffix for display)
        df_cached['bar_size'] = df_cached['timeframe'].apply(lambda x: x.replace(' (tick)', '') if 'tick' in x else x)

        # Create display DataFrame with desired columns
        df_display = df_cached[['symbol', 'start_date', 'end_date', 'bar_size', 'is_tick', 'file_size']].copy()
        df_display.columns = ['Symbol', 'Start Date', 'End Date', 'Bar Size', 'Tick', 'File Size']

        # Add selection column
        df_display.insert(0, 'Select', False)

        # Display editable table with checkboxes
        edited_df = st.data_editor(
            df_display,
            hide_index=True,
            use_container_width=True,
            disabled=['Symbol', 'Start Date', 'End Date', 'Bar Size', 'Tick', 'File Size'],
            column_config={
                "Select": st.column_config.CheckboxColumn(
                    "Select",
                    help="Select files to load or delete",
                    default=False,
                )
            }
        )

        # Get selected indices
        selected_mask = edited_df['Select']
        selected_indices = [i for i, selected in enumerate(selected_mask) if selected]

        st.markdown("")  # Spacing

        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("Load Selected", type="primary", disabled=len(selected_indices) == 0):
                if selected_indices:
                    with st.spinner(f"Loading {len(selected_indices)} file(s)..."):
                        try:
                            selected_files = [cached_files[i] for i in selected_indices]
                            data = load_cached_data(selected_files)

                            if data:
                                st.session_state.loaded_data = data
                                st.session_state.data_source = 'cached'
                                st.success(f"Successfully loaded {len(data)} symbol(s)!")
                                st.rerun()
                            else:
                                st.error("Failed to load data")
                        except Exception as e:
                            st.error(f"Error loading data: {str(e)}")

        with col2:
            if st.button("Delete Selected", type="secondary", disabled=len(selected_indices) == 0):
                if selected_indices:
                    # Confirmation dialog
                    st.warning(f"⚠️ Are you sure you want to delete {len(selected_indices)} file(s)?")
                    col_confirm, col_cancel = st.columns(2)
                    with col_confirm:
                        if st.button("Confirm Delete", key="confirm_delete"):
                            selected_files = [cached_files[i] for i in selected_indices]
                            success, failed = delete_cached_data(selected_files)
                            if success > 0:
                                st.success(f"Deleted {success} file(s)")
                            if failed > 0:
                                st.error(f"Failed to delete {failed} file(s)")
                            st.rerun()

    st.markdown("---")

    # Section 3: Data Preview & Quality
    st.subheader("3. Data Preview & Quality Check")

    if st.session_state.loaded_data is not None:
        data = st.session_state.loaded_data

        # Quality report
        try:
            quality_df = get_data_quality_summary(data)
            st.dataframe(quality_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating quality report: {str(e)}")

        # Quick preview
        st.write("**Data Preview** (first symbol):")
        first_symbol = list(data.keys())[0]
        st.dataframe(data[first_symbol].head(10), use_container_width=True)

        # Clear data button
        if st.button("Clear Loaded Data"):
            reset_loaded_data()
            st.success("Cleared loaded data")
            st.rerun()

    else:
        st.info("Load data to view preview and quality metrics.")
