"""
Data Management UI component for Streamlit dashboard.

Handles data fetching, cached data browsing, and data deletion.
"""

import sys
from pathlib import Path
from datetime import date, timedelta

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
        options=["OHLCV Candles", "Tick Data (Trades)", "Historical Tick (Binance Vision)"],
        index=0,
        horizontal=True,
        help=(
            "OHLCV: Aggregated candles via API. "
            "Tick (Trades): Recent trades via API (~24h only). "
            "Historical Tick: Full-day trade files from Binance public archive (any date, up to 7 days)."
        )
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

            elif data_type == "Tick Data (Trades)":
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

            else:  # Historical Tick (Binance Vision)
                start_date = st.date_input(
                    "Start Date",
                    value=date(2024, 3, 1),
                    help="First day to download. Binance Vision has data from 2017 onwards."
                )
                n_days = st.slider(
                    "Days to download",
                    min_value=1, max_value=7, value=3,
                    help="One parquet file per symbol per day. Full days can be 40–100 MB each for BTC."
                )
                if n_days >= 5:
                    st.warning(
                        f"⚠️ Downloading {n_days} days × {len(symbols) if symbols else '?'} symbols "
                        f"may take several minutes and ~{n_days * len(symbols) * 70 if symbols else '?'} MB."
                    )

                # Show which files are already cached
                data_dir = Path(__file__).parent.parent.parent / "data"
                already_cached = []
                if symbols:
                    for sym in symbols:
                        sym_file = sym.replace('/', '_')
                        for d in range(n_days):
                            day = start_date + timedelta(days=d)
                            day_str = day.strftime('%Y%m%d')
                            fname = f"{sym_file}_tick_{day_str}_{day_str}.parquet"
                            if (data_dir / fname).exists():
                                already_cached.append(fname)
                    if already_cached:
                        st.caption(f"Already cached ({len(already_cached)} files, will be skipped):")
                        st.caption(", ".join(already_cached[:5]) + ("..." if len(already_cached) > 5 else ""))

        with col2:
            # Bar size selection (conditional)
            if data_type == "OHLCV Candles":
                timeframe = st.selectbox(
                    "Bar Size",
                    options=['1m', '5m', '15m', '1h'],
                    index=0,
                    help="Candle bar size (1m is native Binance resolution)"
                )
            elif data_type == "Tick Data (Trades)":
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
            else:  # Historical Tick (Binance Vision)
                st.markdown("**Data format after download:**")
                st.caption("• DatetimeIndex (UTC)")
                st.caption("• `price` column (trade price)")
                st.caption("• `amount` column (trade quantity)")
                st.caption("• One parquet per symbol-day in `data/`")
                st.caption("• Files appear in the cache browser below")

            st.markdown("")  # Spacing
            fetch_button = st.form_submit_button("Fetch Data", type="primary", use_container_width=True)

        # Informational message for tick data
        if data_type == "Tick Data (Trades)":
            st.info(
                "ℹ️ **Tick Data Info:**\n"
                "- API provides last ~24 hours only\n"
                "- For historical data: use 'Historical Tick (Binance Vision)' above\n"
                "- Individual trades are 100-1000x larger than OHLCV"
            )
        elif data_type == "Historical Tick (Binance Vision)":
            st.info(
                "ℹ️ **Binance Vision Info:**\n"
                "- Downloads from https://data.binance.vision (public, no API key needed)\n"
                "- Coverage: 2017-present for major pairs\n"
                "- BTC/USDT: ~40–100 MB compressed per day\n"
                "- Files are saved to `data/` and immediately visible in the cache browser"
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
                            progress_bar.progress(i / len(symbols))
                            status_text.text(f"Fetching {symbol}... ({i+1}/{len(symbols)})")
                            symbol_data = loader.fetch_ohlcv(symbol, timeframe, days_back=days, save=True)
                            data[symbol] = symbol_data
                        progress_bar.progress(1.0)

                        # Store in session state
                        st.session_state.loaded_data = data
                        st.session_state.data_source = 'fetched'

                        status_text.empty()
                        progress_bar.empty()
                        st.success(f"Successfully fetched {len(symbols)} symbol(s)!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error fetching data: {str(e)}")

            elif data_type == "Tick Data (Trades)":
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
                            progress_bar.progress(i / len(symbols))
                            status_text.text(f"Fetching tick data for {symbol}... ({i+1}/{len(symbols)})")
                            ohlcv, raw = loader.fetch_and_resample_trades(
                                symbol,
                                resample_to=freq,
                                hours_back=hours,
                                save_raw=True,  # Always save raw tick data
                                save_resampled=True
                            )
                            data[symbol] = ohlcv
                        progress_bar.progress(1.0)

                        # Store in session state
                        st.session_state.loaded_data = data
                        st.session_state.data_source = 'tick'

                        status_text.empty()
                        progress_bar.empty()
                        st.success(f"Successfully fetched and resampled tick data for {len(symbols)} symbol(s)!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error fetching tick data: {str(e)}")

            else:  # Historical Tick (Binance Vision)
                _fetch_binance_vision(symbols, start_date, n_days)

    st.markdown("---")

    # Section 2: Browse & Load Cached Data
    st.subheader("2. Browse & Load Cached Data")

    cached_files = list_cached_data()

    if not cached_files:
        st.warning("No cached data files found in the data/ directory.")
    else:
        # Filter controls
        all_symbols = sorted({f['symbol'] for f in cached_files})
        col_f1, col_f2, col_f3, col_f4 = st.columns([2, 2, 2, 2])
        with col_f1:
            filter_symbols = st.multiselect("Symbol", all_symbols, key="filter_symbols")
        with col_f2:
            filter_type = st.radio(
                "Type", ["All", "Tick only", "OHLCV only"],
                horizontal=True, key="filter_type"
            )
        with col_f3:
            filter_start = st.date_input("From", value=None, key="filter_start")
        with col_f4:
            filter_end = st.date_input("To", value=None, key="filter_end")

        # Apply filters
        filtered_files = cached_files
        if filter_symbols:
            filtered_files = [f for f in filtered_files if f['symbol'] in filter_symbols]
        if filter_type == "Tick only":
            filtered_files = [f for f in filtered_files if 'tick' in f['timeframe'].lower()]
        elif filter_type == "OHLCV only":
            filtered_files = [f for f in filtered_files if 'tick' not in f['timeframe'].lower()]
        if filter_start:
            filtered_files = [f for f in filtered_files if f['start_date'] >= str(filter_start)]
        if filter_end:
            filtered_files = [f for f in filtered_files if f['start_date'] <= str(filter_end)]

        st.caption(f"Showing {len(filtered_files)} of {len(cached_files)} file(s)")

        # When filters change, bump the table version so the data_editor gets a fresh key
        # (avoids the banned st.session_state write-to-widget-key pattern).
        filter_state = (tuple(filter_symbols), filter_type, str(filter_start), str(filter_end))
        if st.session_state.get("_cached_table_filter_state") != filter_state:
            st.session_state["_cached_table_filter_state"] = filter_state
            st.session_state["_cached_table_version"] = st.session_state.get("_cached_table_version", 0) + 1
            st.session_state["_cached_table_sel_all"] = False

        # Select All / Deselect All — bump version so the editor starts fresh each time
        c1, c2, _ = st.columns([1, 1, 4])
        with c1:
            if st.button("Select All", key="sel_all_btn"):
                st.session_state["_cached_table_version"] = st.session_state.get("_cached_table_version", 0) + 1
                st.session_state["_cached_table_sel_all"] = True
                st.rerun()
        with c2:
            if st.button("Deselect All", key="desel_all_btn"):
                st.session_state["_cached_table_version"] = st.session_state.get("_cached_table_version", 0) + 1
                st.session_state["_cached_table_sel_all"] = False
                st.rerun()

        # Convert to DataFrame and prepare for table display
        df_cached = pd.DataFrame(filtered_files) if filtered_files else pd.DataFrame(cached_files[:0])

        # Add "Tick" column (Yes/No based on whether it's tick data)
        df_cached['is_tick'] = df_cached['timeframe'].apply(lambda x: 'Yes' if 'tick' in x.lower() else 'No')

        # Clean up bar size column (remove " (tick)" suffix for display)
        df_cached['bar_size'] = df_cached['timeframe'].apply(lambda x: x.replace(' (tick)', '') if 'tick' in x else x)

        # Create display DataFrame with desired columns
        df_display = df_cached[['symbol', 'start_date', 'end_date', 'bar_size', 'is_tick', 'file_size', 'date_fetched']].copy()
        df_display.columns = ['Symbol', 'Start Date', 'End Date', 'Bar Size', 'Tick', 'File Size', 'Date Fetched']

        # Initial checkbox state driven by the Select All flag
        sel_all = st.session_state.get("_cached_table_sel_all", False)
        df_display.insert(0, 'Select', sel_all)

        # Versioned key: each Select All / filter change gets a fresh editor instance
        table_key = f"cached_data_table_{st.session_state.get('_cached_table_version', 0)}"

        edited_df = st.data_editor(
            df_display,
            hide_index=True,
            use_container_width=True,
            key=table_key,
            disabled=['Symbol', 'Start Date', 'End Date', 'Bar Size', 'Tick', 'File Size', 'Date Fetched'],
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
                            selected_files = [filtered_files[i] for i in selected_indices]
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
                            selected_files = [filtered_files[i] for i in selected_indices]
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


def _fetch_binance_vision(symbols, start_date, n_days):
    """Download full-day trade files from data.binance.vision and save as parquet."""
    data_dir = Path(__file__).parent.parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    total = len(symbols) * n_days
    done = 0
    skipped = 0
    failed = []

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    merged: dict[str, list] = {sym: [] for sym in symbols}

    for sym in symbols:
        # URL uses BTCUSDT; filename uses BTC_USDT
        sym_url = sym.replace('/', '').replace('_', '')   # BTC/USDT → BTCUSDT
        sym_file = sym.replace('/', '_')                   # BTC/USDT → BTC_USDT

        for d in range(n_days):
            day = start_date + timedelta(days=d)
            day_str = day.strftime('%Y%m%d')
            day_iso = day.strftime('%Y-%m-%d')
            fname = f"{sym_file}_tick_{day_str}_{day_str}.parquet"
            fpath = data_dir / fname

            done += 1
            progress_bar.progress(done / total)

            if fpath.exists():
                status_text.text(f"Skipping {sym_url} {day_iso} (already cached) [{done}/{total}]")
                skipped += 1
                df = pd.read_parquet(fpath)
                merged[sym].append(df)
                continue

            url = (
                f"https://data.binance.vision/data/spot/daily/trades"
                f"/{sym_url}/{sym_url}-trades-{day_iso}.zip"
            )
            status_text.text(f"Downloading {sym_url} {day_iso}... [{done}/{total}]")

            try:
                df = pd.read_csv(
                    url,
                    compression='zip',
                    names=['trade_id', 'price', 'qty', 'quote_qty', 'time',
                           'is_buyer_maker', 'is_best_match'],
                )
                df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
                df = df.rename(columns={'qty': 'amount'}).set_index('time')
                df = df[['price', 'amount']]
                df.to_parquet(fpath, compression='snappy')
                merged[sym].append(df)
            except Exception as e:
                failed.append(f"{sym_url} {day_iso}: {e}")

    progress_bar.progress(1.0)
    status_text.empty()

    # Merge per-symbol and load into session state
    loaded = {}
    for sym, frames in merged.items():
        if frames:
            combined = pd.concat(frames).sort_index()
            combined = combined[~combined.index.duplicated(keep='first')]
            loaded[sym] = combined

    if loaded:
        st.session_state.loaded_data = loaded
        st.session_state.data_source = 'binance_vision'
        n_new = total - skipped - len(failed)
        msg = (
            f"Downloaded {n_new} new file(s), "
            f"skipped {skipped} cached, loaded {len(loaded)} symbol(s)."
        )
        if failed:
            msg += f" {len(failed)} failed."
        st.success(msg)
        if failed:
            with st.expander("Failed downloads"):
                for f in failed:
                    st.text(f)
        st.rerun()
    else:
        st.error("No data downloaded. Check the symbol names and date range.")
        if failed:
            for f in failed:
                st.error(f)
