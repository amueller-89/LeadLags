"""
Results Viewer UI component for Streamlit dashboard.

Displays analysis results from timestamped run directories.
"""

import sys
from pathlib import Path
import json

import streamlit as st
import pandas as pd
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend_utils import list_analysis_runs


def render():
    """Render the Results Viewer tab."""

    st.header("Results Viewer")

    # List available analysis runs
    runs = list_analysis_runs()

    if not runs:
        st.info("No analysis runs found. Run an analysis from the Analysis tab to see results here.")
        return

    # Section 1: Run Selection
    st.subheader("Select Analysis Run")

    # Create dropdown options
    run_options = {}
    for run in runs:
        config = run.get('config', {})
        symbols = config.get('symbols', 'Unknown')
        if isinstance(symbols, list):
            symbols = ', '.join(symbols)
        label = f"{run['timestamp']} - {symbols}"
        run_options[label] = run['run_dir']

    selected_label = st.selectbox(
        "Analysis Run",
        options=list(run_options.keys()),
        help="Select an analysis run to view results"
    )

    selected_run_dir = Path(run_options[selected_label])

    # Load and display run configuration
    with st.expander("Run Configuration"):
        config_path = selected_run_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            st.json(config)
        else:
            st.warning("Configuration file not found")

    st.markdown("---")

    # Section 2: Results by Frequency Band
    st.subheader("Analysis Results")

    # Detect available frequency bands
    csv_files = list(selected_run_dir.glob("*_delays.csv"))
    band_names = [f.stem.replace('_delays', '') for f in csv_files]

    if not band_names:
        st.warning("No results found in this run directory")
        return

    # Create tabs for each frequency band
    tabs = st.tabs(band_names)

    for tab, band_name in zip(tabs, band_names):
        with tab:
            render_band_results(selected_run_dir, band_name)


def render_band_results(run_dir: Path, band_name: str):
    """
    Render results for a specific frequency band.

    Args:
        run_dir: Path to the analysis run directory
        band_name: Name of the frequency band
    """

    # Load CSV files
    delays_path = run_dir / f"{band_name}_delays.csv"
    coherence_path = run_dir / f"{band_name}_coherence.csv"
    relationships_path = run_dir / f"{band_name}_relationships.csv"
    leadership_path = run_dir / f"{band_name}_leadership.csv"

    # Check if files exist
    files_exist = all([
        delays_path.exists(),
        coherence_path.exists(),
        relationships_path.exists(),
        leadership_path.exists()
    ])

    if not files_exist:
        st.warning(f"Some result files are missing for {band_name}")
        return

    # Display visualizations
    st.subheader(f"Visualizations - {band_name}")

    # Find PNG files
    delays_png = run_dir / f"lead_lag_analysis_{band_name}_delays.png"
    leadership_png = run_dir / f"lead_lag_analysis_{band_name}_leadership.png"
    network_png = run_dir / f"lead_lag_analysis_{band_name}_network.png"

    if delays_png.exists():
        st.image(str(delays_png), use_container_width=True)
    else:
        st.warning("Delays visualization not found")

    col1, col2 = st.columns(2)

    with col1:
        if leadership_png.exists():
            st.image(str(leadership_png), use_container_width=True)
        else:
            st.warning("Leadership visualization not found")

    with col2:
        if network_png.exists():
            st.image(str(network_png), use_container_width=True)
        else:
            st.warning("Network visualization not found")

    st.markdown("---")

    # Display data tables
    st.subheader("Data Tables")

    # Top Relationships
    st.write("**Top Lead-Lag Relationships:**")
    relationships_df = pd.read_csv(relationships_path)
    if not relationships_df.empty:
        st.dataframe(relationships_df.head(10), use_container_width=True)
    else:
        st.info("No significant relationships found")

    # Leadership Scores
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Leadership Scores:**")
        leadership_df = pd.read_csv(leadership_path, index_col=0)
        leadership_df = leadership_df.sort_values('score', ascending=False)
        st.dataframe(leadership_df, use_container_width=True)

    with col2:
        # BTC Leadership Analysis (if applicable)
        btc_assets = [idx for idx in leadership_df.index if 'BTC' in idx]
        if btc_assets:
            st.write("**BTC Analysis:**")
            btc_asset = btc_assets[0]
            btc_score = leadership_df.loc[btc_asset, 'score']

            if btc_score > 0:
                st.success(f"BTC LEADS by {btc_score:.2f}s on average")
            elif btc_score < 0:
                st.warning(f"BTC LAGS by {-btc_score:.2f}s on average")
            else:
                st.info("BTC shows no clear lead/lag")

    st.markdown("---")

    # Download options
    st.subheader("Download Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with open(delays_path, 'rb') as f:
            st.download_button(
                "📥 Delays CSV",
                f,
                file_name=f"{band_name}_delays.csv",
                mime="text/csv"
            )

    with col2:
        with open(coherence_path, 'rb') as f:
            st.download_button(
                "📥 Coherence CSV",
                f,
                file_name=f"{band_name}_coherence.csv",
                mime="text/csv"
            )

    with col3:
        with open(relationships_path, 'rb') as f:
            st.download_button(
                "📥 Relationships CSV",
                f,
                file_name=f"{band_name}_relationships.csv",
                mime="text/csv"
            )

    with col4:
        if delays_png.exists():
            with open(delays_png, 'rb') as f:
                st.download_button(
                    "📥 Delays PNG",
                    f,
                    file_name=f"{band_name}_delays.png",
                    mime="image/png"
                )
