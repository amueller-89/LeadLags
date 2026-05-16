"""
Main Streamlit dashboard application for crypto lead-lag analysis.
"""

import sys
from pathlib import Path

import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard import analysis_runner, data_manager, results_viewer, state, tgat_runner


def main():
    """Main dashboard application."""

    # Page configuration
    st.set_page_config(
        page_title="Crypto Lead-Lag Analysis",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    state.initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.title("📈 Crypto Lead-Lag")
        st.markdown("---")

        # Status indicators
        st.subheader("Status")

        # Data status
        if st.session_state.loaded_data is not None:
            data_count = len(st.session_state.loaded_data)
            st.success(f"✅ Data loaded ({data_count} symbols)")
        else:
            st.warning("⚠️ No data loaded")

        # Analysis status
        if st.session_state.current_analysis_run is not None:
            run_name = Path(st.session_state.current_analysis_run).name
            st.success("✅ Analysis complete")
            st.caption(f"Run: {run_name}")
        else:
            st.info("ℹ️ No analysis run yet")

        # TGAT status
        if st.session_state.get("tgat_results") is not None:
            mse = st.session_state.tgat_results.get("test_mse", float("nan"))
            st.success(f"✅ TGAT trained (MSE={mse:.5f})")
        else:
            st.info("ℹ️ TGAT not trained")

        st.markdown("---")

        # Quick info
        st.subheader("About")
        st.markdown("""
        This dashboard provides an interface for:

        - **Fetching** crypto data from Binance
        - **Managing** cached data files
        - **Analyzing** lead-lag relationships using spectral methods
        - **Viewing** results across frequency bands

        Navigate using the tabs above.
        """)

        st.markdown("---")
        st.caption("Powered by Streamlit & Anthropic Claude")

    # Global training banner
    if st.session_state.get("tgat_training_active"):
        st.warning(
            "🏋️ **TGAT training is running.** "
            "Navigate to the TGAT tab to monitor progress or interrupt training."
        )

    # Main content area
    st.title("Crypto Lead-Lag Analysis Dashboard")
    st.markdown("Analyze frequency-specific lead-lag relationships in cryptocurrency markets using spectral methods.")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Management",
        "⚙️ Analysis",
        "📈 Results",
        "🤖 TGAT"
    ])

    with tab1:
        data_manager.render()

    with tab2:
        analysis_runner.render()

    with tab3:
        results_viewer.render()

    with tab4:
        tgat_runner.render()


if __name__ == "__main__":
    main()
