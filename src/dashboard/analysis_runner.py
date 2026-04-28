"""
Analysis Configuration & Execution UI component for Streamlit dashboard.

Handles preprocessing configuration, analysis parameters, and execution.
"""

import sys
from pathlib import Path

import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessor import CryptoPreprocessor, PreprocessConfig
from analyzer import LeadLagAnalyzer, create_crypto_frequency_bands
from backend_utils import save_results_timestamped
from dashboard.state import get_loaded_data_summary


def render():
    """Render the Analysis Configuration & Execution tab."""

    st.header("Analysis Configuration & Execution")

    # Check prerequisites
    if st.session_state.loaded_data is None:
        st.warning("⚠️ No data loaded. Please fetch or load data from the Data Management tab first.")
        return

    # Show current data summary
    current_data_summary = get_loaded_data_summary()
    if current_data_summary:
        st.info("**Analyzing Data:**\n\n" + current_data_summary)

    st.markdown("---")

    # Section 1: Preprocessing Configuration
    st.subheader("1. Preprocessing Configuration")

    col1, col2 = st.columns(2)

    with col1:
        # Resampling is always enabled
        st.checkbox(
            "Resample Data",
            value=True,
            disabled=True,
            help="Resampling is required to ensure aligned timestamps across symbols."
        )

        resample_freq = st.selectbox(
            "Bar Size",
            options=['1s', '5s', '10s', '30s', '1min', '5min', '15min', '1h'],
            index=4,  # Default to 1min
            help="Target bar size for resampling. Cannot upsample OHLCV data."
        )

        remove_outliers = st.checkbox(
            "Remove Outliers",
            value=st.session_state.preprocessing_config.remove_outliers,
            help="Clip returns beyond N standard deviations"
        )

    with col2:
        outlier_std = st.number_input(
            "Outlier Threshold (std)",
            min_value=1.0,
            max_value=20.0,
            value=st.session_state.preprocessing_config.outlier_std,
            step=0.5,
            help="Number of standard deviations for outlier clipping",
            disabled=not remove_outliers
        )

        price_column = st.selectbox(
            "Price Column for Returns",
            options=['close', 'open', 'high', 'low'],
            index=0,
            help="Which OHLCV column to use for return calculation"
        )

    # Update session state config
    st.session_state.preprocessing_config = PreprocessConfig(
        resample_freq=resample_freq,
        remove_outliers=remove_outliers,
        outlier_std=outlier_std,
        price_column=price_column
    )

    st.markdown("---")

    # Section 2: Analysis Configuration
    st.subheader("2. Analysis Configuration")

    min_coherence = st.slider(
        "Minimum Coherence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.min_coherence,
        step=0.05,
        help="Filter relationships with coherence below this threshold (0-1 scale)"
    )

    st.session_state.min_coherence = min_coherence

    # Calculate sampling frequency from resample string
    if resample_freq.endswith('s'):
        # Handle seconds: 1s, 5s, 10s, 30s
        seconds = int(resample_freq[:-1])
        sampling_freq = 1.0 / seconds
        period_str = f"{1/sampling_freq:.0f}s period"
    elif resample_freq.endswith('min'):
        minutes = int(resample_freq[:-3])
        sampling_freq = 1.0 / (minutes * 60)
        period_str = f"{1/sampling_freq:.0f}s period"
    elif resample_freq.endswith('h'):
        hours = int(resample_freq[:-1])
        sampling_freq = 1.0 / (hours * 3600)
        period_str = f"{1/sampling_freq:.0f}s period"
    else:
        # Default to 1 minute
        sampling_freq = 1.0 / 60.0
        period_str = "60s period (default)"

    st.write(f"**Sampling Frequency:** {sampling_freq:.6f} Hz ({period_str})")

    # Preview frequency bands
    with st.expander("Preview Frequency Bands"):
        st.write("The following frequency bands will be analyzed:")
        try:
            # Capture stdout to show band generation info
            import io
            from contextlib import redirect_stdout

            f = io.StringIO()
            with redirect_stdout(f):
                bands = create_crypto_frequency_bands(sampling_freq)
            band_info_output = f.getvalue()

            # Display bands
            for band in bands:
                p_min, p_max = band.period_range
                if p_max < 120:
                    period_str = f"{p_min:.0f}s - {p_max:.0f}s"
                elif p_max < 7200:
                    period_str = f"{p_min/60:.1f}min - {p_max/60:.1f}min"
                else:
                    period_str = f"{p_min/3600:.1f}hr - {p_max/3600:.1f}hr"
                st.write(f"- **{band.name}**: {period_str}")

        except Exception as e:
            st.error(f"Error generating frequency bands: {str(e)}")

    st.markdown("---")

    # Section 3: Execution
    st.subheader("3. Run Analysis")

    if st.button("Run Full Analysis", type="primary", use_container_width=True):
        run_analysis_pipeline()


def run_analysis_pipeline():
    """Execute the full analysis pipeline with progress tracking."""

    data = st.session_state.loaded_data
    config = st.session_state.preprocessing_config
    min_coherence = st.session_state.min_coherence

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Step 1: Preprocessing
        status_text.text("Step 1/3: Preprocessing data...")
        progress_bar.progress(0.1)

        preprocessor = CryptoPreprocessor(config)

        # Determine whether to resample based on config
        should_resample = config.resample_freq is not None

        processed_ohlcv, returns = preprocessor.process_pipeline(
            data,
            align=True,
            resample=should_resample,
            calc_returns=True,
            remove_outliers=config.remove_outliers
        )

        if returns is None or returns.empty:
            st.error("Preprocessing failed: No returns calculated")
            return

        progress_bar.progress(0.4)

        # Step 2: Spectral Analysis
        status_text.text("Step 2/3: Running spectral analysis...")

        # Calculate sampling frequency
        if config.resample_freq is None:
            # No resampling - infer from data
            # TODO: Improve this to actually detect frequency from data
            sampling_freq = 1.0 / 60.0  # Default assumption
        elif config.resample_freq.endswith('s'):
            # Handle seconds: 1s, 5s, 10s, 30s
            seconds = int(config.resample_freq[:-1])
            sampling_freq = 1.0 / seconds
        elif config.resample_freq.endswith('min'):
            minutes = int(config.resample_freq[:-3])
            sampling_freq = 1.0 / (minutes * 60)
        elif config.resample_freq.endswith('h'):
            hours = int(config.resample_freq[:-1])
            sampling_freq = 1.0 / (hours * 3600)
        else:
            sampling_freq = 1.0 / 60.0

        analyzer = LeadLagAnalyzer(sampling_freq)
        bands = create_crypto_frequency_bands(sampling_freq)

        results_raw = analyzer.analyze(returns, bands)

        progress_bar.progress(0.7)

        # Step 3: Process Results
        status_text.text("Step 3/3: Processing results and generating visualizations...")

        # Create band lookup dictionary
        band_info = {band.name: band for band in bands}

        # Process results
        results = {}
        for band_name, (delays_df, coherence_df) in results_raw.items():
            relationships = analyzer.rank_lead_lag_relationships(
                delays_df, coherence_df, min_coherence=min_coherence
            )

            scores = analyzer.get_asset_leadership_score(
                delays_df, coherence_df, min_coherence=min_coherence
            )

            results[band_name] = (delays_df, coherence_df, relationships, scores)

        progress_bar.progress(0.85)

        # Save results to timestamped directory
        symbols = list(data.keys())
        analysis_config = {
            'symbols': symbols,
            'resample_freq': config.resample_freq,
            'remove_outliers': config.remove_outliers,
            'outlier_std': config.outlier_std,
            'min_coherence': min_coherence,
            'n_samples': len(returns)
        }

        run_dir = save_results_timestamped(results, band_info, analysis_config)

        progress_bar.progress(1.0)
        status_text.empty()
        progress_bar.empty()

        # Store in session state
        st.session_state.current_analysis_run = str(run_dir)
        st.session_state.analysis_results = results

        st.success(f"✅ Analysis complete! Results saved to: {run_dir.name}")
        st.info("View results in the **Results** tab.")

    except Exception as e:
        status_text.empty()
        progress_bar.empty()
        st.error(f"Error during analysis: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
