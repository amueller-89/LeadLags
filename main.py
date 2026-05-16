#!/usr/bin/env python3
"""
Main entry point for crypto lead-lag analysis.

This script orchestrates the full pipeline:
1. Fetch data from Binance
2. Preprocess (align, resample, calculate returns)
3. Run spectral lead-lag analysis
4. Generate visualizations
5. Save results
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd

from analyzer import LeadLagAnalyzer, create_crypto_frequency_bands
from data_loader import BinanceDataLoader, get_default_crypto_symbols
from preprocessor import CryptoPreprocessor, PreprocessConfig
from visualizer import LeadLagVisualizer


def main():
    """Run the complete lead-lag analysis pipeline."""

    parser = argparse.ArgumentParser(
        description="Crypto Lead-Lag Analysis using Spectral Methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data fetching arguments
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Crypto symbols to analyze (e.g., BTC/USDT ETH/USDT)",
    )
    parser.add_argument("--days", type=int, default=2, help="Number of days of data to fetch")
    parser.add_argument(
        "--skip-fetch", action="store_true", help="Skip data fetching (use cached data)"
    )

    # Preprocessing arguments
    parser.add_argument(
        "--resample",
        type=str,
        default="1min",
        help="Resampling frequency. Use '1min' (native OHLCV) or downsample to '5min', '15min', '1h'. "
             "Cannot upsample OHLCV to sub-minute without tick data.",
    )
    parser.add_argument("--no-outlier-removal", action="store_true", help="Disable outlier removal")

    # Analysis arguments
    parser.add_argument(
        "--min-coherence",
        type=float,
        default=0.3,
        help="Minimum coherence threshold for relationships",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization generation")

    # TGAT arguments
    parser.add_argument(
        "--train-tgat", action="store_true", help="Train TGAT model after spectral analysis"
    )
    parser.add_argument(
        "--tgat-epochs", type=int, default=50, help="Max training epochs for TGAT"
    )
    parser.add_argument(
        "--tgat-window", type=int, default=300, help="Tick lookback window in seconds for TGAT"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("  CRYPTO LEAD-LAG ANALYSIS")
    print("=" * 70)

    # Determine symbols
    symbols = args.symbols or get_default_crypto_symbols()
    print(f"\nAssets: {', '.join(symbols)}")
    print(f"Period: {args.days} days")
    print(f"Resample frequency: {args.resample}")

    # Step 1: Fetch data
    if not args.skip_fetch:
        print("\n" + "=" * 70)
        print("  Step 1: Fetching Data from Binance")
        print("=" * 70)

        loader = BinanceDataLoader()

        try:
            data = loader.fetch_multiple_symbols(
                symbols=symbols, timeframe="1m", days_back=args.days, save=True
            )
        except Exception as e:
            print(f"\nError fetching data: {e}")
            print("Try using --skip-fetch to use cached data")
            return 1

        if not data:
            print("No data fetched. Exiting.")
            return 1
    else:
        print("\n" + "=" * 70)
        print("  Step 1: Loading Cached Data")
        print("=" * 70)

        # Try to load cached data
        # This is a simplified version - in practice, you'd need to specify
        # which cached files to load
        print("Using --skip-fetch requires manual data loading implementation")
        print("For now, please remove --skip-fetch to fetch fresh data")
        return 1

    # Step 2: Preprocess
    print("\n" + "=" * 70)
    print("  Step 2: Preprocessing Data")
    print("=" * 70)

    config = PreprocessConfig(
        resample_freq=args.resample,
        remove_outliers=not args.no_outlier_removal,
        outlier_std=10.0,
        price_column="close",
    )

    preprocessor = CryptoPreprocessor(config)

    try:
        processed_ohlcv, returns = preprocessor.process_pipeline(
            data,
            align=True,
            resample=True,
            calc_returns=True,
            remove_outliers=not args.no_outlier_removal,
        )
    except Exception as e:
        print(f"\nError during preprocessing: {e}")
        import traceback

        traceback.print_exc()
        return 1

    if returns is None or returns.empty:
        print("No returns calculated. Exiting.")
        return 1

    # Step 3: Spectral Analysis
    print("\n" + "=" * 70)
    print("  Step 3: Spectral Lead-Lag Analysis")
    print("=" * 70)

    # Determine sampling frequency from resample string
    if args.resample.endswith("s") or args.resample.endswith("S"):
        seconds = int(args.resample[:-1])
        sampling_freq = 1.0 / seconds
    elif args.resample.endswith("min"):
        minutes = int(args.resample[:-3])
        sampling_freq = 1.0 / (minutes * 60)
    else:
        print(f"Warning: Cannot parse resample frequency '{args.resample}', assuming 5s")
        sampling_freq = 0.2

    print(f"\nSampling frequency: {sampling_freq:.4f} Hz")
    print(f"Number of samples: {len(returns)}")
    print(f"Duration: {len(returns) / sampling_freq / 60:.1f} minutes")

    analyzer = LeadLagAnalyzer(sampling_freq)
    bands = create_crypto_frequency_bands(sampling_freq)

    print(f"\nAnalyzing {len(bands)} frequency bands:")
    for band in bands:
        print(f"  - {band}")

    try:
        results_raw = analyzer.analyze(returns, bands)
    except Exception as e:
        print(f"\nError during spectral analysis: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Step 4: Process Results
    print("\n" + "=" * 70)
    print("  Step 4: Processing Results")
    print("=" * 70)

    # Create band lookup dictionary
    band_info = {band.name: band for band in bands}

    results = {}
    for band_name, (delays_df, coherence_df) in results_raw.items():
        print(f"\n{band_name}:")
        print("-" * 60)

        # Rank relationships
        relationships = analyzer.rank_lead_lag_relationships(
            delays_df, coherence_df, min_coherence=args.min_coherence
        )

        if not relationships.empty:
            print("\nTop Lead-Lag Relationships:")
            print(relationships.head(10).to_string(index=False))
        else:
            print(f"No relationships found with coherence >= {args.min_coherence}")

        # Leadership scores
        scores = analyzer.get_asset_leadership_score(
            delays_df, coherence_df, min_coherence=args.min_coherence
        )

        print("\nLeadership Scores (positive = leads, negative = lags):")
        for asset, score in scores.items():
            print(f"  {asset:>8}: {score:>8.2f}s")

        # Store for visualization
        results[band_name] = (delays_df, coherence_df, relationships, scores)

        # Save to CSV
        delays_df.to_csv(output_dir / f"{band_name}_delays.csv")
        coherence_df.to_csv(output_dir / f"{band_name}_coherence.csv")
        relationships.to_csv(output_dir / f"{band_name}_relationships.csv", index=False)
        scores.to_csv(output_dir / f"{band_name}_leadership.csv", header=["score"])

    # Step 5: TGAT Training (optional)
    if args.train_tgat:
        print("\n" + "=" * 70)
        print("  Step 5: Training TGAT Model")
        print("=" * 70)

        try:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).parent / "src"))
            from models import TGATModel, GraphBuilder, TGATDataset, TGATTrainer
            from backend_utils import list_cached_data, load_cached_data

            # Find raw tick parquet files
            cached = list_cached_data()
            tick_files = [f for f in cached if f.get("timeframe", "") == "tick"]

            if not tick_files:
                print("No raw tick data found. Fetch tick data first (use the dashboard).")
            else:
                print(f"Found {len(tick_files)} tick data file(s)")
                tick_raw = load_cached_data(tick_files)  # dict[symbol -> DataFrame]

                # Determine which spectral band to use for graph seeding
                first_band = next(iter(results_raw))
                print(f"Seeding graph edges from spectral band: '{first_band}'")

                asset_names = list(returns.columns)
                gb = GraphBuilder(
                    asset_names=asset_names,
                    spectral_results=results_raw,
                    window_seconds=args.tgat_window,
                )

                train_ds = TGATDataset(
                    tick_raw, processed_ohlcv, gb, window_seconds=args.tgat_window, split="train"
                )
                val_ds = TGATDataset(
                    tick_raw, processed_ohlcv, gb, window_seconds=args.tgat_window, split="val"
                )
                test_ds = TGATDataset(
                    tick_raw, processed_ohlcv, gb, window_seconds=args.tgat_window, split="test"
                )

                print(f"Dataset: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test samples")

                if len(train_ds) == 0:
                    print("No training samples available. Ensure tick and 1-min OHLCV data overlap.")
                else:
                    model = TGATModel(n_assets=len(asset_names))
                    trainer = TGATTrainer(
                        model=model,
                        train_dataset=train_ds,
                        val_dataset=val_ds,
                        test_dataset=test_ds,
                        checkpoint_dir=str(output_dir / "tgat_checkpoints"),
                    )

                    history = trainer.train(max_epochs=args.tgat_epochs)
                    test_metrics = history.get("test_metrics", {})
                    print(f"\nTGAT Test MSE:      {test_metrics.get('mse', float('nan')):.6f}")
                    print(f"TGAT Spearman ρ:    {test_metrics.get('spearman_corr', float('nan')):.4f}")

                    # Spearman vs spectral leadership
                    spectral_scores = analyzer.get_asset_leadership_score(
                        list(results_raw.values())[0][0],
                        list(results_raw.values())[0][1],
                        min_coherence=args.min_coherence,
                    )
                    rho_vs_spectral = trainer.compute_spearman_vs_spectral(test_ds, spectral_scores)
                    print(f"TGAT vs Spectral ρ: {rho_vs_spectral:.4f}")

        except ImportError as e:
            print(f"Could not import TGAT modules: {e}")
            print("Ensure torch and torch-geometric are installed.")
        except Exception as e:
            print(f"Error during TGAT training: {e}")
            import traceback
            traceback.print_exc()

    # Step 6: Visualization
    if not args.no_viz:
        print("\n" + "=" * 70)
        print("  Step 5: Generating Visualizations")
        print("=" * 70)

        try:
            visualizer = LeadLagVisualizer(output_dir=output_dir)
            visualizer.create_summary_report(results, band_info=band_info)
        except Exception as e:
            print(f"\nError generating visualizations: {e}")
            import traceback
            traceback.print_exc()
            print("\nContinuing without visualizations...")

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    print(f"\nResults saved to: {output_dir}/")
    print("  - CSV files with delays, coherence, relationships, and leadership scores")
    if not args.no_viz:
        print("  - PNG visualizations of the analysis")

    # BTC analysis (if present)
    btc_assets = [col for col in returns.columns if "BTC" in col]
    if btc_assets:
        btc_asset = btc_assets[0]
        print("\nBTC Leadership Analysis:")
        print("-" * 60)

        for band_name, (delays_df, coherence_df, relationships, scores) in results.items():
            if btc_asset in scores:
                score = scores[btc_asset]
                if score > 0:
                    print(f"  {band_name}: BTC LEADS by {score:.2f}s on average")
                elif score < 0:
                    print(f"  {band_name}: BTC LAGS by {-score:.2f}s on average")
                else:
                    print(f"  {band_name}: BTC shows no clear lead/lag")

    print("\n" + "=" * 70)
    print("  Analysis Complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
