"""
Backend utilities for Streamlit dashboard.

Provides wrappers for data management, cached data loading, and results persistence.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import json

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import BinanceDataLoader


def list_cached_data() -> List[Dict]:
    """
    List all cached parquet files in data/ directory with metadata.

    Returns:
        List of dictionaries with keys: filename, symbol, timeframe, start_date,
        end_date, file_size, filepath
    """
    data_dir = Path(__file__).parent.parent / "data"

    if not data_dir.exists():
        return []

    cached_files = []

    for parquet_file in data_dir.glob("*.parquet"):
        try:
            # Parse filename patterns:
            # OHLCV: BTC_USDT_1m_20260418_20260420.parquet
            # Raw tick: BTC_USDT_tick_20260420_20260420.parquet
            # Resampled tick: BTC_USDT_5s_tick_20260420_20260420.parquet
            parts = parquet_file.stem.split('_')

            if len(parts) >= 5:
                # Detect tick data files
                if 'tick' in parts:
                    # Tick data file
                    tick_idx = parts.index('tick')

                    # Symbol is everything before timeframe/tick
                    if tick_idx >= 3:
                        # Resampled tick: BTC_USDT_5s_tick_...
                        symbol = '_'.join(parts[:tick_idx - 1])
                        timeframe_part = parts[tick_idx - 1]
                        timeframe = f"{timeframe_part} (tick)"
                    else:
                        # Raw tick: BTC_USDT_tick_...
                        symbol = '_'.join(parts[:tick_idx])
                        timeframe = "tick"

                    # Dates are after 'tick'
                    start_date = parts[tick_idx + 1]
                    end_date = parts[tick_idx + 2]
                else:
                    # OHLCV data file
                    timeframe_idx = -3
                    symbol = '_'.join(parts[:timeframe_idx])
                    timeframe = parts[timeframe_idx]
                    start_date = parts[timeframe_idx + 1]
                    end_date = parts[timeframe_idx + 2]

                # Format dates for display
                start_display = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
                end_display = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"

                # Get file size
                file_size_bytes = parquet_file.stat().st_size
                if file_size_bytes < 1024:
                    file_size = f"{file_size_bytes} B"
                elif file_size_bytes < 1024 * 1024:
                    file_size = f"{file_size_bytes / 1024:.1f} KB"
                else:
                    file_size = f"{file_size_bytes / (1024 * 1024):.1f} MB"

                cached_files.append({
                    'filename': parquet_file.name,
                    'symbol': symbol.replace('_', '/'),  # Convert back to BTC/USDT format
                    'timeframe': timeframe,
                    'start_date': start_display,
                    'end_date': end_display,
                    'file_size': file_size,
                    'filepath': str(parquet_file)
                })
        except Exception as e:
            # Skip malformed filenames
            print(f"Warning: Could not parse {parquet_file.name}: {e}")
            continue

    # Sort by symbol, then by start_date (most recent first)
    cached_files.sort(key=lambda x: (x['symbol'], x['start_date']), reverse=True)

    return cached_files


def load_cached_data(file_info_list: List[Dict]) -> Dict[str, pd.DataFrame]:
    """
    Load selected cached parquet files.

    Args:
        file_info_list: List of file info dictionaries from list_cached_data()

    Returns:
        Dictionary mapping symbol to DataFrame

    Raises:
        Exception: If all files fail to load, with details about the errors
    """
    loader = BinanceDataLoader()
    data_dict = {}
    errors = []

    for file_info in file_info_list:
        try:
            filepath = Path(file_info['filepath'])
            df = loader.load_from_parquet(str(filepath))

            # Use symbol from filename as key
            symbol = file_info['symbol']

            # If symbol already exists, merge/append data
            if symbol in data_dict:
                # Combine and remove duplicates
                combined = pd.concat([data_dict[symbol], df])
                combined = combined[~combined.index.duplicated(keep='first')]
                combined = combined.sort_index()
                data_dict[symbol] = combined
            else:
                data_dict[symbol] = df

        except Exception as e:
            error_msg = f"Error loading {file_info['filename']}: {str(e)}"
            print(error_msg)
            errors.append(error_msg)
            continue

    # If all files failed to load, raise an exception with details
    if not data_dict and errors:
        raise Exception(f"Failed to load any data files:\n" + "\n".join(errors))

    return data_dict


def delete_cached_data(file_info_list: List[Dict]) -> Tuple[int, int]:
    """
    Delete selected cached parquet files.

    Args:
        file_info_list: List of file info dictionaries from list_cached_data()

    Returns:
        Tuple of (successful_deletions, failed_deletions)
    """
    success_count = 0
    fail_count = 0

    for file_info in file_info_list:
        try:
            filepath = Path(file_info['filepath'])
            if filepath.exists():
                filepath.unlink()
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"Error deleting {file_info['filename']}: {e}")
            fail_count += 1

    return success_count, fail_count


def save_results_timestamped(
    results: Dict,
    band_info: Dict,
    config: Dict,
    output_base_dir: Path = None
) -> Path:
    """
    Save analysis results to a timestamped subdirectory.

    Args:
        results: Dictionary mapping band name to (delays_df, coherence_df, relationships_df, scores)
        band_info: Dictionary mapping band name to FrequencyBand object
        config: Dictionary with analysis configuration metadata
        output_base_dir: Base results directory (default: results/)

    Returns:
        Path to the created results directory
    """
    if output_base_dir is None:
        output_base_dir = Path(__file__).parent.parent / "results"

    output_base_dir.mkdir(exist_ok=True)

    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_base_dir / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)

    # Save configuration metadata
    config_path = run_dir / "config.json"
    with open(config_path, 'w') as f:
        # Convert any non-serializable objects to strings
        serializable_config = {
            k: str(v) if not isinstance(v, (int, float, str, bool, list, dict, type(None))) else v
            for k, v in config.items()
        }
        json.dump(serializable_config, f, indent=2)

    # Save results for each frequency band
    for band_name, (delays_df, coherence_df, relationships_df, scores) in results.items():
        # Save CSV files
        delays_df.to_csv(run_dir / f"{band_name}_delays.csv")
        coherence_df.to_csv(run_dir / f"{band_name}_coherence.csv")
        relationships_df.to_csv(run_dir / f"{band_name}_relationships.csv", index=False)
        scores.to_csv(run_dir / f"{band_name}_leadership.csv", header=["score"])

    # Generate visualizations using existing visualizer
    from visualizer import LeadLagVisualizer

    try:
        visualizer = LeadLagVisualizer(output_dir=run_dir)
        visualizer.create_summary_report(results, band_info=band_info)
    except Exception as e:
        print(f"Warning: Visualization generation failed: {e}")

    return run_dir


def list_analysis_runs(output_base_dir: Path = None) -> List[Dict]:
    """
    List all timestamped analysis runs.

    Args:
        output_base_dir: Base results directory (default: results/)

    Returns:
        List of dictionaries with keys: timestamp, run_dir, config
    """
    if output_base_dir is None:
        output_base_dir = Path(__file__).parent.parent / "results"

    if not output_base_dir.exists():
        return []

    runs = []

    for run_dir in output_base_dir.glob("run_*"):
        if run_dir.is_dir():
            # Extract timestamp from directory name
            timestamp_str = run_dir.name.replace("run_", "")

            try:
                # Parse timestamp
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                timestamp_display = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            except:
                timestamp_display = timestamp_str

            # Load config if available
            config_path = run_dir / "config.json"
            config = {}
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                except:
                    pass

            runs.append({
                'timestamp': timestamp_display,
                'run_dir': str(run_dir),
                'config': config
            })

    # Sort by timestamp (most recent first)
    runs.sort(key=lambda x: x['timestamp'], reverse=True)

    return runs


def get_data_quality_summary(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Generate a data quality summary for loaded data.

    Args:
        data_dict: Dictionary mapping symbol to DataFrame

    Returns:
        DataFrame with quality metrics
    """
    from preprocessor import CryptoPreprocessor, PreprocessConfig

    preprocessor = CryptoPreprocessor(PreprocessConfig())
    return preprocessor.get_data_quality_report(data_dict)


if __name__ == '__main__':
    # Quick test
    print("Testing backend_utils...")

    print("\n1. Listing cached data:")
    cached = list_cached_data()
    for item in cached[:5]:  # Show first 5
        print(f"  {item['symbol']} {item['timeframe']} {item['start_date']} to {item['end_date']}")

    print(f"\nTotal cached files: {len(cached)}")

    print("\n2. Listing analysis runs:")
    runs = list_analysis_runs()
    for run in runs[:5]:  # Show first 5
        print(f"  {run['timestamp']}: {run.get('config', {}).get('symbols', 'N/A')}")

    print(f"\nTotal analysis runs: {len(runs)}")
