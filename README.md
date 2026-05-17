# Crypto Lead-Lag Analysis

A dashboard for analyzing lead-lag relationships in cryptocurrency markets using both spectral methods and machine learning.

### Key Question

- Can we recover lead-lag relationships with learned (temporal) attention networks?
- What additional structure is learned?
- quantify value predictions (Although we would be very surprised if this approach at this level of sophistication provides any real alpha)

## Project Structure

```
LeadLags/
├── src/
│   ├── data_loader.py        # Binance data fetching
│   ├── preprocessor.py        # Data preprocessing & alignment
│   ├── spectral_engine.py     # FFT-based lead-lag detection
│   ├── analyzer.py            # High-level analysis wrapper
│   ├── visualizer.py          # Visualization functions
│   └── config.py              # Configuration (placeholder)
├── tests/
│   ├── test_spectral.py       # Spectral analysis tests
│   ├── util.py                # Synthetic signal generators
│   └── test_stationarity.py   # Stationarity tests (placeholder)
├── data/                      # Downloaded data (parquet files)
├── results/                   # Analysis outputs
├── notebooks/                 # Jupyter notebooks
├── main.py                    # Main CLI entry point
├── test_crypto_pipeline.py    # End-to-end test script
└── requirements.txt           # Python dependencies
```

## Installation

### Prerequisites

- Python 3.9+
- Virtual environment (recommended)

### Setup

```bash
# Clone or navigate to the project directory
cd LeadLags

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- **Data Science**: numpy, pandas, scipy, statsmodels
- **Data Fetching**: ccxt (cryptocurrency exchange API)
- **Visualization**: matplotlib, seaborn
- **Storage**: pyarrow (parquet files)
- **ML** (coming soon): torch, torch_geometric, torch_geometric_temporal

## Quick Start

### Option 1: Streamlit Dashboard (Recommended)

The easiest way to use this project is through the interactive Streamlit dashboard:

```bash
# Activate virtual environment
source venv/bin/activate

# Launch the dashboard
streamlit run streamlit_app.py
```

This will open a web interface in your browser with three tabs:
- **Data Management** - Fetch new data or load cached data
- **Analysis** - Configure and run spectral analysis
- **Results** - View and download analysis results

### Option 2: Command Line Interface

```bash
# Activate virtual environment
source venv/bin/activate

# Run analysis with default settings (BTC, ETH, BNB, SOL, XRP for 2 days)
python main.py

# Custom analysis
python main.py --symbols BTC/USDT ETH/USDT --days 7 --resample 5min
```

### Command Line Options

```
usage: main.py [-h] [--symbols SYMBOLS [SYMBOLS ...]] [--days DAYS]
               [--skip-fetch] [--resample RESAMPLE] [--no-outlier-removal]
               [--min-coherence MIN_COHERENCE] [--output-dir OUTPUT_DIR]
               [--no-viz]

Options:
  --symbols            Crypto symbols (default: BTC/USDT ETH/USDT BNB/USDT SOL/USDT XRP/USDT)
  --days               Days of data to fetch (default: 2)
  --skip-fetch         Use cached data instead of fetching
  --resample           Resampling frequency (default: 1min)
                       Examples: 1min (native), 5min, 15min, 1h (downsampling only)
  --no-outlier-removal Disable outlier removal
  --min-coherence      Minimum coherence threshold (default: 0.3)
  --output-dir         Results directory (default: results/)
  --no-viz             Skip visualization generation
```

### Test the Pipeline

```bash
# Run end-to-end test with real Binance data
python test_crypto_pipeline.py
```

## Usage Examples

### 1. Basic Analysis

```python
from src.data_loader import BinanceDataLoader
from src.preprocessor import CryptoPreprocessor, PreprocessConfig
from src.analyzer import LeadLagAnalyzer, create_crypto_frequency_bands

# Fetch data
loader = BinanceDataLoader()
data = loader.fetch_multiple_symbols(
    symbols=['BTC/USDT', 'ETH/USDT'],
    timeframe='1m',
    days_back=3
)

# Preprocess
config = PreprocessConfig(resample_freq='1min')
preprocessor = CryptoPreprocessor(config)
_, returns = preprocessor.process_pipeline(data)

# Analyze
analyzer = LeadLagAnalyzer(sampling_frequency=1/60)  # 1-minute bars = 0.0167 Hz
bands = create_crypto_frequency_bands(1/60)
results = analyzer.analyze(returns, bands)

# Display results
for band_name, (delays_df, coherence_df) in results.items():
    print(f"\n{band_name}:")
    print(delays_df)
```

### 2. Custom Frequency Bands

```python
from src.analyzer import FrequencyBand, LeadLagAnalyzer

# Define custom frequency bands
bands = [
    FrequencyBand(name="30sec_window", f_min=1/60, f_max=1/30),
    FrequencyBand(name="5min_window", f_min=1/600, f_max=1/300),
]

analyzer = LeadLagAnalyzer(sampling_frequency=0.2)
results = analyzer.analyze(returns, bands)
```

### 3. Visualization

```python
from src.visualizer import LeadLagVisualizer

visualizer = LeadLagVisualizer(output_dir='results')

# Plot delay matrix
visualizer.plot_delay_matrix(
    delays_df,
    coherence_df,
    title="BTC vs Altcoins - 1-10 Minute Trends",
    save_path="results/btc_altcoins.png"
)

# Plot leadership scores
scores = analyzer.get_asset_leadership_score(delays_df, coherence_df)
visualizer.plot_leadership_scores(scores)

# Plot relationship network
relationships = analyzer.rank_lead_lag_relationships(delays_df, coherence_df)
visualizer.plot_relationship_network(relationships)
```

## How It Works

### Spectral Analysis Method

The spectral method detects lead-lag relationships using FFT-based cross-spectral density:

1. **FFT Computation**: Apply Hann window and compute FFT for each time series
2. **Cross-Spectrum**: Build cross-spectrum matrix `S_xy = X(f) * conj(Y(f))`
3. **Phase Extraction**: `phase = angle(S_xy)`
4. **Delay Calculation**: `delay = phase / (2π * freq)`
5. **Frequency Banding**: Average delays within specified frequency bands
6. **Coherence**: Compute `|mean(S_xy)|² / (mean(S_xx) * mean(S_yy))` as confidence metric

**Interpretation**:
- **Positive delay[i,j]**: Series i leads series j
- **Negative delay[i,j]**: Series i lags series j
- **Coherence**: 0-1 scale, higher = stronger relationship

### Data Pipeline

```
Binance API → 1min OHLCV → Align Timestamps → Resample (1min or downsample) →
Calculate Returns → Remove Outliers → Spectral Analysis → Visualize
```

## Output Files

After running the analysis, the `results/` directory will contain:

### CSV Files (per frequency band)
- `{band}_delays.csv` - Delay matrix (seconds)
- `{band}_coherence.csv` - Coherence matrix (0-1)
- `{band}_relationships.csv` - Ranked lead-lag pairs
- `{band}_leadership.csv` - Leadership scores per asset

### Visualizations
- `{band}_delays.png` - Heatmaps of delays and coherence
- `{band}_leadership.png` - Bar chart of leadership scores
- `{band}_network.png` - Network diagram of relationships

## Testing

```bash
# Run spectral analysis tests
python -m pytest tests/test_spectral.py -v

# Run all tests
python -m pytest tests/ -v
```


---

**Note**: This project is for research and educational purposes. Not financial advice.
