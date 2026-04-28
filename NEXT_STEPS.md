# Next Steps - Crypto Lead-Lag Analysis

## ✅ What's Been Completed (Phase 1 & 2)

### Binance Data Integration
- ✅ **data_loader.py**: Full Binance data fetcher using ccxt
  - Fetches OHLCV data for multiple symbols
  - Handles rate limiting and pagination
  - Saves to parquet format
  - Default symbols: BTC, ETH, BNB, SOL, XRP

- ✅ **preprocessor.py**: Complete preprocessing pipeline
  - Timestamp alignment across assets
  - Configurable resampling (default: 5-second bars)
  - Returns calculation (log returns)
  - Outlier removal
  - Data quality reporting

### Spectral Analysis
- ✅ **spectral_engine.py**: Already implemented (from your original code)
  - FFT-based cross-spectral density
  - Frequency band analysis
  - Coherence calculation
  - Well-tested (14 test cases)

- ✅ **analyzer.py**: High-level analysis wrapper
  - Frequency band definitions for crypto
  - Lead-lag relationship ranking
  - Leadership score calculation
  - Clean DataFrame outputs

### Visualization & Reporting
- ✅ **visualizer.py**: Comprehensive plotting
  - Delay/coherence heatmaps
  - Leadership score bar charts
  - Network diagrams
  - Time series with lag overlay

- ✅ **main.py**: CLI entry point
  - Full pipeline orchestration
  - Command-line arguments
  - CSV exports
  - Automatic visualization generation

- ✅ **test_crypto_pipeline.py**: End-to-end test script

- ✅ **README.md**: Complete documentation

## 🚀 How to Run (Once Dependencies Install)

### 1. Wait for Installation to Complete

The pip installation is currently running in the background. Once it completes:

```bash
# Verify installation
source venv/bin/activate
python -c "import numpy, pandas, ccxt; print('Dependencies OK!')"
```

### 2. Test with Real Data

```bash
# Run the end-to-end test (fetches 2 days of data)
python test_crypto_pipeline.py
```

Expected output:
- Fetches BTC, ETH, BNB, SOL, XRP from Binance
- Preprocesses to 5-second bars
- Runs spectral analysis on 3 frequency bands
- Displays delay matrices, coherence, and rankings
- Shows whether BTC leads or lags other assets

### 3. Run Full Analysis with Visualization

```bash
# Complete analysis with all features
python main.py --days 3

# Results will be in results/ directory:
# - CSV files with numerical results
# - PNG visualizations
```

### 4. Experiment with Parameters

```bash
# Different assets
python main.py --symbols BTC/USDT ETH/USDT DOGE/USDT --days 7

# Different time resolution
python main.py --resample 10S --days 5

# Different coherence threshold
python main.py --min-coherence 0.5
```

## 📊 Expected Results

Based on the spec and typical crypto market behavior, you should see:

1. **BTC tends to lead major altcoins** at medium frequencies (5-15 min)
2. **Higher coherence between similar assets** (e.g., ETH-BNB vs BTC-DOGE)
3. **Frequency-dependent relationships** (different leads/lags at different timescales)
4. **Coherence values 0.3-0.7** for strong crypto pairs

## 🔄 Phase 3: ML Infrastructure (Next Priority)

Since you haven't decided on ML methods yet, here's a suggested roadmap:

### Option A: Start Simple (Recommended for 1-2 weeks)
1. **Feature Engineering**
   - Create file: `src/features.py`
   - Node features: returns, volatility, volume, momentum
   - Edge features: correlation, spectral coherence
   - Temporal windows

2. **Baseline ML Models**
   - Simple LSTM or GRU for comparison
   - Linear regression on engineered features
   - Establish baseline performance

3. **Compare with Spectral**
   - Use spectral results as ground truth
   - Evaluate ML predictions
   - Correlation analysis

### Option B: Full GAT/TGAT (Per Original Spec)
1. **Add PyTorch Dependencies**
   ```bash
   pip install torch torch-geometric torch-geometric-temporal
   ```

2. **Implement GAT** (`src/models/gat.py`)
   - Graph construction from time series
   - 2-layer GAT with 4 attention heads (per spec)
   - Loss function: MSE on spectral delays

3. **Implement TGAT** (`src/models/tgat.py`)
   - Event stream format
   - Temporal attention mechanism
   - Dynamic graph updates

4. **Training Pipeline** (`src/train.py`)
   - Data loaders
   - Training loop
   - Validation & checkpointing

5. **Comparison Analysis** (`src/compare_methods.py`)
   - Spectral vs GAT vs TGAT
   - Spearman correlation of rankings
   - Attention weight visualization

### Option C: Explore First, Decide Later
1. **Run Analysis on Longer Periods**
   ```bash
   python main.py --days 14  # 2 weeks of data
   python main.py --days 30  # 1 month
   ```

2. **Analyze Results**
   - What frequencies show strongest lead-lag?
   - Which asset pairs have highest coherence?
   - Are results stable over time?

3. **Based on findings, choose ML approach**
   - If relationships are simple → GAT
   - If time-varying → TGAT
   - If spectral works well → enhance it instead

## 🐛 Troubleshooting

### If Data Fetch Fails
```bash
# Check ccxt installation
python -c "import ccxt; print(ccxt.__version__)"

# Test Binance connection
python -c "import ccxt; ex = ccxt.binance(); print(ex.fetch_ticker('BTC/USDT'))"
```

### If Spectral Analysis Fails
```bash
# Run unit tests
python -m pytest tests/test_spectral.py -v

# Check data shape
python -c "from src.data_loader import *; loader = BinanceDataLoader(); print('OK')"
```

### If Visualization Fails
```bash
# Test matplotlib
python -c "import matplotlib.pyplot as plt; print('OK')"

# Run without viz
python main.py --no-viz
```

## 📝 My Recommendations

Given your goals (research comparison, 1-2 weeks, M4 Mac):

### Week 1 (Current Status: ✅ Done!)
- ✅ Data pipeline
- ✅ Spectral analysis
- ✅ Visualization
- **Next**: Run on real data and analyze results

### Week 2 (Your Choice)
**Path A - Quick ML Baseline:**
1. Day 1-2: Simple feature engineering
2. Day 3-4: LSTM/GRU baseline
3. Day 5-7: Comparison with spectral

**Path B - Full GAT:**
1. Day 1-2: PyTorch + PyG setup
2. Day 3-5: GAT implementation
3. Day 6-7: Training and evaluation

**Path C - Enhanced Analysis:**
1. Day 1-3: Longer time periods, more assets
2. Day 4-5: Time-varying analysis (rolling windows)
3. Day 6-7: Stability analysis and write-up

## 💡 Quick Wins

If you want immediate results:

```bash
# 1. Run quick test (5 min)
python test_crypto_pipeline.py

# 2. Full 7-day analysis (15 min)
python main.py --days 7

# 3. Explore results/
open results/  # View generated plots
cat results/*_relationships.csv  # See top relationships
```

## 📚 Code Structure Summary

```
Core Pipeline:
  data_loader.py → preprocessor.py → analyzer.py → visualizer.py

Entry Points:
  main.py              (production CLI)
  test_crypto_pipeline.py  (testing/validation)

Future ML Extension:
  src/features.py      (feature engineering)
  src/models/          (GAT, TGAT, etc.)
  src/train.py         (training pipeline)
  src/compare_methods.py  (benchmarking)
```

---

**Status**: Phase 1-2 complete! Ready to run on real crypto data.
**Next**: Choose ML path based on initial spectral results.
