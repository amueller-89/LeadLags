# TGAT Training Guide

## What the Training Interface Does

The TGAT tab trains a Temporal Graph Attention Network end-to-end on raw tick data. The goal is
not a production trading system — it is a research question: **does the model learn, from tick
dynamics alone, the same lead-lag structure that the spectral engine identifies from OHLCV
returns?** The primary metric is Spearman rank correlation between TGAT's attention weights and
the spectral leadership scores, not predictive accuracy.

---

## Prerequisites

Three things must be in place before training can start:

| Prerequisite | What it is | How to get it |
|---|---|---|
| Spectral analysis | Delay + coherence matrices for each frequency band | Run Analysis tab |
| OHLCV data | 1-minute candles loaded into session | Load or fetch in Data Management |
| Raw tick data | Per-trade price/quantity files | Fetch via "Historical Tick (Binance Vision)" or "Tick Data (Trades)" in Data Management |

The tick data provides the **input events** to the model. The OHLCV data provides the **labels**
(next 1-minute bar log return per asset). The spectral results provide the **static edge features**
that seed the graph structure.

---

## Configuration Options

### Tick Lookback Window (seconds)
**Default: 300 s. Recommended range: 60–600 s.**

For each 1-minute bar boundary (the query time), the model looks back this far and collects all
trades in the window `[t - W, t)`. These trades become the input events.

- **Shorter windows (60–120 s):** Capture only very recent microstructure. Better for detecting
  sub-minute lead-lag. Fewer ticks per asset → faster training.
- **Longer windows (300–600 s):** Cover slower lead-lag signals. More ticks → richer context but
  slower per-sample computation (inference time scales linearly with tick count).
- **Rule of thumb:** Match the window to the frequency band you care about. The short-term band
  (lags of ~30–120 s) is well covered by 300 s. Going above 600 s rarely helps with 1-minute OHLCV labels.

### Max Epochs
**Default: 50. Recommended range: 30–200.**

An epoch is one complete pass over all training query times. With early stopping active (see
below), training typically stops well before the maximum.

- With **< 2 weeks of tick data** (< ~10 000 training samples), 50 epochs is usually enough.
- With **more data**, training longer helps but early stopping will handle it automatically.
- Setting this very high (200+) is safe — early stopping will terminate training when it stops
  improving.

### Early Stopping Patience
**Default: 10. Recommended range: 5–20.**

Training halts if validation MSE has not improved for this many consecutive epochs. The best
checkpoint (lowest val MSE) is restored at the end.

- **Lower patience (5–7):** Stops faster, risks stopping before the model has converged on noisy
  financial data.
- **Higher patience (15–20):** Gives the model more time to escape local plateaus. Recommended if
  training loss is still decreasing but val loss is noisy.
- With very small datasets (< 1 000 training samples), use patience ≥ 10 — val MSE is volatile.

### Learning Rate
**Default: 1e-3. Options: 1e-4, 5e-4, 1e-3, 5e-3.**

Step size for the Adam optimiser.

| Rate | When to use |
|---|---|
| `1e-3` | Standard starting point. Use this unless training is unstable. |
| `5e-4` | If loss oscillates or spikes during training. |
| `1e-4` | If the model is not learning at all (loss flat from epoch 1). Rare — usually indicates a data problem instead. |
| `5e-3` | Aggressive. Can converge faster on large datasets but risks instability. Not recommended. |

Gradient norms are clipped to 1.0, which provides some protection against large learning rates,
but `1e-3` is almost always the right default.

### Use Spectral Edge Seeding
**Default: OFF (clean mode — recommended for the primary experiment).**

This toggle controls whether spectral delay/coherence matrices are passed to the model as
static edge features. The choice fundamentally changes what the evaluation metrics mean.

#### The circular dependency problem

When seeding is ON, the inter-asset attention score for asset `j` is:
```
score_j = a^T LeakyReLU(W_inter · [h_i || h_j_agg || e_ij])
```
where `e_ij = [delay_norm, coherence]` from spectral analysis. A model can score high on
"vs Spectral ρ" simply by learning to upweight the `coherence` dimension of `e_ij` — without
learning anything from tick dynamics. The metric we care most about is the one most easily
gamed by the seeding.

#### Seeding OFF (recommended first run)

Edge features are all zeros. The model receives no spectral information. "vs Spectral ρ"
becomes a genuine independent validation: positive values mean tick-based attention and
spectral phase analysis agree on the leadership ranking, purely from different information
sources.

#### Seeding ON

Edge features carry spectral delay/coherence as a structural prior. The model may converge
faster and to a better return-prediction MSE. But "vs Spectral ρ" is now confounded —
agreement with spectral rankings may just mean the model learned to copy the prior. In seeded
mode, the meaningful comparison is **test MSE between seeded and unseeded runs**: if seeding
lowers test MSE, the spectral prior genuinely helped prediction.

#### Spectral Band for Edge Seeding
Only relevant when seeding is ON.

- **`short_term` (lags 30–120 s):** Best match for the 300 s lookback window.
- **`medium_term` (lags 120–600 s):** Appropriate with a 600 s window and several weeks of data.
- **`long_term`:** Multi-minute structure that tick data in a 5-minute window cannot resolve.
  Edge seeding from this band is unlikely to help.

If the spectral result for a pair is NaN, the edge feature is `[0, 0]` — model treats that pair as unknown.

---

## Algorithm Steps (What Happens When You Click Train)

### Step 1 — Data loading and alignment

The training pipeline constructs a timeline of **query times**: all 1-minute bar boundaries where
every asset has a valid current close price *and* a valid next-bar close price. These are
derived from the intersection of all assets' OHLCV indices.

The timeline is split chronologically (not randomly):

| Split | Fraction | Purpose |
|---|---|---|
| Train | 60 % (earliest) | Gradient updates |
| Validation | 20 % (middle) | Early stopping |
| Test | 20 % (latest) | Final evaluation |

Chronological splitting is critical for financial data — random splits leak future information
into the training set.

### Step 2 — Event batch construction (per query time)

For each query time `t`, the model receives one **EventBatch**:

- For each of the N assets, all trades in `[t - W, t)` are collected.
- Each trade becomes two features: `log_return` (log price change from previous trade) and
  `log_volume` (log of trade quantity).
- The time of each trade is converted to a **time lag** `Δt = t - τ` in seconds.
- Assets with no trades in the window contribute an empty tensor — handled as a zero context
  vector in the attention layer.
- Static edge features `(N, N, 2)` are attached from the spectral results. These are the same
  for every query time.

### Step 3 — Two-level temporal attention (per query)

Each TGAT layer applies two rounds of attention. See `TGAT_Design.md` for the mathematical
details; the key intuition is:

**Level 1 — intra-asset:** For each source asset `j`, all of `j`'s ticks in the window are
compressed into a single context vector via attention. This prevents high-volume assets (BTC has
~10× more trades than SOL) from dominating simply because they contribute more tokens.

**Level 2 — inter-asset:** The N per-asset context vectors are combined via a second round of
attention, incorporating the static edge features. The resulting scalar weight `β[i, j]` is how
much query node `i` attends to asset `j`. These weights are the interpretable output.

The model stacks 2 such layers with LayerNorm + residual connections between them.

### Step 4 — Prediction and loss

A small MLP maps each node's embedding to a scalar: the predicted next-bar log return. The loss
is MSE over all N assets × all training query times.

Gradients are clipped to norm 1.0 before each Adam step to prevent instability from
high-variance trades.

### Step 5 — Validation and early stopping

After each epoch, the model is evaluated on the validation set (no gradient updates). If
validation MSE improves, the model checkpoint is saved. If it does not improve for `patience`
consecutive epochs, training stops and the best checkpoint is restored.

### Step 6 — Post-training evaluation

Three metrics are computed on the held-out test set:

**`Test MSE`** — mean squared error of predicted vs actual next-bar log returns. Expect values
in the range 1e-5 to 1e-4 for 1-minute crypto returns. The absolute value is not very
meaningful; use it only to monitor training stability.

**`Prediction Spearman ρ`** — Spearman rank correlation between predicted and actual log returns
across all (time, asset) pairs. Financial return prediction is inherently noisy. Values of
0.02–0.10 are realistic; anything above 0.15 is exceptional and should be inspected for data
leakage.

**`vs Spectral Spearman ρ`** *(primary metric — only meaningful with seeding OFF)* — For each
asset, compute the mean attention it receives across all query nodes and all test-set queries.
This gives an "attention leadership score" per asset. Correlate this ranking with the spectral
leadership scores from the chosen frequency band.

**Seeding OFF:** A positive ρ is a genuine result — TGAT independently discovered from tick
dynamics the same leadership ordering that spectral phase analysis found from OHLCV returns.
Values of 0.4–0.8 would be a strong result.

**Seeding ON:** This metric is confounded. The model was given spectral coherence as input and
may have learned to copy it. Do not interpret a high ρ as independent confirmation in this
mode. Instead compare test MSE between seeded and unseeded runs to assess whether the prior
helped.

---

## Practical Recommendations

### Minimum viable dataset
One week of tick data for 3–5 assets yields ~6 000 training samples. Expect noisy but
interpretable results. For the spectral comparison to be meaningful, use at least 2 weeks.

### First run checklist
1. Use default settings (W=300 s, lr=1e-3, patience=10, `short_term` band).
2. Watch the loss chart — train and val MSE should both decrease over the first 5–10 epochs. If
   train MSE decreases but val MSE does not, the model is overfitting (reduce epochs or add
   dropout; the default dropout=0.1 is already conservative).
3. If train MSE is flat from epoch 1, the data pipeline has a problem (check that tick data and
   OHLCV overlap in time).
4. Check `vs Spectral Spearman ρ`. If it is NaN, there is an asset name mismatch between tick
   data keys and spectral result indices.

### If training is too slow
With a 300 s window and BTC at ~1 trade/second, each query processes ~300 ticks × N assets.
For 5 assets and 864 training samples, one epoch can take 2–5 minutes on CPU.

- Reduce the window to 120 s (fewer ticks per query, ~3× speedup).
- Use fewer assets (3 instead of 5).
- Reduce `max_epochs` if you just want a quick sanity check — the overfit test
  (`test_tgat_overfit_tiny`) is faster than a full training run.

### Interpreting the attention heatmap
The heatmap shows `β[i, j]` averaged over the test set. Read it as: "when predicting asset `i`'s
next return, how much does the model look at asset `j`'s recent tick history?"

A high off-diagonal value in row `i`, column `j` (with `j ≠ i`) means the model found `j`'s
recent trades predictive for `i`. If this pattern aligns with the spectral delay matrix (asset
`j` leads asset `i`), that is the confirmation the research question asks for.
