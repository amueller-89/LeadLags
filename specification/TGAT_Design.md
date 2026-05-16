# TGAT Design: Lead-Lag Detection on Crypto Tick Data

## Background: TGAT vs TGN

This project uses a TGAT-inspired architecture, but it is worth understanding how TGAT relates to the more general TGN framework, since the two are often confused.

**TGAT** (Xu et al., 2020 — *Inductive Representation Learning on Temporal Graphs*)  
Purely attention-based. At query time `t`, node `i` aggregates from its K most recent temporal neighbours using time-encoded attention. No persistent memory. Each query is computed fresh from the lookback window.

**TGN** (Rossi et al., 2020 — *Temporal Graph Networks for Deep Learning on Dynamic Graphs*)  
Adds a per-node **memory module** (a GRU state updated after every event) on top of the same temporal attention. The memory captures history beyond the fixed lookback window and is reset only at construction. TGN is strictly more expressive than TGAT: TGAT is a special case of TGN with no memory.

**Why TGAT here?**  
Lead-lag relationships in crypto operate at sub-minute timescales (confirmed by the spectral engine). A 5-minute lookback window captures the relevant dynamics without needing cross-day memory. TGAT is also significantly simpler to implement correctly. A TGN extension would be valuable if rolling regime changes (e.g. BTC's leadership flipping on macro news) turn out to matter.

---

## The Problem Formulation

Given a set of N crypto assets, each emitting trades at irregular timestamps:

- **Nodes**: assets (BTC, ETH, SOL, ...)
- **Events**: each trade `(asset_j, time τ, log_return, log_volume)` is a timestamped event on node `j`
- **Query**: at each 1-minute bar boundary `t`, compute updated node embeddings → predict next-bar log return per asset

The spectral engine already tells us *statically* which assets lead which and by how long. TGAT's job is to *learn* these relationships end-to-end from raw tick dynamics, then be compared against the spectral baseline via Spearman rank correlation.

---

## Our TGAT Variant

The architecture departs from the original TGAT paper in several deliberate ways suited to the lead-lag problem.

### Cross-asset attention

In the original TGAT, node `i` aggregates only from events that node `i` itself participated in (its own interaction history). For the lead-lag problem this makes no sense: we *want* node `i` (ETH) to attend to node `j`'s (BTC) recent tick history to detect that BTC moved first.

We therefore aggregate across all assets' tick histories. For query node `i` at time `t`, the temporal neighbourhood is the union of all ticks from all assets `j` in the window `[t - W, t)`.

### Two-level attention (cross-asset softmax imbalance fix)

The naive approach puts all ticks from all assets into a single softmax. BTC trades roughly 10× more frequently than SOL. In a flat softmax, BTC dominates not because the model learned it leads, but because it contributes 10× more tokens. This is a data-volume artefact, not a learned signal.

**Fix: two-level attention.**

**Level 1 — intra-asset** (within each source asset `j`):  
Softmax only over asset `j`'s own ticks. Produces a single context vector `h_j_agg` per asset, regardless of how many ticks `j` had.

```
α_jk = softmax_k( a_intra · LeakyReLU( W_intra · [h_i || h_jk || Φ(Δt_jk)] ) )
h_j_agg = Σ_k  α_jk · W_v_intra · [h_jk || Φ(Δt_jk)]
```

Assets with zero ticks in the window contribute a zero vector at this stage.

**Level 2 — inter-asset** (across assets):  
Softmax over the N per-asset summaries, incorporating static edge features (spectral delay, coherence).

```
β_j = softmax_j( a_inter · LeakyReLU( W_inter · [h_i || h_j_agg || e_ij] ) )
out_i = Σ_j  β_j · W_v_inter · h_j_agg
```

Each asset now contributes exactly one vector at the inter-asset stage. A high-frequency asset and a low-frequency asset are on equal footing in the inter-asset softmax.

The inter-asset weights `β[i, j]` are the directly interpretable quantities: "how much does query node `i` attend to asset `j`'s history?" These are what we compare against spectral leadership scores via Spearman correlation.

### Spectral seeding of edge features

The edge feature vector `e_ij = [delay_norm, coherence]` is computed once from the spectral engine output (chosen frequency band, default: `short_term`) and passed as a static input to the inter-asset attention at every query. Specifically:

- `delay_norm = delay_seconds / W` where `W` is the lookback window, so delays are O(1)
- `coherence ∈ [0, 1]` from the cross-spectral magnitude

This gives the model a strong structural prior: pairs with high spectral coherence and known delay are likely to have non-trivial inter-asset attention. The model can confirm, refine, or contradict this prior through training.

Diagonal entries: `[0.0, 1.0]` (zero self-delay, full self-coherence).  
Pairs with NaN spectral output (no signal in band): `[0.0, 0.0]`.

### Fixed lookback window (vs K-neighbour sampling)

The original TGAT samples exactly K most recent temporal neighbours per node. We instead take *all* ticks in a fixed time window `[t - W, t)`. This is simpler and avoids the need to choose K, but means window size W is the key hyperparameter. A narrow window misses the longer lags the spectral engine detects; a wide window includes stale ticks and is slower.

Recommended range: 60–600 seconds. Default: 300 s (5 minutes), which covers all frequency bands the spectral engine identifies for 1-min OHLCV data.

### Discrete-ish query times

Queries are issued at 1-minute bar boundaries (matching the OHLCV label frequency), not at every tick. This means the model is trained in a discrete rhythm even though the input events are continuous-time. The prediction target is the next 1-min bar log return per asset.

### No temporal neighbourhood sampling

No K-cap on the number of ticks per asset per window. For BTC at ~1 trade/second this can be ~300 ticks per 5-minute window. The intra-asset softmax handles arbitrarily many ticks, but inference time grows linearly with tick volume. If speed matters, capping at K=50 most-recent ticks per asset is a straightforward extension.

---

## Model Architecture Summary

```
Input: N assets, each with T_j ticks in [t-W, t), plus static (N, N, 2) edge features

Layer 0 query init:
  h_i^(0) = W_init · mean(event_feats_i)    [mean of i's own ticks]
           = empty_embedding                  [if no ticks for asset i]

For each TGAT layer l:
  [Two-level attention as above → (N, n_heads * out_dim)]
  h^(l+1) = LayerNorm( h^(l) + W_res · layer_output )

Prediction head:
  ŷ_i = MLP( h_i^(L) )    [scalar next-bar log return]

Loss: MSE( ŷ, y )
```

Default hyperparameters: `node_embed_dim=64`, `time_enc_dim=64`, `n_heads=4`, `n_layers=2`, `mlp_hidden_dim=64`, `dropout=0.1`.

---

## Design Choices and Open Issues

| # | Choice | Rationale | Known limitation |
|---|--------|-----------|-----------------|
| 1 | Two-level attention | Removes tick-volume bias across assets | Intra-asset softmax still biased toward recent ticks (earlier ticks compete equally despite being older) |
| 2 | Spectral edge seeding | Gives model structural priors from spectral analysis | Static — does not update as lead-lag relationships shift over time |
| 3 | Fixed time window | Simple; no K to tune | Linear cost in tick volume; very large windows may be slow |
| 4 | TGAT (no memory) | Simpler; sufficient for sub-minute lags | Cannot capture regime shifts across hours/days |
| 5 | Cross-asset attention | Directly models inter-asset influence | Original TGAT attends only within a node's own history |
| 6 | 1-min bar query cadence | Aligns with OHLCV label frequency | Many ticks between queries are "wasted" as context, not as separate training signals |
| 7 | MSE regression target | Clean differentiable objective | Return prediction is very low SNR; Spearman-vs-spectral is more meaningful for research |
| 8 | batch_size=1 | Avoids ragged-tensor batching complexity | Slower training; batching with padding is the natural next step |
| 9 | Static graph topology | All assets always connected | Missing assets (0 ticks in window) participate via zero context vector |

---

## Training and Evaluation Guide

### Data requirements

One day of tick data yields ~1,440 query times → ~864 training samples after 60/20/20 split. This is too small for meaningful generalisation. Aim for at least **2 weeks** of tick data (>10,000 training samples). Fetch with the dashboard's Tick Data mode.

### Sanity checks before real training

1. Run `pytest tests/test_tgat.py::test_tgat_overfit_tiny` — confirms gradients flow.
2. Set `max_epochs=10`, `patience=999` on 50 training samples. Train loss should decrease monotonically. If it doesn't, the model or data pipeline has a bug.
3. Check that `spearman_vs_spectral` is computable (not NaN). NaN usually means asset name mismatches between spectral results and tick data keys.

### Metrics in order of importance

**`spearman_vs_spectral`** (primary)  
Rank correlation between TGAT's inter-asset attention weights and spectral leadership scores. This is the core research question: does the model learn the same structure the spectral engine finds? Values of 0.5–0.8 would be a strong result.

**`val_mse`** during training  
Use only for early stopping. Return prediction is inherently noisy; absolute MSE values are not meaningful on their own.

**`test_spearman`** (prediction rank correlation)  
Spearman ρ between predicted and actual next-bar returns. Expect 0.02–0.10 for financial return prediction. This is a hard task; do not use this as the primary success criterion.

### Hyperparameters that matter

- **Window `W`**: try 60 s, 300 s, 600 s. Short windows favour high-frequency lead-lag; long windows favour slower macro-structure.
- **`n_heads`**: 4 is a reasonable default. Attention heads learn complementary lag patterns.
- **`lr`**: 1e-3 with Adam is a good start. If training is unstable, drop to 5e-4.
- **`patience`**: 10–15 epochs. With 864 training samples an epoch is fast; don't stop too early.

### Comparing against spectral baseline

After training, compare `spearman_vs_spectral` on the test set. The spectral engine provides leadership scores via `LeadLagAnalyzer.get_asset_leadership_score()`. A positive ρ means TGAT's attention and spectral phases agree on the leadership ranking. A near-zero or negative ρ is also interesting — it could mean TGAT is learning something the spectral engine misses (or that there is insufficient data to train on).
