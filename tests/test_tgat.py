"""Unit tests for the TGAT implementation."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.graph_builder import EventBatch, GraphBuilder
from models.dataset import TGATDataset
from models.time_encoder import TimeEncoder
from models.tgat import TGATModel
from models.tgat_layer import TGATLayer
from models.trainer import TGATTrainer
from .util import generate_synthetic_tick_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spectral_results(asset_names):
    """Build synthetic spectral result dict with known delays and coherences."""
    N = len(asset_names)
    delay_arr = np.random.uniform(-10, 10, (N, N))
    np.fill_diagonal(delay_arr, 0.0)
    delays = pd.DataFrame(delay_arr, index=asset_names, columns=asset_names)

    coh_arr = np.abs(np.random.uniform(0.3, 0.8, (N, N)))
    np.fill_diagonal(coh_arr, 1.0)
    coherence = pd.DataFrame(coh_arr, index=asset_names, columns=asset_names)

    return {"short_term": (delays, coherence)}


def _make_event_batch(n_assets=3, n_ticks=10, window=300):
    """Make a synthetic EventBatch."""
    asset_names = [f"A{i}" for i in range(n_assets)]
    spectral = _make_spectral_results(asset_names)
    gb = GraphBuilder(asset_names, spectral, window_seconds=window)
    tick_data, _ = generate_synthetic_tick_data(n_assets=n_assets, n_ticks_per_asset=n_ticks)
    # Rename keys to match asset_names
    tick_data = {f"A{i}": v for i, v in enumerate(tick_data.values())}
    query_time = list(tick_data.values())[0].index[-1] + pd.Timedelta(seconds=1)
    return gb.build_event_batch(tick_data, query_time), gb


# ---------------------------------------------------------------------------
# TimeEncoder tests
# ---------------------------------------------------------------------------

def test_time_encoder_shape():
    enc = TimeEncoder(64)
    out = enc(torch.zeros(10))
    assert out.shape == (10, 64), f"Expected (10, 64), got {out.shape}"

    out2 = enc(torch.tensor([0.0, 100.0, 300.0]))
    assert out2.shape == (3, 64)


def test_time_encoder_zero_delta():
    enc = TimeEncoder(64)
    out = enc(torch.tensor([0.0]))  # (1, 64)
    cos_vals = out[0, :32]   # first half: cos
    sin_vals = out[0, 32:]   # second half: sin
    assert torch.allclose(cos_vals, torch.ones(32), atol=1e-6), "cos(0) should be 1"
    assert torch.allclose(sin_vals, torch.zeros(32), atol=1e-6), "sin(0) should be 0"


def test_time_encoder_output_range():
    enc = TimeEncoder(32)
    deltas = torch.rand(100) * 1000
    out = enc(deltas)
    assert out.shape == (100, 32)
    assert out.abs().max().item() <= 1.0 + 1e-5, "cos/sin values must be in [-1, 1]"


# ---------------------------------------------------------------------------
# GraphBuilder tests
# ---------------------------------------------------------------------------

def test_graph_builder_edge_features():
    asset_names = ["BTC", "ETH", "SOL"]
    spectral = _make_spectral_results(asset_names)
    gb = GraphBuilder(asset_names, spectral, window_seconds=300)

    ef = gb.edge_feats
    assert ef.shape == (3, 3, 2), f"Expected (3, 3, 2), got {ef.shape}"

    # Diagonal: delay=0, coherence=1
    for i in range(3):
        assert ef[i, i, 0].item() == pytest.approx(0.0), "diagonal delay should be 0"
        assert ef[i, i, 1].item() == pytest.approx(1.0), "diagonal coherence should be 1"

    # Off-diagonal coherence in [0, 1]
    for i in range(3):
        for j in range(3):
            if i != j:
                assert 0.0 <= ef[i, j, 1].item() <= 1.0


def test_graph_builder_event_batch():
    batch, gb = _make_event_batch(n_assets=3, n_ticks=50)

    assert len(batch.event_feats) == 3
    assert len(batch.event_times) == 3

    for j in range(3):
        feats = batch.event_feats[j]
        times = batch.event_times[j]
        if feats.shape[0] > 0:
            assert feats.shape[1] == 2, "tick feats should have 2 columns"
            assert (times >= 0).all(), "Δt must be non-negative"
            assert (times <= gb.window_seconds).all(), "Δt must not exceed window"


def test_graph_builder_missing_asset():
    """GraphBuilder should return empty tensors for assets with no tick data."""
    asset_names = ["A", "B", "C"]
    spectral = _make_spectral_results(asset_names)
    gb = GraphBuilder(asset_names, spectral, window_seconds=300)

    # Only provide tick data for asset A
    tick_data_a, _ = generate_synthetic_tick_data(n_assets=1)
    tick_data = {"A": list(tick_data_a.values())[0]}
    query_time = list(tick_data_a.values())[0].index[-1] + pd.Timedelta(seconds=1)
    batch = gb.build_event_batch(tick_data, query_time)

    assert batch.event_feats[1].shape[0] == 0, "Missing asset B should have 0 ticks"
    assert batch.event_feats[2].shape[0] == 0, "Missing asset C should have 0 ticks"


# ---------------------------------------------------------------------------
# TGATLayer tests
# ---------------------------------------------------------------------------

def test_tgat_layer_output_shape():
    layer = TGATLayer(node_feat_dim=16, time_enc_dim=16, edge_feat_dim=2, out_dim=8, n_heads=4)
    N = 3
    query = torch.randn(N, 16)
    event_feats = [torch.randn(10, 2) for _ in range(N)]
    event_times = [torch.rand(10) * 300 for _ in range(N)]
    edge_feats = torch.rand(N, N, 2)

    out = layer(query, event_feats, event_times, edge_feats)
    assert out.shape == (N, 4 * 8), f"Expected ({N}, 32), got {out.shape}"


def test_tgat_layer_empty_window():
    """Layer should not crash when some or all assets have no ticks."""
    layer = TGATLayer(node_feat_dim=16, time_enc_dim=16, edge_feat_dim=2, out_dim=8, n_heads=2)
    N = 3
    query = torch.randn(N, 16)
    # All empty
    event_feats = [torch.zeros(0, 2) for _ in range(N)]
    event_times = [torch.zeros(0) for _ in range(N)]
    edge_feats = torch.rand(N, N, 2)

    out = layer(query, event_feats, event_times, edge_feats)
    assert out.shape == (N, 2 * 8)
    assert torch.isfinite(out).all(), "Output should be finite even with empty window"


def test_tgat_layer_return_attn_weights():
    layer = TGATLayer(node_feat_dim=16, time_enc_dim=16, edge_feat_dim=2, out_dim=8, n_heads=2)
    N = 3
    query = torch.randn(N, 16)
    event_feats = [torch.randn(5, 2) for _ in range(N)]
    event_times = [torch.rand(5) * 100 for _ in range(N)]
    edge_feats = torch.rand(N, N, 2)

    out, attn = layer(query, event_feats, event_times, edge_feats, return_attn_weights=True)
    assert out.shape == (N, 2 * 8)
    assert attn.shape == (N, N), f"Expected ({N}, {N}), got {attn.shape}"


# ---------------------------------------------------------------------------
# TGATModel tests
# ---------------------------------------------------------------------------

def test_tgat_model_forward():
    N = 5
    model = TGATModel(n_assets=N, node_embed_dim=32, time_enc_dim=32, n_heads=2, n_layers=2)
    event_feats = [torch.randn(15, 2) for _ in range(N)]
    event_times = [torch.rand(15) * 300 for _ in range(N)]
    edge_feats = torch.rand(N, N, 2)

    preds = model(event_feats, event_times, edge_feats)
    assert preds.shape == (N,), f"Expected ({N},), got {preds.shape}"
    assert torch.isfinite(preds).all()


def test_tgat_model_return_attn_weights():
    N = 3
    model = TGATModel(n_assets=N, node_embed_dim=16, time_enc_dim=16, n_heads=2, n_layers=2)
    event_feats = [torch.randn(8, 2) for _ in range(N)]
    event_times = [torch.rand(8) * 100 for _ in range(N)]
    edge_feats = torch.rand(N, N, 2)

    preds, attn = model(event_feats, event_times, edge_feats, return_attn_weights=True)
    assert preds.shape == (N,)
    assert attn.shape == (N, N)


# ---------------------------------------------------------------------------
# TGATDataset tests
# ---------------------------------------------------------------------------

def test_dataset_split_chronological():
    n_assets = 3
    tick_data, ohlcv_data = generate_synthetic_tick_data(n_assets=n_assets, n_ticks_per_asset=500)
    asset_names = list(tick_data.keys())
    spectral = _make_spectral_results(asset_names)
    gb = GraphBuilder(asset_names, spectral, window_seconds=60)

    train_ds = TGATDataset(tick_data, ohlcv_data, gb, window_seconds=60, split="train")
    val_ds = TGATDataset(tick_data, ohlcv_data, gb, window_seconds=60, split="val")
    test_ds = TGATDataset(tick_data, ohlcv_data, gb, window_seconds=60, split="test")

    # Must have at least some samples in each split (may be 0 if data is too sparse)
    # The key invariant: train times < val times < test times
    if len(train_ds) > 0 and len(val_ds) > 0:
        assert train_ds.query_times[-1] <= val_ds.query_times[0], \
            "Train times must precede val times"
    if len(val_ds) > 0 and len(test_ds) > 0:
        assert val_ds.query_times[-1] <= test_ds.query_times[0], \
            "Val times must precede test times"


def test_dataset_getitem():
    n_assets = 2
    tick_data, ohlcv_data = generate_synthetic_tick_data(n_assets=n_assets, n_ticks_per_asset=400)
    asset_names = list(tick_data.keys())
    spectral = _make_spectral_results(asset_names)
    gb = GraphBuilder(asset_names, spectral, window_seconds=60)

    ds = TGATDataset(tick_data, ohlcv_data, gb, window_seconds=60, split="train")
    if len(ds) == 0:
        pytest.skip("No valid query times in synthetic data — too sparse")

    batch, labels = ds[0]
    assert isinstance(batch, EventBatch)
    assert labels.shape == (n_assets,)
    assert torch.isfinite(labels).all()


# ---------------------------------------------------------------------------
# Overfit test (validates backward pass)
# ---------------------------------------------------------------------------

def test_tgat_overfit_tiny():
    """Train on 5 samples and verify the loss decreases."""
    n_assets = 2
    tick_data, ohlcv_data = generate_synthetic_tick_data(
        n_assets=n_assets, n_ticks_per_asset=500, seed=0
    )
    asset_names = list(tick_data.keys())
    spectral = _make_spectral_results(asset_names)
    gb = GraphBuilder(asset_names, spectral, window_seconds=60)
    ds = TGATDataset(tick_data, ohlcv_data, gb, window_seconds=60, split="train")

    if len(ds) < 3:
        pytest.skip("Not enough query times for overfit test")

    # Tiny subset
    ds.query_times = ds.query_times[:5]

    model = TGATModel(n_assets=n_assets, node_embed_dim=16, time_enc_dim=16, n_heads=2, n_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.MSELoss()

    losses = []
    for epoch in range(20):
        for i in range(len(ds)):
            batch, labels = ds[i]
            optimizer.zero_grad()
            preds = model(batch.event_feats, batch.event_times, batch.edge_feats)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
        # Record loss on last sample
        with torch.no_grad():
            batch, labels = ds[0]
            preds = model(batch.event_feats, batch.event_times, batch.edge_feats)
            losses.append(criterion(preds, labels).item())

    assert losses[-1] < losses[0], (
        f"Loss should decrease during overfitting: {losses[0]:.6f} → {losses[-1]:.6f}"
    )
