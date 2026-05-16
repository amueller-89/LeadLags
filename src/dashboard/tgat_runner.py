"""TGAT Training UI component for the Streamlit dashboard."""

import json
import sys
import time
import threading
import traceback as tb_module
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend_utils import list_cached_data, load_cached_data, save_tgat_run, list_tgat_runs


def _fmt_secs(s: float) -> str:
    s = int(s)
    return f"{s // 60}m {s % 60:02d}s"


def render():
    """Render the TGAT Training tab."""
    st.header("TGAT Model Training")

    # Training monitor takes over the whole tab while active
    if st.session_state.get("tgat_training_active"):
        _render_training_monitor()
        return

    # Training setup section (gated by prerequisites; doesn't block the rest)
    _render_training_section()

    # Results — visible even if prerequisites are no longer met
    tgat_results = st.session_state.get("tgat_results")
    if tgat_results:
        st.markdown("---")
        st.subheader("Results")
        _render_results(tgat_results)

    # Inference — visible when a model is loaded
    if st.session_state.get("tgat_inference_ctx"):
        st.markdown("---")
        st.subheader("Run Inference")
        _render_inference_ui()

    # Previous runs browser — always shown when runs exist
    tgat_runs = list_tgat_runs()
    if tgat_runs:
        st.markdown("---")
        st.subheader("Previous Runs")
        _render_previous_runs(tgat_runs)


# ── Training section ──────────────────────────────────────────────────────────

def _render_training_section():
    """Render training prerequisites, configuration and Train button."""
    st.subheader("Train")
    st.markdown(
        "Train a Temporal Graph Attention Network on raw tick data to predict next "
        "1-minute bar returns, and compare learned attention weights to spectral lead-lag rankings."
    )

    analysis_results = st.session_state.get("analysis_results")
    loaded_data = st.session_state.get("loaded_data")
    # Only show tick files for symbols that are in the currently loaded dataset
    # (loaded_data keys are "BTC/USDT" format, matching list_cached_data() symbol field)
    analyzed_symbols = set(loaded_data.keys()) if loaded_data else set()
    tick_files = [
        f for f in list_cached_data()
        if f.get("timeframe") == "tick"
        and (not analyzed_symbols or f.get("symbol") in analyzed_symbols)
    ]

    col_spec, col_ohlcv, col_tick = st.columns(3)
    with col_spec:
        if analysis_results:
            st.success("✅ Spectral analysis ready")
        else:
            st.error("❌ Run spectral analysis first")
    with col_ohlcv:
        if loaded_data:
            st.success(f"✅ OHLCV data loaded ({len(loaded_data)} symbols)")
        else:
            st.error("❌ Load OHLCV data first")
    with col_tick:
        if tick_files:
            st.success(f"✅ {len(tick_files)} tick file(s) available")
        else:
            st.error("❌ No raw tick data found")

    if not (analysis_results and loaded_data and tick_files):
        st.info(
            "**To train TGAT:**\n"
            "1. Fetch tick data (Data Management → Historical Tick (Binance Vision))\n"
            "2. Load OHLCV data and run spectral analysis (Analysis tab)\n\n"
            "Previous runs can be loaded in the section below."
        )
        return

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        window_seconds = st.slider(
            "Tick Lookback Window (seconds)",
            min_value=60, max_value=600, value=300, step=30,
            help="How far back from each bar boundary to look for tick events"
        )
        max_epochs = st.number_input(
            "Max Epochs", min_value=5, max_value=500, value=50, step=5
        )
        patience = st.number_input(
            "Early Stopping Patience", min_value=3, max_value=50, value=10
        )

    with col2:
        lr = st.select_slider(
            "Learning Rate",
            options=[1e-4, 5e-4, 1e-3, 5e-3],
            value=1e-3,
            format_func=lambda x: f"{x:.0e}"
        )
        band_names = list(analysis_results.keys())
        band_name = st.selectbox(
            "Spectral Band for Edge Seeding",
            options=band_names,
            index=0,
            help="Which spectral band's delays/coherences to use as initial edge weights"
        )
        tick_symbols = sorted({f["symbol"] for f in tick_files})
        st.write(f"**Tick data available for:** {', '.join(tick_symbols)}")

    use_spectral_seeding = st.checkbox(
        "Use spectral edge seeding",
        value=False,
        help=(
            "When OFF (recommended): edge features are all zeros — the model learns lead-lag "
            "purely from tick dynamics, so 'vs Spectral ρ' is a genuine independent check. "
            "When ON: spectral delay/coherence are fed as inputs, which confounds the metric "
            "but lets you measure whether the prior improves return prediction (compare test MSE)."
        ),
    )
    if use_spectral_seeding:
        st.warning(
            "Seeding ON — spectral coherence is an input feature. "
            "'vs Spectral ρ' is confounded: high values may just mean the model learned to "
            "copy the prior. Use test MSE to judge whether seeding helps."
        )
    else:
        st.info(
            "Seeding OFF (clean mode) — TGAT receives no spectral information. "
            "'vs Spectral ρ' is a genuine independent validation of whether tick dynamics "
            "agree with spectral phase analysis."
        )

    if st.button("Train TGAT", type="primary", use_container_width=True):
        _run_training(
            tick_files=tick_files,
            loaded_data=loaded_data,
            analysis_results=analysis_results,
            band_name=band_name,
            window_seconds=window_seconds,
            max_epochs=max_epochs,
            patience=patience,
            lr=lr,
            use_spectral_seeding=use_spectral_seeding,
        )


# ── Live training monitor ─────────────────────────────────────────────────────

def _render_training_monitor():
    """Render live training progress. Polls progress dict every 0.5 s via st.rerun()."""
    progress = st.session_state.tgat_training_progress
    stop_event: threading.Event = st.session_state.tgat_stop_event
    start_time: float = st.session_state.tgat_train_start_time

    if progress["done"]:
        _finalize_training(progress)
        return

    col_msg, col_btn = st.columns([4, 1])
    with col_msg:
        st.warning("Training in progress — other controls are disabled.")
    with col_btn:
        if st.button("Interrupt", type="secondary", use_container_width=True):
            stop_event.set()
            st.rerun()

    epoch = progress["epoch"]
    max_epochs = progress["max_epochs"]
    batch = progress["batch"]
    n_batches = progress["n_batches"]

    total_steps = max_epochs * n_batches
    done_steps = epoch * n_batches + batch
    frac = done_steps / total_steps if total_steps > 0 else 0.0
    st.progress(min(frac, 1.0))

    elapsed = time.time() - start_time
    eta_str = _fmt_secs(elapsed / frac * (1 - frac)) if frac > 0.01 else "estimating..."
    st.text(
        f"Epoch {epoch}/{max_epochs} | Batch {batch}/{n_batches} | "
        f"Elapsed: {_fmt_secs(elapsed)} | ETA: {eta_str}"
    )

    lh = progress["loss_history"]
    if lh["epoch"]:
        st.line_chart(
            pd.DataFrame({"train": lh["train_mse"], "val": lh["val_mse"]}, index=lh["epoch"])
        )

    time.sleep(0.5)
    st.rerun()


def _finalize_training(progress: dict):
    """Called once training thread finishes. Runs post-training steps in the main thread."""
    st.session_state.tgat_training_active = False

    if progress["interrupted"]:
        st.warning(f"Training interrupted after {progress['epoch']} epoch(s).")
        lh = progress["loss_history"]
        if lh["epoch"]:
            st.line_chart(
                pd.DataFrame({"train": lh["train_mse"], "val": lh["val_mse"]}, index=lh["epoch"])
            )
        _cleanup_training_refs()
        return

    if progress["error"]:
        st.error(f"Training failed: {progress['error']}")
        st.code(progress["tb"])
        _cleanup_training_refs()
        return

    trainer = st.session_state.tgat_trainer_ref
    test_ds = st.session_state.tgat_test_ds_ref
    analysis_results = st.session_state.tgat_analysis_results_ref
    asset_names = st.session_state.tgat_asset_names_ref
    spectral_seeding = st.session_state.tgat_spectral_seeding_ref
    gb = st.session_state.get("tgat_gb_ref")
    run_config_ref = st.session_state.get("tgat_run_config_ref", {})
    history = progress["history"]

    _, _, _, spectral_scores = list(analysis_results.values())[0]
    try:
        rho_vs_spectral = trainer.compute_spearman_vs_spectral(test_ds, spectral_scores)
    except Exception:
        rho_vs_spectral = float("nan")

    try:
        attn_matrix = _extract_mean_attention(trainer, test_ds)
    except Exception:
        attn_matrix = None

    lh = progress["loss_history"]
    test_metrics = history.get("test_metrics", {})

    tgat_results = {
        "test_mse": test_metrics.get("mse", float("nan")),
        "test_spearman": test_metrics.get("spearman_corr", float("nan")),
        "spearman_vs_spectral": rho_vs_spectral,
        "asset_names": asset_names,
        "attn_matrix": attn_matrix,
        "loss_history": {"epoch": lh["epoch"], "train_mse": lh["train_mse"], "val_mse": lh["val_mse"]},
        "spectral_seeding": spectral_seeding,
    }
    st.session_state.tgat_results = tgat_results
    st.session_state.tgat_trainer = trainer

    # Auto-save to disk
    edge_feats_np = gb.edge_feats.detach().cpu().numpy() if gb is not None else None
    save_config = {
        "asset_names": asset_names,
        "n_assets": len(asset_names),
        "window_seconds": run_config_ref.get("window_seconds", 300),
        "band_name": run_config_ref.get("band_name", ""),
        "spectral_seeding": spectral_seeding,
        "lr": run_config_ref.get("lr", 1e-3),
        "max_epochs": run_config_ref.get("max_epochs", 50),
        "patience": run_config_ref.get("patience", 10),
        "n_heads": 4,
        "n_layers": 2,
        "node_embed_dim": 64,
        "tick_files": run_config_ref.get("tick_files", []),
        "spectral_run": run_config_ref.get("spectral_run", ""),
    }
    save_results = {
        "test_mse": tgat_results["test_mse"],
        "test_spearman": tgat_results["test_spearman"],
        "spearman_vs_spectral": tgat_results["spearman_vs_spectral"],
        "loss_history": tgat_results["loss_history"],
        "attn_matrix": attn_matrix,
        "edge_feats": edge_feats_np,
    }
    run_dir = None
    try:
        run_dir = save_tgat_run(trainer.model, save_config, save_results)
        st.toast(f"Run saved: {run_dir.name}", icon="💾")
    except Exception as e:
        st.warning(f"Could not save run to disk: {e}")

    # Set inference context
    st.session_state.tgat_inference_ctx = {
        "model": trainer.model,
        "edge_feats": gb.edge_feats if gb is not None else None,
        "asset_names": asset_names,
        "window_seconds": run_config_ref.get("window_seconds", 300),
        "run_dir": str(run_dir) if run_dir else None,
        "tick_files": run_config_ref.get("tick_files", []),
        "config": save_config,
        "model_loaded": True,
    }

    _cleanup_training_refs()
    st.toast("Training complete!", icon="✅")
    st.rerun()


def _cleanup_training_refs():
    for key in [
        "tgat_training_progress", "tgat_stop_event", "tgat_train_start_time",
        "tgat_trainer_ref", "tgat_test_ds_ref",
        "tgat_analysis_results_ref", "tgat_asset_names_ref", "tgat_spectral_seeding_ref",
        "tgat_gb_ref", "tgat_run_config_ref",
    ]:
        st.session_state.pop(key, None)


# ── Background training thread launcher ──────────────────────────────────────

def _run_training(
    tick_files, loaded_data, analysis_results, band_name,
    window_seconds, max_epochs, patience, lr, use_spectral_seeding=False
):
    try:
        from models import TGATModel, GraphBuilder, TGATDataset, TGATTrainer
        from models.trainer import TrainingInterrupted
        from preprocessor import CryptoPreprocessor, PreprocessConfig
    except ImportError as e:
        st.error(f"Could not import TGAT modules: {e}\n\nEnsure torch and torch-geometric are installed.")
        return

    status = st.empty()

    status.info("Loading tick data...")
    try:
        tick_raw = load_cached_data(tick_files)
    except Exception as e:
        st.error(f"Failed to load tick data: {e}")
        return

    spectral_for_graph = {
        bname: (delays, coh)
        for bname, (delays, coh, _, _) in analysis_results.items()
    }

    status.info("Preprocessing OHLCV data for labels...")
    try:
        cfg = PreprocessConfig(resample_freq="1min", remove_outliers=True, outlier_std=10.0)
        preprocessor = CryptoPreprocessor(cfg)
        processed_ohlcv, _ = preprocessor.process_pipeline(
            loaded_data, align=True, resample=True, calc_returns=False, remove_outliers=False
        )
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        return

    first_delays_df = list(spectral_for_graph.values())[0][0]
    asset_names = list(first_delays_df.index)
    if not asset_names:
        st.error("No assets found in spectral results.")
        return

    status.info("Building temporal graph...")
    try:
        gb = GraphBuilder(
            asset_names=asset_names,
            spectral_results=spectral_for_graph,
            window_seconds=window_seconds,
            band_name=band_name,
            spectral_seeding=use_spectral_seeding,
        )
    except Exception as e:
        st.error(f"Graph builder failed: {e}")
        return

    status.info("Building datasets...")
    try:
        train_ds = TGATDataset(tick_raw, processed_ohlcv, gb, window_seconds=window_seconds, split="train")
        val_ds   = TGATDataset(tick_raw, processed_ohlcv, gb, window_seconds=window_seconds, split="val")
        test_ds  = TGATDataset(tick_raw, processed_ohlcv, gb, window_seconds=window_seconds, split="test")
    except Exception as e:
        st.error(f"Dataset construction failed: {e}")
        return

    if len(train_ds) == 0:
        st.error(
            "No training samples found. The tick data and OHLCV data may not overlap in time. "
            "Ensure you have both raw tick data and 1-min OHLCV data for the same time period."
        )
        return

    st.write(
        f"Dataset: **{len(train_ds)}** train / **{len(val_ds)}** val / **{len(test_ds)}** test samples"
    )

    model = TGATModel(n_assets=len(asset_names))
    trainer = TGATTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        lr=lr,
        patience=patience,
        checkpoint_dir="checkpoints",
    )
    status.empty()

    progress = {
        "epoch": 0, "max_epochs": int(max_epochs),
        "batch": 0, "n_batches": max(1, len(train_ds)),
        "train_mse": None, "val_mse": None,
        "loss_history": {"epoch": [], "train_mse": [], "val_mse": []},
        "done": False, "interrupted": False, "error": None, "tb": None,
        "history": None,
    }
    stop_event = threading.Event()

    def _batch_callback(epoch, total_epochs, batch, n_batches):
        progress["epoch"] = epoch
        progress["batch"] = batch
        progress["n_batches"] = n_batches

    def _epoch_callback(epoch, total, train_mse, val_mse):
        progress["epoch"] = epoch
        progress["train_mse"] = train_mse
        progress["val_mse"] = val_mse
        lh = progress["loss_history"]
        lh["epoch"].append(epoch)
        lh["train_mse"].append(train_mse)
        lh["val_mse"].append(val_mse)

    def _train_fn():
        try:
            history = trainer.train(
                max_epochs=int(max_epochs),
                progress_callback=_epoch_callback,
                batch_callback=_batch_callback,
                stop_event=stop_event,
            )
            progress["history"] = history
        except TrainingInterrupted:
            progress["interrupted"] = True
        except Exception as exc:
            progress["error"] = str(exc)
            progress["tb"] = tb_module.format_exc()
        finally:
            progress["done"] = True

    st.session_state.tgat_training_active = True
    st.session_state.tgat_training_progress = progress
    st.session_state.tgat_stop_event = stop_event
    st.session_state.tgat_train_start_time = time.time()
    st.session_state.tgat_trainer_ref = trainer
    st.session_state.tgat_test_ds_ref = test_ds
    st.session_state.tgat_analysis_results_ref = analysis_results
    st.session_state.tgat_asset_names_ref = asset_names
    st.session_state.tgat_spectral_seeding_ref = use_spectral_seeding
    st.session_state.tgat_gb_ref = gb
    st.session_state.tgat_run_config_ref = {
        "window_seconds": window_seconds,
        "band_name": band_name,
        "lr": lr,
        "max_epochs": int(max_epochs),
        "patience": patience,
        "tick_files": [f["filepath"] for f in tick_files],
        "spectral_run": str(st.session_state.get("current_analysis_run", "")),
    }

    threading.Thread(target=_train_fn, daemon=True).start()
    st.rerun()


# ── Attention extraction helper ───────────────────────────────────────────────

def _extract_mean_attention(trainer, dataset) -> np.ndarray:
    """Average (N, N) attention matrix over all samples in dataset."""
    import torch
    from torch.utils.data import DataLoader
    from models.trainer import _collate_fn

    model = trainer.model
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_collate_fn)
    N = model.n_assets
    attn_sum = np.zeros((N, N))
    count = 0

    with torch.no_grad():
        for batch in loader:
            event_batch, _ = batch
            ef = [f.to(trainer.device) for f in event_batch.event_feats]
            et = [t.to(trainer.device) for t in event_batch.event_times]
            edge = event_batch.edge_feats.to(trainer.device)
            _, attn = model(ef, et, edge, return_attn_weights=True)
            attn_sum += attn.cpu().numpy()
            count += 1

    return attn_sum / max(count, 1)


# ── Results rendering ─────────────────────────────────────────────────────────

def _render_results(results: dict):
    col1, col2, col3 = st.columns(3)
    col1.metric("Test MSE", f"{results['test_mse']:.6f}")
    col2.metric("Prediction Spearman ρ", f"{results['test_spearman']:.4f}")
    col3.metric("vs Spectral Spearman ρ", f"{results['spearman_vs_spectral']:.4f}")

    seeding = results.get("spectral_seeding", True)
    if seeding:
        st.warning(
            "**Seeding was ON.** 'vs Spectral ρ' is confounded — the model had access to spectral "
            "coherence as an input. A high value may just reflect the prior being copied. "
            "Re-run with seeding OFF to get a genuine independent validation; compare test MSE "
            "between runs to assess whether seeding helped."
        )
    else:
        st.info(
            "**Seeding was OFF.** 'vs Spectral ρ' is a genuine independent check — TGAT had no "
            "access to spectral information. A positive value means tick dynamics and spectral "
            "phase analysis agree on the leadership ranking."
        )

    lh = results.get("loss_history", {})
    if lh and lh.get("epoch"):
        st.markdown("**Training history**")
        st.line_chart(
            pd.DataFrame({"train": lh["train_mse"], "val": lh["val_mse"]}, index=lh["epoch"])
        )

    attn = results.get("attn_matrix")
    asset_names = results.get("asset_names", [])
    if attn is not None and len(asset_names) > 0:
        st.markdown("**Attention weight heatmap (test set average)**")
        st.caption(
            "Row = query asset (predicting), column = source asset (attended to). "
            "Off-diagonal highs suggest the source asset's recent trades help predict the query asset's returns."
        )
        _render_attention_heatmap(attn, asset_names, "TGAT Mean Attention Weights (test set)")


def _render_attention_heatmap(attn: np.ndarray, asset_names: list, title: str):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(max(4, len(asset_names)), max(3, len(asset_names))))
        sns.heatmap(
            attn,
            xticklabels=asset_names,
            yticklabels=asset_names,
            annot=True,
            fmt=".3f",
            cmap="Blues",
            ax=ax,
            cbar_kws={"label": "Mean attention weight"},
        )
        ax.set_xlabel("Source asset (attended to)")
        ax.set_ylabel("Query asset (attending)")
        ax.set_title(title)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.warning(f"Could not render attention heatmap: {e}")


# ── Previous runs browser ─────────────────────────────────────────────────────

def _load_tgat_run(run_dir: str):
    """Load a saved TGAT run into session state (results + inference context)."""
    import torch
    from models import TGATModel

    run_path = Path(run_dir)

    with open(run_path / "config.json") as f:
        config = json.load(f)

    results = {}
    results_path = run_path / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

    # Reconstruct model
    model = TGATModel(n_assets=config["n_assets"])
    state_path = run_path / "model.pt"
    model_loaded = state_path.exists()
    if model_loaded:
        state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    model.eval()

    attn_path = run_path / "attention.npy"
    attn_matrix = np.load(attn_path) if attn_path.exists() else None

    ef_path = run_path / "edge_feats.npy"
    edge_feats_np = np.load(ef_path) if ef_path.exists() else None
    edge_feats = torch.tensor(edge_feats_np, dtype=torch.float32) if edge_feats_np is not None else None

    st.session_state.tgat_results = {
        "test_mse": results.get("test_mse", float("nan")),
        "test_spearman": results.get("test_spearman", float("nan")),
        "spearman_vs_spectral": results.get("spearman_vs_spectral", float("nan")),
        "asset_names": config.get("asset_names", []),
        "attn_matrix": attn_matrix,
        "loss_history": results.get("loss_history", {"epoch": [], "train_mse": [], "val_mse": []}),
        "spectral_seeding": config.get("spectral_seeding", True),
    }
    st.session_state.tgat_inference_ctx = {
        "model": model,
        "edge_feats": edge_feats,
        "asset_names": config.get("asset_names", []),
        "window_seconds": config.get("window_seconds", 300),
        "run_dir": run_dir,
        "tick_files": config.get("tick_files", []),
        "config": config,
        "model_loaded": model_loaded,
    }


def _render_previous_runs(runs: list):
    rows = []
    for run in runs:
        cfg = run.get("config", {})
        res = run.get("results", {})
        assets = cfg.get("asset_names", [])
        assets_str = ", ".join(assets[:3]) + ("…" if len(assets) > 3 else "")
        mse_val = res.get("test_mse")
        try:
            mse_str = f"{float(mse_val):.6f}" if mse_val is not None else "—"
        except Exception:
            mse_str = "—"
        rows.append({
            "Timestamp": run["timestamp"],
            "Assets": assets_str,
            "Window (s)": cfg.get("window_seconds", "?"),
            "Band": cfg.get("band_name", "?"),
            "Seeding": "ON" if cfg.get("spectral_seeding", True) else "OFF",
            "Test MSE": mse_str,
        })

    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    run_labels = [r["timestamp"] for r in runs]
    selected_label = st.selectbox("Select run to load", run_labels, key="prev_run_select")
    selected_run = next((r for r in runs if r["timestamp"] == selected_label), None)

    if selected_run and st.button("Load selected run", key="load_prev_run_btn"):
        try:
            _load_tgat_run(selected_run["run_dir"])
            st.rerun()
        except Exception as e:
            st.error(f"Failed to load run: {e}")


# ── Inference ─────────────────────────────────────────────────────────────────

def _load_tick_files_for_inference(tick_file_paths: list) -> dict:
    """Load raw tick parquet files by absolute path → {symbol: DataFrame}."""
    tick_data = {}
    for p_str in tick_file_paths:
        p = Path(p_str)
        if not p.exists():
            continue
        df = pd.read_parquet(p_str)
        # BTC_USDT_tick_... → BTC/USDT
        parts = p.stem.split("_")
        if "tick" in parts:
            tick_idx = parts.index("tick")
            symbol = "_".join(parts[:tick_idx]).replace("_", "/")
        else:
            symbol = p.stem
        if symbol in tick_data:
            combined = pd.concat([tick_data[symbol], df])
            tick_data[symbol] = combined[~combined.index.duplicated(keep="first")].sort_index()
        else:
            tick_data[symbol] = df
    return tick_data


def _run_inference(
    model,
    edge_feats,
    tick_data: dict,
    asset_names: list,
    window_seconds: int,
    query_time: pd.Timestamp,
):
    """Single forward pass. Returns (predicted_returns ndarray, attn_matrix ndarray)."""
    import torch
    from models.graph_builder import GraphBuilder

    if edge_feats is None:
        N = len(asset_names)
        edge_feats_t = torch.zeros(N, N, 2, dtype=torch.float32)
    elif isinstance(edge_feats, torch.Tensor):
        edge_feats_t = edge_feats
    else:
        edge_feats_t = torch.tensor(np.array(edge_feats), dtype=torch.float32)

    gb = GraphBuilder(
        asset_names=asset_names,
        window_seconds=window_seconds,
        edge_feats_override=edge_feats_t,
    )
    batch = gb.build_event_batch(tick_data, query_time)

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        ef = [f.to(device) for f in batch.event_feats]
        et = [t.to(device) for t in batch.event_times]
        edge = batch.edge_feats.to(device)
        preds, attn = model(ef, et, edge, return_attn_weights=True)

    return preds.cpu().numpy().flatten(), attn.cpu().numpy()


def _render_inference_ui():
    """Render query-time picker and inference results."""
    ctx = st.session_state.tgat_inference_ctx
    asset_names = ctx["asset_names"]
    window_seconds = ctx["window_seconds"]
    tick_file_paths = ctx.get("tick_files", [])

    if not ctx.get("model_loaded", True):
        st.warning("Model weights not found for this run — weights are randomly initialized.")

    st.markdown(
        "Run the trained model at a chosen bar boundary to get predicted next-minute returns "
        "and attention weights for that specific moment."
    )

    if not tick_file_paths:
        st.warning("No tick file paths saved in this run's config — cannot run inference.")
        return

    existing_paths = [p for p in tick_file_paths if Path(p).exists()]
    if not existing_paths:
        st.warning(
            "Tick files from this training run no longer exist at their original paths:\n"
            + "\n".join(f"- `{p}`" for p in tick_file_paths)
        )
        return

    # Cache tick data in session state to avoid reloading on every widget interaction
    cache_key = f"tgat_infer_tick_{ctx.get('run_dir', 'live')}"
    if cache_key not in st.session_state:
        with st.spinner("Loading tick data for inference..."):
            st.session_state[cache_key] = _load_tick_files_for_inference(existing_paths)
    tick_data = st.session_state[cache_key]

    if not tick_data:
        st.warning("Could not load any tick data.")
        return

    all_max = [df.index.max() for df in tick_data.values() if not df.empty]
    all_min = [df.index.min() for df in tick_data.values() if not df.empty]
    if not all_max:
        st.warning("Tick data is empty.")
        return

    latest_time = max(all_max).floor("1min")
    earliest_time = min(all_min).ceil("1min") + pd.Timedelta(seconds=window_seconds)

    col1, col2 = st.columns([3, 1])
    with col1:
        query_input = st.text_input(
            "Query time (UTC, minute boundary)",
            value=latest_time.strftime("%Y-%m-%d %H:%M:%S"),
            key="tgat_infer_query_time",
            help=(
                f"Available range: {earliest_time.strftime('%Y-%m-%d %H:%M')} "
                f"to {latest_time.strftime('%Y-%m-%d %H:%M')} UTC"
            ),
        )
    with col2:
        run_btn = st.button(
            "Run Inference", type="primary", use_container_width=True, key="tgat_infer_run_btn"
        )

    if run_btn:
        try:
            query_time = pd.Timestamp(query_input, tz="UTC")
        except Exception:
            st.error(f"Invalid query time: {query_input!r}")
            return

        if query_time < earliest_time or query_time > latest_time:
            st.error(
                f"Query time is outside the available tick data range.  \n"
                f"Pick a time between **{earliest_time.strftime('%Y-%m-%d %H:%M')}** "
                f"and **{latest_time.strftime('%Y-%m-%d %H:%M')} UTC**."
            )
            return

        with st.spinner("Running inference..."):
            try:
                preds, attn = _run_inference(
                    ctx["model"], ctx["edge_feats"],
                    tick_data, asset_names, window_seconds, query_time,
                )
            except Exception as e:
                st.error(f"Inference failed: {e}")
                st.code(tb_module.format_exc())
                return

        N = len(asset_names)
        if N > 0 and np.allclose(attn, 1.0 / N, atol=0.02):
            st.warning(
                f"Attention weights are uniform (≈ 1/{N} everywhere) — "
                f"no tick events were found in the {window_seconds}s lookback window. "
                "The model has no information to differentiate assets at this query time."
            )

        st.markdown(
            f"**Query time:** `{query_time.strftime('%Y-%m-%d %H:%M:%S UTC')}`  "
            f"| window: {window_seconds} s"
        )
        st.dataframe(
            pd.DataFrame({
                "Asset": asset_names,
                "Predicted log return": [f"{p:.6f}" for p in preds],
            }),
            hide_index=True,
            use_container_width=True,
        )

        if attn is not None:
            st.markdown("**Attention weights at this query time**")
            _render_attention_heatmap(
                attn, asset_names,
                f"TGAT Attention — {query_time.strftime('%H:%M UTC')}",
            )
