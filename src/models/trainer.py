import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from .dataset import TGATDataset
from .tgat import TGATModel


class TrainingInterrupted(Exception):
    pass


def _collate_fn(batch):
    """Identity collate — returns a single (EventBatch, label) pair unchanged."""
    return batch[0]


class TGATTrainer:
    """Training loop for TGATModel with early stopping and Spearman evaluation."""

    def __init__(
        self,
        model: TGATModel,
        train_dataset: TGATDataset,
        val_dataset: TGATDataset,
        test_dataset: TGATDataset,
        lr: float = 1e-3,
        patience: int = 10,
        checkpoint_dir: str = "checkpoints",
        device: str = "auto",
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_ckpt = os.path.join(checkpoint_dir, "tgat_best.pt")

    def _forward_batch(self, batch):
        event_batch, labels = batch
        labels = labels.to(self.device)
        # Move event tensors to device
        event_feats = [f.to(self.device) for f in event_batch.event_feats]
        event_times = [t.to(self.device) for t in event_batch.event_times]
        edge_feats = event_batch.edge_feats.to(self.device)
        preds = self.model(event_feats, event_times, edge_feats)
        return preds, labels

    def train(self, max_epochs: int = 100, progress_callback=None, batch_callback=None,
              stop_event=None) -> dict:
        """Train the model with optional progress callbacks.

        Args:
            max_epochs: Maximum number of training epochs.
            progress_callback: callable(epoch, max_epochs, train_mse, val_mse) — called after
                each epoch with loss values. Useful for updating loss charts.
            batch_callback: callable(epoch, max_epochs, batch, n_batches) — called ~20 times
                per epoch during the training loop. Useful for intra-epoch progress bars.
            stop_event: threading.Event — if set, training raises TrainingInterrupted at the
                next batch boundary. Used for clean interruption from another thread.
        """
        best_val_mse = float("inf")
        epochs_no_improve = 0
        history = {"train_mse": [], "val_mse": []}

        train_loader = DataLoader(
            self.train_dataset, batch_size=1, shuffle=False, collate_fn=_collate_fn
        )
        n_batches = len(train_loader)
        # ~20 UI updates per epoch regardless of dataset size
        update_every = max(1, n_batches // 20)

        for epoch in range(max_epochs):
            # Train
            self.model.train()
            train_losses = []
            for batch_idx, batch in enumerate(train_loader):
                if stop_event is not None and stop_event.is_set():
                    raise TrainingInterrupted()
                self.optimizer.zero_grad()
                preds, labels = self._forward_batch(batch)
                loss = self.criterion(preds, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_losses.append(loss.item())

                if batch_callback is not None and (batch_idx + 1) % update_every == 0:
                    batch_callback(epoch, max_epochs, batch_idx + 1, n_batches)

            train_mse = float(np.mean(train_losses))
            val_metrics = self.evaluate(self.val_dataset)
            val_mse = val_metrics["mse"]

            history["train_mse"].append(train_mse)
            history["val_mse"].append(val_mse)

            print(f"Epoch {epoch+1:3d}/{max_epochs} | train_mse={train_mse:.6f} | val_mse={val_mse:.6f}")
            if progress_callback is not None:
                progress_callback(epoch + 1, max_epochs, train_mse, val_mse)

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                epochs_no_improve = 0
                torch.save(
                    {"model_state": self.model.state_dict(), "epoch": epoch, "val_mse": val_mse},
                    self.best_ckpt,
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Restore best checkpoint
        ckpt = torch.load(self.best_ckpt, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state"])
        print(f"Restored best checkpoint from epoch {ckpt['epoch']+1} (val_mse={ckpt['val_mse']:.6f})")

        test_metrics = self.evaluate(self.test_dataset)
        history["test_metrics"] = test_metrics
        return history

    def evaluate(self, dataset: TGATDataset) -> dict:
        self.model.eval()
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_collate_fn)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                preds, labels = self._forward_batch(batch)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        preds_arr = np.concatenate(all_preds, axis=0)    # (n_samples * N,)
        labels_arr = np.concatenate(all_labels, axis=0)

        mse = float(np.mean((preds_arr - labels_arr) ** 2))

        if len(preds_arr) > 2:
            spearman, _ = spearmanr(preds_arr, labels_arr)
        else:
            spearman = float("nan")

        return {"mse": mse, "spearman_corr": spearman}

    def compute_spearman_vs_spectral(
        self,
        dataset: TGATDataset,
        spectral_scores: pd.Series,
    ) -> float:
        """Correlate TGAT attention-based leadership with spectral leadership scores.

        For each query, extract the mean attention each asset j receives across all
        query nodes i (how much j is attended to). Average over all queries to get
        a per-asset 'attention leadership' score, then correlate with spectral scores.
        """
        self.model.eval()
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_collate_fn)
        N = self.model.n_assets
        attn_sums = np.zeros(N, dtype=np.float64)
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                event_batch, _ = batch
                event_feats = [f.to(self.device) for f in event_batch.event_feats]
                event_times = [t.to(self.device) for t in event_batch.event_times]
                edge_feats = event_batch.edge_feats.to(self.device)

                _, attn = self.model(
                    event_feats, event_times, edge_feats, return_attn_weights=True
                )
                # attn: (N, N) — attn[i, j] = attention node i pays to asset j
                # sum over query nodes i to get total attention asset j receives
                attn_sums += attn.sum(dim=0).cpu().numpy()
                n_batches += 1

        if n_batches == 0:
            return float("nan")

        attn_leadership = attn_sums / n_batches  # (N,) — mean attention received per asset
        asset_names = dataset.asset_names

        # Align with spectral scores
        spectral_vals = []
        attn_vals = []
        for i, asset in enumerate(asset_names):
            if asset in spectral_scores.index:
                spectral_vals.append(float(spectral_scores[asset]))
                attn_vals.append(attn_leadership[i])

        if len(spectral_vals) < 3:
            return float("nan")

        rho, _ = spearmanr(attn_vals, spectral_vals)
        return float(rho)
