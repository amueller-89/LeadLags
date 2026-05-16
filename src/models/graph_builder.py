from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass
class EventBatch:
    event_feats: list  # N tensors of shape (T_j, 2): [log_return, log_volume]
    event_times: list  # N tensors of shape (T_j,): Δt in seconds (positive, ≤ window)
    edge_feats: torch.Tensor  # (N, N, 2): [delay_normalized, coherence]
    asset_names: list
    query_time: pd.Timestamp


class GraphBuilder:
    """Builds the static graph edge features from spectral output and event batches from tick data."""

    def __init__(
        self,
        asset_names: list,
        spectral_results: dict = None,
        window_seconds: int = 300,
        band_name: str = "short_term",
        spectral_seeding: bool = True,
        edge_feats_override: torch.Tensor = None,
    ):
        self.asset_names = asset_names
        self.n_assets = len(asset_names)
        self.window_seconds = window_seconds
        self.spectral_seeding = spectral_seeding

        if edge_feats_override is not None:
            self.edge_feats = edge_feats_override
            self.delays_df = None
            self.coherence_df = None
            return

        if spectral_results is None:
            raise ValueError("spectral_results is required when edge_feats_override is not provided")

        # Extract delay and coherence DataFrames for the chosen band
        if band_name not in spectral_results:
            # Fall back to the first available band
            band_name = next(iter(spectral_results))
        delays_df, coherence_df = spectral_results[band_name]

        self.delays_df = delays_df
        self.coherence_df = coherence_df
        self.edge_feats = self.build_edge_features()

    def build_edge_features(self) -> torch.Tensor:
        """Build (N, N, 2) edge feature tensor [delay_normalized, coherence].

        When spectral_seeding=False, returns an all-zeros tensor so the model
        receives no spectral information and must learn structure from tick dynamics alone.
        Delays are normalized by window_seconds so they are O(1).
        Diagonal: [0.0, 1.0] (self-loop: zero delay, full coherence).
        Missing/NaN pairs: [0.0, 0.0].
        """
        N = self.n_assets

        if not self.spectral_seeding:
            return torch.zeros(N, N, 2, dtype=torch.float32)

        edge = torch.zeros(N, N, 2, dtype=torch.float32)

        for i, asset_i in enumerate(self.asset_names):
            for j, asset_j in enumerate(self.asset_names):
                if i == j:
                    edge[i, j, 0] = 0.0
                    edge[i, j, 1] = 1.0
                    continue

                delay = 0.0
                coherence = 0.0

                if asset_i in self.delays_df.index and asset_j in self.delays_df.columns:
                    d = self.delays_df.loc[asset_i, asset_j]
                    c = self.coherence_df.loc[asset_i, asset_j]
                    if not (np.isnan(d) or np.isnan(c)):
                        delay = float(d) / self.window_seconds  # normalize
                        coherence = float(c)

                edge[i, j, 0] = delay
                edge[i, j, 1] = coherence

        return edge

    def build_event_batch(
        self,
        tick_data: dict,
        query_time: pd.Timestamp,
    ) -> EventBatch:
        """Build an EventBatch for a given query time.

        For each asset, slices ticks in [query_time - window_seconds, query_time),
        computes log-return and log-volume, and converts timestamps to Δt in seconds.
        """
        window_start = query_time - pd.Timedelta(seconds=self.window_seconds)
        event_feats = []
        event_times = []

        for asset in self.asset_names:
            df = tick_data.get(asset)
            if df is None or df.empty:
                event_feats.append(torch.zeros(0, 2, dtype=torch.float32))
                event_times.append(torch.zeros(0, dtype=torch.float32))
                continue

            # Slice to window (exclusive of query_time itself)
            mask = (df.index >= window_start) & (df.index < query_time)
            window = df.loc[mask].copy()

            if window.empty:
                event_feats.append(torch.zeros(0, 2, dtype=torch.float32))
                event_times.append(torch.zeros(0, dtype=torch.float32))
                continue

            window = window.sort_index()

            # Log returns: first tick in window gets 0 (no prior price available)
            prices = window["price"].values
            log_returns = np.zeros(len(prices), dtype=np.float32)
            if len(prices) > 1:
                log_returns[1:] = np.log(prices[1:] / np.maximum(prices[:-1], 1e-10))

            # Log volume
            amounts = window["amount"].values.astype(np.float32)
            log_volumes = np.log(amounts + 1e-8)

            feats = torch.tensor(
                np.stack([log_returns, log_volumes], axis=1), dtype=torch.float32
            )

            # Δt in seconds (positive: how far before query_time)
            delta_ts = (query_time - window.index).total_seconds().astype(np.float32)
            delta_ts = np.clip(delta_ts, 0.0, self.window_seconds)

            event_feats.append(feats)
            event_times.append(torch.tensor(delta_ts, dtype=torch.float32))

        return EventBatch(
            event_feats=event_feats,
            event_times=event_times,
            edge_feats=self.edge_feats,
            asset_names=self.asset_names,
            query_time=query_time,
        )
