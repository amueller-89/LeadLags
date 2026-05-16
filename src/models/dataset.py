import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .graph_builder import GraphBuilder, EventBatch


def _normalize_asset_name(key: str) -> str:
    """Strip _USDT suffix to match preprocessor output (BTC_USDT → BTC)."""
    return key.replace("_USDT", "").replace("/USDT", "")


class TGATDataset(Dataset):
    """Dataset of (EventBatch, label) pairs for TGAT training.

    Each sample corresponds to one 1-min bar boundary. The event batch contains
    all ticks in the lookback window. The label is the next-bar log return per asset.
    """

    def __init__(
        self,
        tick_data: dict,
        ohlcv_data: dict,
        graph_builder: GraphBuilder,
        window_seconds: int = 300,
        split: str = "train",
        train_frac: float = 0.6,
        val_frac: float = 0.2,
    ):
        self.graph_builder = graph_builder
        self.window_seconds = window_seconds
        self.asset_names = graph_builder.asset_names

        # Normalize keys so tick_data and ohlcv_data both use short names (BTC, ETH, ...)
        self.tick_data = {_normalize_asset_name(k): v for k, v in tick_data.items()}
        self.ohlcv_data = {_normalize_asset_name(k): v for k, v in ohlcv_data.items()}

        self.query_times = self._build_query_times()

        # Chronological split
        n = len(self.query_times)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        if split == "train":
            self.query_times = self.query_times[:n_train]
        elif split == "val":
            self.query_times = self.query_times[n_train : n_train + n_val]
        elif split == "test":
            self.query_times = self.query_times[n_train + n_val :]
        else:
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

    def _build_query_times(self) -> list:
        """Find all 1-min bar boundaries where every asset has a valid next bar."""
        # Intersect all OHLCV indices (minute-aligned)
        indices = [
            self.ohlcv_data[a].index
            for a in self.asset_names
            if a in self.ohlcv_data
        ]
        if not indices:
            return []

        common = indices[0]
        for idx in indices[1:]:
            common = common.intersection(idx)

        common = common.sort_values()
        one_min = pd.Timedelta(minutes=1)

        # Keep only times where the next bar exists for all assets
        valid = []
        for t in common:
            t_next = t + one_min
            if all(t_next in self.ohlcv_data[a].index for a in self.asset_names if a in self.ohlcv_data):
                valid.append(t)

        return valid

    def __len__(self) -> int:
        return len(self.query_times)

    def __getitem__(self, idx: int):
        query_time = self.query_times[idx]
        batch = self.graph_builder.build_event_batch(self.tick_data, query_time)

        # Build label: next-bar log return for each asset
        one_min = pd.Timedelta(minutes=1)
        t_next = query_time + one_min
        labels = []
        for asset in self.asset_names:
            ohlcv = self.ohlcv_data.get(asset)
            if ohlcv is not None and query_time in ohlcv.index and t_next in ohlcv.index:
                close_now = ohlcv.loc[query_time, "close"]
                close_next = ohlcv.loc[t_next, "close"]
                ret = float(np.log(max(close_next, 1e-10) / max(close_now, 1e-10)))
            else:
                ret = 0.0
            labels.append(ret)

        label_tensor = torch.tensor(labels, dtype=torch.float32)
        return batch, label_tensor
