import torch
import torch.nn as nn

from .tgat_layer import TGATLayer


class TGATModel(nn.Module):
    """Full Temporal Graph Attention Network for next-bar return prediction.

    Stacks n_layers TGAT layers with LayerNorm + residual connections, then
    applies a per-node MLP to predict next 1-min bar log returns.
    """

    def __init__(
        self,
        n_assets: int,
        tick_feat_dim: int = 2,
        node_embed_dim: int = 64,
        time_enc_dim: int = 64,
        edge_feat_dim: int = 2,
        n_heads: int = 4,
        n_layers: int = 2,
        mlp_hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.node_embed_dim = node_embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Project raw tick features to node_embed_dim for initial query embedding
        self.W_init = nn.Linear(tick_feat_dim, node_embed_dim, bias=False)
        # Fallback embedding for assets with no ticks in window
        self.empty_embedding = nn.Parameter(torch.zeros(node_embed_dim))

        layer_out_dim = node_embed_dim  # out_dim per head
        self.layers = nn.ModuleList([
            TGATLayer(
                node_feat_dim=node_embed_dim,
                time_enc_dim=time_enc_dim,
                edge_feat_dim=edge_feat_dim,
                out_dim=layer_out_dim,
                n_heads=n_heads,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Project concatenated multi-head output back to node_embed_dim (residual)
        self.W_res = nn.ModuleList([
            nn.Linear(n_heads * layer_out_dim, node_embed_dim, bias=False)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(node_embed_dim) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        # Prediction MLP: node_embed_dim → mlp_hidden_dim → 1
        self.mlp = nn.Sequential(
            nn.Linear(node_embed_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, 1),
        )

    def _init_query_embeddings(self, event_feats: list, device: torch.device) -> torch.Tensor:
        """Build initial query embedding from mean of tick features in window."""
        embeddings = []
        for j, feats in enumerate(event_feats):
            if feats.shape[0] == 0:
                embeddings.append(self.empty_embedding)
            else:
                mean_feat = feats.to(device).mean(dim=0)  # (2,)
                embeddings.append(self.W_init(mean_feat))  # (node_embed_dim,)
        return torch.stack(embeddings, dim=0)  # (N, node_embed_dim)

    def forward(
        self,
        event_feats: list,
        event_times: list,
        edge_feats: torch.Tensor,
        return_attn_weights: bool = False,
    ):
        """
        Args:
            event_feats: N tensors of (T_j, 2)
            event_times: N tensors of (T_j,) — Δt in seconds
            edge_feats: (N, N, 2) — [delay_normalized, coherence]
            return_attn_weights: if True, also return (N, N) mean attention

        Returns:
            predictions: (N,) next-bar log returns
            attn_weights (optional): (N, N)
        """
        device = edge_feats.device
        h = self._init_query_embeddings(event_feats, device)  # (N, node_embed_dim)

        last_attn = None
        for l_idx, layer in enumerate(self.layers):
            if return_attn_weights and l_idx == self.n_layers - 1:
                layer_out, last_attn = layer(
                    h, event_feats, event_times, edge_feats, return_attn_weights=True
                )
            else:
                layer_out = layer(h, event_feats, event_times, edge_feats)

            # Residual + LayerNorm
            h = self.norms[l_idx](h + self.dropout(self.W_res[l_idx](layer_out)))

        predictions = self.mlp(h).squeeze(-1)  # (N,)

        if return_attn_weights:
            return predictions, last_attn
        return predictions
