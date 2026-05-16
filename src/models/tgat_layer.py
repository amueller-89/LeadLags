import torch
import torch.nn as nn
import torch.nn.functional as F

from .time_encoder import TimeEncoder


class TGATLayer(nn.Module):
    """Two-level temporal graph attention layer.

    Fixes the cross-asset softmax imbalance of the naive single-level approach by
    separating attention into two stages:

    Level 1 — intra-asset (within each source asset j):
        Attends over asset j's own ticks, softmax only within j.
        Produces a single context vector h_j_agg per asset, regardless of tick count.

    Level 2 — inter-asset (across assets):
        Attends over the N per-asset context vectors using learned scores that also
        incorporate static edge features (spectral delay, coherence).
        Softmax over assets produces the final per-query-node output.

    This ensures a high-volume asset (e.g. BTC with 10× more ticks) cannot dominate
    simply by contributing more tokens to a shared softmax. Each asset contributes
    exactly one vector at the inter-asset stage.

    The inter-asset attention weights (β) are the interpretable quantities: β[i, j]
    is directly "how much query node i attends to asset j", comparable to spectral
    leadership scores.
    """

    def __init__(
        self,
        node_feat_dim: int,
        time_enc_dim: int,
        edge_feat_dim: int,
        out_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.node_feat_dim = node_feat_dim
        self.time_enc_dim = time_enc_dim

        self.time_encoder = TimeEncoder(time_enc_dim)
        # Projects raw tick features (2: log_return, log_volume) to node_feat_dim
        self.W_feat = nn.Linear(2, node_feat_dim, bias=False)

        # --- Level 1: intra-asset tick attention (per head) ---
        # Score input: [h_i (query) || h_jk_tick || Φ(Δt_jk)]
        intra_in = node_feat_dim + node_feat_dim + time_enc_dim
        self.W_intra = nn.ModuleList([nn.Linear(intra_in, out_dim, bias=False) for _ in range(n_heads)])
        self.a_intra = nn.ModuleList([nn.Linear(out_dim, 1, bias=False) for _ in range(n_heads)])
        # Value: [h_jk_tick || Φ(Δt_jk)]
        self.W_v_intra = nn.ModuleList([
            nn.Linear(node_feat_dim + time_enc_dim, out_dim, bias=False) for _ in range(n_heads)
        ])

        # --- Level 2: inter-asset attention (per head) ---
        # Score input: [h_i || h_j_agg || e_ij]
        inter_in = node_feat_dim + out_dim + edge_feat_dim
        self.W_inter = nn.ModuleList([nn.Linear(inter_in, out_dim, bias=False) for _ in range(n_heads)])
        self.a_inter = nn.ModuleList([nn.Linear(out_dim, 1, bias=False) for _ in range(n_heads)])
        # Value: h_j_agg → out_dim
        self.W_v_inter = nn.ModuleList([nn.Linear(out_dim, out_dim, bias=False) for _ in range(n_heads)])

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(
        self,
        query_embeddings: torch.Tensor,
        event_feats: list,
        event_times: list,
        edge_feats: torch.Tensor,
        return_attn_weights: bool = False,
    ):
        """
        Args:
            query_embeddings: (N, node_feat_dim)
            event_feats: list of N tensors (T_j, 2)
            event_times: list of N tensors (T_j,) — Δt in seconds
            edge_feats: (N, N, 2) — static spectral features [delay_norm, coherence]
            return_attn_weights: if True, also return (N, N) inter-asset attention

        Returns:
            output: (N, n_heads * out_dim)
            attn_weights (optional): (N, N) — β[i,j] = attention node i pays to asset j
        """
        N = query_embeddings.shape[0]
        device = query_embeddings.device

        # Pre-project tick features and time-encode upfront (shared across heads/queries)
        proj = []     # per asset: (T_j, node_feat_dim) or None
        tenc = []     # per asset: (T_j, time_enc_dim) or None

        for j in range(N):
            if event_feats[j].shape[0] == 0:
                proj.append(None)
                tenc.append(None)
            else:
                f = event_feats[j].to(device)
                t = event_times[j].to(device)
                proj.append(self.W_feat(f))          # (T_j, node_feat_dim)
                tenc.append(self.time_encoder(t))    # (T_j, time_enc_dim)

        head_outputs = []
        head_inter_attn = [] if return_attn_weights else None

        for h in range(self.n_heads):
            node_outputs = []
            node_inter_attn = [] if return_attn_weights else None

            for i in range(N):
                h_i = query_embeddings[i]  # (node_feat_dim,)

                # --- Level 1: aggregate each source asset j into h_j_agg ---
                asset_agg = []  # will hold (out_dim,) per asset j

                for j in range(N):
                    if proj[j] is None:
                        # No ticks: zero context vector
                        asset_agg.append(torch.zeros(self.out_dim, device=device))
                        continue

                    T_j = proj[j].shape[0]
                    h_i_exp = h_i.unsqueeze(0).expand(T_j, -1)   # (T_j, node_feat_dim)
                    intra_in = torch.cat([h_i_exp, proj[j], tenc[j]], dim=-1)  # (T_j, intra_in)

                    scores = self.a_intra[h](
                        self.leaky_relu(self.W_intra[h](intra_in))
                    ).squeeze(-1)  # (T_j,)

                    alpha = F.softmax(scores, dim=0)   # softmax within j only
                    alpha = self.dropout(alpha)

                    vals = self.W_v_intra[h](
                        torch.cat([proj[j], tenc[j]], dim=-1)
                    )  # (T_j, out_dim)
                    h_j_agg = (alpha.unsqueeze(-1) * vals).sum(dim=0)  # (out_dim,)
                    asset_agg.append(h_j_agg)

                asset_agg_t = torch.stack(asset_agg, dim=0)  # (N, out_dim)

                # --- Level 2: attend over the N per-asset summaries ---
                h_i_exp_inter = h_i.unsqueeze(0).expand(N, -1)  # (N, node_feat_dim)
                e_i = edge_feats[i].to(device)                   # (N, edge_feat_dim)
                inter_in = torch.cat([h_i_exp_inter, asset_agg_t, e_i], dim=-1)  # (N, inter_in)

                scores_inter = self.a_inter[h](
                    self.leaky_relu(self.W_inter[h](inter_in))
                ).squeeze(-1)  # (N,)

                beta = F.softmax(scores_inter, dim=0)   # softmax over assets
                beta = self.dropout(beta)

                vals_inter = self.W_v_inter[h](asset_agg_t)  # (N, out_dim)
                out_i = (beta.unsqueeze(-1) * vals_inter).sum(dim=0)  # (out_dim,)
                node_outputs.append(out_i)

                if return_attn_weights:
                    node_inter_attn.append(beta.detach())  # (N,)

            head_outputs.append(torch.stack(node_outputs, dim=0))  # (N, out_dim)
            if return_attn_weights:
                head_inter_attn.append(torch.stack(node_inter_attn, dim=0))  # (N, N)

        output = torch.cat(head_outputs, dim=-1)  # (N, n_heads * out_dim)

        if return_attn_weights:
            # Average inter-asset attention across heads: (N, N)
            mean_attn = torch.stack(head_inter_attn, dim=0).mean(dim=0)
            return output, mean_attn

        return output
