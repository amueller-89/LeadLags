import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    """Functional time encoding from TGAT (Xu et al. 2020).

    Encodes a time delta (in seconds) as a vector of interleaved cosines and sines
    at geometrically spaced frequencies: Φ(Δt)_k = cos/sin(ω_k · Δt),
    ω_k = 1 / 10^(k / (out_dim/2)).
    """

    def __init__(self, out_dim: int):
        super().__init__()
        assert out_dim % 2 == 0, "out_dim must be even"
        d = out_dim // 2
        # ω_k = 1 / 10^(k/d) for k = 0 .. d-1, shape (d,)
        omega = 1.0 / (10.0 ** (torch.arange(d, dtype=torch.float32) / d))
        self.register_buffer("omega", omega)
        self.out_dim = out_dim

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        # delta_t: (...,) float32, seconds
        # returns: (..., out_dim)
        angles = delta_t.unsqueeze(-1) * self.omega  # (..., d)
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)  # (..., out_dim)
