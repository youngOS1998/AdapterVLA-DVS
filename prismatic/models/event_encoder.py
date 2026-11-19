"""Event network used the same way as DVSVLA's event fusion bridge.

A square polarity-time volume becomes one token per cell. The current object
center is added to every token that the action bridge reads. A separate head
pools those cells before the center is added and regresses four image-plane
displacements.
"""

import torch
from torch import nn

# Matches prismatic.vla.datasets.event_volume.DISPLACEMENT_COUNT.
DISPLACEMENT_COUNT = 4


class CenterHead(nn.Module):
    """Current object center from the pooled action-query state and proprio."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.up = nn.Linear(input_dim, hidden_dim)
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.act(self.up(self.norm(features)))
        hidden = hidden + self.block(hidden)
        return torch.sigmoid(self.down(self.out_norm(hidden)))


class FutureProjector(nn.Module):
    """Map each ``(dx, dy)`` displacement into the action-bridge width."""

    def __init__(self, llm_dim: int, point_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(point_dim, llm_dim, bias=True)
        self.fc2 = nn.Linear(llm_dim, llm_dim, bias=True)
        self.act = nn.GELU()

    def forward(self, point: torch.Tensor) -> torch.Tensor:
        hidden = self.act(self.fc1(point))
        return self.fc2(hidden)


class EventMotionEncoder(nn.Module):
    """Tokens from a square polarity-time-image volume, conditioned on the object center.

    Input volume is ``(batch, 2, time, grid, grid)``. ``query`` is the current
    object center in normalized image ``(x, y)``. An 8-bin, 32-grid volume is
    ``(64, 4, 16, 16)`` after the convolutions.
    """

    def __init__(self, out_dim: int, context_dim: int | None = None):
        super().__init__()
        del context_dim
        self.conv = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=(2, 1, 1), padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.MaxPool3d(kernel_size=3, stride=(2, 1, 1), padding=1),
        )
        self.proj = nn.Linear(64, out_dim)
        self.query_proj = nn.Linear(2, out_dim)
        self.displacement = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, DISPLACEMENT_COUNT * 2),
        )

    def forward(
        self,
        volume: torch.Tensor,
        query: torch.Tensor,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del context, context_mask
        if volume.ndim == 4:
            volume = volume.unsqueeze(0)
        dtype = self.proj.weight.dtype
        volume = volume.to(dtype=dtype)
        hidden = self.conv(volume)
        tokens = self.proj(hidden.flatten(2).transpose(1, 2))
        occupied = (volume.sum(dim=1, keepdim=True) > 0).to(dtype=volume.dtype)
        valid = self.pool(occupied).flatten(2).squeeze(1) > 0
        empty = ~valid.any(dim=1)
        if empty.any():
            valid = valid.clone()
            valid[empty, 0] = True
            tokens = tokens.clone()
            tokens[empty, 0] = 0
        future = self._displacements(tokens, valid)
        if query.ndim == 1:
            query = query.unsqueeze(0)
        point = self.query_proj(query.to(dtype=tokens.dtype)).unsqueeze(1)
        tokens = tokens + point
        return {"tokens": tokens, "valid": valid, "future": future}

    def _displacements(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """``(batch, 4, 2)`` image displacements from the event cells alone.

        The current center is added to ``tokens`` only after this pool, so this
        head cannot answer by copying that center.
        """
        weight = valid.to(dtype=tokens.dtype).unsqueeze(-1)
        pooled = (tokens * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        raw = self.displacement(pooled)
        return raw.view(tokens.shape[0], DISPLACEMENT_COUNT, 2)
