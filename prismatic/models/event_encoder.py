"""Event network used the same way as DVSVLA's event fusion bridge.

A square polarity-time volume becomes one token per cell. The current object
center is added to every token that the action bridge reads. A separate head
pools those cells before the center is added. The pool plus the current center
cross-attends to the VLM's last layer, and one MLP reads that sum as four
image-plane displacements.
"""

import math

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


class GoalCrossAttention(nn.Module):
    """Attend from the event pool and the current center into the VLM last layer."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"位移交叉注意力的宽度 {dim} 不能被 {num_heads} 个头整除")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, pooled, center, context, context_mask):
        query = pooled + center
        batch, length, _ = context.shape
        q = self.q(query).view(batch, self.num_heads, 1, self.head_dim)
        k = self.k(context).view(batch, length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(context).view(batch, length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(self.head_dim)
        if context_mask is not None:
            mask = context_mask.to(device=scores.device).bool()
            if mask.shape != (batch, length):
                raise RuntimeError(
                    f"视觉和指令 mask 是 {tuple(mask.shape)}，VLM 隐状态是 {(batch, length)}"
                )
            scores = scores.masked_fill(~mask[:, None, None, :], -1.0e4)
            readable = mask.any(dim=1)
        else:
            readable = torch.ones(batch, dtype=torch.bool, device=scores.device)
        attn = torch.softmax(scores, dim=-1).to(dtype=context.dtype)
        mixed = torch.matmul(attn, v).reshape(batch, -1)
        mixed = self.out(mixed)
        return mixed * readable.to(dtype=mixed.dtype).unsqueeze(-1)


class EventMotionEncoder(nn.Module):
    """Tokens from a square polarity-time-image volume, conditioned on the object center.

    Input volume is ``(batch, 2, time, grid, grid)``. ``query`` is the current
    object center in normalized image ``(x, y)``. An 8-bin, 32-grid volume is
    ``(64, 4, 16, 16)`` after the convolutions.
    """

    def __init__(self, out_dim: int, context_dim: int | None = None):
        super().__init__()
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
        self.readout = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, DISPLACEMENT_COUNT * 2),
        )
        if context_dim is None or context_dim == out_dim:
            self.context_in = nn.Identity()
        else:
            self.context_in = nn.Linear(context_dim, out_dim)
        self.goal_attn = GoalCrossAttention(out_dim)

    def forward(
        self,
        volume: torch.Tensor,
        query: torch.Tensor,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
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
        if query.ndim == 1:
            query = query.unsqueeze(0)
        future = self._displacements(tokens, valid, query, context, context_mask)
        point = self.query_proj(query.to(dtype=tokens.dtype)).unsqueeze(1)
        tokens = tokens + point
        return {"tokens": tokens, "valid": valid, "future": future}

    def _pool(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weight = valid.to(dtype=tokens.dtype).unsqueeze(-1)
        return (tokens * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

    def _displacements(
        self,
        tokens: torch.Tensor,
        valid: torch.Tensor,
        query: torch.Tensor,
        context: torch.Tensor | None,
        context_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """``(batch, 4, 2)`` image displacements.

        The event pool is the query, together with the current center. It
        cross-attends to the vision patches and the instruction. The readout
        sees the pool and that attended state together, so the belt motion
        stays in the vector that produces ``(dx, dy)``.
        """
        if context is None:
            raise RuntimeError("未来位移要读 VLM 最后一层，这一步却没有隐状态")
        pooled = self._pool(tokens, valid)
        context = self.context_in(context.to(device=pooled.device, dtype=pooled.dtype))
        center = self.query_proj(query.to(device=pooled.device, dtype=pooled.dtype))
        attended = self.goal_attn(pooled, center, context, context_mask)                 # attended 是从画面和指令里读来的目标信息
        raw = self.readout(pooled + attended)
        return raw.view(tokens.shape[0], DISPLACEMENT_COUNT, 2)
