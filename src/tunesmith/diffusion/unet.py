"""Conditional U-Net that predicts the noise in a noisy piano roll.

    input  x_t: (B, channels, frames, pitches)   noisy roll, values around [-1, 1]
           t:   (B,)                             diffusion step, 0..T-1
           cond:(B, n_fields)                    control-token ids (same vocabulary as the
                                                 Transformer's prompt)
    output eps: (B, channels, frames, pitches)   predicted noise

Architecture (DDPM / ADM style):

    conv_in -> [ResBlock x n_res (+ self-attention) -> Downsample] per level
            -> middle: ResBlock, self-attention, ResBlock
            -> [ResBlock x (n_res + 1) with skip connections (+ attention) -> Upsample] per level
            -> GroupNorm, SiLU, conv_out

- The time step goes through a sinusoidal encoding and an MLP. Each control token is
  embedded, the field embeddings are concatenated (so the model knows which instrument
  belongs to which slot) and mapped by an MLP to the same size. The sum of the two
  embeddings scales and shifts the normalised features in every residual block (FiLM).
- Self-attention runs at the coarsest resolutions only (16 x 12 for an 8-bar roll), where
  every position can see the whole piece.
- Convolutions along the pitch axis are shared across pitches, so a chord shape learnt in
  one key is the same pattern in another; along the time axis, a rhythm is the same
  pattern in any bar.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-23. See ATTRIBUTION.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UNetConfig:
    in_channels: int = 9
    base: int = 64
    mults: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    n_res: int = 2
    attn_levels: list[int] = field(default_factory=lambda: [3])   # level indices with self-attention
    n_heads: int = 4
    dropout: float = 0.1
    cond_vocab: int = 504
    n_fields: int = 11
    field_dim: int = 64       # embedding size per control token


def _norm(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, ch // 4), ch)


def _zero(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, emb_dim: int, dropout: float):
        super().__init__()
        self.norm1, self.conv1 = _norm(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * cout)
        self.norm2 = _norm(cout)
        self.drop = nn.Dropout(dropout)
        self.conv2 = _zero(nn.Conv2d(cout, cout, 3, padding=1))
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.drop(F.silu(h)))
        return self.skip(x) + h


class AttnBlock(nn.Module):
    def __init__(self, ch: int, n_heads: int):
        super().__init__()
        self.norm, self.n_heads = _norm(ch), n_heads
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = _zero(nn.Conv2d(ch, ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, self.n_heads, C // self.n_heads, H * W).unbind(1)
        out = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.proj(out.transpose(-1, -2).reshape(B, C, H, W))


class Down(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class RollUNet(nn.Module):
    def __init__(self, cfg: UNetConfig):
        super().__init__()
        self.cfg = cfg
        base, emb_dim = cfg.base, 4 * cfg.base
        self.time_mlp = nn.Sequential(nn.Linear(base, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.cond_emb = nn.Embedding(cfg.cond_vocab, cfg.field_dim)
        self.cond_mlp = nn.Sequential(nn.Linear(cfg.n_fields * cfg.field_dim, emb_dim), nn.SiLU(),
                                      nn.Linear(emb_dim, emb_dim))
        self.conv_in = nn.Conv2d(cfg.in_channels, base, 3, padding=1)

        chans = [base * m for m in cfg.mults]
        self.down = nn.ModuleList()
        skips, ch = [base], base
        for lvl, c in enumerate(chans):
            for _ in range(cfg.n_res):
                blocks = nn.ModuleList([ResBlock(ch, c, emb_dim, cfg.dropout)])
                if lvl in cfg.attn_levels:
                    blocks.append(AttnBlock(c, cfg.n_heads))
                self.down.append(blocks)
                ch = c
                skips.append(ch)
            if lvl < len(chans) - 1:
                self.down.append(nn.ModuleList([Down(ch)]))
                skips.append(ch)

        self.mid = nn.ModuleList([ResBlock(ch, ch, emb_dim, cfg.dropout), AttnBlock(ch, cfg.n_heads),
                                  ResBlock(ch, ch, emb_dim, cfg.dropout)])

        self.up = nn.ModuleList()
        for lvl, c in reversed(list(enumerate(chans))):
            for i in range(cfg.n_res + 1):
                blocks = nn.ModuleList([ResBlock(ch + skips.pop(), c, emb_dim, cfg.dropout)])
                ch = c
                if lvl in cfg.attn_levels:
                    blocks.append(AttnBlock(c, cfg.n_heads))
                if lvl > 0 and i == cfg.n_res:
                    blocks.append(Up(ch))
                self.up.append(blocks)
        self.out = nn.Sequential(_norm(ch), nn.SiLU(), _zero(nn.Conv2d(ch, cfg.in_channels, 3, padding=1)))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def downsample_factor(self) -> int:
        return 2 ** (len(self.cfg.mults) - 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(timestep_embedding(t, self.cfg.base))
        emb = emb + self.cond_mlp(self.cond_emb(cond).flatten(1))
        h = self.conv_in(x)
        hs = [h]
        for blocks in self.down:
            for b in blocks:
                h = b(h, emb) if isinstance(b, ResBlock) else b(h)
            hs.append(h)
        for b in self.mid:
            h = b(h, emb) if isinstance(b, ResBlock) else b(h)
        for blocks in self.up:
            h = torch.cat([h, hs.pop()], dim=1)
            for b in blocks:
                h = b(h, emb) if isinstance(b, ResBlock) else b(h)
        return self.out(h)
