"""Reference implementation of the Transformer composer (`model.type: transformer_ref`).

A GPT-style decoder-only Transformer over the control-prefix + REMI token sequence:
token embedding + learned positional embedding -> n_layers pre-LayerNorm blocks
(multi-head causal self-attention, GELU MLP) -> final LayerNorm -> LM head tied to the
token embedding. Supports a KV cache for fast generation.

Attention uses PyTorch's fused `F.scaled_dot_product_attention` (flash / memory-efficient
kernels). An explicit softmax(QK^T)V keeps a (B, heads, T, T) matrix per layer for the
backward pass and does not fit an 8 GB GPU at batch 24 x 1024 tokens; the fused kernel
needs about 4 GB.

This file exists alongside the student's own implementation in `model/transformer.py`
(same interface and the same `ComposerConfig`), so both can be trained and compared.

AI attribution: written by Claude Code (Claude Opus 5.5) on 2026-11-14 at the student's
request, as a reference to train before the student's own version is finished.
See ATTRIBUTION.md.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import ComposerConfig, KVCache


class MLP(nn.Module):
    def __init__(self, cfg: ComposerConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.d_ff)
        self.proj = nn.Linear(cfg.d_ff, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ComposerConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads, self.head_dim = cfg.n_heads, cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.resid_drop = nn.Dropout(cfg.dropout)
        # No dropout on the attention probabilities: with dropout the fused SDPA kernels are not
        # always available (seen on Windows) and memory use nearly doubles. Dropout stays on the
        # residual and MLP outputs.
        self.attn_p = 0.0

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, heads, T, head_dim)
        q, k, v = (t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) for t in (q, k, v))
        P = 0
        if past_kv is not None:
            P = past_kv[0].size(2)
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        if P == 0:
            mask, causal = None, True
        elif T == 1:
            mask, causal = None, False  # the single new query may see every cached position
        else:
            # query i (absolute position P + i) may see keys 0 .. P + i
            mask = torch.ones(T, P + T, dtype=torch.bool, device=x.device).tril(P)
            causal = False
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal,
                                           dropout_p=self.attn_p if self.training else 0.0)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.resid_drop(self.proj(y)), ((k, v) if use_cache else None)


class Block(nn.Module):
    def __init__(self, cfg: ComposerConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, past_kv=None, use_cache=False):
        a, present = self.attn(self.ln_1(x), past_kv, use_cache)
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x, present


class Composer(nn.Module):
    def __init__(self, cfg: ComposerConfig):
        super().__init__()
        if cfg.positional != "learned":
            raise ValueError("the reference implementation supports positional='learned' only")
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init_weights)
        # GPT-2: scale the residual output projections by 1/sqrt(2 * n_layers)
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        past_kv: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[KVCache]]:
        B, T = idx.shape
        P = past_kv[0][0].size(2) if past_kv else 0
        if P + T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {P + T} exceeds max_seq_len {self.cfg.max_seq_len}")
        pos = torch.arange(P, P + T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        presents: KVCache = []
        for i, block in enumerate(self.blocks):
            x, present = block(x, past_kv[i] if past_kv else None, use_cache)
            presents.append(present)
        logits = self.head(self.ln_f(x))
        return logits, (presents if use_cache else None)

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - self.pos_emb.weight.numel() if exclude_embeddings else n
