"""Baseline composer architecture: a multi-layer GRU language model.

Same interface as the Transformer composer, so training, generation and evaluation
code treat both the same way. Used (1) as the architecture comparison for the
Transformer and (2) to exercise the whole pipeline before the Transformer exists.
The "cache" is the GRU hidden state.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class GRUConfig:
    vocab_size: int
    d_model: int = 640
    n_layers: int = 3
    dropout: float = 0.1
    max_seq_len: int = 1024


class GRUComposer(nn.Module):
    def __init__(self, cfg: GRUConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.gru = nn.GRU(cfg.d_model, cfg.d_model, num_layers=cfg.n_layers, batch_first=True,
                          dropout=cfg.dropout if cfg.n_layers > 1 else 0.0)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        nn.init.normal_(self.tok_emb.weight, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        past_kv: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.drop(self.tok_emb(idx))
        out, h = self.gru(x, past_kv)
        logits = self.head(self.ln_f(self.drop(out)))
        return logits, (h if use_cache else None)

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        return sum(p.numel() for p in self.parameters())
