"""The composer: a GPT-style decoder-only Transformer over control + REMI tokens.

>>> STUDENT-IMPLEMENTED FILE <<<
This file is a skeleton: the signatures, shapes and docstrings are given, the bodies
are yours to write (every `raise NotImplementedError` marks a TODO). The rest of the
project already calls this interface:

    model = Composer(ComposerConfig(vocab_size=V, ...))
    logits, present = model(idx)                                 # training
    logits, present = model(new_idx, past_kv=present, use_cache=True)  # generation

Check your work with:  pixi run pytest tests/test_transformer.py -q
(tests are skipped while a TODO still raises NotImplementedError).

Suggested order: MLP -> CausalSelfAttention (no cache) -> Block -> Composer.forward
-> train a tiny model (configs/composer_tiny.yaml with `type: transformer`)
-> optional: KV cache, then RoPE.

Background: course slides 08-Transformer (masked self-attention, multi-head attention,
residual connections, layer normalisation) and 09-LanguageModels (causal LM).

AI attribution: skeleton and docstrings drafted with Claude Code (Claude Opus 5.5),
2026-11-10; method bodies to be written by the student. Update ATTRIBUTION.md when done.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# One (key, value) pair per layer, each of shape (B, n_heads, T_past, head_dim).
KVCache = list[tuple[torch.Tensor, torch.Tensor]]


@dataclass
class ComposerConfig:
    vocab_size: int
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 1024
    positional: str = "learned"  # "learned" (required) | "rope" (optional extension)

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        return self.d_model // self.n_heads


class MLP(nn.Module):
    """Position-wise feed-forward network: Linear(d_model, d_ff) -> GELU -> Linear(d_ff, d_model) -> Dropout."""

    def __init__(self, cfg: ComposerConfig):
        super().__init__()
        # TODO(student): create the two linear layers and the dropout.
        raise NotImplementedError("TODO(student): MLP.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model)."""
        raise NotImplementedError("TODO(student): MLP.forward")


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention.

    Steps (x has shape (B, T, d_model)):
      1. project x to queries, keys, values (one Linear(d_model, 3 * d_model) or three
         separate ones), then reshape each to (B, n_heads, T, head_dim);
      2. if `past_kv` is given (generation), concatenate the cached keys/values in front
         along the time axis, so k, v have length T_past + T;
      3. attention weights softmax(q k^T / sqrt(head_dim)) with a CAUSAL mask: position
         t may only attend to positions <= t (counting cached positions). Hint:
         F.scaled_dot_product_attention(q, k, v, is_causal=...) is allowed, but write
         the masked version by hand once to make sure you understand it. Careful: with a
         cache and T == 1 the single new query may attend to everything, so is_causal
         must be False in that case;
      4. merge heads back to (B, T, d_model), apply the output Linear and dropout.

    Returns (y, present) where present = (k, v) of the full length if use_cache else None.
    """

    def __init__(self, cfg: ComposerConfig):
        super().__init__()
        self.cfg = cfg
        # TODO(student): projections, output projection, attention/residual dropout.
        raise NotImplementedError("TODO(student): CausalSelfAttention.__init__")

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        raise NotImplementedError("TODO(student): CausalSelfAttention.forward")


class Block(nn.Module):
    """Pre-LayerNorm Transformer block:

        x = x + attn(ln_1(x))
        x = x + mlp(ln_2(x))
    """

    def __init__(self, cfg: ComposerConfig):
        super().__init__()
        raise NotImplementedError("TODO(student): Block.__init__")

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        raise NotImplementedError("TODO(student): Block.forward")


class Composer(nn.Module):
    """Token embedding + positional embedding -> n_layers Blocks -> LayerNorm -> LM head.

    Requirements checked by tests/test_transformer.py:
      - forward(idx) returns logits of shape (B, T, vocab_size) and present (None unless use_cache);
      - causality: changing token t must not change the logits at positions < t;
      - with use_cache=True, feeding tokens one at a time with past_kv reproduces the
        logits of a single full forward pass (only if you implement the cache);
      - weight tying: the LM head shares its weight with the token embedding
        (self.head.weight = self.tok_emb.weight);
      - positions: with a cache of length P, the new tokens are at positions P, P+1, ...
    """

    def __init__(self, cfg: ComposerConfig):
        super().__init__()
        self.cfg = cfg
        # TODO(student): self.tok_emb, self.pos_emb (if positional == "learned"), self.drop,
        # self.blocks (nn.ModuleList of Block), self.ln_f, self.head (no bias), weight tying,
        # then call self.apply(self._init_weights).
        raise NotImplementedError("TODO(student): Composer.__init__")

    def forward(
        self,
        idx: torch.Tensor,
        past_kv: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[KVCache]]:
        """idx: (B, T) token ids -> (logits (B, T, vocab_size), present cache or None).

        Raise a ValueError if past length + T exceeds cfg.max_seq_len.
        If you do not implement the cache, return None as the cache; generation then
        re-feeds the whole sequence every step (correct but slower).
        """
        raise NotImplementedError("TODO(student): Composer.forward")

    # ------------------------------------------------------------ provided helpers

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """GPT-2 style initialisation (provided)."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings and hasattr(self, "pos_emb"):
            n -= self.pos_emb.weight.numel()
        return n


__all__ = ["ComposerConfig", "Composer", "CausalSelfAttention", "Block", "MLP", "KVCache"]
_ = (math, F)  # imported for your implementation
