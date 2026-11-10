"""Non-neural baseline composers with the same interface as the learned models.

- UniformComposer: every token equally likely (only the grammar and key bias shape it).
- BigramComposer: token bigram LM fitted on the training split (add-k smoothing),
  i.e. a first-order Markov chain over REMI tokens.

Both plug into `model.generate.generate`, so they get exactly the same grammar, key
bias and stopping rules as the Transformer and the GRU: a controlled comparison.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..tokenizer import TuneTokenizer


@dataclass
class _Cfg:
    max_seq_len: int = 2048


class UniformComposer(nn.Module):
    name = "uniform"

    def __init__(self, vocab_size: int):
        super().__init__()
        self.cfg = _Cfg()
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.vocab_size = vocab_size

    def forward(self, idx, past_kv=None, use_cache=False):
        return torch.zeros(*idx.shape, self.vocab_size, device=idx.device), None


class BigramComposer(nn.Module):
    name = "bigram"

    def __init__(self, log_probs: np.ndarray):
        super().__init__()
        self.cfg = _Cfg()
        self.table = nn.Parameter(torch.from_numpy(log_probs.astype(np.float32)), requires_grad=False)

    def forward(self, idx, past_kv=None, use_cache=False):
        return self.table[idx], None

    @classmethod
    def fit(cls, data_dir: str | Path, tok: TuneTokenizer, alpha: float = 0.1, max_sequences: int = 200000):
        data_dir = Path(data_dir)
        index = np.load(data_dir / "index_train.npy")
        tokens = np.memmap(data_dir / "tokens_train.bin", dtype=np.uint16, mode="r")
        V = tok.vocab_size
        counts = np.full((V, V), alpha, dtype=np.float64)
        for start, length in index[:max_sequences]:
            seq = np.asarray(tokens[start + tok.prompt_len - 1:start + length], dtype=np.int64)
            np.add.at(counts, (seq[:-1], seq[1:]), 1.0)
        log_probs = np.log(counts / counts.sum(axis=1, keepdims=True))
        return cls(log_probs)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.table.detach().cpu().numpy())

    @classmethod
    def load(cls, path: str | Path):
        return cls(np.load(path))
