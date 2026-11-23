"""PyTorch dataset over prepared token sequences, with augmentation.

Augmentation (training split only):
- transposition by k semitones: every Pitch token and the KEY control token are shifted
  together, so the label stays correct (skipped if a pitch would leave the range);
- attribute dropout: each control token is replaced by its field's NULL token with
  probability `attr_dropout`, and the whole prompt with probability `uncond_prob`.
  This teaches the model to work with partial descriptions and enables
  classifier-free guidance at generation time.

The loss is computed on music tokens only: targets for the prompt are -100.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..tokenizer import TuneTokenizer

IGNORE = -100


class TokenDataset(Dataset):
    def __init__(self, data_dir: str | Path, split: str, tok: TuneTokenizer,
                 augment: Optional[dict[str, Any]] = None):
        data_dir = Path(data_dir)
        self.tok = tok
        self.index = np.load(data_dir / f"index_{split}.npy")
        self.bin_path = data_dir / f"tokens_{split}.bin"
        self.tokens = self._open_tokens()
        self.augment = augment or None
        self.slots = [tok.slot(f) for f in tok.fields]  # control-token positions for this mode
        self.nulls = np.array([tok.null_id(f) for f in tok.fields], dtype=np.int64)
        self.tables: dict[int, np.ndarray] = {}
        if self.augment:
            lo, hi = self.augment.get("transpose_range", [0, 0])
            self.tables = {k: tok.transpose_table(k) for k in range(int(lo), int(hi) + 1) if k != 0}

    def _open_tokens(self) -> np.ndarray:
        if self.bin_path.stat().st_size == 0:
            return np.zeros(0, dtype=np.uint16)
        return np.memmap(self.bin_path, dtype=np.uint16, mode="r")

    def __getstate__(self) -> dict[str, Any]:
        # DataLoader workers on Windows receive the dataset by pickling. Pickling a memmap copies
        # the whole token file (1.25 GB for the ensemble data), so send the path and reopen it.
        state = self.__dict__.copy()
        state["tokens"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.tokens = self._open_tokens()

    def __len__(self) -> int:
        return len(self.index)

    def raw(self, i: int) -> np.ndarray:
        start, length = self.index[i]
        return np.array(self.tokens[start:start + length], dtype=np.int64)

    def __getitem__(self, i: int) -> torch.Tensor:
        seq = self.raw(i)
        a = self.augment
        if a:
            rng = np.random.default_rng()
            if self.tables and rng.random() < float(a.get("transpose_prob", 0.0)):
                k = int(rng.choice(list(self.tables)))
                shifted = self.tables[k][seq]
                if (shifted >= 0).all():
                    seq = shifted
            if rng.random() < float(a.get("uncond_prob", 0.0)):
                seq[self.slots] = self.nulls
            else:
                drop = rng.random(len(self.slots)) < float(a.get("attr_dropout", 0.0))
                seq[np.array(self.slots)[drop]] = self.nulls[drop]
        return torch.from_numpy(seq)


def collate(batch: list[torch.Tensor], pad_id: int, prompt_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad to the longest sequence; return (inputs, targets) shifted by one."""
    L = max(len(x) for x in batch)
    x = torch.full((len(batch), L), pad_id, dtype=torch.long)
    for i, seq in enumerate(batch):
        x[i, : len(seq)] = seq
    inputs, targets = x[:, :-1].clone(), x[:, 1:].clone()
    targets[targets == pad_id] = IGNORE
    targets[:, : prompt_len - 1] = IGNORE  # do not train on predicting the given prompt
    return inputs, targets
