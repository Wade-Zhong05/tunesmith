"""Piano-roll dataset for the diffusion composer, read from the prepared ensemble windows.

Reuses data/processed/midicaps_ensemble (the token files the ensemble Transformer is
trained on): each window is decoded straight from its tokens into a roll
(`roll.tokens_to_roll`), so no second copy of the data is stored. Only 4/4 windows are
used (about 85% of the data); the roll has a fixed grid of 8 bars of 4/4.

Each item is (roll, condition):
- roll: uint8 tensor (channels, frames, pitches) with values {0, 1};
- condition: the 11 control-token ids of the window's prompt (key, meter, tempo,
  inst1-4, drums, mood, genre, bars), the same tokens the Transformer reads.

Augmentation (training split only), as for the Transformer:
- transposition by k semitones: pitched channels shift along the pitch axis and the KEY
  token is shifted with them (drums are not transposed); skipped if notes would leave
  the roll;
- attribute dropout: each control token -> its NULL token with probability
  `attr_dropout`, the whole condition with probability `uncond_prob` (for guidance).

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-23. See ATTRIBUTION.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from ..data.dataset import TokenDataset
from ..tokenizer import TuneTokenizer
from .roll import RollSpec, tokens_to_roll


def condition_ids(tok: TuneTokenizer, prompt: list[int]) -> list[int]:
    """The control tokens of a prompt (BOS and SEP removed)."""
    return list(prompt[1:tok.prompt_len - 1])


class RollDataset(TokenDataset):
    def __init__(self, data_dir: str | Path, split: str, tok: TuneTokenizer, spec: RollSpec,
                 augment: Optional[dict[str, Any]] = None, max_items: Optional[int] = None, seed: int = 0):
        if not tok.ensemble:
            raise ValueError("the diffusion composer is trained on ensemble-mode data")
        super().__init__(data_dir, split, tok, None)
        self.spec = spec
        self.augment = augment or None
        # keep 4/4 windows: the time signature token follows the prompt and the first Bar
        if len(self.index):
            ts = np.asarray(self.tokens[self.index[:, 0] + tok.prompt_len + 1], dtype=np.int64)
            self.index = self.index[ts == tok.timesig_id("4/4")]
        if max_items and len(self.index) > max_items:
            keep = np.random.default_rng(seed).choice(len(self.index), max_items, replace=False)
            self.index = self.index[np.sort(keep)]
        self.key_tables = {}
        if self.augment:
            lo, hi = self.augment.get("transpose_range", [0, 0])
            self.key_tables = {k: tok.transpose_table(k) for k in range(int(lo), int(hi) + 1) if k != 0}
        self.cond_nulls = np.array(condition_ids(tok, tok.null_prompt_ids()), dtype=np.int64)

    def item(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """(roll, condition ids) without augmentation."""
        ids = self.raw(i).tolist()
        roll = tokens_to_roll(ids, self.tok, self.spec)
        if roll is None:  # cannot happen after the 4/4 filter; keep the loader alive anyway
            roll = np.zeros(self.spec.shape, dtype=np.uint8)
        return roll, np.array(condition_ids(self.tok, ids), dtype=np.int64)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        roll, cond = self.item(i)
        a = self.augment
        if a:
            rng = np.random.default_rng()
            if self.key_tables and rng.random() < float(a.get("transpose_prob", 0.0)):
                k = int(rng.choice(list(self.key_tables)))
                shifted = transpose_roll(roll, k, self.spec)
                if shifted is not None:
                    roll, cond = shifted, self.key_tables[k][cond]
            if rng.random() < float(a.get("uncond_prob", 0.0)):
                cond = self.cond_nulls.copy()
            else:
                drop = rng.random(len(cond)) < float(a.get("attr_dropout", 0.0))
                cond[drop] = self.cond_nulls[drop]
        return torch.from_numpy(np.ascontiguousarray(roll)), torch.from_numpy(cond)


def transpose_roll(roll: np.ndarray, k: int, spec: RollSpec) -> Optional[np.ndarray]:
    """Shift the pitched channels by k semitones (drums stay); None if a note would leave
    the roll."""
    pitched = roll[: spec.drum_channel]
    rows = np.nonzero(pitched.any(axis=(0, 1)))[0]
    if len(rows) and (rows.min() + k < 0 or rows.max() + k >= spec.n_pitch):
        return None
    out = np.zeros_like(roll)
    out[spec.drum_channel] = roll[spec.drum_channel]
    if k >= 0:
        out[: spec.drum_channel, :, k:] = pitched[:, :, : spec.n_pitch - k]
    else:
        out[: spec.drum_channel, :, :k] = pitched[:, :, -k:]
    return out
