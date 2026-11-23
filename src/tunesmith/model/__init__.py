"""Composer models: construction, checkpoint saving and loading.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..tokenizer import TuneTokenizer


class NotImplementedYet(RuntimeError):
    """Raised when a student-implemented component is still a TODO."""


def _pick(cls, cfg: dict[str, Any]) -> dict[str, Any]:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in cfg.items() if k in names}


def build_model(model_cfg: dict[str, Any], vocab_size: int) -> nn.Module:
    kind = model_cfg.get("type", "transformer")
    if kind == "transformer":
        from .transformer import Composer, ComposerConfig

        cfg = ComposerConfig(**_pick(ComposerConfig, {**model_cfg, "vocab_size": vocab_size}))
        try:
            return Composer(cfg)
        except NotImplementedError as e:
            raise NotImplementedYet(
                "The Transformer composer is still a skeleton (src/tunesmith/model/transformer.py, "
                f"{e}). Implement it, or train the GRU baseline with model.type: gru."
            ) from e
    if kind == "transformer_ref":
        from .transformer import ComposerConfig
        from .transformer_ref import Composer as RefComposer

        return RefComposer(ComposerConfig(**_pick(ComposerConfig, {**model_cfg, "vocab_size": vocab_size})))
    if kind == "gru":
        from .rnn import GRUComposer, GRUConfig

        return GRUComposer(GRUConfig(**_pick(GRUConfig, {**model_cfg, "vocab_size": vocab_size})))
    raise ValueError(f"unknown model type {kind!r}")


def save_checkpoint(path: str | Path, model: nn.Module, model_cfg: dict[str, Any],
                    tokenizer: TuneTokenizer, extra: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_cfg": model_cfg,
        "model_state": model.state_dict(),
        "tokenizer": tokenizer.config_dict(),
        "arch_cfg": asdict(model.cfg),
        **(extra or {}),
    }, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    """Returns (model, tokenizer, checkpoint dict)."""
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    tok_cfg = {k: v for k, v in ckpt["tokenizer"].items() if k != "vocab_size"}
    tokenizer = TuneTokenizer(tok_cfg)
    if tokenizer.vocab_size != ckpt["tokenizer"]["vocab_size"]:
        raise ValueError("tokenizer vocabulary does not match the checkpoint")
    model = build_model(ckpt["model_cfg"], tokenizer.vocab_size)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, tokenizer, ckpt


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
