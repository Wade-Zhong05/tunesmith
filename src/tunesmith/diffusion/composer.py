"""The diffusion composer: attributes -> multi-track piano roll -> MIDI score.

    composer = DiffusionComposer.load("models/diffusion/ensemble-unet/best.pt")
    scores = composer.compose([attrs], seed=0)                  # new pieces
    edited = composer.edit(score, attrs, bars=[4])              # rewrite bar 5, keep the rest
    backing = composer.edit(score, attrs, keep_slots=[0])       # keep the lead, rewrite the others

All pieces are 8 bars of 4/4 (the roll grid the model is trained on).

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-23. See ATTRIBUTION.md.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from symusic import Score

from ..attributes import MusicAttributes
from ..model import pick_device
from ..tokenizer import TuneTokenizer
from .data import condition_ids
from .roll import RollSpec, allowed_mask, band_for, roll_to_score, score_to_roll
from .schedule import Diffusion
from .unet import RollUNet, UNetConfig

KIND = "diffusion"
SAMPLE_DEFAULTS = {"steps": 50, "cfg_scale": 2.0, "threshold": 0.5}


def save_diffusion_checkpoint(path: str | Path, model: RollUNet, spec: RollSpec, tok: TuneTokenizer,
                              timesteps: int, extra: Optional[dict[str, Any]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": KIND, "unet_cfg": asdict(model.cfg), "model_state": model.state_dict(),
                "spec": spec.to_dict(), "tokenizer": tok.config_dict(), "timesteps": int(timesteps),
                **(extra or {})}, path)


def is_diffusion_checkpoint(ckpt: dict[str, Any]) -> bool:
    return ckpt.get("kind") == KIND


class DiffusionComposer:
    def __init__(self, model: RollUNet, tok: TuneTokenizer, spec: RollSpec, diffusion: Diffusion,
                 defaults: Optional[dict[str, Any]] = None):
        self.model, self.tok, self.spec, self.diffusion = model.eval(), tok, spec, diffusion
        self.defaults = {**SAMPLE_DEFAULTS, **(defaults or {})}
        self.device = next(model.parameters()).device

    @classmethod
    def from_checkpoint(cls, ckpt: dict[str, Any], device: str | torch.device = "cpu") -> "DiffusionComposer":
        tok = TuneTokenizer({k: v for k, v in ckpt["tokenizer"].items() if k != "vocab_size"})
        model = RollUNet(UNetConfig(**ckpt["unet_cfg"]))
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()
        return cls(model, tok, RollSpec.from_dict(ckpt["spec"]), Diffusion(ckpt["timesteps"]), ckpt.get("sample"))

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> "DiffusionComposer":
        dev = pick_device(device)
        return cls.from_checkpoint(torch.load(Path(path), map_location=dev, weights_only=False), dev)

    # ------------------------------------------------------------ conditioning

    def request(self, attrs: MusicAttributes, families: Optional[list[str]] = None) -> MusicAttributes:
        """The attributes as the model will see them: the instrument slots filled (from the
        request, or a default band for its style), 4/4 and 8 bars."""
        families = families or band_for(attrs, self.spec.n_slots)
        return attrs.merged({"instruments": families, "meter": f"{self.spec.beats_per_bar}/4",
                             "bars": self.spec.bars})

    def _cond(self, attrs_list: list[MusicAttributes]) -> tuple[torch.Tensor, torch.Tensor]:
        cond = [condition_ids(self.tok, self.tok.prompt_ids(a)) for a in attrs_list]
        null = condition_ids(self.tok, self.tok.null_prompt_ids())
        return (torch.tensor(cond, device=self.device),
                torch.tensor([null] * len(attrs_list), device=self.device))

    def _sample(self, req: list[MusicAttributes], seed: Optional[int], steps: Optional[int],
                cfg_scale: Optional[float], known: Optional[np.ndarray] = None,
                known_mask: Optional[np.ndarray] = None) -> np.ndarray:
        spec = self.spec
        allowed = np.stack([allowed_mask(a.instruments or [], a.drums, spec) for a in req])
        cond, null = self._cond(req)
        gen = torch.Generator().manual_seed(int(seed)) if seed is not None else None
        dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else None
        x0 = self.diffusion.sample(
            self.model, cond, null, (len(req), *spec.shape),
            steps=int(steps or self.defaults["steps"]),
            cfg_scale=float(self.defaults["cfg_scale"] if cfg_scale is None else cfg_scale),
            allowed=torch.from_numpy(allowed).to(self.device),
            known=None if known is None else torch.from_numpy(known).float().to(self.device) * 2 - 1,
            known_mask=None if known_mask is None else torch.from_numpy(known_mask).to(self.device),
            generator=gen, autocast=dtype)
        return ((x0 + 1) / 2).float().cpu().numpy()

    def _score(self, roll: np.ndarray, attrs: MusicAttributes, threshold: Optional[float]) -> Score:
        qpm = float(attrs.tempo_bpm) if attrs.tempo_bpm else 100.0
        return roll_to_score(roll, attrs.instruments or [], self.spec, qpm,
                             float(self.defaults["threshold"] if threshold is None else threshold))

    # ------------------------------------------------------------ public API

    def compose(self, attrs_list: list[MusicAttributes], *, seed: Optional[int] = None, steps: Optional[int] = None,
                cfg_scale: Optional[float] = None, threshold: Optional[float] = None) -> list[Score]:
        req = [self.request(a) for a in attrs_list]
        rolls = self._sample(req, seed, steps, cfg_scale)
        return [self._score(r, a, threshold) for r, a in zip(rolls, req)]

    def edit(self, score: Score, attrs: Optional[MusicAttributes] = None, *, bars: Optional[list[int]] = None,
             keep_slots: Optional[list[int]] = None, seed: Optional[int] = None, steps: Optional[int] = None,
             cfg_scale: Optional[float] = None, threshold: Optional[float] = None) -> Score:
        """Rewrite part of an existing piece and keep the rest note for note.

        bars:       0-based bars to rewrite (all instruments); everything else is kept.
        keep_slots: instrument slots to keep (0 = the part with most notes); the other
                    parts, drums included, are rewritten.
        """
        spec = self.spec
        roll, families, has_drums = score_to_roll(score, spec)
        attrs = attrs or MusicAttributes()
        if attrs.drums is None and has_drums:
            attrs = attrs.merged({"drums": True})
        req = self.request(attrs, families or None)
        keep = np.ones(spec.shape, dtype=bool)
        if bars is not None:
            steps_per_bar = spec.beats_per_bar * spec.steps_per_beat
            for b in bars:
                keep[:, b * steps_per_bar:(b + 1) * steps_per_bar, :] = False
        if keep_slots is not None:
            kept_channels = [c for s in keep_slots for c in (2 * s, 2 * s + 1)]
            for c in range(spec.channels):
                if c not in kept_channels:
                    keep[c] = False
        rolls = self._sample([req], seed, steps, cfg_scale, known=roll[None], known_mask=keep[None])
        return self._score(rolls[0], req, threshold)
