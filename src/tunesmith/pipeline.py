"""End-to-end pipeline: text -> attributes -> music -> MIDI / WAV / piano-roll data.

Two composer modes:
- melody:   one monophonic part (`composer_checkpoint`, `fallback_checkpoint`);
- ensemble: up to four instrument families plus drums (`ensemble_checkpoint`,
            `ensemble_fallback`), loaded on first use.

Used by the CLI, the web app and the smoke test. Works before any training: without a
checkpoint a mode falls back to its bigram baseline (if fitted) or the random baseline,
and says so in the result.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10; ensemble mode
added 2026-11-22. See ATTRIBUTION.md.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from .attributes import COMPOSER_MODES, MusicAttributes, program_to_family
from .eval.baselines import BigramComposer, UniformComposer
from .eval.metrics import music_metrics
from .model import load_checkpoint, pick_device
from .model.generate import generate
from .paths import DATA, load_json, load_yaml, resolve
from .render.pianoroll import notes_for_ui
from .render.render import preload_soundfont, render_wav
from .tokenizer import TuneTokenizer

log = logging.getLogger(__name__)

# where each mode looks for prepared data (for its tokenizer) and its fitted bigram baseline
_MODE_DATA = {"melody": "midicaps", "ensemble": "midicaps_ensemble"}
_MODE_BIGRAM = {"melody": "models/baselines/bigram.npy", "ensemble": "models/baselines/bigram_ensemble.npy"}
_UI_METRICS = ("n_notes", "bars", "detected_key", "key_match", "in_scale", "meter", "repetition",
               "instrument_recall", "drums_match", "n_tracks", "polyphony")


def make_parser(kind: str, cfg: Optional[dict[str, Any]] = None):
    cfg = cfg or load_yaml("configs/app.yaml")
    if kind == "rules":
        from .understand.rules import RuleParser

        return RuleParser()
    if kind == "lm":
        from .understand.lm_parser import LMParser

        return LMParser(cfg["parser_base"], cfg.get("parser_adapter"), cfg.get("parser_mode", "lora"),
                        cfg.get("device", "auto"))
    raise ValueError(f"unknown parser {kind!r}")


def _default_tokenizer(mode: str = "melody") -> TuneTokenizer:
    meta = DATA / "processed" / _MODE_DATA[mode] / "meta.json"
    if meta.exists():
        return TuneTokenizer(load_json(meta)["tokenizer"])
    return TuneTokenizer({"mode": mode} if mode != "melody" else None)


def _as_list(value) -> list[str]:
    if not value:
        return []
    return [value] if isinstance(value, str) else list(value)


@dataclass
class Composer:
    """A loaded composer for one mode."""

    mode: str
    model: torch.nn.Module
    tok: TuneTokenizer
    name: str
    trained: bool


class TuneSmith:
    def __init__(self, config: str | dict[str, Any] = "configs/app.yaml"):
        self.cfg = load_yaml(config) if isinstance(config, (str, Path)) else dict(config)
        self.device = pick_device(self.cfg.get("device", "auto"))
        self.out_dir = resolve(self.cfg.get("output_dir", "outputs/app"))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.parse_lock = threading.Lock()
        self.parse_cache: dict[str, MusicAttributes] = {}
        self.warnings: list[str] = []
        self.composers: dict[str, Composer] = {}
        render_cfg = self.cfg.get("render", {})
        preload_soundfont(render_cfg.get("soundfont"), int(render_cfg.get("sample_rate", 44100)))
        self._load_parser()
        melody = self.composer("melody")
        # melody-mode aliases (kept for the CLI, tests and older callers)
        self.model, self.tok, self.composer_name, self.trained = melody.model, melody.tok, melody.name, melody.trained

    # ------------------------------------------------------------ loading

    def _load_parser(self) -> None:
        kind = self.cfg.get("parser", "rules")
        try:
            self.parser = make_parser(kind, self.cfg)
        except Exception as e:
            from .understand.rules import RuleParser

            self.warnings.append(f"LM parser unavailable ({type(e).__name__}); using the rule parser")
            log.warning("parser %s failed to load: %s", kind, e)
            self.parser = RuleParser()
        self.parser_name = getattr(self.parser, "name", kind)

    def composer(self, mode: str = "melody") -> Composer:
        """The composer for a mode, loaded on first use."""
        if mode not in COMPOSER_MODES:
            raise ValueError(f"unknown mode {mode!r}; choose one of {', '.join(COMPOSER_MODES)}")
        if mode not in self.composers:
            with self.lock:
                if mode not in self.composers:
                    self.composers[mode] = self._load_composer(mode)
        return self.composers[mode]

    def _load_composer(self, mode: str) -> Composer:
        if mode == "melody":
            primary = self.cfg.get("composer_checkpoint", "")
            fallback = _as_list(self.cfg.get("fallback_checkpoint"))
        else:
            primary = self.cfg.get("ensemble_checkpoint", "")
            fallback = _as_list(self.cfg.get("ensemble_fallback"))
        ckpt = resolve(primary) if primary else None
        for path in ([ckpt] if ckpt else []) + [resolve(f) for f in fallback]:
            if path.is_file():
                model, tok, meta = load_checkpoint(path, self.device)
                if tok.mode != mode:
                    self.warnings.append(f"{path} is a {tok.mode} checkpoint; skipped for {mode} mode")
                    continue
                if ckpt is not None and path != ckpt:
                    self.warnings.append(f"{mode} checkpoint not found at {ckpt}; using {path}")
                return Composer(mode, model, tok, f"{meta['model_cfg']['type']} ({path.parent.name}/{path.name})", True)
        tok = _default_tokenizer(mode)
        bigram = resolve(_MODE_BIGRAM[mode])
        if self.cfg.get("fallback_composer", "markov") == "markov" and bigram.exists():
            model, name = BigramComposer.load(bigram).to(self.device), "bigram baseline (no trained composer yet)"
        else:
            model, name = UniformComposer(tok.vocab_size).to(self.device), "random baseline (no trained composer yet)"
        self.warnings.append(f"no {mode} composer checkpoint found; using the {name}")
        return Composer(mode, model, tok, name, False)

    def status(self) -> dict[str, Any]:
        """Loaded composers and which checkpoints exist (without loading the ensemble model)."""
        out = {m: {"name": c.name, "trained": c.trained, "loaded": True} for m, c in self.composers.items()}
        if "ensemble" not in out:
            paths = [self.cfg.get("ensemble_checkpoint")] + _as_list(self.cfg.get("ensemble_fallback"))
            available = any(p and resolve(p).is_file() for p in paths)
            out["ensemble"] = {"name": "not loaded yet", "trained": available, "loaded": False}
        return out

    # ------------------------------------------------------------ use

    def parse(self, text: str) -> MusicAttributes:
        """Attributes for a description. The language-model parser is deterministic (greedy
        decoding), so results are cached: composing again with the same text skips it."""
        text = text.strip()
        if not text:
            return MusicAttributes()
        with self.parse_lock:
            if text not in self.parse_cache:
                if len(self.parse_cache) >= 256:
                    self.parse_cache.pop(next(iter(self.parse_cache)))
                self.parse_cache[text] = self.parser.parse(text)
            return self.parse_cache[text].model_copy()

    def compose(self, text: str = "", overrides: Optional[dict[str, Any]] = None, n: int = 1,
                seed: Optional[int] = None, temperature: Optional[float] = None, top_k: Optional[int] = None,
                top_p: Optional[float] = None, cfg_scale: Optional[float] = None,
                key_bias: Optional[float] = None, render: bool = True, mode: str = "melody") -> dict[str, Any]:
        comp = self.composer(mode)
        d = self.cfg.get("defaults", {})
        t0 = time.time()
        attrs = self.parse(text).merged(overrides)
        if attrs.lyrics:
            raise ValueError("lyrics mode is not available yet: the instrumental version comes first")
        t_parse = time.time() - t0
        if mode == "ensemble":
            attrs = attrs.model_copy(update={"bars": int(d.get("ensemble_bars", 8))})  # trained on 8-bar windows
        elif attrs.bars is None:
            attrs = attrs.model_copy(update={"bars": int(d.get("bars", 16))})
        seed = int(seed) if seed is not None else int(time.time() * 1000) % 1_000_000
        with self.lock:
            seqs = generate(comp.model, comp.tok, [attrs] * n,
                            temperature=d.get("temperature", 1.0) if temperature is None else temperature,
                            top_k=d.get("top_k", 0) if top_k is None else top_k,
                            top_p=d.get("top_p", 0.95) if top_p is None else top_p,
                            cfg_scale=(d.get("cfg_scale", 1.5) if cfg_scale is None else cfg_scale) if comp.trained else 1.0,
                            key_bias=d.get("key_bias", 1.0) if key_bias is None else key_bias,
                            seed=seed)
        t_gen = time.time() - t0 - t_parse
        playback = attrs if attrs.tempo_bpm else attrs.model_copy(update={"tempo_bpm": 100})
        rid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        pieces = []
        render_cfg = self.cfg.get("render", {})
        for k, ids in enumerate(seqs):
            score = comp.tok.decode_for(ids, playback)
            midi = self.out_dir / f"{rid}_{k}.mid"
            score.dump_midi(str(midi))
            metrics = music_metrics(score, attrs, comp.tok.count_bars(ids), ensemble=comp.tok.ensemble)
            piece: dict[str, Any] = {
                "midi": midi.name,
                "notes": notes_for_ui(score),
                "metrics": {k2: v for k2, v in metrics.items() if k2 in _UI_METRICS},
                "tracks": [{"family": "drums" if t.is_drum else (program_to_family(int(t.program)) or t.name),
                            "program": int(t.program), "is_drum": bool(t.is_drum), "n_notes": len(t.notes)}
                           for t in sorted(score.tracks, key=lambda t: t.is_drum) if len(t.notes)],
            }
            if render:
                wav = self.out_dir / f"{rid}_{k}.wav"
                piece["engine"] = render_wav(score, wav, render_cfg.get("soundfont"),
                                             int(render_cfg.get("sample_rate", 44100)))
                piece["wav"] = wav.name
            pieces.append(piece)
        return {
            "id": rid,
            "mode": mode,
            "text": text,
            "attributes": attrs.model_dump(),
            "playback_tempo": playback.tempo_bpm,
            "parser": self.parser_name,
            "composer": comp.name,
            "trained": comp.trained,
            "seed": seed,
            "pieces": pieces,
            "timings": {"parse_s": round(t_parse, 3), "generate_s": round(t_gen, 3),
                        "total_s": round(time.time() - t0, 3)},
            "warnings": list(self.warnings),
        }
