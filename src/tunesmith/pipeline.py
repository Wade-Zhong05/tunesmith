"""End-to-end pipeline: text -> attributes -> melody -> MIDI / WAV / piano-roll data.

Used by the CLI, the web app and the smoke test. Works before any training: without a
composer checkpoint it falls back to the bigram baseline (if fitted) or the random
baseline, and says so in the result.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from .attributes import MusicAttributes
from .eval.baselines import BigramComposer, UniformComposer
from .eval.metrics import music_metrics
from .model import load_checkpoint, pick_device
from .model.generate import generate
from .paths import DATA, load_json, load_yaml, resolve
from .render.pianoroll import notes_for_ui
from .render.render import render_wav
from .tokenizer import TuneTokenizer

log = logging.getLogger(__name__)


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


def _default_tokenizer() -> TuneTokenizer:
    meta = DATA / "processed" / "midicaps" / "meta.json"
    return TuneTokenizer(load_json(meta)["tokenizer"]) if meta.exists() else TuneTokenizer()


class TuneSmith:
    def __init__(self, config: str | dict[str, Any] = "configs/app.yaml"):
        self.cfg = load_yaml(config) if isinstance(config, (str, Path)) else dict(config)
        self.device = pick_device(self.cfg.get("device", "auto"))
        self.out_dir = resolve(self.cfg.get("output_dir", "outputs/app"))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.warnings: list[str] = []
        self._load_parser()
        self._load_composer()

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

    def _load_composer(self) -> None:
        ckpt = resolve(self.cfg.get("composer_checkpoint", ""))
        if ckpt.is_file():
            self.model, self.tok, meta = load_checkpoint(ckpt, self.device)
            self.composer_name = f"{meta['model_cfg']['type']} ({ckpt.parent.name}/{ckpt.name})"
            self.trained = True
            return
        self.tok = _default_tokenizer()
        bigram = resolve("models/baselines/bigram.npy")
        if self.cfg.get("fallback_composer", "markov") == "markov" and bigram.exists():
            self.model = BigramComposer.load(bigram).to(self.device)
            self.composer_name = "bigram baseline (no trained composer yet)"
        else:
            self.model = UniformComposer(self.tok.vocab_size).to(self.device)
            self.composer_name = "random baseline (no trained composer yet)"
        self.trained = False
        self.warnings.append(f"composer checkpoint not found at {ckpt}; using the {self.composer_name}")

    # ------------------------------------------------------------ use

    def parse(self, text: str) -> MusicAttributes:
        return self.parser.parse(text) if text.strip() else MusicAttributes()

    def compose(self, text: str = "", overrides: Optional[dict[str, Any]] = None, n: int = 1,
                seed: Optional[int] = None, temperature: Optional[float] = None, top_k: Optional[int] = None,
                top_p: Optional[float] = None, cfg_scale: Optional[float] = None,
                key_bias: Optional[float] = None, render: bool = True) -> dict[str, Any]:
        d = self.cfg.get("defaults", {})
        t0 = time.time()
        attrs = self.parse(text).merged(overrides)
        if attrs.lyrics:
            raise ValueError("lyrics mode is not available yet: the instrumental version comes first")
        t_parse = time.time() - t0
        if attrs.bars is None:
            attrs = attrs.model_copy(update={"bars": int(d.get("bars", 16))})
        seed = int(seed) if seed is not None else int(time.time() * 1000) % 1_000_000
        with self.lock:
            seqs = generate(self.model, self.tok, [attrs] * n,
                            temperature=d.get("temperature", 1.0) if temperature is None else temperature,
                            top_k=d.get("top_k", 0) if top_k is None else top_k,
                            top_p=d.get("top_p", 0.95) if top_p is None else top_p,
                            cfg_scale=(d.get("cfg_scale", 1.5) if cfg_scale is None else cfg_scale) if self.trained else 1.0,
                            key_bias=d.get("key_bias", 1.0) if key_bias is None else key_bias,
                            seed=seed)
        t_gen = time.time() - t0 - t_parse
        playback = attrs if attrs.tempo_bpm else attrs.model_copy(update={"tempo_bpm": 100})
        rid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        pieces = []
        render_cfg = self.cfg.get("render", {})
        for k, ids in enumerate(seqs):
            score = self.tok.decode_for(ids, playback)
            midi = self.out_dir / f"{rid}_{k}.mid"
            score.dump_midi(str(midi))
            piece: dict[str, Any] = {"midi": midi.name, "notes": notes_for_ui(score),
                                     "metrics": {k2: v for k2, v in music_metrics(score, attrs, self.tok.count_bars(ids)).items()
                                                 if k2 in ("n_notes", "bars", "detected_key", "key_match",
                                                           "in_scale", "meter", "repetition")}}
            if render:
                wav = self.out_dir / f"{rid}_{k}.wav"
                piece["engine"] = render_wav(score, wav, render_cfg.get("soundfont"),
                                             int(render_cfg.get("sample_rate", 44100)))
                piece["wav"] = wav.name
            pieces.append(piece)
        return {
            "id": rid,
            "text": text,
            "attributes": attrs.model_dump(),
            "playback_tempo": playback.tempo_bpm,
            "parser": self.parser_name,
            "composer": self.composer_name,
            "trained": self.trained,
            "seed": seed,
            "pieces": pieces,
            "timings": {"parse_s": round(t_parse, 3), "generate_s": round(t_gen, 3),
                        "total_s": round(time.time() - t0, 3)},
            "warnings": list(self.warnings),
        }
