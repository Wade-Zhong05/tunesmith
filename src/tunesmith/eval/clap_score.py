"""Text-audio agreement with CLAP (optional metric, as in Text2midi's evaluation).

Uses the `transformers` port laion/larger_clap_music_and_speech. (The port
laion/larger_clap_music was reported to have a collapsed text tower.) Audio is rendered
at 48 kHz, CLAP's sample rate. Downloads about 800 MB on first use.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

MODEL_ID = "laion/larger_clap_music_and_speech"
SAMPLE_RATE = 48000


@lru_cache(maxsize=1)
def _load(device: str):
    from transformers import ClapModel, ClapProcessor

    return ClapModel.from_pretrained(MODEL_ID).to(device).eval(), ClapProcessor.from_pretrained(MODEL_ID)


def _features(out) -> torch.Tensor:
    if not isinstance(out, torch.Tensor):  # some versions return a model output object
        out = getattr(out, "pooler_output", None) if getattr(out, "pooler_output", None) is not None else out[0]
    return torch.nn.functional.normalize(out, dim=-1)


@torch.no_grad()
def clap_scores(texts: list[str], wav_paths: list[str | Path], device: str = "cpu") -> list[float]:
    from ..render.synth import read_wav

    model, proc = _load(device)
    audios = []
    for p in wav_paths:
        audio, sr = read_wav(p)
        if sr != SAMPLE_RATE:
            idx = np.linspace(0, len(audio) - 1, int(len(audio) * SAMPLE_RATE / sr))
            audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
        audios.append(audio[: SAMPLE_RATE * 10])
    t = proc(text=texts, return_tensors="pt", padding=True).to(device)
    try:  # transformers >= 5 uses `audio`, 4.x used `audios`
        a = proc(audio=audios, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(device)
    except TypeError:
        a = proc(audios=audios, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(device)
    te = _features(model.get_text_features(**t))
    ae = _features(model.get_audio_features(**a))
    return (te * ae).sum(-1).cpu().tolist()
