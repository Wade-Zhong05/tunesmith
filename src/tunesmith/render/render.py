"""Render a symusic Score to WAV.

Engines, in order of preference:
1. tinysoundfont (MIT) with the MuseScore General SoundFont (`pixi run fetch-soundfont`).
   Loading the SF3 takes about 30 s (its samples are Ogg-compressed); rendering then takes
   about 0.1 s per piece, so the loaded synthesiser is cached per process and can be
   preloaded in the background (`preload_soundfont`).
2. symusic's built-in synthesiser (fails on some Windows setups with symusic 0.6.0).
3. the numpy additive synthesiser (always available).

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10; tinysoundfont backend
added 2026-11-23. See ATTRIBUTION.md.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from symusic import Score

from ..paths import resolve
from .synth import mix, preset_for_program, render_drums, render_notes, write_wav

log = logging.getLogger(__name__)
_SYNTHS: dict[tuple[str, int], object] = {}
_TSF: dict[tuple[str, int], object] = {}
_TSF_LOCK = threading.Lock()
DRUM_CHANNEL = 9


def _tsf_synth(sf: Path, sample_rate: int):
    """Loaded tinysoundfont synthesiser (cached; callers must hold _TSF_LOCK)."""
    key = (str(sf), sample_rate)
    if key not in _TSF:
        import tinysoundfont

        synth = tinysoundfont.Synth(samplerate=sample_rate)
        synth.sfload(str(sf))
        _TSF[key] = synth
    return _TSF[key]


def preload_soundfont(soundfont: Optional[str | Path], sample_rate: int = 44100) -> None:
    """Load the SoundFont in a background thread so the first request does not wait 30 s."""
    sf = resolve(soundfont) if soundfont else None
    if sf is None or not sf.exists():
        return

    def load():
        try:
            with _TSF_LOCK:
                _tsf_synth(sf, sample_rate)
            log.info("soundfont loaded: %s", sf.name)
        except Exception as e:
            log.warning("soundfont preload failed (%s)", e)

    threading.Thread(target=load, daemon=True, name="soundfont-preload").start()


def render_tinysoundfont(score: Score, sf: Path, sample_rate: int = 44100) -> np.ndarray:
    """Stereo float32 audio (2, samples), peak-normalised to 0.9."""
    import tinysoundfont

    fd, tmp = tempfile.mkstemp(suffix=".mid")
    os.close(fd)
    try:
        score.dump_midi(tmp)
        with _TSF_LOCK:
            synth = _tsf_synth(sf, sample_rate)
            synth.sounds_off()
            seq = tinysoundfont.Sequencer(synth)
            synth.program_change(DRUM_CHANNEL, 0, True)  # GM drum kit on channel 10
            seq.midi_load(tmp)
            block, tail, chunks = sample_rate // 10, 0, []
            while tail < 20:  # about 2 s after the last event, for release tails
                chunks.append(np.frombuffer(synth.generate(block), dtype=np.float32).copy())
                if seq.is_empty():
                    tail += 1
            synth.sounds_off()
    finally:
        os.unlink(tmp)
    audio = np.concatenate(chunks).reshape(-1, 2).T
    peak = float(np.abs(audio).max())
    return audio / peak * 0.9 if peak > 0 else audio


def _sec_per_tick(score: Score) -> float:
    qpm = float(score.tempos[0].qpm) if len(score.tempos) else 120.0
    return 60.0 / qpm / score.tpq


def score_to_seconds(score: Score) -> list[tuple[float, float, int, int]]:
    spt = _sec_per_tick(score)
    return [(n.time * spt, n.duration * spt, int(n.pitch), int(n.velocity))
            for tr in score.tracks if not tr.is_drum for n in tr.notes]


def render_numpy(score: Score, sample_rate: int = 44100) -> "np.ndarray":
    """Every pitched track with its family's preset, plus synthesised drums, mixed."""
    spt = _sec_per_tick(score)
    parts, gains = [], []
    for tr in score.tracks:
        if not len(tr.notes):
            continue
        if tr.is_drum:
            parts.append(render_drums([(n.time * spt, int(n.pitch), int(n.velocity)) for n in tr.notes], sample_rate))
            gains.append(0.8)
        else:
            notes = [(n.time * spt, n.duration * spt, int(n.pitch), int(n.velocity)) for n in tr.notes]
            audio = render_notes(notes, sample_rate, preset_for_program(int(tr.program)))
            parts.append(audio)
            gains.append(1.0)
    if not parts:
        return render_notes([], sample_rate)
    if len(parts) == 1:
        return parts[0]
    return mix(parts, [g / len(parts) ** 0.5 for g in gains])


def _soundfont_synth(sf: Path, sample_rate: int):
    key = (str(sf), sample_rate)
    if key not in _SYNTHS:
        import symusic

        _SYNTHS[key] = symusic.Synthesizer(str(sf), sample_rate)
    return _SYNTHS[key]


def render_wav(score: Score, path: str | Path, soundfont: Optional[str | Path] = None,
               sample_rate: int = 44100) -> str:
    """Write a WAV file; returns the engine used ("soundfont" or "numpy")."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf = resolve(soundfont) if soundfont else None
    if sf is not None and sf.exists():
        try:
            write_wav(path, render_tinysoundfont(score, sf, sample_rate), sample_rate)
            return "soundfont"
        except ImportError:
            pass
        except Exception as e:
            log.warning("tinysoundfont rendering failed (%s); trying symusic", e)
        try:
            import symusic

            audio = _soundfont_synth(sf, sample_rate).render(score, stereo=True)
            symusic.dump_wav(str(path), audio, sample_rate, use_int16=True)
            return "soundfont"
        except Exception as e:  # fall back rather than fail the request
            log.warning("soundfont rendering failed (%s); using numpy synth", e)
    write_wav(path, render_numpy(score, sample_rate), sample_rate)
    return "numpy"
