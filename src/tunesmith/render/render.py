"""Render a symusic Score to WAV.

Preferred: symusic's built-in SoundFont synthesizer (Prestosynth) with MuseScore General
(`pixi run fetch-soundfont`). Fallback: the numpy additive synthesizer (always available).

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from symusic import Score

from ..paths import resolve
from .synth import preset_for_program, render_notes, write_wav

log = logging.getLogger(__name__)
_SYNTHS: dict[tuple[str, int], object] = {}


def score_to_seconds(score: Score) -> list[tuple[float, float, int, int]]:
    qpm = float(score.tempos[0].qpm) if len(score.tempos) else 120.0
    sec_per_tick = 60.0 / qpm / score.tpq
    return [(n.time * sec_per_tick, n.duration * sec_per_tick, int(n.pitch), int(n.velocity))
            for tr in score.tracks if not tr.is_drum for n in tr.notes]


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
            import symusic

            audio = _soundfont_synth(sf, sample_rate).render(score, stereo=True)
            symusic.dump_wav(str(path), audio, sample_rate, use_int16=True)
            return "soundfont"
        except Exception as e:  # fall back rather than fail the request
            log.warning("soundfont rendering failed (%s); using numpy synth", e)
    program = int(score.tracks[0].program) if len(score.tracks) else 0
    audio = render_notes(score_to_seconds(score), sample_rate, preset_for_program(program))
    write_wav(path, audio, sample_rate)
    return "numpy"
