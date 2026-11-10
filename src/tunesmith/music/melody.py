"""MIDI loading, melody reduction and bar windows (symusic, tick time).

The composer is trained on single-track excerpts:
- `melody` mode picks the most melody-like non-drum track and makes it monophonic
  (skyline: at each onset keep the highest note, cut notes at the next onset);
- `piano` mode merges all non-drum tracks into one part, capping polyphony.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

from symusic import Note, Score, Tempo, TimeSignature, Track

NoteTuple = tuple[int, int, int, int]  # (time, duration, pitch, velocity) in ticks


@dataclass
class Excerpt:
    """A monophonic or polyphonic single-track excerpt starting at tick 0."""

    tpq: int
    numerator: int
    denominator: int
    qpm: float
    program: int
    notes: list[NoteTuple]
    bars: int

    def to_score(self) -> Score:
        score = Score(self.tpq)
        track = Track(name="melody", program=int(self.program), is_drum=False)
        for t, d, p, v in self.notes:
            track.notes.append(Note(int(t), max(1, int(d)), int(p), int(v)))
        score.tracks.append(track)
        score.time_signatures.append(TimeSignature(0, int(self.numerator), int(self.denominator)))
        score.tempos.append(Tempo(0, float(self.qpm)))
        return score


def load_score(path: str | Path) -> Score:
    return Score(str(path))


def bar_ticks(tpq: int, numerator: int, denominator: int) -> int:
    return int(round(tpq * 4 * numerator / denominator))


def first_meter(score: Score) -> tuple[int, int, int | None]:
    """(numerator, denominator, tick of the next time-signature change or None)."""
    sigs = sorted(score.time_signatures, key=lambda s: s.time)
    if not sigs:
        return 4, 4, None
    first = sigs[0]
    change = next((s.time for s in sigs[1:] if (s.numerator, s.denominator)
                   != (first.numerator, first.denominator)), None)
    return int(first.numerator), int(first.denominator), change


def first_qpm(score: Score, default: float = 120.0) -> float:
    tempos = sorted(score.tempos, key=lambda t: t.time)
    return float(tempos[0].qpm) if tempos else default


def _track_notes(track) -> list[NoteTuple]:
    return [(int(n.time), int(n.duration), int(n.pitch), int(n.velocity)) for n in track.notes]


def _monophony(notes: list[NoteTuple]) -> float:
    if not notes:
        return 0.0
    onsets: dict[int, int] = {}
    for t, _, _, _ in notes:
        onsets[t] = onsets.get(t, 0) + 1
    return sum(1 for c in onsets.values() if c == 1) / len(onsets)


def pick_melody_track(score: Score, min_notes: int = 16):
    """Heuristic lead-track choice: high register, mostly monophonic, enough notes, not bass."""
    best, best_score = None, -math.inf
    for track in score.tracks:
        if track.is_drum or 32 <= int(track.program) <= 39:
            continue
        notes = _track_notes(track)
        if len(notes) < min_notes:
            continue
        mean_pitch = sum(n[2] for n in notes) / len(notes)
        s = mean_pitch / 12.0 + 3.0 * _monophony(notes) + math.log(len(notes))
        if s > best_score:
            best, best_score = track, s
    return best


def skyline(notes: list[NoteTuple]) -> list[NoteTuple]:
    """Monophonic top line: highest note per onset; each note ends at the next onset."""
    notes = sorted(notes, key=lambda n: (n[0], -n[2]))
    out: list[NoteTuple] = []
    for t, d, p, v in notes:
        if out and out[-1][0] == t:
            continue
        if out:
            pt, pd, pp, pv = out[-1]
            if t < pt + pd:
                out[-1] = (pt, t - pt, pp, pv)
        out.append((t, d, p, v))
    return [n for n in out if n[1] > 0]


def merge_piano(score: Score, max_poly: int = 4) -> list[NoteTuple]:
    notes = [n for tr in score.tracks if not tr.is_drum for n in _track_notes(tr)]
    by_onset: dict[int, list[NoteTuple]] = {}
    for n in notes:
        by_onset.setdefault(n[0], []).append(n)
    out = []
    for t in sorted(by_onset):
        out.extend(sorted(by_onset[t], key=lambda n: -n[2])[:max_poly])
    return out


def reduce(score: Score, mode: str = "melody") -> tuple[list[NoteTuple], int] | None:
    """Reduce a multi-track score to one part. Returns (notes, program) or None."""
    if mode == "melody":
        track = pick_melody_track(score)
        if track is None:
            return None
        return skyline(_track_notes(track)), int(track.program)
    if mode == "piano":
        notes = merge_piano(score)
        return (notes, 0) if notes else None
    raise ValueError(f"unknown reduce mode {mode!r}")


def windows(
    score: Score,
    bars_options: list[int],
    mode: str = "melody",
    max_windows: int = 4,
    min_notes: int = 8,
    rng: random.Random | None = None,
) -> list[Excerpt]:
    """Non-overlapping excerpts of `bars` bars, cut at bar lines of the first meter."""
    rng = rng or random.Random(0)
    reduced = reduce(score, mode)
    if reduced is None:
        return []
    notes, program = reduced
    num, den, change = first_meter(score)
    bt = bar_ticks(score.tpq, num, den)
    if bt <= 0:
        return []
    end = change if change is not None else max((n[0] + n[1] for n in notes), default=0)
    total_bars = end // bt
    qpm = first_qpm(score)
    candidates = []
    for bars in bars_options:
        for start in range(0, total_bars - bars + 1, bars):
            candidates.append((start, bars))
    rng.shuffle(candidates)
    taken: list[tuple[int, int]] = []
    out: list[Excerpt] = []
    for start, bars in candidates:
        if len(out) >= max_windows:
            break
        if any(start < s + b and s < start + bars for s, b in taken):
            continue
        lo, hi = start * bt, (start + bars) * bt
        sel = [(t - lo, min(d, hi - t), p, v) for t, d, p, v in notes if lo <= t < hi]
        if len(sel) < min_notes:
            continue
        taken.append((start, bars))
        out.append(Excerpt(score.tpq, num, den, qpm, program, sel, bars))
    return out


def score_notes(score: Score) -> list[NoteTuple]:
    return [n for tr in score.tracks if not tr.is_drum for n in _track_notes(tr)]
