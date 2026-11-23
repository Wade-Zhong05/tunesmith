"""Multi-instrument ("ensemble") excerpts for the ensemble composer.

From a multi-track MIDI file:
1. group the non-drum tracks by instrument family (`program_to_family`) and merge tracks
   of the same family, removing exact duplicates;
2. cap the polyphony of each family at every onset (highest notes kept; lowest for bass);
3. keep the `max_families` families with the most notes, plus the drum tracks;
4. cut non-overlapping windows aligned to bar lines of the first time signature.

Each family is played by its representative GM program (`attributes.INSTRUMENTS`), so the
tokenizer's Program tokens map one-to-one onto families.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-22. See ATTRIBUTION.md.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from symusic import Note, Score, Tempo, TimeSignature, Track

from ..attributes import INSTRUMENTS, program_to_family
from .melody import NoteTuple, bar_ticks, first_meter, first_qpm


@dataclass
class EnsembleExcerpt:
    tpq: int
    numerator: int
    denominator: int
    qpm: float
    parts: dict[str, list[NoteTuple]]          # family -> notes, starting at tick 0
    drums: list[NoteTuple] = field(default_factory=list)
    bars: int = 8

    @property
    def families(self) -> list[str]:
        """Families present in this excerpt, most notes first."""
        return [f for f, notes in sorted(self.parts.items(), key=lambda kv: -len(kv[1])) if notes]

    @property
    def pitched_notes(self) -> list[NoteTuple]:
        return [n for notes in self.parts.values() for n in notes]

    def to_score(self) -> Score:
        score = Score(self.tpq)
        for fam in self.families:
            track = Track(name=fam, program=INSTRUMENTS[fam], is_drum=False)
            for t, d, p, v in sorted(self.parts[fam]):
                track.notes.append(Note(int(t), max(1, int(d)), int(p), int(v)))
            score.tracks.append(track)
        if self.drums:
            track = Track(name="drums", program=0, is_drum=True)
            for t, d, p, v in sorted(self.drums):
                track.notes.append(Note(int(t), max(1, int(d)), int(p), int(v)))
            score.tracks.append(track)
        score.time_signatures.append(TimeSignature(0, int(self.numerator), int(self.denominator)))
        score.tempos.append(Tempo(0, float(self.qpm)))
        return score


def cap_polyphony(notes: list[NoteTuple], max_poly: int, keep_low: bool = False) -> list[NoteTuple]:
    """At most `max_poly` notes per onset; duplicates (same onset and pitch) removed."""
    by_onset: dict[int, dict[int, NoteTuple]] = {}
    for n in notes:
        by_onset.setdefault(n[0], {}).setdefault(n[2], n)
    out: list[NoteTuple] = []
    for t in sorted(by_onset):
        chord = sorted(by_onset[t].values(), key=lambda n: n[2], reverse=not keep_low)
        out.extend(chord[:max_poly])
    return out


def group_families(score: Score, max_poly: int = 6) -> tuple[dict[str, list[NoteTuple]], list[NoteTuple]]:
    """(family -> merged notes, drum notes) for a whole score."""
    fams: dict[str, list[NoteTuple]] = {}
    drums: list[NoteTuple] = []
    for track in score.tracks:
        notes = [(int(n.time), int(n.duration), int(n.pitch), int(n.velocity)) for n in track.notes]
        if track.is_drum:
            drums.extend(notes)
            continue
        fam = program_to_family(int(track.program))
        if fam is not None:
            fams.setdefault(fam, []).extend(notes)
    fams = {f: cap_polyphony(ns, max_poly, keep_low=(f == "bass")) for f, ns in fams.items()}
    drums = cap_polyphony(drums, max_poly)
    return fams, drums


def ensemble_windows(
    score: Score,
    bars_options: list[int],
    max_families: int = 4,
    max_windows: int = 6,
    min_notes: int = 16,
    min_family_notes: int = 8,
    max_poly: int = 6,
    rng: random.Random | None = None,
) -> list[EnsembleExcerpt]:
    rng = rng or random.Random(0)
    fams, drums = group_families(score, max_poly)
    top = [f for f, ns in sorted(fams.items(), key=lambda kv: -len(kv[1])) if len(ns) >= min_family_notes][:max_families]
    if not top:
        return []
    num, den, change = first_meter(score)
    bt = bar_ticks(score.tpq, num, den)
    if bt <= 0:
        return []
    all_notes = [n for f in top for n in fams[f]] + drums
    end = change if change is not None else max(n[0] + n[1] for n in all_notes)
    total_bars = end // bt
    qpm = first_qpm(score)
    candidates = [(start, bars) for bars in bars_options for start in range(0, total_bars - bars + 1, bars)]
    rng.shuffle(candidates)
    taken: list[tuple[int, int]] = []
    out: list[EnsembleExcerpt] = []

    def cut(notes: list[NoteTuple], lo: int, hi: int) -> list[NoteTuple]:
        return [(t - lo, min(d, hi - t), p, v) for t, d, p, v in notes if lo <= t < hi]

    for start, bars in candidates:
        if len(out) >= max_windows:
            break
        if any(start < s + b and s < start + bars for s, b in taken):
            continue
        lo, hi = start * bt, (start + bars) * bt
        parts = {f: cut(fams[f], lo, hi) for f in top}
        parts = {f: ns for f, ns in parts.items() if ns}
        if sum(len(ns) for ns in parts.values()) < min_notes:
            continue
        taken.append((start, bars))
        out.append(EnsembleExcerpt(score.tpq, num, den, qpm, parts, cut(drums, lo, hi), bars))
    return out
