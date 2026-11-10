"""Music metrics for generated pieces, measured against the requested attributes.

Control (did it follow the request?): key match (Krumhansl-Schmuckler), key match up to
the relative key, share of notes in the requested scale, meter match, bar-count match.
Musical statistics (compared with real data): note density, pitch range, pitch-class
entropy, repetition (share of 4-note interval/duration patterns that occur again),
empty-bar rate.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Optional

import numpy as np
from symusic import Score

from ..attributes import MusicAttributes, key_pitch_classes, normalize_meter
from ..music.keyfinding import detect_key, relative_key
from ..music.melody import bar_ticks, score_notes


def repetition_ratio(notes, n: int = 4) -> float:
    seq = [(b[2] - a[2], b[1]) for a, b in zip(notes, notes[1:])]
    grams = [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]
    if not grams:
        return 0.0
    counts = Counter(grams)
    return sum(c for c in counts.values() if c > 1) / len(grams)


def music_metrics(score: Score, attrs: MusicAttributes, n_bars: Optional[int] = None) -> dict[str, Any]:
    """`n_bars`: bars the model generated (Bar tokens); default: from the note extent."""
    notes = sorted(score_notes(score))
    m: dict[str, Any] = {"n_notes": len(notes), "valid": len(notes) > 0}
    ts = score.time_signatures[0] if len(score.time_signatures) else None
    num, den = (int(ts.numerator), int(ts.denominator)) if ts else (4, 4)
    bt = bar_ticks(score.tpq, num, den)
    end = max((t + d for t, d, _, _ in notes), default=0)
    if n_bars is None:
        n_bars = max(1, math.ceil(end / bt)) if notes else 0
    m["bars"] = n_bars
    m["meter"] = normalize_meter(f"{num}/{den}")
    if not notes:
        return m
    pitches = np.array([p for _, _, p, _ in notes])
    m["note_density"] = len(notes) / n_bars
    m["pitch_range"] = int(pitches.max() - pitches.min())
    pc = np.bincount(pitches % 12, minlength=12) / len(pitches)
    m["pitch_class_entropy"] = float(-(pc[pc > 0] * np.log2(pc[pc > 0])).sum())
    m["repetition"] = repetition_ratio(notes)
    used_bars = {t // bt for t, _, _, _ in notes}
    m["empty_bar_rate"] = 1 - len(used_bars) / n_bars
    key, corr = detect_key(notes)
    m["detected_key"] = key
    if attrs.key:
        m["key_match"] = key == attrs.key
        m["key_match_relative"] = key in (attrs.key, relative_key(attrs.key))
        scale = key_pitch_classes(attrs.key)
        m["in_scale"] = float(np.mean([p % 12 in scale for p in pitches]))
    if attrs.meter and attrs.meter != "other":
        m["meter_match"] = m["meter"] == attrs.meter
    if attrs.bars:
        m["bars_match"] = n_bars == attrs.bars
    return m


def summarize(rows: list[dict[str, Any]]) -> dict[str, Optional[float]]:
    keys = ["valid", "key_match", "key_match_relative", "in_scale", "meter_match", "bars_match",
            "note_density", "pitch_range", "pitch_class_entropy", "repetition", "empty_bar_rate", "n_notes"]
    out: dict[str, Optional[float]] = {"n": len(rows)}
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r and r[k] is not None]
        out[k] = float(np.mean(vals)) if vals else None
    return out
