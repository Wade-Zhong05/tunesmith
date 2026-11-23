"""Music metrics for generated pieces, measured against the requested attributes.

Control (did it follow the request?): key match (Krumhansl-Schmuckler), key match up to
the relative key, share of notes in the requested scale, meter match, bar-count match.
Musical statistics (compared with real data): note density, pitch range, pitch-class
entropy, repetition (share of 4-note interval/duration patterns that occur again),
empty-bar rate.
Ensemble mode adds: instrument precision / recall / F1 (requested vs present families),
drums match, number of tracks, polyphony (pitched notes per onset) and track balance
(share of notes in the busiest track). Key and pitch statistics use pitched notes only;
repetition uses the top line (skyline) so it stays comparable with melody mode.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10; ensemble metrics
added 2026-11-22. See ATTRIBUTION.md.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Optional

import numpy as np
from symusic import Score

from ..attributes import MusicAttributes, key_pitch_classes, normalize_meter, program_to_family
from ..music.keyfinding import detect_key, relative_key
from ..music.melody import bar_ticks, score_notes, skyline


def repetition_ratio(notes, n: int = 4) -> float:
    seq = [(b[2] - a[2], b[1]) for a, b in zip(notes, notes[1:])]
    grams = [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]
    if not grams:
        return 0.0
    counts = Counter(grams)
    return sum(c for c in counts.values() if c > 1) / len(grams)


def ensemble_metrics(score: Score, attrs: MusicAttributes) -> dict[str, Any]:
    """Instrumentation metrics for a multi-track score."""
    present: list[str] = []
    counts: list[int] = []
    drums_present = False
    for track in score.tracks:
        if not len(track.notes):
            continue
        if track.is_drum:
            drums_present = True
            continue
        fam = program_to_family(int(track.program))
        if fam and fam not in present:
            present.append(fam)
        counts.append(len(track.notes))
    m: dict[str, Any] = {"n_tracks": len(present), "drums_present": drums_present,
                         "families": ",".join(present)}
    if counts:
        m["track_balance"] = max(counts) / sum(counts)
    if attrs.instruments:
        req, got = set(attrs.instruments), set(present)
        hit = len(req & got)
        m["instrument_precision"] = hit / len(got) if got else 0.0
        m["instrument_recall"] = hit / len(req)
        p, r = m["instrument_precision"], m["instrument_recall"]
        m["instrument_f1"] = 2 * p * r / (p + r) if p + r else 0.0
    if attrs.drums is not None:
        m["drums_match"] = drums_present == attrs.drums
    return m


def music_metrics(score: Score, attrs: MusicAttributes, n_bars: Optional[int] = None,
                  ensemble: bool = False) -> dict[str, Any]:
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
    if ensemble:
        m.update(ensemble_metrics(score, attrs))
        onsets = Counter(t for t, _, _, _ in notes)
        if onsets:
            m["polyphony"] = len(notes) / len(onsets)
    if not notes:
        return m
    pitches = np.array([p for _, _, p, _ in notes])
    m["note_density"] = len(notes) / n_bars
    m["pitch_range"] = int(pitches.max() - pitches.min())
    pc = np.bincount(pitches % 12, minlength=12) / len(pitches)
    m["pitch_class_entropy"] = float(-(pc[pc > 0] * np.log2(pc[pc > 0])).sum())
    m["repetition"] = repetition_ratio(skyline(notes) if ensemble else notes)
    all_notes = notes
    if ensemble:  # a bar with only drums is not empty
        all_notes = notes + [(int(n.time), 0, 0, 0) for tr in score.tracks if tr.is_drum for n in tr.notes]
    used_bars = {t // bt for t, _, _, _ in all_notes}
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
            "note_density", "pitch_range", "pitch_class_entropy", "repetition", "empty_bar_rate", "n_notes",
            "instrument_precision", "instrument_recall", "instrument_f1", "drums_match", "n_tracks",
            "polyphony", "track_balance"]
    out: dict[str, Optional[float]] = {"n": len(rows)}
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r and r[k] is not None]
        out[k] = float(np.mean(vals)) if vals else None
    return out
