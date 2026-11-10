"""Key detection with the Krumhansl-Schmuckler algorithm (Krumhansl & Kessler profiles).

Used as an evaluation metric (did the composer follow the requested key?) and,
optionally, to relabel training windows (`key_source: detected` in configs/data.yaml).

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

import numpy as np

from ..attributes import KEYS, TONICS

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def pitch_class_histogram(notes, weight_by_duration: bool = True) -> np.ndarray:
    """notes: iterable of (time, duration, pitch, velocity)."""
    hist = np.zeros(12)
    for _, d, p, _ in notes:
        hist[int(p) % 12] += float(d) if weight_by_duration else 1.0
    return hist


def key_scores(hist: np.ndarray) -> dict[str, float]:
    """Pearson correlation between the histogram and each of the 24 rotated profiles."""
    out = {}
    if hist.sum() <= 0 or np.allclose(hist, hist[0]):
        return {k: 0.0 for k in KEYS}
    for i, tonic in enumerate(TONICS):
        for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            rotated = np.roll(profile, i)
            out[f"{tonic} {mode}"] = float(np.corrcoef(hist, rotated)[0, 1])
    return out


def detect_key(notes) -> tuple[str | None, float]:
    """Best key and its correlation; (None, 0.0) for empty or flat input."""
    scores = key_scores(pitch_class_histogram(notes))
    best = max(scores, key=scores.get)
    return (best, scores[best]) if scores[best] > 0 else (None, 0.0)


def relative_key(key: str) -> str:
    tonic, mode = key.split()
    i = TONICS.index(tonic)
    return f"{TONICS[(i + 9) % 12]} minor" if mode == "major" else f"{TONICS[(i + 3) % 12]} major"
