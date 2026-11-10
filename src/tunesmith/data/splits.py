"""Deterministic file-level splits shared by the composer data and the parser data.

MidiCaps marks its own test files (`test_set`); validation files are picked from the
rest by a hash of the path, so a file never contributes windows to two splits.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

import hashlib


def stable_fraction(text: str) -> float:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def split_of(location: str, test_flag, val_fraction: float = 0.02) -> str:
    if str(test_flag).strip().lower() in ("true", "1", "yes"):
        return "test"
    return "val" if stable_fraction(location) < val_fraction else "train"
