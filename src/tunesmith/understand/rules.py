"""Rule-based parser (baseline): keyword and regex matching over English text.

Fields are matched in a fixed order and each match is blanked out of the text, so a
phrase is used once (e.g. "very slow" sets the tempo once, and "slow" inside it is not
matched again).

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Optional

from ..attributes import MusicAttributes, normalize_key, normalize_meter, tempo_from_bin
from ..paths import CONFIGS, load_yaml

_METER = re.compile(r"\b(\d{1,2})\s*/\s*(\d{1,2})\b")
_KEY = re.compile(r"\b([A-Ga-g])\s*(#|b|-?sharp|-?flat)?\s*(major|minor)\b", re.IGNORECASE)


@lru_cache(maxsize=1)
def load_keywords_raw() -> dict[str, Any]:
    return load_yaml(CONFIGS / "keywords.yaml")


@lru_cache(maxsize=4)
def load_keywords(path: Optional[str] = None) -> dict[str, dict[str, list[str]]]:
    data = load_yaml(path) if path else load_keywords_raw()
    return {field: {str(k): [str(w) for w in v] for k, v in table.items()}
            for field, table in data.items() if field != "patterns"}


@lru_cache(maxsize=1)
def _patterns() -> dict[str, re.Pattern]:
    pats = load_keywords_raw().get("patterns", {})

    def alt(words):
        return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))

    return {
        "bpm": re.compile(r"(\d{2,3})\s*(?:" + alt(pats.get("bpm_units", ["bpm"])) + r")\b", re.IGNORECASE),
        "bars": re.compile(r"(\d{1,2})\s*(?:" + alt(pats.get("bar_units", ["bars"])) + r")\b", re.IGNORECASE),
    }


@lru_cache(maxsize=4096)
def _compiled(word: str) -> re.Pattern:
    return re.compile(r"(?<![A-Za-z])" + re.escape(word) + r"(?![A-Za-z])", re.IGNORECASE)


def find_first(text: str, table: dict[str, list[str]]) -> tuple[Optional[str], Optional[tuple[int, int]]]:
    """Earliest keyword occurrence in the text (the longest keyword wins at the same position)."""
    best: tuple[int, int, str] | None = None  # (start, -length, value)
    best_span = None
    for value, words in table.items():
        for w in words:
            m = _compiled(w).search(text)
            if m and (best is None or (m.start(), -len(w)) < best[:2]):
                best, best_span = (m.start(), -len(w), value), m.span()
    return (best[2], best_span) if best else (None, None)


def _blank(text: str, span: tuple[int, int]) -> str:
    return text[: span[0]] + " " * (span[1] - span[0]) + text[span[1]:]


def _key(text: str, kw: dict) -> tuple[Optional[str], str]:
    m = _KEY.search(text)
    if m:
        letter, acc, mode = m.group(1), (m.group(2) or ""), m.group(3).lower()
        acc = {"sharp": "#", "-sharp": "#", "flat": "b", "-flat": "b"}.get(acc.lower(), acc)
        key = normalize_key(f"{letter.upper()}{acc} {mode}")
        if key:
            return key, _blank(text, m.span())
    mode, span = find_first(text, kw.get("mode", {}))
    if mode:  # "a minor-key tune" without a tonic -> default tonic
        return ("A minor" if mode == "minor" else "C major"), _blank(text, span)
    return None, text


def parse(text: str, keywords: Optional[dict] = None) -> MusicAttributes:
    kw = keywords or load_keywords()
    pats = _patterns()
    t = text
    data: dict[str, Any] = {}

    data["key"], t = _key(t, kw)
    m = pats["bpm"].search(t)
    if m:
        data["tempo_bpm"] = int(m.group(1))
        t = _blank(t, m.span())
    m = pats["bars"].search(t)
    if m:
        data["bars"] = int(m.group(1))
        t = _blank(t, m.span())
    m = _METER.search(t)
    if m:
        data["meter"] = normalize_meter(f"{m.group(1)}/{m.group(2)}")
        t = _blank(t, m.span())

    for field in ("instrument", "mood", "genre"):
        value, span = find_first(t, kw[field])
        if value:
            data[field] = value
            t = _blank(t, span)
    if "meter" not in data:
        value, span = find_first(t, kw["meter"])
        if value:
            data["meter"] = value
            t = _blank(t, span)
    if "bars" not in data:
        value, span = find_first(t, kw["bars"])
        if value:
            data["bars"] = int(value)
            t = _blank(t, span)
    if "tempo_bpm" not in data:
        value, span = find_first(t, kw["tempo"])
        if value:
            data["tempo_bpm"] = tempo_from_bin(value)
    return MusicAttributes.lenient(data)


class RuleParser:
    name = "rules"

    def parse(self, text: str) -> MusicAttributes:
        return parse(text)


def mentions(text: str, field: str, value: Any, keywords: Optional[dict] = None) -> bool:
    """Does the text mention this field value? Used to clean parser training labels."""
    kw = keywords or load_keywords()
    if value is None:
        return False
    if field == "key":
        found, _ = _key(text, {"mode": {}})
        return found == value
    if field == "tempo_bpm":
        return bool(_patterns()["bpm"].search(text)) or find_first(text, kw["tempo"])[0] is not None
    if field == "meter":
        return str(value) in text or find_first(text, {str(value): kw["meter"].get(str(value), [])})[0] is not None
    if field in ("instrument", "mood", "genre"):
        return find_first(text, {str(value): kw[field].get(str(value), [])})[0] is not None
    if field == "bars":
        m = _patterns()["bars"].search(text)
        return bool(m and int(m.group(1)) == int(value))
    return False
