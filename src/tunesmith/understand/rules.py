"""Rule-based parser (baseline): keyword and regex matching over English text.

Fields are matched in a fixed order and each match is blanked out of the text, so a
phrase is used once (e.g. "very slow" sets the tempo once, and "slow" inside it is not
matched again).

Before matching, words that are not keywords are repaired when the repair is unambiguous
(`normalize_text`):
- stretched letters ("saaddd", "sloooow", "veeery") are squeezed back to a keyword ("sad",
  "slow", "very");
- a single missing, extra or swapped letter in a longer word ("guita", "violn", "pinao")
  is corrected to the only keyword that close. Substituted letters are not corrected,
  because those often make another real word ("horse" is not "house"), and neither are
  word forms such as plurals ("guitars", "voice" vs "voices", "orchestral"): whether
  those count is a question for the keyword tables, not for spelling repair.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

import re
from functools import lru_cache
from itertools import groupby, product
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


def find_all(text: str, table: dict[str, list[str]], limit: int = 4) -> tuple[list[str], str]:
    """Distinct values in order of first mention (longest keyword first at each spot), at most
    `limit`; returns (values, text with every match blanked)."""
    found: list[str] = []
    while len(found) < limit:
        value, span = find_first(text, table)
        if value is None:
            break
        text = _blank(text, span)
        if value not in found:
            found.append(value)
    return found, text


_WORD = re.compile(r"[A-Za-z]+")
_MIN_FUZZY = 5  # only words this long are corrected for a missing, extra or swapped letter


@lru_cache(maxsize=1)
def _vocabulary() -> frozenset[str]:
    """Every word that occurs in a keyword."""
    words: set[str] = set()
    for table in load_keywords().values():
        for keywords in table.values():
            for k in keywords:
                words.update(w.lower() for w in _WORD.findall(k))
    return frozenset(words)


def _squeezed(word: str) -> list[str]:
    """Forms of a word with every run of a repeated letter kept once or twice."""
    runs = [(ch, len(list(g))) for ch, g in groupby(word)]
    if all(n == 1 for _, n in runs) or sum(n > 1 for _, n in runs) > 6:
        return []
    return ["".join(p) for p in product(*[[ch] if n == 1 else [ch, ch * 2] for ch, n in runs])]


def _one_edit(a: str, b: str) -> bool:
    """b is a with one letter inserted or deleted, or two adjacent letters swapped."""
    if abs(len(a) - len(b)) == 1:
        short, long_ = (a, b) if len(a) < len(b) else (b, a)
        return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))
    if len(a) == len(b) and a != b:
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        return len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]
    return False


# word = keyword + suffix: an inflected form of the keyword ("guitars", "dancer", "orchestral")
_LONGER_FORMS = ("s", "es", "y", "l", "al", "d", "ed", "r", "er", "ly")
# keyword = word + suffix: the keyword is a plural or adjective of a real word ("voice" -> "voices",
# "space" -> "spacey"); a dropped final letter of other kinds is a typo ("guita" -> "guitar")
_SHORTER_FORMS = ("s", "es", "y", "al")


def _word_form(word: str, keyword: str) -> bool:
    """Is `word` another form of `keyword` (or the reverse) rather than a misspelling?"""
    if len(word) > len(keyword):
        return any(word == keyword + suf for suf in _LONGER_FORMS)
    return any(keyword == word + suf for suf in _SHORTER_FORMS)


@lru_cache(maxsize=4096)
def _repair(word: str) -> str:
    vocab = _vocabulary()
    if word in vocab:
        return word
    forms = [f for f in _squeezed(word) if f in vocab]
    if forms:
        return min(forms, key=len)
    if len(word) >= _MIN_FUZZY:
        near = [v for v in vocab if len(v) >= _MIN_FUZZY and _one_edit(word, v) and not _word_form(word, v)]
        if len(near) == 1:
            return near[0]
    return word


def normalize_text(text: str) -> str:
    """Repair stretched and slightly misspelt keywords; other words are left alone."""
    return _WORD.sub(lambda m: m.group(0) if m.group(0).lower() in _vocabulary() else _repair(m.group(0).lower()),
                     text)


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
    t = normalize_text(text)
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

    instruments, t = find_all(t, kw["instrument"])
    if instruments:
        data["instruments"] = instruments
        data["instrument"] = instruments[0]
    if "drums" in kw:
        for flag in ("no", "yes"):  # negated phrases first ("no drums" contains "drums")
            value, span = find_first(t, {flag: kw["drums"][flag]})
            if value:
                data["drums"] = flag == "yes"
                t = _blank(t, span)
                break
    for field in ("mood", "genre"):
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
