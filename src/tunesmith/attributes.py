"""Music attributes: the interface between text understanding and composition.

Stage 1 (the parser) turns a description into a `MusicAttributes` object. Stage 2 (the
composer) is conditioned on the attributes through one control token per field.
Every field is optional: a missing field becomes a NULL control token, which the
composer learns through attribute dropout.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

# ---------------------------------------------------------------- vocabularies

TONICS = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
MODES = ("major", "minor")
KEYS = tuple(f"{t} {m}" for m in MODES for t in TONICS)
METERS = ("4/4", "3/4", "2/4", "6/8", "other")
BARS = (8, 16, 32)
MOODS = ("happy", "sad", "calm", "energetic", "dark", "epic", "romantic", "dreamy", "playful", "tense")
GENRES = ("classical", "pop", "rock", "electronic", "jazz", "folk", "ambient", "soundtrack", "world", "hiphop")

# name, lower bound (inclusive), upper bound (exclusive), representative bpm
TEMPO_BINS = (
    ("very_slow", 0, 60, 52),
    ("slow", 60, 80, 70),
    ("moderate", 80, 100, 90),
    ("lively", 100, 120, 110),
    ("fast", 120, 140, 130),
    ("very_fast", 140, 170, 155),
    ("presto", 170, 1000, 185),
)

# instrument family -> General MIDI program used for playback
INSTRUMENTS = {
    "piano": 0,
    "electric_piano": 4,
    "mallet": 11,
    "organ": 19,
    "accordion": 21,
    "guitar": 25,
    "electric_guitar": 27,
    "bass": 33,
    "violin": 40,
    "cello": 42,
    "harp": 46,
    "strings": 48,
    "choir": 52,
    "trumpet": 56,
    "brass": 61,
    "sax": 65,
    "woodwind": 71,
    "flute": 73,
    "synth_lead": 80,
    "synth_pad": 89,
    "ethnic": 107,
}

# MidiCaps mood / genre tags (Essentia classifiers) -> our reduced classes.
MIDICAPS_MOOD_MAP = {
    "happy": "happy", "positive": "happy", "upbeat": "happy", "uplifting": "happy",
    "party": "happy", "summer": "happy", "christmas": "happy", "holiday": "happy",
    "hopeful": "happy", "motivational": "happy", "inspiring": "happy",
    "fun": "playful", "funny": "playful", "children": "playful", "game": "playful",
    "sad": "sad", "melancholic": "sad", "emotional": "sad", "ballad": "sad",
    "calm": "calm", "relaxing": "calm", "meditative": "calm", "soft": "calm",
    "nature": "calm", "background": "calm", "slow": "calm",
    "energetic": "energetic", "fast": "energetic", "action": "energetic", "sport": "energetic",
    "powerful": "energetic", "groovy": "energetic", "heavy": "energetic",
    "dark": "dark", "deep": "dark",
    "epic": "epic", "trailer": "epic", "adventure": "epic", "film": "epic", "movie": "epic",
    "documentary": "epic",
    "dramatic": "tense", "drama": "tense",
    "romantic": "romantic", "love": "romantic", "sexy": "romantic",
    "dream": "dreamy", "space": "dreamy", "soundscape": "dreamy",
}

MIDICAPS_GENRE_MAP = {
    "classical": "classical",
    "pop": "pop",
    "rock": "rock", "metal": "rock", "punk": "rock",
    "electronic": "electronic", "dance": "electronic", "techno": "electronic",
    "house": "electronic", "trance": "electronic",
    "jazz": "jazz", "blues": "jazz", "funk": "jazz", "soul": "jazz", "funksoul": "jazz",
    "folk": "folk", "country": "folk",
    "ambient": "ambient", "newage": "ambient", "chillout": "ambient", "lounge": "ambient",
    "soundtrack": "soundtrack", "film": "soundtrack",
    "world": "world", "latin": "world", "reggae": "world",
    "hiphop": "hiphop", "rap": "hiphop",
}

CONTROL_FIELDS = ("key", "meter", "tempo", "instrument", "mood", "genre", "bars")

KeyName = Literal[KEYS]
MeterName = Literal[METERS]
MoodName = Literal[MOODS]
GenreName = Literal[GENRES]
InstrumentName = Literal[tuple(INSTRUMENTS)]
BarsCount = Literal[BARS]

# ---------------------------------------------------------------- helpers

_ENHARMONIC = {"DB": "C#", "D#": "Eb", "GB": "F#", "G#": "Ab", "A#": "Bb",
               "CB": "B", "FB": "E", "E#": "F", "B#": "C"}
_KEY_RE = re.compile(
    r"^\s*([A-Ga-g])\s*(#|b|sharp|flat)?\s*(major|minor|maj|min|m)?\s*$", re.IGNORECASE)


def normalize_key(text: Any) -> Optional[str]:
    """'f# minor', 'F#m', 'Gb major' -> canonical key name; None if not a key."""
    if text is None:
        return None
    s = str(text).strip().replace("_", " ")
    if s in KEYS:
        return s
    m = _KEY_RE.match(s)
    if not m:
        return None
    letter = m.group(1).upper()
    acc_raw = (m.group(2) or "").lower()
    acc = "#" if acc_raw in ("#", "sharp") else ("b" if acc_raw in ("b", "flat") else "")
    mode_raw = m.group(3)
    if mode_raw is None or mode_raw == "M":
        mode = "major"
    elif mode_raw == "m" or mode_raw.lower() in ("minor", "min"):
        mode = "minor"
    else:
        mode = "major"
    name = letter + acc
    name = _ENHARMONIC.get(name.upper(), name)
    key = f"{name} {mode}"
    return key if key in KEYS else None


def key_tonic_pc(key: str) -> int:
    return TONICS.index(key.split()[0])


def transpose_key(key: str, semitones: int) -> str:
    tonic, mode = key.split()
    return f"{TONICS[(TONICS.index(tonic) + semitones) % 12]} {mode}"


def key_pitch_classes(key: str) -> set[int]:
    """Scale pitch classes; minor allows natural and harmonic minor (raised 7th)."""
    tonic = key_tonic_pc(key)
    steps = (0, 2, 4, 5, 7, 9, 11) if key.endswith("major") else (0, 2, 3, 5, 7, 8, 10, 11)
    return {(tonic + s) % 12 for s in steps}


def tempo_bin(bpm: Optional[float]) -> Optional[str]:
    if bpm is None:
        return None
    for name, lo, hi, _ in TEMPO_BINS:
        if lo <= bpm < hi:
            return name
    return TEMPO_BINS[-1][0]


def tempo_from_bin(name: str) -> int:
    for n, _, _, rep in TEMPO_BINS:
        if n == name:
            return rep
    raise KeyError(name)


def normalize_meter(text: Any) -> Optional[str]:
    if text is None:
        return None
    s = str(text).strip().replace(" ", "")
    if s in METERS:  # includes "other"
        return s
    if not re.fullmatch(r"\d+/\d+", s):
        return None
    return s if s in METERS else "other"


def program_to_family(program: int) -> Optional[str]:
    """Map a General MIDI program number (0-127) to an instrument family."""
    p = int(program)
    table = (
        (0, 3, "piano"), (4, 7, "electric_piano"), (8, 15, "mallet"), (16, 20, "organ"),
        (21, 23, "accordion"), (24, 25, "guitar"), (26, 31, "electric_guitar"), (32, 39, "bass"),
        (40, 40, "violin"), (41, 41, "strings"), (42, 43, "cello"), (44, 45, "strings"),
        (46, 46, "harp"), (47, 47, "mallet"), (48, 51, "strings"), (52, 54, "choir"),
        (55, 55, "strings"), (56, 56, "trumpet"), (57, 63, "brass"), (64, 67, "sax"),
        (68, 71, "woodwind"), (72, 79, "flute"), (80, 87, "synth_lead"), (88, 103, "synth_pad"),
        (104, 111, "ethnic"), (112, 119, "mallet"),
    )
    for lo, hi, fam in table:
        if lo <= p <= hi:
            return fam
    return None


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [value]
    return list(value) if isinstance(value, (list, tuple)) else [value]


# ---------------------------------------------------------------- schema


class MusicAttributes(BaseModel):
    """What the user asked for. `lyrics` is reserved and must stay None for now."""

    model_config = ConfigDict(extra="ignore")

    key: Optional[KeyName] = None
    tempo_bpm: Optional[int] = Field(default=None, ge=30, le=260)
    meter: Optional[MeterName] = None
    instrument: Optional[InstrumentName] = None
    mood: Optional[MoodName] = None
    genre: Optional[GenreName] = None
    bars: Optional[BarsCount] = None
    lyrics: Optional[str] = None

    @field_validator("key", mode="before")
    @classmethod
    def _key(cls, v):
        if v is None or v == "":
            return None
        k = normalize_key(v)
        if k is None:
            raise ValueError(f"not a key: {v!r}")
        return k

    @field_validator("meter", mode="before")
    @classmethod
    def _meter(cls, v):
        if v is None or v == "":
            return None
        m = normalize_meter(v)
        if m is None:
            raise ValueError(f"not a meter: {v!r}")
        return m

    @field_validator("tempo_bpm", mode="before")
    @classmethod
    def _tempo(cls, v):
        if v is None or v == "":
            return None
        return int(round(float(v)))

    @field_validator("bars", mode="before")
    @classmethod
    def _bars(cls, v):
        if v is None or v == "":
            return None
        n = int(v)
        return min(BARS, key=lambda b: abs(b - n))

    @field_validator("instrument", "mood", "genre", mode="before")
    @classmethod
    def _lower(cls, v):
        if v is None or v == "":
            return None
        return str(v).strip().lower().replace(" ", "_")

    # -------------------------------------------------------------- conversion

    @classmethod
    def lenient(cls, data: dict[str, Any]) -> "MusicAttributes":
        """Build from untrusted input (e.g. LLM output): drop fields that do not validate."""
        good: dict[str, Any] = {}
        for name in cls.model_fields:
            if name in data and data[name] is not None:
                try:
                    cls(**{name: data[name]})
                    good[name] = data[name]
                except (ValidationError, ValueError, TypeError):
                    pass
        return cls(**good)

    def merged(self, overrides: Optional[dict[str, Any]]) -> "MusicAttributes":
        if not overrides:
            return self.model_copy()
        data = self.model_dump()
        data.update({k: v for k, v in overrides.items() if v not in (None, "")})
        return MusicAttributes.lenient(data)

    def field_value(self, field: str) -> Optional[str]:
        """String value of a control field ('tempo' is binned)."""
        if field == "tempo":
            return tempo_bin(self.tempo_bpm)
        v = getattr(self, field)
        if v is None:
            return None
        return str(v).replace(" ", "_")

    def control_tokens(self) -> list[str]:
        out = []
        for field in CONTROL_FIELDS:
            v = self.field_value(field)
            out.append(f"{field.upper()}_{v if v is not None else 'NULL'}")
        return out

    def summary(self) -> str:
        parts = [f"{k}={v}" for k, v in self.model_dump().items() if v is not None]
        return ", ".join(parts) if parts else "(no attributes)"


def control_vocab() -> dict[str, list[str]]:
    """All control tokens per field, NULL first."""
    values = {
        "key": [k.replace(" ", "_") for k in KEYS],
        "meter": list(METERS),
        "tempo": [b[0] for b in TEMPO_BINS],
        "instrument": list(INSTRUMENTS),
        "mood": list(MOODS),
        "genre": list(GENRES),
        "bars": [str(b) for b in BARS],
    }
    return {f: [f"{f.upper()}_NULL"] + [f"{f.upper()}_{v}" for v in values[f]] for f in CONTROL_FIELDS}


def from_midicaps(row: dict[str, Any]) -> MusicAttributes:
    """File-level attributes from a MidiCaps metadata row (bars are set per window later)."""
    key = normalize_key(row.get("key"))
    try:
        tempo = int(round(float(row.get("tempo")))) if row.get("tempo") not in (None, "") else None
        if tempo is not None and not 30 <= tempo <= 260:
            tempo = None
    except (TypeError, ValueError):
        tempo = None
    meter = normalize_meter(row.get("time_signature"))
    instrument = None
    for prog in _as_list(row.get("instrument_numbers_sorted")):
        try:
            prog = int(prog)
        except (TypeError, ValueError):
            continue
        if prog < 128:
            instrument = program_to_family(prog)
            if instrument:
                break
    mood = next((MIDICAPS_MOOD_MAP[m] for m in map(str, _as_list(row.get("mood")))
                 if m.lower() in MIDICAPS_MOOD_MAP), None)
    genre = next((MIDICAPS_GENRE_MAP[g.lower().replace("/", "").replace(" ", "")]
                  for g in map(str, _as_list(row.get("genre")))
                  if g.lower().replace("/", "").replace(" ", "") in MIDICAPS_GENRE_MAP), None)
    return MusicAttributes(key=key, tempo_bpm=tempo, meter=meter, instrument=instrument,
                           mood=mood, genre=genre)
