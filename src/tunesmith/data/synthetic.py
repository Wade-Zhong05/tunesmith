"""Synthetic toy dataset in MidiCaps format, for smoke tests and unit tests.

    pixi run synth-data --n 80 --out outputs/smoke/raw

Each file has a lead melody (motif A A' B A in a random key and meter), a bass track
and a drum track, so melody extraction is exercised. Not used for real training.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from symusic import Note, Score, Tempo, TimeSignature, Track

from ..attributes import INSTRUMENTS, KEYS, key_pitch_classes, key_tonic_pc
from ..paths import resolve, write_jsonl

RHYTHMS = {  # bar patterns in quarter notes
    "4/4": [[1, 1, 1, 1], [0.5, 0.5, 1, 2], [1.5, 0.5, 1, 1], [2, 1, 1]],
    "3/4": [[1, 1, 1], [2, 1], [1.5, 0.5, 1]],
    "2/4": [[1, 1], [0.5, 0.5, 1], [0.5, 0.5, 0.5, 0.5]],
    "6/8": [[1.5, 1.5], [1, 0.5, 1, 0.5], [0.5, 0.5, 0.5, 1.5]],
}
BAR_QUARTERS = {"4/4": 4, "3/4": 3, "2/4": 2, "6/8": 3}
MOODS = ["happy", "sad", "calm", "energetic", "dark", "epic", "romantic", "dream", "fun", "dramatic"]
GENRES = ["classical", "pop", "rock", "electronic", "jazz", "folk", "ambient", "soundtrack", "world", "hiphop"]
GM_NAMES = {0: "Acoustic Grand Piano", 4: "Electric Piano 1", 11: "Vibraphone", 19: "Church Organ",
            21: "Accordion", 25: "Acoustic Guitar (steel)", 27: "Electric Guitar (clean)", 33: "Electric Bass (finger)",
            40: "Violin", 42: "Cello", 46: "Orchestral Harp", 48: "String Ensemble 1", 52: "Choir Aahs",
            56: "Trumpet", 61: "Brass Section", 65: "Alto Sax", 71: "Clarinet", 73: "Flute",
            80: "Lead 1 (square)", 89: "Pad 2 (warm)", 107: "Koto"}


def _scale(key: str, low: int = 60, high: int = 84) -> list[int]:
    pcs = key_pitch_classes(key)
    if key.endswith("minor"):
        pcs.discard((key_tonic_pc(key) + 11) % 12)  # natural minor for melodies
    return [p for p in range(low, high + 1) if p % 12 in pcs]


def make_piece(rng: random.Random, tpq: int = 480) -> tuple[Score, dict]:
    key = rng.choice(KEYS)
    meter = rng.choice(list(RHYTHMS))
    tempo = rng.randint(60, 160)
    program = rng.choice(list(INSTRUMENTS.values()))
    scale = _scale(key)
    tonic_idx = next(i for i, p in enumerate(scale) if p % 12 == key_tonic_pc(key))

    def phrase(start_idx: int) -> list[tuple[float, int]]:
        notes, idx = [], start_idx
        for _ in range(2):
            for dur in rng.choice(RHYTHMS[meter]):
                notes.append((dur, scale[idx]))
                idx = max(0, min(len(scale) - 1, idx + rng.choice([-2, -1, -1, 0, 1, 1, 2])))
        return notes

    a = phrase(tonic_idx)
    a2 = [(d, scale[max(0, min(len(scale) - 1, scale.index(p) + (1 if i == len(a) - 1 else 0)))])
          for i, (d, p) in enumerate(a)]
    b = phrase(min(len(scale) - 1, tonic_idx + 4))
    melody = a + a2 + b + a
    melody = melody + melody  # 16 bars

    score = Score(tpq)
    lead = Track(name="lead", program=program, is_drum=False)
    t = 0
    for dur, pitch in melody:
        ticks = int(dur * tpq)
        lead.notes.append(Note(t, int(ticks * 0.9), pitch, rng.randint(70, 100)))
        t += ticks
    total = t
    bass = Track(name="bass", program=33, is_drum=False)
    bar = int(BAR_QUARTERS[meter] * tpq)
    root = 36 + key_tonic_pc(key)
    for start in range(0, total, bar):
        bass.notes.append(Note(start, bar, root + rng.choice([0, 5, 7]), 70))
    drums = Track(name="drums", program=0, is_drum=True)
    for start in range(0, total, tpq):
        drums.notes.append(Note(start, tpq // 4, 36 if (start // tpq) % 2 == 0 else 38, 80))
    for tr in (lead, bass, drums):
        score.tracks.append(tr)
    num, den = (6, 8) if meter == "6/8" else (int(meter[0]), 4)
    score.time_signatures.append(TimeSignature(0, num, den))
    score.tempos.append(Tempo(0, float(tempo)))

    mood, genre = rng.choice(MOODS), rng.choice(GENRES)
    caption = (f"A {mood} {genre} piece in {key} with a {meter} time signature at {tempo} bpm, "
               f"featuring {GM_NAMES.get(program, 'piano').lower()}, bass and drums.")
    row = {
        "caption": caption, "genre": [genre], "genre_prob": [0.5], "mood": [mood], "mood_prob": [0.5],
        "key": key, "time_signature": meter, "tempo": str(tempo), "tempo_word": "",
        "instrument_summary": [GM_NAMES.get(program, "Piano"), "Electric Bass (finger)", "Drums"],
        "instrument_numbers_sorted": [program, 33, 128],
    }
    return score, row


def generate_dataset(out_dir: Path, n: int = 80, seed: int = 0, test_every: int = 10) -> Path:
    rng = random.Random(seed)
    (out_dir / "synth").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        score, row = make_piece(rng)
        location = f"synth/{i:04d}.mid"
        score.dump_midi(str(out_dir / location))
        rows.append({"location": location, **row, "test_set": i % test_every == 0})
    path = out_dir / "metadata.jsonl"
    write_jsonl(rows, path)
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/smoke/raw")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    path = generate_dataset(resolve(args.out), args.n, args.seed)
    print(f"wrote {args.n} pieces, metadata at {path}")


if __name__ == "__main__":
    main()
