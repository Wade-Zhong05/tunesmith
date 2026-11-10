"""Build the text -> attributes dataset for the parser (data/parser/{train,val,test}.jsonl).

Two sources:
1. MidiCaps captions with *cleaned* labels: a metadata attribute is kept as the target
   only if the caption actually mentions it (otherwise the target is null), so the model
   is never trained to hallucinate attributes the text does not contain.
2. Template queries (configs/parser_templates.yaml), which look more like what a user
   types than MidiCaps' long captions do.

Splits follow the composer's file-level split (MidiCaps test files -> test).

    pixi run make-parser-data
    pixi run make-parser-data --metadata outputs/smoke/raw/metadata.jsonl --out outputs/smoke/parser --midicaps-train 50

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter
from typing import Any, Optional

from ..attributes import BARS, INSTRUMENTS, KEYS, METERS, MOODS, GENRES, TEMPO_BINS, from_midicaps
from ..data.splits import split_of
from ..paths import CONFIGS, load_yaml, read_jsonl, resolve, save_json, write_jsonl
from .prompts import FIELDS
from .rules import find_first, load_keywords, mentions


def clean_target(caption: str, row: dict[str, Any], kw: dict) -> dict[str, Any]:
    attrs = from_midicaps(row).model_dump()
    target: dict[str, Any] = {f: None for f in FIELDS}
    for field in ("key", "tempo_bpm", "meter", "mood", "genre"):
        if attrs.get(field) is not None and mentions(caption, field, attrs[field], kw):
            target[field] = attrs[field]
    # instrument: the first instrument the caption names (the lead, from the user's point of view)
    inst, _ = find_first(caption, kw["instrument"])
    target["instrument"] = inst
    return target


def midicaps_rows(metadata: str, val_fraction: float, limits: dict[str, int], seed: int, kw: dict):
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for row in read_jsonl(metadata):
        split = split_of(row["location"], row.get("test_set"), val_fraction)
        cap = row.get("caption") or ""
        if cap:
            buckets[split].append({"text": cap, "row": row})
    out = {}
    for split, items in buckets.items():
        rng.shuffle(items)
        out[split] = [{"text": it["text"], "target": clean_target(it["text"], it["row"], kw), "source": "midicaps"}
                      for it in items[: limits[split]]]
    return out


def template_rows(n: int, rng: random.Random, kw: dict, templates: list[str]) -> list[dict]:
    rows = []
    slot_re = re.compile(r"\{(\w+)\}")
    for _ in range(n):
        tpl = rng.choice(templates)
        target: dict[str, Any] = {f: None for f in FIELDS}
        fill: dict[str, str] = {}
        for slot in slot_re.findall(tpl):
            if slot == "mood":
                v = rng.choice(MOODS)
                target["mood"], fill["mood"] = v, rng.choice(kw["mood"][v])
            elif slot == "instrument":
                v = rng.choice(list(INSTRUMENTS))
                target["instrument"], fill["instrument"] = v, rng.choice(kw["instrument"][v])
            elif slot == "genre":
                v = rng.choice(GENRES)
                target["genre"], fill["genre"] = v, rng.choice(kw["genre"][v])
            elif slot == "tempo":
                name, lo, hi, rep = rng.choice(TEMPO_BINS)
                if rng.random() < 0.3:
                    bpm = rng.randint(max(40, lo), min(200, hi - 1))
                    target["tempo_bpm"], fill["tempo"] = bpm, f"{bpm} bpm"
                else:
                    target["tempo_bpm"], fill["tempo"] = rep, rng.choice(kw["tempo"][name])
            elif slot == "key":
                v = rng.choice(KEYS)
                target["key"], fill["key"] = v, v
            elif slot == "meter":
                v = rng.choice(METERS[:4])
                target["meter"], fill["meter"] = v, rng.choice(kw["meter"][v])
            elif slot == "bars":
                v = rng.choice(BARS)
                target["bars"], fill["bars"] = v, str(v)
        rows.append({"text": tpl.format(**fill), "target": target, "source": "template"})
    return rows


def build(metadata: Optional[str], out: str, limits: dict[str, int], templates: dict[str, int],
          val_fraction: float = 0.02, seed: int = 0) -> dict[str, Any]:
    kw = load_keywords()
    tpl = load_yaml(CONFIGS / "parser_templates.yaml")
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    if metadata:
        for split, rows in midicaps_rows(metadata, val_fraction, limits, seed, kw).items():
            splits[split].extend(rows)
    for split, n in templates.items():
        splits[split].extend(template_rows(n, rng, kw, tpl["templates"]))
    stats: dict[str, Any] = {}
    for split, rows in splits.items():
        rng.shuffle(rows)
        write_jsonl(rows, resolve(out) / f"{split}.jsonl")
        coverage = Counter(f for r in rows for f in FIELDS if r["target"][f] is not None)
        stats[split] = {"rows": len(rows), "sources": dict(Counter(r["source"] for r in rows)),
                        "field_coverage": {f: round(coverage[f] / max(1, len(rows)), 3) for f in FIELDS}}
    save_json(stats, resolve(out) / "stats.json")
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", default="data/raw/midicaps/train.json")
    ap.add_argument("--no-midicaps", action="store_true")
    ap.add_argument("--out", default="data/parser")
    ap.add_argument("--midicaps-train", type=int, default=20000)
    ap.add_argument("--midicaps-eval", type=int, default=1000)
    ap.add_argument("--templates-train", type=int, default=10000)
    ap.add_argument("--templates-eval", type=int, default=500)
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    limits = {"train": args.midicaps_train, "val": args.midicaps_eval, "test": args.midicaps_eval}
    templates = {"train": args.templates_train, "val": args.templates_eval, "test": args.templates_eval}
    stats = build(None if args.no_midicaps else args.metadata, args.out, limits, templates,
                  args.val_fraction, args.seed)
    for split, s in stats.items():
        print(split, s["rows"], s["sources"])


if __name__ == "__main__":
    main()
