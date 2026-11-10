"""Prepare composer training data: MIDI -> melody windows -> token sequences.

    pixi run prepare-data                         # full MidiCaps, configs/data.yaml
    pixi run prepare-data --limit 2000            # quick subset
    pixi run prepare-data --metadata outputs/smoke/raw/metadata.jsonl --midi-root outputs/smoke/raw \
        --out outputs/smoke/processed             # any MidiCaps-format metadata (e.g. synthetic)

Output folder:
    tokens_{split}.bin   uint16 token ids, all sequences concatenated
    index_{split}.npy    int64 (N, 2): start offset and length of each sequence
    files_{split}.jsonl  one row per source file: location, caption, file-level attributes
    meta.json            tokenizer config, vocab size, counts, the config used

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from ..attributes import from_midicaps, normalize_meter, program_to_family
from ..music.keyfinding import detect_key
from ..music.melody import load_score, windows
from ..paths import load_yaml, read_jsonl, resolve, save_json
from ..tokenizer import TuneTokenizer
from .splits import split_of, stable_fraction

KEEP_FIELDS = ("location", "caption", "key", "tempo", "time_signature", "instrument_numbers_sorted",
               "instrument_summary", "mood", "genre", "test_set")
SPLITS = ("train", "val", "test")

_TOK: TuneTokenizer | None = None
_CFG: dict[str, Any] = {}


def _init_worker(cfg: dict[str, Any]) -> None:
    global _TOK, _CFG
    _CFG = cfg
    _TOK = TuneTokenizer(cfg["tokenizer"])


def process_row(row: dict[str, Any]) -> dict[str, Any]:
    """One metadata row -> token sequences for its windows (runs in a worker process)."""
    cfg, tok = _CFG, _TOK
    assert tok is not None
    location = row["location"]
    split = split_of(location, row.get("test_set"), cfg["val_fraction"])
    info = {"location": location, "caption": row.get("caption", ""), "split": split}
    try:
        score = load_score(Path(cfg["midi_root"]) / location)
    except Exception as e:  # corrupt or missing MIDI
        return {**info, "error": f"load: {type(e).__name__}", "seqs": []}
    try:
        base = from_midicaps(row)
    except Exception as e:  # malformed metadata row: skip the file, keep the run going
        return {**info, "error": f"metadata: {type(e).__name__}", "seqs": []}
    info["attrs"] = base.model_dump()
    rng = random.Random(int(stable_fraction(location) * 1e9) ^ int(cfg["seed"]))
    try:
        excerpts = windows(score, list(cfg["window_bars"]), cfg["reduce"], int(cfg["max_windows_per_file"]),
                           int(cfg["min_notes_per_window"]), rng)
    except Exception as e:
        return {**info, "error": f"windows: {type(e).__name__}", "seqs": []}
    seqs = []
    for ex in excerpts:
        update = {
            "bars": ex.bars,
            "meter": normalize_meter(f"{ex.numerator}/{ex.denominator}"),
            "instrument": program_to_family(ex.program) or base.instrument,
        }
        if cfg["key_source"] == "detected":
            key, corr = detect_key(ex.notes)
            if key and corr > 0.5:
                update["key"] = key
        attrs = base.model_copy(update=update)
        try:
            ids = tok.encode(ex.to_score(), attrs)
        except Exception:
            continue
        if len(ids) <= int(cfg["max_seq_len"]) and len(ids) > tok.prompt_len + 4:
            seqs.append(np.asarray(ids, dtype=np.uint16))
    info["n_windows"] = len(seqs)
    return {**info, "seqs": seqs}


def _batched(it: Iterable, n: int) -> Iterator[list]:
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def prepare(cfg: dict[str, Any], limit: int | None = None) -> dict[str, Any]:
    out_dir = resolve(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {**cfg, "midi_root": str(resolve(cfg["midi_root"]))}
    tok = TuneTokenizer(cfg["tokenizer"])

    rows = ({k: r.get(k) for k in KEEP_FIELDS} for r in read_jsonl(cfg["metadata"]))
    if limit:
        rows = (r for i, r in zip(range(limit), rows))

    bins = {s: open(out_dir / f"tokens_{s}.bin", "wb") for s in SPLITS}
    files = {s: open(out_dir / f"files_{s}.jsonl", "w", encoding="utf-8") for s in SPLITS}
    index: dict[str, list[tuple[int, int]]] = {s: [] for s in SPLITS}
    offsets = {s: 0 for s in SPLITS}
    stats = {"files": 0, "errors": 0, "files_without_windows": 0, "tokens": {s: 0 for s in SPLITS}}
    workers = int(cfg.get("workers") or 0) or max(1, (os.cpu_count() or 2) - 1)
    t0 = time.time()

    def consume(result: dict[str, Any]) -> None:
        import json

        stats["files"] += 1
        if "error" in result:
            stats["errors"] += 1
            return
        split = result["split"]
        if not result["seqs"]:
            stats["files_without_windows"] += 1
        for arr in result["seqs"]:
            bins[split].write(arr.tobytes())
            index[split].append((offsets[split], len(arr)))
            offsets[split] += len(arr)
            stats["tokens"][split] += len(arr)
        row = {k: result.get(k) for k in ("location", "caption", "attrs", "n_windows")}
        files[split].write(json.dumps(row, ensure_ascii=False) + "\n")
        if stats["files"] % 2000 == 0:
            rate = stats["files"] / max(1e-9, time.time() - t0)
            print(f"  {stats['files']} files, {rate:.0f} files/s, windows "
                  f"{ {s: len(index[s]) for s in SPLITS} }", flush=True)

    try:
        if workers == 1:
            _init_worker(cfg)
            for row in rows:
                consume(process_row(row))
        else:
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(cfg,)) as ex:
                for batch in _batched(rows, 2000):
                    for result in ex.map(process_row, batch, chunksize=32):
                        consume(result)
    finally:
        for f in (*bins.values(), *files.values()):
            f.close()

    for s in SPLITS:
        np.save(out_dir / f"index_{s}.npy", np.asarray(index[s], dtype=np.int64).reshape(-1, 2))
    meta = {
        "tokenizer": tok.config,
        "vocab_size": tok.vocab_size,
        "prompt_len": tok.prompt_len,
        "windows": {s: len(index[s]) for s in SPLITS},
        "stats": stats,
        "config": {k: v for k, v in cfg.items() if k != "tokenizer"},
        "seconds": round(time.time() - t0, 1),
    }
    save_json(meta, out_dir / "meta.json")
    print(f"done in {meta['seconds']} s: windows {meta['windows']}, errors {stats['errors']}")
    return meta


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--metadata")
    ap.add_argument("--midi-root")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--key-source", choices=["metadata", "detected"])
    ap.add_argument("--reduce", choices=["melody", "piano"])
    ap.add_argument("--val-fraction", type=float)
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    for name, key in (("metadata", "metadata"), ("midi_root", "midi_root"), ("out", "out_dir"),
                      ("workers", "workers"), ("key_source", "key_source"), ("reduce", "reduce"),
                      ("val_fraction", "val_fraction")):
        if getattr(args, name) is not None:
            cfg[key] = getattr(args, name)
    prepare(cfg, args.limit)


if __name__ == "__main__":
    main()
