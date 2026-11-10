"""Evaluate composers: does the music follow the requested attributes, and does it look
like real music? Every system goes through the same generation code (grammar, key
bias, stopping rule), so differences come from the model.

    pixi run evaluate --system models/composer/transformer-base/best.pt --name transformer
    pixi run evaluate --system models/composer/gru-base/best.pt --name gru
    pixi run evaluate --system bigram       # Markov-chain baseline (fitted on the training split)
    pixi run evaluate --system uniform      # random baseline
    pixi run evaluate --system data         # the real test windows: reference statistics
    ... --parser rules|lm                   # end-to-end: attributes parsed from the captions
    ... --render --clap                     # also WAV files and CLAP text-audio scores

Writes outputs/eval/<name>/: samples.csv, summary.json, midi/, a few piano-roll PNGs.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..attributes import MusicAttributes
from ..model import load_checkpoint, pick_device
from ..model.generate import generate
from ..paths import load_json, read_jsonl, resolve, save_json
from ..render.pianoroll import plot_pianoroll
from ..render.render import render_wav
from ..tokenizer import TuneTokenizer
from .baselines import BigramComposer, UniformComposer
from .metrics import music_metrics, summarize


def load_system(spec: str, data_dir: Path, device: torch.device):
    if spec in ("uniform", "bigram", "data"):
        tok = TuneTokenizer(load_json(data_dir / "meta.json")["tokenizer"])
        if spec == "uniform":
            return UniformComposer(tok.vocab_size).to(device), tok
        if spec == "bigram":
            path = resolve("models/baselines/bigram.npy")
            if path.exists():
                model = BigramComposer.load(path)
            else:
                print("fitting bigram baseline on the training split ...")
                model = BigramComposer.fit(data_dir, tok)
                model.save(path)
            return model.to(device), tok
        return None, tok
    model, tok, _ = load_checkpoint(resolve(spec), device)
    return model, tok


def load_prompts(path: Path, n: int, bars: int, seed: int) -> list[dict[str, Any]]:
    rows = [r for r in read_jsonl(path) if r.get("attrs")]
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows[:n]:
        attrs = MusicAttributes.lenient(r["attrs"])
        if attrs.bars is None:
            attrs = attrs.model_copy(update={"bars": bars})
        out.append({"caption": r.get("caption", ""), "location": r.get("location"), "attrs": attrs})
    return out


def real_windows(data_dir: Path, tok: TuneTokenizer, n: int, seed: int):
    index = np.load(data_dir / "index_test.npy")
    tokens = np.memmap(data_dir / "tokens_test.bin", dtype=np.uint16, mode="r")
    order = list(range(len(index)))
    random.Random(seed).shuffle(order)
    for i in order[:n]:
        start, length = index[i]
        ids = [int(x) for x in tokens[start:start + length]]
        yield ids, tok.attrs_from_prompt(ids[: tok.prompt_len])


def flatten(d: dict[str, Any]) -> dict[str, Any]:
    return {k: (v if not isinstance(v, bool) else int(v)) for k, v in d.items()}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True, help="checkpoint path | bigram | uniform | data")
    ap.add_argument("--name")
    ap.add_argument("--data-dir", default="data/processed/midicaps")
    ap.add_argument("--prompts", help="default: <data-dir>/files_test.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--bars", type=int, default=16)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--parser", default="oracle", choices=["oracle", "rules", "lm"])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--cfg-scale", type=float, default=1.5)
    ap.add_argument("--key-bias", type=float, default=1.0)
    ap.add_argument("--no-grammar", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--clap", action="store_true")
    ap.add_argument("--soundfont", default="models/soundfonts/MuseScore_General.sf3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-root", default="outputs/eval")
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    data_dir = resolve(args.data_dir)
    name = args.name or Path(args.system).stem
    out = resolve(args.out_root) / name
    (out / "midi").mkdir(parents=True, exist_ok=True)
    model, tok = load_system(args.system, data_dir, device)

    items: list[tuple[list[int], MusicAttributes, str]] = []
    t0 = time.time()
    if args.system == "data":
        for ids, attrs in real_windows(data_dir, tok, args.n, args.seed):
            items.append((ids, attrs, ""))
    else:
        prompts = load_prompts(resolve(args.prompts) if args.prompts else data_dir / "files_test.jsonl",
                               args.n, args.bars, args.seed)
        if args.parser != "oracle":
            from ..pipeline import make_parser

            parser = make_parser(args.parser)
            for p in prompts:
                parsed = parser.parse(p["caption"])
                p["attrs"] = parsed if parsed.bars else parsed.model_copy(update={"bars": args.bars})
        for i in range(0, len(prompts), args.batch):
            chunk = prompts[i:i + args.batch]
            seqs = generate(model, tok, [p["attrs"] for p in chunk], temperature=args.temperature,
                            top_k=args.top_k, top_p=args.top_p, cfg_scale=args.cfg_scale,
                            key_bias=args.key_bias, grammar=not args.no_grammar, seed=args.seed + i)
            items.extend((s, p["attrs"], p["caption"]) for s, p in zip(seqs, chunk))
            print(f"  generated {len(items)}/{len(prompts)}", flush=True)
    gen_seconds = time.time() - t0

    rows, wavs, captions = [], [], []
    for i, (ids, attrs, caption) in enumerate(items):
        try:
            score = tok.decode_for(ids, attrs)
        except Exception as e:
            rows.append({"i": i, "valid": False, "error": type(e).__name__})
            continue
        m = music_metrics(score, attrs, tok.count_bars(ids))
        midi_path = out / "midi" / f"{i:04d}.mid"
        score.dump_midi(str(midi_path))
        if i < 6:
            plot_pianoroll(score, out / f"pianoroll_{i}.png", title=attrs.summary())
        if args.render or args.clap:
            wav = out / "wav" / f"{i:04d}.wav"
            render_wav(score, wav, args.soundfont, sample_rate=48000)
            wavs.append(wav)
            captions.append(caption or attrs.summary())
        rows.append({"i": i, **{f"req_{k}": v for k, v in attrs.model_dump().items() if k != "lyrics"},
                     **flatten(m), "tokens": len(ids)})

    summary: dict[str, Any] = {"system": args.system, "name": name, "parser": args.parser,
                               "settings": {k: getattr(args, k) for k in ("temperature", "top_k", "top_p", "cfg_scale",
                                                                          "key_bias", "bars", "seed")},
                               "grammar": not args.no_grammar, **summarize(rows),
                               "seconds_per_piece": gen_seconds / max(1, len(items)), "device": str(device)}
    if args.clap and wavs:
        from .clap_score import clap_scores

        scores = clap_scores(captions, wavs, device=str(device))
        summary["clap"] = float(np.mean(scores))
    with open(out / "samples.csv", "w", newline="", encoding="utf-8") as f:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    save_json(summary, out / "summary.json")
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items() if k != "settings"})


if __name__ == "__main__":
    main()
