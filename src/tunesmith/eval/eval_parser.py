"""Evaluate text -> attribute parsers on the parser test set, overall and per source.

    pixi run evaluate-parser --parser rules
    pixi run evaluate-parser --parser lm --mode zero
    pixi run evaluate-parser --parser lm --mode few
    pixi run evaluate-parser --parser lm --mode lora --adapter models/parser/lora-r16
    ... --test my_human_written.jsonl   # any JSONL with {"text", "target"} rows

Writes outputs/eval_parser/<name>/summary.json and predictions.jsonl.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

from ..attributes import MusicAttributes
from ..paths import load_yaml, read_jsonl, resolve, save_json, write_jsonl
from .parser_metrics import field_accuracy


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parser", default="rules", choices=["rules", "lm"])
    ap.add_argument("--mode", default="lora", choices=["zero", "few", "lora"])
    ap.add_argument("--base")
    ap.add_argument("--adapter")
    ap.add_argument("--test", default="data/parser/test.jsonl")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--name")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    rows = list(read_jsonl(args.test))[: args.limit] if args.limit else list(read_jsonl(args.test))
    texts = [r["text"] for r in rows]
    t0 = time.time()
    if args.parser == "rules":
        from ..understand.rules import parse

        preds = [parse(t) for t in texts]
        raws = [None] * len(texts)
        name = args.name or "rules"
    else:
        from ..understand.lm_parser import LMParser

        app = load_yaml("configs/app.yaml")
        parser = LMParser(args.base or app["parser_base"], args.adapter or app["parser_adapter"], args.mode, args.device)
        results = parser.parse_batch(texts)
        preds = [r.attrs if r.valid_json else None for r in results]
        raws = [r.raw for r in results]
        name = args.name or parser.name
    seconds = time.time() - t0
    golds = [MusicAttributes.lenient(r["target"]) for r in rows]

    summary = {"parser": name, "overall": field_accuracy(preds, golds), "ms_per_query": 1000 * seconds / max(1, len(rows))}
    by_source = defaultdict(list)
    for i, r in enumerate(rows):
        by_source[r.get("source", "unknown")].append(i)
    summary["by_source"] = {src: field_accuracy([preds[i] for i in idx], [golds[i] for i in idx])
                            for src, idx in by_source.items()}
    out = resolve("outputs/eval_parser") / name
    save_json(summary, out / "summary.json")
    write_jsonl(({"text": r["text"], "target": r["target"], "pred": p.model_dump() if p else None, "raw": raw}
                 for r, p, raw in zip(rows, preds, raws)), out / "predictions.jsonl")
    o = summary["overall"]
    print(f"{name}: mean field acc {o['mean_field_accuracy']:.3f}, exact {o['exact_match']:.3f}, "
          f"valid {o['valid_rate']:.3f}, {summary['ms_per_query']:.1f} ms/query")
    for src, s in summary["by_source"].items():
        print(f"  {src}: mean field acc {s['mean_field_accuracy']:.3f} (n={s['n']})")


if __name__ == "__main__":
    main()
