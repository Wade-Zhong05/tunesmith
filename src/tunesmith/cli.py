"""Command line: parse a description, or generate music from it.

    pixi run parse "a sad piano melody in A minor, waltz"
    pixi run generate "an epic brass theme at 140 bpm" --n 2 --seed 7

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="tunesmith", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parse", help="text -> attributes (JSON)")
    p.add_argument("text")
    p.add_argument("--parser", default=None, choices=["rules", "lm"])
    g = sub.add_parser("generate", help="text -> MIDI + WAV")
    g.add_argument("text")
    g.add_argument("--mode", default="melody", choices=["melody", "ensemble"])
    g.add_argument("--n", type=int, default=1)
    g.add_argument("--seed", type=int)
    g.add_argument("--config", default="configs/app.yaml")
    g.add_argument("--checkpoint")
    g.add_argument("--no-wav", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "parse":
        from .paths import load_yaml
        from .pipeline import make_parser

        kind = args.parser or load_yaml("configs/app.yaml").get("parser", "rules")
        attrs = make_parser(kind).parse(args.text)
        print(json.dumps(attrs.model_dump(), indent=2))  # ASCII-only output (safe on any console)
        return

    from .paths import load_yaml
    from .pipeline import TuneSmith

    cfg = load_yaml(args.config)
    if args.checkpoint:
        cfg["composer_checkpoint"] = args.checkpoint
    smith = TuneSmith(cfg)
    res = smith.compose(args.text, n=args.n, seed=args.seed, render=not args.no_wav, mode=args.mode)
    print(f"parser: {res['parser']}   composer: {res['composer']}")
    print("attributes:", json.dumps({k: v for k, v in res["attributes"].items() if v is not None}))
    for p in res["pieces"]:
        files = [str(smith.out_dir / p["midi"])] + ([str(smith.out_dir / p["wav"])] if "wav" in p else [])
        print(" ", " | ".join(files), "|", p["metrics"])
    for w in res["warnings"]:
        print("warning:", w)


if __name__ == "__main__":
    main()
