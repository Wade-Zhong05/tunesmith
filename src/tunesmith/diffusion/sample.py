"""Write pieces with the diffusion composer, or rewrite part of an existing MIDI file.

    pixi run -e cuda diffusion-sample --text "an upbeat pop song with piano, electric guitar and bass, with drums"
    pixi run -e cuda diffusion-sample --text "..." --n 4 --seed 3 --cfg-scale 3
    pixi run -e cuda diffusion-sample --edit song.mid --bars 5 6          # rewrite bars 5 and 6
    pixi run -e cuda diffusion-sample --edit song.mid --keep 1            # keep part 1, rewrite the rest

Parts are numbered by note count (1 = the part with most notes), as in the roll's
instrument slots. Writes MIDI, WAV and piano-roll PNG files to --out.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-23. See ATTRIBUTION.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from symusic import Score

from ..attributes import MusicAttributes
from ..paths import load_yaml, resolve
from ..render.pianoroll import plot_pianoroll
from ..render.render import render_wav
from .composer import DiffusionComposer


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="models/diffusion/ensemble-unet/best.pt")
    ap.add_argument("--text", default="", help="description, read by the rule parser (or --parser lm)")
    ap.add_argument("--parser", default="rules", choices=["rules", "lm"])
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--cfg-scale", type=float)
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--edit", help="MIDI file to rewrite part of")
    ap.add_argument("--bars", type=int, nargs="+", help="1-based bars to rewrite (with --edit)")
    ap.add_argument("--keep", type=int, nargs="+", help="1-based parts to keep (with --edit)")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--out", default="outputs/diffusion")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    composer = DiffusionComposer.load(resolve(args.checkpoint), args.device)
    attrs = MusicAttributes()
    if args.text:
        from ..pipeline import make_parser

        attrs = make_parser(args.parser).parse(args.text)
    print(f"attributes: {attrs.summary()}")
    kw = dict(steps=args.steps, cfg_scale=args.cfg_scale, threshold=args.threshold)
    if args.edit:
        source = Score(str(args.edit))
        bars = [b - 1 for b in args.bars] if args.bars else None
        keep = [k - 1 for k in args.keep] if args.keep else None
        if bars is None and keep is None:
            ap.error("--edit needs --bars or --keep")
        scores = [composer.edit(source, attrs, bars=bars, keep_slots=keep, seed=args.seed + i, **kw)
                  for i in range(args.n)]
        stem = Path(args.edit).stem + "_edit"
    else:
        scores = composer.compose([attrs] * args.n, seed=args.seed, **kw)
        stem = "piece"

    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    soundfont = load_yaml("configs/app.yaml").get("render", {}).get("soundfont")
    for i, score in enumerate(scores):
        base = out / f"{stem}_{args.seed}_{i}"
        score.dump_midi(str(base.with_suffix(".mid")))
        plot_pianoroll(score, base.with_suffix(".png"), title=args.text or attrs.summary())
        if not args.no_audio:
            render_wav(score, base.with_suffix(".wav"), soundfont)
        parts = ", ".join(f"{t.name} {len(t.notes)}" for t in score.tracks)
        print(f"{base.with_suffix('.mid')}: {parts}")


if __name__ == "__main__":
    main()
