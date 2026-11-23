"""Blind A/B listening page: real excerpts vs generated ones for the same attributes.

    pixi run -e cuda listening --mode melody   --checkpoint models/composer/transformer-ref/best.pt
    pixi run -e cuda listening --mode ensemble --checkpoint models/composer/ensemble-ref/best.pt

For each of `--n` test windows, the window's own attributes are the prompt. The page plays
the real window and one generated piece per sampling setting, all rendered with the same
synthesiser, in a random order per row; a button reveals which is which. Output:
outputs/app/listening/<mode>/index.html (served by the app at /files/listening/<mode>/).

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-23. See ATTRIBUTION.md.
"""

from __future__ import annotations

import argparse
import html
import json
import random

from ..model import load_checkpoint, pick_device
from ..model.generate import generate
from ..paths import load_yaml, resolve
from ..render.render import render_wav
from .run_eval import real_windows

SETTINGS = {
    "plain": {"label": "model, plain sampling (temperature 1.0, no top-p)", "temperature": 1.0, "top_p": 1.0},
    "nucleus": {"label": "model, temperature 0.85 + top-p 0.9", "temperature": 0.85, "top_p": 0.9},
}

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>tunesmith listening test ({mode})</title>
<style>
body {{ font: 16px/1.5 system-ui, sans-serif; max-width: 980px; margin: 30px auto; padding: 0 16px; color: #1e2427; background: #e2e3de; }}
h1 {{ font-size: 28px; margin: 0 0 6px; }} .row {{ background: #fbfbf9; border: 1px solid #b9bcb5; border-radius: 12px; padding: 14px 16px; margin: 14px 0; }}
.prompt {{ color: #586266; font-size: 14px; margin: 0 0 10px; }} .clips {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
.clip p {{ margin: 0 0 4px; font-weight: 600; }} .answer {{ display: none; color: #586266; font-weight: 400; }}
body.revealed .answer {{ display: inline; }} audio {{ width: 100%; }} button {{ font: inherit; padding: 8px 14px; border-radius: 8px; border: 1px solid #9a9f99; background: #eeebe0; cursor: pointer; }}
</style></head><body>
<h1>Listening test: {mode} mode</h1>
<p>Each row has three clips for the same request: one real excerpt from the test set and two generated
pieces. They are shuffled. Listen, decide which you prefer, then reveal the answers.</p>
<button onclick="document.body.classList.toggle('revealed')">Show or hide which clip is which</button>
{rows}
</body></html>
"""


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="melody", choices=["melody", "ensemble"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--cfg-scale", type=float, default=1.5)
    ap.add_argument("--key-bias", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    data_dir = resolve(args.data_dir or ("data/processed/midicaps" if args.mode == "melody"
                                         else "data/processed/midicaps_ensemble"))
    out = resolve("outputs/app/listening") / args.mode
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    model, tok, _ = load_checkpoint(resolve(args.checkpoint), device)
    soundfont = load_yaml("configs/app.yaml").get("render", {}).get("soundfont")
    rng = random.Random(args.seed)

    real = list(real_windows(data_dir, tok, args.n, args.seed))
    attrs = [a.model_copy(update={"tempo_bpm": a.tempo_bpm or 100}) for _, a in real]
    generated = {name: generate(model, tok, attrs, temperature=s["temperature"], top_p=s["top_p"],
                                cfg_scale=args.cfg_scale, key_bias=args.key_bias, seed=args.seed)
                 for name, s in SETTINGS.items()}
    rows, answers = [], []
    for i, ((ids, _), a) in enumerate(zip(real, attrs)):
        clips = [("real", "real excerpt from the test set", ids)]
        clips += [(name, SETTINGS[name]["label"], generated[name][i]) for name in SETTINGS]
        rng.shuffle(clips)
        cells = []
        for j, (name, label, seq) in enumerate(clips):
            wav = out / f"{i:02d}_{j}.wav"
            render_wav(tok.decode_for(seq, a), wav, soundfont)
            cells.append(f'<div class="clip"><p>Clip {"ABC"[j]} <span class="answer">: {html.escape(label)}</span></p>'
                         f'<audio controls preload="none" src="{wav.name}"></audio></div>')
            answers.append({"row": i, "clip": "ABC"[j], "source": name})
        rows.append(f'<div class="row"><p class="prompt">{html.escape(a.summary())}</p>'
                    f'<div class="clips">{"".join(cells)}</div></div>')
    (out / "index.html").write_text(PAGE.format(mode=args.mode, rows="\n".join(rows)), encoding="utf-8")
    (out / "answers.json").write_text(json.dumps(answers, indent=1), encoding="utf-8")
    print(f"wrote {out / 'index.html'}  (open http://127.0.0.1:8000/files/listening/{args.mode}/index.html with `pixi run app`)")


if __name__ == "__main__":
    main()
