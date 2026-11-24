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
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>tunesmith listening test ({mode})</title>
<style>
@font-face {{ font-family: "Instrument Sans"; src: url("/fonts/InstrumentSans-Variable.ttf") format("truetype"); font-weight: 400 700; }}
body {{ font: 16px/1.55 "Instrument Sans", system-ui, sans-serif; max-width: 980px; margin: 36px auto; padding: 0 20px;
        color: #1d1d1b; background: #fff; }}
a {{ color: #e55408; }}
h1 {{ font-size: 32px; letter-spacing: -0.02em; margin: 0 0 6px; }}
.lede {{ color: #6e6d68; max-width: 70ch; }}
.warn {{ display: none; background: #fff0e6; border-radius: 12px; padding: 12px 16px; margin: 16px 0; }}
.warn.on {{ display: block; }}
.row {{ border-top: 1px solid #e6e5e0; padding: 18px 0; }}
.prompt {{ color: #6e6d68; font-size: 14px; margin: 0 0 10px; }}
.clips {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }}
.clip p {{ margin: 0 0 4px; font-weight: 600; }}
.answer {{ display: none; color: #e55408; font-weight: 500; }}
body.revealed .answer {{ display: inline; }}
audio {{ width: 100%; }}
button {{ font: 600 15px "Instrument Sans", system-ui, sans-serif; padding: 10px 16px; border-radius: 999px; border: 0;
          background: #ff6b1a; color: #fff; cursor: pointer; }}
button:hover {{ background: #e55408; }}
</style></head><body>
<p><a href="/">Back to tunesmith</a></p>
<h1>Listening test: {mode_label}</h1>
<p class="lede">Each row has three clips for the same request: one real excerpt from the test set and two generated
pieces, in random order, all played through the same synthesiser. Listen, decide which you prefer, then reveal
which is which.</p>
<p class="warn" id="warn">The clips cannot be loaded when this file is opened straight from disk. Start the app with
<code>pixi run app</code> and open <code>http://127.0.0.1:8000/files/listening/{mode}/index.html</code>
(or use the link at the bottom of the tunesmith page).</p>
<button onclick="document.body.classList.toggle('revealed')">Show or hide which clip is which</button>
{rows}
<script>
  const warn = document.getElementById("warn");
  if (location.protocol === "file:") warn.classList.add("on");
  document.querySelectorAll("audio").forEach((a) => a.addEventListener("error", () => warn.classList.add("on")));
</script>
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
                         f'<audio controls preload="metadata" src="{wav.name}"></audio></div>')
            answers.append({"row": i, "clip": "ABC"[j], "source": name})
        rows.append(f'<div class="row"><p class="prompt">{html.escape(a.summary())}</p>'
                    f'<div class="clips">{"".join(cells)}</div></div>')
    label = "band" if args.mode == "ensemble" else "melody"
    (out / "index.html").write_text(PAGE.format(mode=args.mode, mode_label=label, rows="\n".join(rows)),
                                    encoding="utf-8")
    (out / "answers.json").write_text(json.dumps(answers, indent=1), encoding="utf-8")
    print(f"wrote {out / 'index.html'}  (open http://127.0.0.1:8000/files/listening/{args.mode}/index.html with `pixi run app`)")


if __name__ == "__main__":
    main()
