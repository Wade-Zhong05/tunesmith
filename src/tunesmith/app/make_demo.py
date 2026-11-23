"""Build the demo piece the web page previews before the first request (web/demo/).

    pixi run make-demo

Composes a few ensemble candidates with the trained ensemble composer, keeps the liveliest one
(least repetitive, most distinct pitches), then writes
web/demo/demo.json (the same shape as an /api/generate response, with local file names)
and web/demo/demo.wav (mono, 22.05 kHz, to keep the file small).

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-23. See ATTRIBUTION.md.
"""

from __future__ import annotations

import argparse
import json

from ..paths import ROOT, load_yaml
from ..pipeline import TuneSmith
from ..render.render import render_wav
from ..render.synth import read_wav, write_wav

TEXT = "an upbeat pop song with piano, electric guitar and bass, with drums"


def _variety(piece: dict) -> int:
    return len({n["pitch"] for n in piece["notes"]["notes"] if not n.get("drum")})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", default=TEXT)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--candidates", type=int, default=6)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    cfg = load_yaml("configs/app.yaml")
    cfg.update(device=args.device, output_dir="outputs/demo")
    smith = TuneSmith(cfg)
    res = smith.compose(args.text, n=args.candidates, seed=args.seed, mode="ensemble", temperature=0.9,
                        top_p=0.92, render=False)
    out = ROOT / "web" / "demo"
    out.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(res["pieces"]):
        print(f"candidate {i}: repetition {p['metrics'].get('repetition')}, {_variety(p)} distinct pitches")
    piece = max(res["pieces"], key=lambda p: (p["metrics"].get("repetition", 1.0) <= 0.6, _variety(p),
                                               -p["metrics"].get("repetition", 1.0)))
    from symusic import Score

    score = Score(str(smith.out_dir / piece["midi"]))
    tmp = smith.out_dir / "demo_stereo.wav"
    render_wav(score, tmp, cfg.get("render", {}).get("soundfont"), sample_rate=22050)
    audio, sr = read_wav(tmp)  # read_wav returns mono
    write_wav(out / "demo.wav", audio, sr)
    score.dump_midi(str(out / "demo.mid"))
    piece.update(wav_url="demo/demo.wav", midi_url="demo/demo.mid")
    demo = {k: res[k] for k in ("mode", "text", "attributes", "composer", "seed")}
    demo["pieces"] = [piece]
    (out / "demo.json").write_text(json.dumps(demo), encoding="utf-8")
    print(f"demo written to {out}: {piece['metrics']}")


if __name__ == "__main__":
    main()
