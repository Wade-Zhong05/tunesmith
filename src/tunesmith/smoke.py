"""CPU smoke test of the whole pipeline on synthetic data (about a minute).

    pixi run smoke                                        # melody mode, GRU composer
    pixi run smoke --model-type transformer               # once model/transformer.py is implemented
    pixi run smoke --mode ensemble --model-type transformer_ref   # multi-instrument mode

synthetic MIDI -> prepare -> train a tiny composer -> generate from text -> MIDI + WAV.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10; ensemble mode
added 2026-11-22. See ATTRIBUTION.md.
"""

from __future__ import annotations

import argparse

from .data.prepare import prepare
from .data.synthetic import generate_dataset
from .paths import load_yaml, resolve
from .pipeline import TuneSmith
from .train.train_composer import train

PROMPTS = {
    "melody": "a sad piano melody in A minor, waltz, 8 bars",
    "ensemble": "an upbeat pop song with piano, strings and bass, with drums",
}


def run(model_type: str = "gru", steps: int = 40, root: str = "outputs/smoke", mode: str = "melody") -> dict:
    root_p = resolve(root)
    meta_path = generate_dataset(root_p / "raw", n=60, seed=0)
    data_cfg = load_yaml("configs/data.yaml" if mode == "melody" else "configs/data_ensemble.yaml")
    data_cfg.update(metadata=str(meta_path), midi_root=str(root_p / "raw"), out_dir=str(root_p / f"processed-{mode}"),
                    workers=1, val_fraction=0.2, min_notes_per_window=8)
    prepare(data_cfg)
    name = f"composer-{mode}-{model_type}"
    cfg = load_yaml("configs/composer_tiny.yaml")
    cfg.update(data_dir=str(root_p / f"processed-{mode}"), out_dir=str(root_p / name))
    cfg["model"]["type"] = model_type
    cfg["train"]["max_steps"] = steps
    train(cfg, device_name="cpu")
    app_cfg = load_yaml("configs/app.yaml")
    ckpt = str(root_p / name / "best.pt")
    app_cfg.update(parser="rules", output_dir=str(root_p / "app"), device="cpu",
                   composer_checkpoint=ckpt if mode == "melody" else str(root_p / "none.pt"), fallback_checkpoint=None,
                   ensemble_checkpoint=ckpt if mode == "ensemble" else str(root_p / "none.pt"), ensemble_fallback=None)
    smith = TuneSmith(app_cfg)
    res = smith.compose(PROMPTS[mode], n=2, seed=0, mode=mode)
    print("parsed:", {k: v for k, v in res["attributes"].items() if v is not None})
    print("composer:", res["composer"])
    for p in res["pieces"]:
        print(" ", p["midi"], p.get("wav"), p.get("engine"), [t["family"] for t in p["tracks"]], p["metrics"])
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="melody", choices=["melody", "ensemble"])
    ap.add_argument("--model-type", default="gru", choices=["gru", "transformer", "transformer_ref"])
    ap.add_argument("--steps", type=int, default=40)
    args = ap.parse_args(argv)
    run(args.model_type, args.steps, mode=args.mode)
    print("smoke test passed")


if __name__ == "__main__":
    main()
