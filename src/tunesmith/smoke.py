"""CPU smoke test of the whole pipeline on synthetic data (about a minute).

    pixi run smoke                       # GRU composer
    pixi run smoke --model-type transformer   # once model/transformer.py is implemented

synthetic MIDI -> prepare -> train a tiny composer -> generate from text -> MIDI + WAV.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

import argparse

from .data.prepare import prepare
from .data.synthetic import generate_dataset
from .paths import load_yaml, resolve
from .pipeline import TuneSmith
from .train.train_composer import train


def run(model_type: str = "gru", steps: int = 40, root: str = "outputs/smoke") -> dict:
    root_p = resolve(root)
    meta_path = generate_dataset(root_p / "raw", n=60, seed=0)
    data_cfg = load_yaml("configs/data.yaml")
    data_cfg.update(metadata=str(meta_path), midi_root=str(root_p / "raw"), out_dir=str(root_p / "processed"),
                    workers=1, val_fraction=0.2)
    prepare(data_cfg)
    cfg = load_yaml("configs/composer_tiny.yaml")
    cfg.update(data_dir=str(root_p / "processed"), out_dir=str(root_p / f"composer-{model_type}"))
    cfg["model"]["type"] = model_type
    cfg["train"]["max_steps"] = steps
    train(cfg, device_name="cpu")
    app_cfg = load_yaml("configs/app.yaml")
    app_cfg.update(parser="rules", composer_checkpoint=str(root_p / f"composer-{model_type}" / "best.pt"),
                   output_dir=str(root_p / "app"), device="cpu")
    smith = TuneSmith(app_cfg)
    res = smith.compose("a sad piano melody in A minor, waltz, 8 bars", n=2, seed=0)
    print("parsed:", {k: v for k, v in res["attributes"].items() if v is not None})
    for p in res["pieces"]:
        print(" ", p["midi"], p.get("wav"), p.get("engine"), p["metrics"])
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-type", default="gru", choices=["gru", "transformer"])
    ap.add_argument("--steps", type=int, default=40)
    args = ap.parse_args(argv)
    run(args.model_type, args.steps)
    print("smoke test passed")


if __name__ == "__main__":
    main()
