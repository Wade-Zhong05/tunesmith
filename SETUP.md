# Setup

## Install

[pixi](https://pixi.sh) installs Python and every package. Install pixi once (on Windows,
`winget install prefix-dev.pixi`), then in the repository:

```bash
pixi install            # CPU
pixi install -e cuda    # PyTorch with CUDA 12.6, for training and fast inference
```

Add `-e cuda` to any task to run it on the GPU. Without pixi, install PyTorch, run
`pip install -r requirements.txt`, and start the modules listed under `[tool.pixi.tasks]`
in `pyproject.toml` with `python -m`.

## Trained models

```bash
pixi run download-models --all
```

This downloads the `models-v1` GitHub release (melody and band composers, GRU, bigram
baselines, LoRA adapter), plus the Qwen2.5-0.5B base model and the MuseScore SoundFont
from their original sources. Every file is checked against the SHA-256 hashes in
`configs/model_release.yaml`, and a second run skips files that are already correct.
Other options: `--with-parser-base` and `--with-soundfont` (both included in `--all`),
`--with-training-runs` for logs and loss curves, `--base-url` for a mirror.

Without the models the app still starts, but it shows a notice and composes with a random
baseline.

## Run

```bash
pixi run app                                         # web app at http://127.0.0.1:8000
pixi run generate "an epic brass theme at 140 bpm" --n 2
pixi run generate "a calm piece for strings and harp, no drums" --mode ensemble
pixi run parse "a playful flute tune in D major"
pixi run test                                        # unit tests
pixi run smoke                                       # tiny synthetic training run, about a minute on CPU
```

The app reads descriptions with the LoRA parser (`parser: lm` in `configs/app.yaml`).
A new description takes about 0.5 s on a GPU and 7–10 s on a CPU; repeated ones are
cached. If the parser model is missing, the app falls back to keyword rules.

## Train from scratch

Data (MidiCaps from the Hugging Face Hub, about 2 GB):

```bash
pixi run download-data
pixi run prepare-data         # melody windows
pixi run prepare-ensemble     # multi-track windows
pixi run make-parser-data
pixi run download-parser-base
```

Models, with times on an RTX 4070 laptop GPU:

```bash
pixi run -e cuda train-composer --config configs/composer_transformer_ref.yaml   # reference Transformer, 1.3 h
pixi run -e cuda train-composer --config configs/composer_base.yaml              # student Transformer
pixi run -e cuda train-composer --config configs/composer_gru.yaml               # GRU baseline, 5 h
pixi run -e cuda train-ensemble                                                 # band Transformer, 6.5 h
pixi run -e cuda train-diffusion                                                # diffusion U-Net, 4.7 h
pixi run -e cuda train-parser --out-dir models/parser/lora-r16-v2               # LoRA parser, 2.6 h
```

Add `--resume` to continue a run from its `last.pt`. On Duke CS OnDemand, start
JupyterLab on the `compsci-gpu` partition, install pixi with
`curl -fsSL https://pixi.sh/install.sh | bash`, clone the repository and run
`pixi install -e cuda`.

The diffusion model can also edit an existing MIDI file:

```bash
pixi run -e cuda diffusion-sample --text "a jazz groove with electric piano, sax and bass" --n 2
pixi run -e cuda diffusion-sample --edit song.mid --bars 5 6    # rewrite bars 5 and 6
pixi run -e cuda diffusion-sample --edit song.mid --keep 1      # keep part 1, rewrite the rest
```

## Evaluate

```bash
pixi run evaluate --system data --name reference
pixi run evaluate --system bigram
pixi run evaluate --system models/composer/transformer-ref/best.pt --name transformer
pixi run evaluate --data-dir data/processed/midicaps_ensemble --system models/composer/ensemble-ref/best.pt --name band
pixi run evaluate-parser --parser lm --mode lora
pixi run listening --mode ensemble --checkpoint models/composer/ensemble-ref/best.pt
```

Results are written to `outputs/eval/<name>/` and `outputs/eval_parser/<name>/`. Add
`--render --clap` to `evaluate` for audio files and CLAP text-audio scores.

## Troubleshooting

- `torch.cuda.is_available()` is False in the cuda environment: update the NVIDIA driver.
  No CUDA toolkit is needed.
- Hugging Face is unreachable: download the data and the Qwen model on another network
  (OnDemand works) and copy `data/raw/midicaps` and `models/parser/base`.
- The audio sounds harsh and synthetic: the SoundFont is missing and the built-in synth is
  used. Run `pixi run fetch-soundfont`.
