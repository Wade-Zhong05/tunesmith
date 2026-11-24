# models/

Download the trained models with `pixi run download-models --all`. The
[models-v1 release](https://github.com/Wade-Zhong05/tunesmith/releases/tag/models-v1)
is described by `configs/model_release.yaml`, including exact sizes, SHA-256 hashes
and destination paths. Model binaries are distributed as release assets.

| Release asset | Installed path |
|---|---|
| `transformer-ref.pt` | `models/composer/transformer-ref/best.pt` |
| `ensemble-ref.pt` | `models/composer/ensemble-ref/best.pt` |
| `gru-base.pt` | `models/composer/gru-base/best.pt` |
| `bigram.npy` | `models/baselines/bigram.npy` |
| `bigram_ensemble.npy` | `models/baselines/bigram_ensemble.npy` |
| `lora-r16-v2.zip` | Extracted into `models/parser/lora-r16-v2/` |
| `training-runs.zip` | Optionally extracted into `models/training-runs/` |

The default command installs the composers, bigrams and adapter. `--with-parser-base`
also fetches Qwen2.5-0.5B-Instruct from its original source. `--with-soundfont` fetches
MuseScore General. `--all` enables both. Use `--with-training-runs` for the optional
training records and `--base-url` to select a mirror. Each asset and extracted member
is checked against the manifest. Existing matching files are skipped.

Composer weights and derived baselines are released under CC BY-SA 4.0, with attribution
to MidiCaps by Melechovsky, Roy and Herremans (ISMIR 2024, dataset CC BY-SA 4.0).
The LoRA adapter is fine-tuned from Qwen2.5-0.5B-Instruct (Apache-2.0) on data derived
from MidiCaps. The Qwen base and SoundFont retain their upstream licences.

Training from scratch uses the following paths.

| Folder | Created by |
|---|---|
| `composer/transformer-base/` | `pixi run -e cuda train-composer --config configs/composer_base.yaml` |
| `composer/gru-base/` | `pixi run -e cuda train-composer --config configs/composer_gru.yaml` |
| `baselines/bigram.npy` | first `pixi run evaluate --system bigram` |
| `parser/base/Qwen2.5-0.5B-Instruct/` | `pixi run download-parser-base` |
| `parser/lora-r16/` | `pixi run -e cuda train-parser` (LoRA adapter, a few MB) |
| `soundfonts/MuseScore_General.sf3` | `pixi run fetch-soundfont` (optional) |

Each composer folder holds `best.pt` (lowest validation loss, used by the app), `last.pt`
(with optimizer state for `--resume`), `log.csv`, `curves.png` and `config.yaml`.
