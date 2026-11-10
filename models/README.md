# models/

Weights are not committed (see `.gitignore`); this file documents where each run writes.

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
