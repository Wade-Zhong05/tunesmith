# Setup

## 1. Install

The project uses [pixi](https://pixi.sh) for every dependency, including Python itself.
Install pixi once (Windows: `winget install prefix-dev.pixi`, or see the pixi website),
then from the repository root:

```bash
pixi install            # default environment: CPU PyTorch (app, tests, evaluation on CPU)
pixi install -e cuda    # CUDA environment: PyTorch with CUDA 12.6 (training); needs an NVIDIA driver
```

The two environments are identical apart from the PyTorch build. Use the CUDA one by
adding `-e cuda` to any task, e.g. `pixi run -e cuda train-composer`. Check the GPU:

```bash
pixi run -e cuda python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

No pixi available (Kaggle, Colab, a plain virtualenv)? Install a matching PyTorch build,
then `pip install -r requirements.txt`, and run the same modules with
`python -m tunesmith.<module>` (the pixi tasks in `pyproject.toml` list them).

## 2. Check that everything works

```bash
pixi run test      # unit tests (the student TODO tests show as "skipped" until implemented)
pixi run smoke     # synthetic data -> tiny GRU composer -> MIDI + WAV, about one minute on CPU
pixi run app       # http://127.0.0.1:8000 (random baseline until a composer is trained)
```

## 3. Data

```bash
pixi run download-data      # MidiCaps from the Hugging Face Hub -> data/raw/midicaps (train.json 428 MB + MIDI 1.6 GB)
pixi run prepare-data       # -> data/processed/midicaps (133 s with 31 worker processes)
```

`configs/data.yaml` controls melody extraction, window lengths, the label source for the
key and the validation fraction. For a quick run use `pixi run prepare-data --limit 5000`.
MidiCaps' own `test_set` files form the test split; 2% of the remaining files (chosen by a
hash of the path) form the validation split.

## 4. Train the composer (attributes -> melody)

```bash
pixi run -e cuda train-composer --config configs/composer_base.yaml   # Transformer (student part)
pixi run -e cuda train-composer --config configs/composer_gru.yaml    # GRU baseline
pixi run -e cuda train-composer --config configs/composer_base.yaml --resume
```

Each run writes `best.pt`, `last.pt`, `log.csv`, `curves.png` and `config.yaml` to the
config's `out_dir`. The script prints tokens per second and an ETA. Expected on the RTX
4070 laptop GPU (8 GB): a usable model within about an hour, a full run in 2–4 hours.

### Training on Duke CS OnDemand

1. Open <https://ondemand.cs.duke.edu>, start **Interactive Apps -> JupyterLab** with
   partition `compsci-gpu` and 1 GPU, then open a terminal.
2. Install pixi in your home directory and get the code:
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash && source ~/.bashrc
   git clone <your repository URL> tunesmith && cd tunesmith
   pixi install -e cuda
   ```
3. Run the data and training tasks as above. Sessions have a time limit: `last.pt` is
   saved every `save_every` steps, and `--resume` continues from it.

### Ensemble (multi-instrument) mode

```bash
pixi run prepare-ensemble          # -> data/processed/midicaps_ensemble (318 s with 31 worker processes, about 0.7 GB)
pixi run -e cuda train-ensemble    # configs/composer_ensemble.yaml -> models/composer/ensemble-ref (about 5.5 h on the 4070)
pixi run -e cuda train-composer --config configs/composer_ensemble_medium.yaml   # 86M variant, for a >=16 GB GPU
pixi run generate "a calm piece for strings, harp and synth pad, no drums" --mode ensemble
pixi run smoke --mode ensemble --model-type transformer_ref   # one-minute CPU check
```

Evaluate with `pixi run evaluate --data-dir data/processed/midicaps_ensemble --system ...`
(same systems as below). The default length is 8 bars, as in training.

### Diffusion composer (experimental)

Uses the prepared ensemble data; no extra preprocessing. See `docs/architecture.md` §11.

```bash
pixi run -e cuda train-diffusion       # configs/diffusion_ensemble.yaml -> models/diffusion/ensemble-unet (about 4.6 h on the 4070)
pixi run train-diffusion --config configs/diffusion_tiny.yaml --device cpu   # one-minute check
pixi run -e cuda diffusion-sample --text "a jazz groove with electric piano, sax and bass, with drums" --n 2
pixi run -e cuda diffusion-sample --edit song.mid --bars 5 6     # rewrite bars 5-6, keep the rest
pixi run -e cuda diffusion-sample --edit song.mid --keep 1       # keep part 1, rewrite the others
pixi run -e cuda evaluate --data-dir data/processed/midicaps_ensemble --system models/diffusion/ensemble-unet/best.pt --cfg-scale 2.0 --name ens_diffusion
```

Training writes a few MIDI files to `models/diffusion/ensemble-unet/samples/` at every
evaluation, so the model can be heard while it trains.

## 5. Train the parser (text -> attributes)

```bash
pixi run make-parser-data          # -> data/parser/{train,val,test}.jsonl (needs data/raw/midicaps/train.json)
pixi run download-parser-base      # Qwen/Qwen2.5-0.5B-Instruct -> models/parser/base/
pixi run -e cuda train-parser --out-dir models/parser/lora-r16-v2   # LoRA adapter (about 2.5 h on the 4070)
```

**Which parser the app uses.**
- The app reads descriptions with this adapter (`parser: lm` and `parser_adapter` in
  `configs/app.yaml`).
- If the base model or the adapter is missing, it falls back to the keyword rules
  (`parser: rules`) and says so in `/api/health`.
- Parsing takes about 0.5 s per new description on the GPU (`pixi run -e cuda app`) and
  7–10 s on the CPU. Repeated descriptions come from a cache.

## 6. Evaluate

```bash
pixi run evaluate --system data --name reference                 # statistics of real test melodies
pixi run evaluate --system uniform
pixi run evaluate --system bigram
pixi run evaluate --system models/composer/gru-base/best.pt --name gru
pixi run evaluate --system models/composer/transformer-base/best.pt --name transformer
pixi run evaluate-parser --parser rules
pixi run evaluate-parser --parser lm --mode zero
pixi run evaluate-parser --parser lm --mode few
pixi run evaluate-parser --parser lm --mode lora
```

Results go to `outputs/eval/<name>/summary.json` and `outputs/eval_parser/<name>/`.
Add `--render --clap` to `evaluate` for audio and CLAP text-audio scores (downloads a CLAP
model on first use).

## 7. Better audio (optional)

```bash
pixi run fetch-soundfont           # MuseScore General SF3 (MIT) -> models/soundfonts/
```

Without it, WAV previews use a small built-in synthesizer.

## Troubleshooting

- **`torch.cuda.is_available()` is False in the cuda environment:** update the NVIDIA
  driver. The CUDA 12.6 wheels need a recent driver; the bundled CUDA runtime means no
  CUDA toolkit is required.
- **Hugging Face downloads fail:** the dataset and the Qwen model come from
  huggingface.co. Download them on a network that reaches it (Duke CS OnDemand does), then
  copy `data/raw/midicaps` and `models/parser/base`.
- **"soundfont rendering failed (Unable to convert function return value ...)":**
  symusic 0.6.0's SoundFont synthesizer returns a type its Python binding cannot convert
  on this setup (seen on Windows, 2026-11-13). WAV previews then fall back to the
  built-in numpy synthesizer. The MIDI files are unaffected: open them in MuseScore or a
  media player for better sound.
- **"The Transformer composer is still a skeleton":** implement
  `src/tunesmith/model/transformer.py` (see docs/student-guide.md), or train the GRU.
