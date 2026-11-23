# tunesmith

Describe a piece of music in plain English and get a melody you can play, see and edit.
A fine-tuned small language model turns the description into musical attributes, and a
Transformer composer, trained from scratch on 168K captioned MIDI files, writes a
melody that follows them.

> **Status (2026-11-23): melody (GRU, reference Transformer) and ensemble composers and the LoRA parser are
> trained and evaluated; the student's own Transformer and sampling filters are in progress.**
> All data, training, evaluation and app code runs end to end.
> Two model components are left for the student to implement (see
> [docs/student-guide.md](docs/student-guide.md)). Numbers below will be filled in as
> experiments finish.

## What it Does

tunesmith is a web app and command-line tool for text-to-melody generation. You type a
request such as "a gentle piano waltz in A minor, slow and a little sad". A small
instruction-tuned language model (Qwen2.5-0.5B with LoRA, fine-tuned on MidiCaps
captions) extracts the attributes it mentions: key, tempo, meter, instrument, mood, genre
and length. A decoder-only Transformer then writes the melody as a sequence of REMI music
tokens, conditioned on one control token per attribute. It was trained on melody excerpts
extracted from MidiCaps, the 168K-file captioned subset of the Lakh MIDI dataset.

**Ensemble mode** writes for up to four instruments plus drums: "an upbeat pop song with
piano, electric guitar and bass, with drums". It uses a second Transformer of the same
design, trained on multi-track 8-bar windows from the same MidiCaps files. The token stream
interleaves every part, and a Program token in front of each note names its instrument.
The parser returns the list of instruments and whether drums are wanted. The decoder lets
only those instruments play, each in its own pitch range, and keeps each requested part
from staying silent.

**Diffusion composer (experimental).** A second ensemble composer with a different
architecture: a 26.6M-parameter U-Net that denoises an 8-bar multi-track piano roll, the
whole piece at once, conditioned on the same attributes. Because it does not write left to
right, it can also rewrite chosen bars of an existing piece, or write new parts around a
part it keeps. Code and tests are done; training and evaluation are pending.

Generation is constrained by a token grammar, so every output decodes to valid MIDI with
the requested meter and number of bars. Classifier-free guidance and an optional in-key
bias make the melody follow the request more closely. The result is shown as a piano roll
with playback and can be downloaded as MIDI.

Every stage is evaluated against baselines:
- **text understanding:** a keyword parser, and zero-shot and few-shot prompting;
- **composition:** a random composer, a bigram Markov chain and a GRU, all generating
  through the same constrained decoder.

## Quick Start

Requirements: [pixi](https://pixi.sh). For training, an NVIDIA GPU. Full instructions:
[SETUP.md](SETUP.md).

```bash
pixi install           # CPU environment from pixi.lock (Python 3.12, PyTorch, miditok, FastAPI ...)
pixi run test          # unit tests
pixi run smoke         # one-minute end-to-end check on synthetic data
pixi run app           # web app on http://127.0.0.1:8000
```

From the command line:

```bash
pixi run parse "an epic brass theme at 140 bpm"
pixi run generate "a gentle piano waltz in A minor, slow" --n 2
```

Full pipeline (data, training, evaluation):

```bash
pixi run download-data                        # MidiCaps: 2 GB
pixi run prepare-data                         # melody windows -> tokens (about 10 min, CPU)
pixi run -e cuda train-composer --config configs/composer_base.yaml
pixi run make-parser-data
pixi run download-parser-base
pixi run -e cuda train-parser
pixi run evaluate --system models/composer/transformer-base/best.pt --name transformer
```

Ensemble (multi-instrument) mode:

```bash
pixi run prepare-ensemble                     # multi-track windows -> data/processed/midicaps_ensemble (about 5 min)
pixi run -e cuda train-ensemble               # 26M Transformer, 2,048-token context
pixi run generate "an upbeat pop song with piano, electric guitar and bass, with drums" --mode ensemble
```

## Video Links

- Demo video (3–5 min): *to be recorded*, `videos/demo.mp4`
- Technical walkthrough (5–10 min): *to be recorded*, `videos/walkthrough.mp4`

## Evaluation

Rows marked TBD are still to run. Protocol in
[docs/architecture.md §6](docs/architecture.md#6-evaluation).

**Composer.** 200 MidiCaps test prompts with the attributes taken from the metadata, 16
bars each, temperature 1.0, no top-p, seed 0. The same constrained decoder is used for
every system.

| System | key match | key match (incl. relative) | in-scale notes | pitch-class entropy | 4-gram repetition | notes / bar | empty-bar rate | val. perplexity |
|---|---|---|---|---|---|---|---|---|
| real test melodies (reference) | 0.40 | 0.50 | 0.93 | 2.38 | 0.27 | 6.8 | 0.11 | — |
| uniform random (grammar only) | 0.02 | 0.08 | 0.62 | 3.41 | 0.00 | 3.1 | 0.03 | — |
| bigram Markov chain | 0.08 | 0.14 | 0.67 | 3.27 | 0.00 | 2.4 | 0.12 | 12.58 |
| GRU (24.4M), no guidance | 0.29 | 0.42 | 0.92 | 2.36 | 0.17 | 5.1 | 0.14 | 2.33 |
| GRU, guidance 1.5 | 0.45 | 0.53 | 0.96 | 2.37 | 0.15 | 5.1 | 0.17 | 2.33 |
| GRU, guidance 1.5 + key bias 1.0 | 0.45 | 0.57 | 1.00 | 2.28 | 0.15 | 5.0 | 0.17 | 2.33 |
| Transformer (26.0M, reference impl.), no guidance | 0.32 | 0.42 | 0.92 | 2.36 | **0.29** | 4.7 | 0.18 | **2.13** |
| Transformer, guidance 1.5 | **0.50** | 0.57 | 0.96 | 2.37 | 0.24 | 4.4 | 0.22 | **2.13** |
| Transformer, guidance 1.5 + key bias 1.0 | **0.53** | **0.59** | 0.99 | 2.28 | 0.25 | 4.4 | 0.22 | **2.13** |
| Transformer, guidance 1.5 + rest limit | 0.50 | 0.57 | 0.96 | 2.44 | 0.25 | 5.5 | 0.06 | **2.13** |
| **Transformer, guidance 1.5 + key bias 1.0 + rest limit (app default)** | **0.53** | 0.58 | 0.99 | 2.37 | 0.25 | 5.3 | 0.06 | **2.13** |
| Transformer, guidance 1.5 + rest limit + key bias 1.0, top-p 0.95 | 0.49 | 0.57 | 1.00 | 2.06 | 0.52 | 6.5 | 0.04 | **2.13** |
| same, temperature 0.9, top-p 0.92 (earlier web defaults) | 0.42 | 0.54 | 1.00 | 1.83 | 0.75 | 7.1 | 0.02 | **2.13** |
| Transformer (student implementation) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

- Meter match is 1.00 for every system, because the grammar enforces it.
- Bar-count match is 0.98–0.995 for the learned models; the misses are sequences that
  reach the 1,024-token limit.
- Both learned composers were trained for 40,000 steps of 48 sequences on the same data.
  The Transformer run used length-bucketed batches, the GRU run random batches.

**Observations**
- **Transformer vs GRU at equal size and steps.**
  - Validation perplexity is 2.13 vs 2.33.
  - Motif repetition is much closer to real music: 0.29 vs 0.17, against 0.27 for real
    melodies. This is the long-range structure attention was expected to help with.
  - Key match with guidance is higher: 0.50 vs 0.45.
  - Training took 76 min vs 307 min, because attention parallelises over time and
    bucketing removes padding.
- **Guidance and baselines.**
  - Classifier-free guidance raises key match for both models (GRU 0.29 → 0.45,
    Transformer 0.32 → 0.50) and leaves pitch statistics unchanged.
  - Both learned models are far better than the baselines (bigram perplexity 12.58).
- **Empty bars and the rest limit.**
  - Without a limit, the Transformer leaves more bars empty than real music (0.22 vs
    0.11) and writes fewer notes per bar (4.4 vs 6.8).
  - Some pieces started with 9 to 11 silent bars, which was very noticeable in the app.
  - The grammar now requires notes in bar 1 and allows at most one silent bar in a row
    (`run_eval.py --max-empty-bars`, default 1; -1 turns it off).
  - This lowers the empty-bar rate from 0.22 to 0.06 and raises notes per bar to 5.5.
    Key match, pitch statistics and repetition stay where they were.
- **Truncated sampling makes melodies repetitive.** The last two rows use the reference
  top-p filter:
  - With top-p 0.95, 4-gram repetition doubles (0.25 → 0.52, against 0.27 for real
    melodies), and pitch-class entropy and pitch range drop.
  - With the web page's earlier defaults (temperature 0.9, top-p 0.92) it reaches 0.75.
    This matches the student's impression that the app's melodies sounded odd and
    monotonous.
  - The app now samples at temperature 1.0 without top-p truncation, with guidance 1.5,
    key bias 1.0 and the rest limit (the bold row): the best key match, with repetition,
    pitch-class entropy and pitch range close to real melodies. The sliders still allow
    lower values.
- **Key labels.** Even real melodies match their MidiCaps key label only 40% of the time.
  The labels were computed on whole songs, and key finding on a single melody is
  ambiguous. So 0.40–0.50 is roughly the ceiling for this metric, and
  `key_source: detected` is worth trying.
- **Training curves.** Neither model overfits: `models/composer/*/curves.png`.
  - GRU: 5.1 h; validation loss 1.23 → 0.846.
  - Transformer: 1.3 h; validation loss 1.27 → 0.757, still decreasing slowly at
    40,000 steps.

**Parser.** 1,500 test rows: 1,000 MidiCaps captions from held-out test files and 500
template queries.
- *Field accuracy*: on fields the gold label specifies.
- *Hallucination*: a value predicted where the gold label is null.
- *Latency*: GPU, batch 8, including model loading.

| Parser | mean field accuracy | exact match | hallucination | tempo accuracy | instruments F1 | drums accuracy | exact match (captions) | ms / query |
|---|---|---|---|---|---|---|---|---|
| keyword rules | 0.945 | 0.561 | 0.102 | 0.69 | 1.00* | 1.00* | 0.372 | 1.3 |
| Qwen2.5-0.5B zero-shot | 0.519 | 0.000 | 0.667 | 0.35 | 0.30 | 0.69 | 0.000 | 392 |
| Qwen2.5-0.5B few-shot (5 examples) | 0.626 | 0.029 | 0.282 | 0.42 | 0.37 | 0.66 | 0.004 | 359 |
| **Qwen2.5-0.5B + LoRA (r=16, 8.8M trainable)** | **0.953** | **0.755** | **0.016** | **0.77** | 1.00 | 1.00 | **0.632** | 646 |

\* The instrument-list and drums labels were derived with the same keyword tables as
the rule parser, so the rules score 1.00 on them by construction.

The parser was retrained for the ensemble schema (instrument lists and drums) on
2026-11-23 (`models/parser/lora-r16-v2`). All four rows were re-run on the rebuilt test
set; the earlier single-instrument results are kept in `outputs/eval_parser/v1/`.

**Observations**
- **LoRA fine-tuning is what makes the small LLM usable.** Without it, the 0.5B model
  rarely produces a fully correct answer: exact match is 0.00 zero-shot and 0.02
  few-shot. It also invents attributes and uses values outside the allowed vocabulary.
  Five in-context examples do not help a model this small.
- **LoRA vs rules.**
  - The fine-tuned parser beats the rules on exact match (0.76 vs 0.56; 0.63 vs 0.37 on
    real captions) and tempo (0.77 vs 0.69).
  - It almost never hallucinates (0.016 vs 0.102).
  - The rules remain better on mood (0.94 vs 0.91) and are about 500 times faster.
- **Instrument lists.** Without fine-tuning the base model finds the requested
  instruments poorly (F1 0.30 zero-shot, 0.37 few-shot); the LoRA parser matches the
  labels almost perfectly (0.999).
- **Caveat.** The test labels were cleaned with the same keyword tables the rule parser
  uses, which favours the rules on key, meter, instruments and drums. A human-written
  test set is still needed.
- **Training cost.** 3,750 steps (2 epochs) in 156 minutes on an RTX 4070 laptop GPU
  (the first, single-instrument adapter took 62 minutes). Validation loss went from
  0.033 to 0.023; the best checkpoint (validation field accuracy 0.947) was at step 2,700.

**Ensemble composer.**
- **Setup.** 200 MidiCaps test prompts. Each requests the instruments and drums listed in
  the file's metadata. 8 bars each, temperature 1.0, no top-p. These runs predate the
  rest limit; the ensemble model rarely leaves bars empty (0.018), so it changes little.
- **Grammar ablation.** "No grammar" rows decode with only an end-of-sequence limit, so the
  instrument metrics there measure what the model itself learned.

| System | instrument recall | instrument precision | drums match | key match | in-scale notes | notes per onset | notes / bar | 4-gram repetition | empty-bar rate | bars match | val. perplexity |
|---|---|---|---|---|---|---|---|---|---|---|---|
| real test windows (reference) | 1.00 | 1.00 | 1.00 | 0.52 | 0.93 | 2.85 | 22.8 | 0.20 | 0.009 | — | — |
| uniform random (grammar only) | 1.00 | 1.00 | 1.00 | 0.02 | 0.61 | 2.11 | 5.8 | 0.00 | 0.032 | 1.00 | — |
| bigram Markov chain | 0.99 | 1.00 | 1.00 | 0.06 | 0.66 | 2.95 | 8.8 | 0.00 | 0.013 | 1.00 | 15.97 |
| Transformer (26.5M), no guidance | 0.98 | 1.00 | 0.995 | 0.50 | 0.92 | 2.97 | 22.5 | 0.13 | 0.024 | 0.99 | **1.72** |
| **Transformer, guidance 1.5** | **1.00** | 1.00 | **1.00** | **0.60** | 0.96 | 2.84 | 21.8 | 0.14 | 0.018 | 1.00 | 1.72 |
| Transformer, no guidance, no grammar | 0.97 | 0.95 | 0.99 | 0.48 | 0.93 | 2.89 | 21.5 | 0.12 | 0.022 | 0.66 | 1.72 |
| Transformer, guidance 1.5, no grammar | 1.00 | 0.97 | 1.00 | 0.62 | 0.96 | 2.84 | 21.3 | 0.12 | 0.021 | 0.75 | 1.72 |
| Diffusion U-Net (26.6M), guidance 2.0 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | — |

**Observations (ensemble)**
- **The model learned instrument control itself.** With the grammar switched off it still
  plays the requested instruments (recall 0.97–1.00, precision 0.95–0.97) and follows
  the drums request (0.99–1.00). The grammar's real contribution is structure: without
  it the requested bar count is met only 66–75% of the time, against 99–100% with it.
- **It matches real ensembles closely** on density (21.8 vs 22.8 notes per bar),
  polyphony (2.84 vs 2.85 notes per onset), pitch statistics and empty bars. The bigram
  baseline is far off (8.8 notes per bar, key match 0.06, perplexity 15.97).
- **Guidance helps key adherence**: key match rises from 0.50 to 0.60, above the 0.52 of
  the real windows, at no cost to the other statistics.
- **Weak spot**: motif repetition (0.13–0.14 vs 0.20), as in melody mode, but less
  severe.
- **Training**: 40,000 steps (about 3 epochs of 625M tokens) in 6.5 h on the RTX 4070
  laptop GPU. Validation loss 1.57 → 0.542 with no overfitting
  (`models/composer/ensemble-ref/curves.png`). Generation takes about 4 s per 8-bar piece
  on the GPU, in batches of 16 with guidance.

**Listening test.** `pixi run listening --mode melody|ensemble` builds a blind A/B page
under `outputs/app/listening/` (served by the app at `/files/listening/...`). Each row has
three shuffled clips for the same attributes: the real test excerpt, plain sampling, and
temperature 0.85 with top-p 0.9.

**Data (measured):**
- All 168,385 MidiCaps files were prepared in 133 s with no errors.
- That gives 452,863 / 9,598 / 51,180 melody windows (train / val / test), and 118.7M
  training tokens.
- 12.8% of files yield no usable window.

The parser dataset has 30,000 training rows and 1,500 rows each for validation and test.

## Individual Contributions

*Solo project unless stated otherwise; this section is required only for groups.*

---

### Repository layout

| Path | Contents |
|---|---|
| `src/tunesmith/` | the package: `attributes`, `tokenizer`, `music/`, `data/`, `model/`, `train/`, `understand/`, `render/`, `eval/`, `app/`, `diffusion/`, `pipeline.py`, `cli.py` |
| `configs/` | data, model, training, parser and app settings; keyword tables and prompt templates |
| `web/` | single-page frontend (no build step) |
| `tests/` | unit and end-to-end tests |
| `data/`, `models/` | downloaded and generated data and weights (git-ignored; see their READMEs) |
| `notebooks/` | analysis notebooks |
| `docs/` | [architecture](docs/architecture.md), [student guide](docs/student-guide.md) |
| `videos/` | demo and walkthrough videos |

See [ATTRIBUTION.md](ATTRIBUTION.md) for datasets, models, libraries and AI-tool use.
