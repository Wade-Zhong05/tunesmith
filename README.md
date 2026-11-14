# tunesmith

Describe a piece of music in plain English and get a melody you can play, see and edit.
A fine-tuned small language model turns the description into musical attributes, and a
Transformer composer, trained from scratch on 168K captioned MIDI files, writes a
melody that follows them.

> **Status (2026-11-13): GRU composer and LoRA parser trained and evaluated; Transformer not trained yet.**
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

## Video Links

- Demo video (3–5 min): *to be recorded*, `videos/demo.mp4`
- Technical walkthrough (5–10 min): *to be recorded*, `videos/walkthrough.mp4`

## Evaluation

Rows marked TBD are still to run. Protocol in
[docs/architecture.md §6](docs/architecture.md#6-evaluation).

**Composer.** 200 MidiCaps test prompts with the attributes taken from the metadata, 16
bars each, temperature 1.0, no top-p, seed 0. The same constrained decoder is used for
every system.

| System | key match | key match (incl. relative) | in-scale notes | pitch-class entropy | 4-gram repetition | notes / bar | pitch range | val. perplexity |
|---|---|---|---|---|---|---|---|---|
| real test melodies (reference) | 0.40 | 0.50 | 0.93 | 2.38 | 0.27 | 6.8 | 17.5 | — |
| uniform random (grammar only) | 0.02 | 0.08 | 0.62 | 3.41 | 0.00 | 3.1 | 58.3 | — |
| bigram Markov chain | 0.08 | 0.14 | 0.67 | 3.27 | 0.00 | 2.4 | 45.4 | 12.58 |
| GRU, no guidance | 0.29 | 0.42 | 0.92 | 2.36 | 0.17 | 5.1 | 16.7 | 2.33 |
| GRU, guidance 1.5 | **0.45** | 0.53 | 0.96 | 2.37 | 0.15 | 5.1 | 18.4 | 2.33 |
| GRU, guidance 1.5 + key bias 1.0 | 0.45 | 0.57 | 1.00 | 2.28 | 0.15 | 5.0 | 18.3 | 2.33 |
| **Transformer (8 x 512)** | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Meter match is 1.00 for every system, because the grammar enforces it. Bar-count match
is 0.98 for the GRU; the misses are sequences that reach the 1,024-token limit.

**Observations**
- The trained GRU is close to the real melodies on every musical statistic: pitch range,
  pitch-class entropy and in-scale share.
- It is far better than the bigram baseline: perplexity 2.33 vs 12.58.
- Classifier-free guidance raises key match from 0.29 to 0.45 and leaves the other
  statistics unchanged.
- Even real melodies match their MidiCaps key label only 40% of the time. The labels were
  computed on whole songs, and key finding on a single melody is ambiguous. So 0.40–0.50
  is the practical ceiling for this metric, and `key_source: detected` is worth trying.
- The GRU repeats motifs (0.15) less than real music (0.27). Long-range structure is
  where the Transformer is expected to help.
- GRU training: 40,000 steps (about 4 epochs) in 5.1 h on an RTX 4070 laptop GPU. The
  validation loss went from 1.23 to 0.846 with no overfitting
  (`models/composer/gru-base/curves.png`).

**Parser.** 1,500 test rows: 1,000 MidiCaps captions from held-out test files and 500
template queries.
- *Field accuracy*: on fields the gold label specifies.
- *Hallucination*: a value predicted where the gold label is null.
- *Latency*: GPU, batch 8, including model loading.

| Parser | mean field accuracy | exact match | hallucination | tempo accuracy | exact match (captions) | ms / query |
|---|---|---|---|---|---|---|
| keyword rules | 0.946 | 0.553 | 0.113 | 0.69 | 0.372 | 0.6 |
| Qwen2.5-0.5B zero-shot | 0.652 | 0.000 | 0.689 | 0.46 | 0.000 | 322 |
| Qwen2.5-0.5B few-shot (5 examples) | 0.624 | 0.015 | 0.334 | 0.44 | 0.007 | 346 |
| **Qwen2.5-0.5B + LoRA (r=16, 8.8M trainable)** | **0.952** | **0.751** | **0.013** | **0.79** | **0.626** | 439 |

**Observations**
- **LoRA fine-tuning is what makes the small LLM usable.** Without it, the 0.5B model
  rarely produces a fully correct answer: exact match is 0.00 zero-shot and 0.02
  few-shot. It also invents attributes and uses values outside the allowed vocabulary.
  Five in-context examples do not help a model this small.
- **LoRA vs rules.** The fine-tuned parser beats the rules on exact match (0.75 vs 0.55)
  and tempo (0.79 vs 0.69), and almost never hallucinates (0.013 vs 0.113). The rules
  remain slightly better on mood (0.92 vs 0.83 on captions) and are about 700 times
  faster.
- **Caveat.** The test labels were cleaned with the same keyword tables the rule parser
  uses, which favours the rules on key, instrument and meter. A human-written test set
  is still needed.
- **Training cost.** 3,750 steps (2 epochs) in 62 minutes on an RTX 4070 laptop GPU.
  Validation loss went from 0.042 to 0.031; the best checkpoint was at step 2,700.

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
| `src/tunesmith/` | the package: `attributes`, `tokenizer`, `music/`, `data/`, `model/`, `train/`, `understand/`, `render/`, `eval/`, `app/`, `pipeline.py`, `cli.py` |
| `configs/` | data, model, training, parser and app settings; keyword tables and prompt templates |
| `web/` | single-page frontend (no build step) |
| `tests/` | unit and end-to-end tests |
| `data/`, `models/` | downloaded and generated data and weights (git-ignored; see their READMEs) |
| `notebooks/` | analysis notebooks |
| `docs/` | [architecture](docs/architecture.md), [student guide](docs/student-guide.md) |
| `videos/` | demo and walkthrough videos |

See [ATTRIBUTION.md](ATTRIBUTION.md) for datasets, models, libraries and AI-tool use.
