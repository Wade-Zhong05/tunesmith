# tunesmith

Describe a piece of music in plain English and get a melody you can play, see and edit.
A fine-tuned small language model turns the description into musical attributes, and a
Transformer composer, trained from scratch on 168K captioned MIDI files, writes a
melody that follows them.

> **Status (2026-11-10): framework complete, models not trained yet.** All data, training,
> evaluation and app code runs end to end; the web app currently uses a random baseline.
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

*Planned; "TBD" is replaced as experiments finish.* Protocol in
[docs/architecture.md §6](docs/architecture.md#6-evaluation).

**Composer** (MidiCaps test files, attributes taken from the metadata):

| System | key match | in-scale notes | meter match | bars match | repetition | val. perplexity |
|---|---|---|---|---|---|---|
| real test melodies (reference) | TBD | TBD | TBD | TBD | TBD | — |
| uniform random (grammar only) | TBD | TBD | TBD | TBD | TBD | — |
| bigram Markov chain | TBD | TBD | TBD | TBD | TBD | TBD |
| GRU (3 x 1152, about 24M params) | TBD | TBD | TBD | TBD | TBD | TBD |
| **Transformer (8 x 512)** | TBD | TBD | TBD | TBD | TBD | TBD |

**Parser** (field accuracy on MidiCaps captions and template queries):

| Parser | mean field accuracy | exact match | hallucination | ms / query |
|---|---|---|---|---|
| keyword rules | TBD | TBD | TBD | TBD |
| Qwen2.5-0.5B zero-shot | TBD | TBD | TBD | TBD |
| Qwen2.5-0.5B few-shot (5 examples) | TBD | TBD | TBD | TBD |
| **Qwen2.5-0.5B + LoRA** | TBD | TBD | TBD | TBD |

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
