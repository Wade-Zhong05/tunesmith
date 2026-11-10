# tunesmith: Architecture

> Version 0.2 (2026-11-10). Framework implemented and tested end to end; models not yet trained.

## 1. Problem and approach

**Goal.** Natural-language description in, instrumental melody out (MIDI, audio
preview, piano roll).

**Research question.** Does splitting text-to-music into two stages work better than
mapping text directly to notes, and how much does each component contribute?
1. *Understanding*: a small fine-tuned language model maps text to explicit musical
   attributes.
2. *Composition*: a Transformer trained from scratch generates music conditioned on
   those attributes.

Splitting the problem has three benefits:
- each stage can be trained on data that fits it;
- each stage can be evaluated separately (attribute accuracy; does the music follow the
  attributes);
- the user can see and override what was understood.

Related work:
- MuseCoco (text -> attributes -> music) uses the same two-stage idea.
- Text2midi (AAAI 2025) is the end-to-end alternative and supplies the comparison
  metrics.

## 2. Pipeline

```
"a gentle piano waltz in A minor, slow and a little sad"
   |
   | (1) parser:  keyword rules | Qwen2.5-0.5B zero/few-shot | Qwen2.5-0.5B + LoRA (ours)
   v
MusicAttributes {key: A minor, tempo_bpm: 70, meter: 3/4, instrument: piano, mood: sad, genre: null, bars: 16}
   |                                (the user may override any field in the UI)
   | (2) prompt:  BOS KEY_A_minor METER_3/4 TEMPO_slow INSTRUMENT_piano MOOD_sad GENRE_NULL BARS_16 SEP
   v
composer (decoder-only Transformer, ~25M params; GRU / bigram / uniform as baselines)
   | (3) constrained decoding: REMI grammar mask + key bias + classifier-free guidance
   |     + temperature / top-k / top-p
   v
REMI tokens: Bar TimeSig_3/4 Position_0 Pitch_69 Velocity_79 Duration_1.0.8 ... EOS
   |
   | (4) decode -> MIDI (symusic), clip at the last bar, set program and tempo
   v
MIDI + WAV preview (SoundFont or numpy synth) + piano-roll JSON -> web UI
```

| Module | Responsibility |
|---|---|
| `attributes.py` | the attribute schema (pydantic), vocabularies, MidiCaps label mapping, key/tempo helpers |
| `tokenizer.py` | one vocabulary: 91 special and control tokens + miditok REMI tokens (421 in total); transposition tables; token-type tables for the grammar |
| `music/melody.py` | lead-track choice, skyline monophonisation, bar-aligned windows |
| `music/keyfinding.py` | Krumhansl-Schmuckler key detection |
| `data/` | download, preparation (multiprocessing), synthetic toy data, PyTorch dataset with augmentation |
| `model/transformer.py` | **student-implemented** composer |
| `model/rnn.py` | GRU composer (architecture baseline) |
| `model/sampling_filters.py` | **student-implemented** temperature / top-k / top-p |
| `model/generate.py` | grammar state machine, key bias, CFG, batched decoding with KV cache |
| `train/train_composer.py` | training loop |
| `understand/` | rule parser, prompts, LLM parser, parser dataset builder, LoRA training |
| `render/` | WAV rendering, piano-roll data and plots |
| `eval/` | music metrics, parser metrics, baselines, CLAP score, evaluation scripts |
| `pipeline.py`, `cli.py`, `app/server.py`, `web/` | end-to-end use |

## 3. Data

### 3.1 MidiCaps → composer training data

1. **Lead-track choice.** Among non-drum, non-bass tracks with at least 16 notes, pick
   the one maximising `mean pitch / 12 + 3 * monophony + log(#notes)`.
2. **Skyline.** Keep the highest note at each onset and cut every note at the next
   onset, giving a monophonic melody.
3. **Windows.** Non-overlapping 8- or 16-bar windows aligned to bar lines of the first
   time signature (stopping at the first meter change), at most 4 per file, at least 8
   notes each.
4. **Labels.** Key, tempo, mood and genre come from the MidiCaps metadata (mood and
   genre mapped to 10 classes each). Meter and bars come from the window. Instrument is
   the family of the melody track's GM program (21 families). Option
   `key_source: detected` relabels the key by key-finding on the window itself (an
   ablation).
5. **Tokens.** REMI with 8 positions per beat, 8 velocity bins, time signatures and no
   tempo tokens (tempo is a control token). Windows longer than 1,024 tokens are
   dropped.
6. **Splits.** MidiCaps' `test_set` files form the test split; 2% of the other files
   (hash of the path) form validation; the rest is training. All windows of a file stay
   in one split.

**Measured on the full dataset (2026-11-10, 31 worker processes):**

| Quantity | Value |
|---|---|
| source files | 168,385 in 133 s (1,286 files/s), 0 errors |
| files without a usable window | 21,509 (12.8%), mostly no melody-like track |
| windows train / val / test | 452,863 / 9,598 / 51,180 |
| tokens train / val / test | 118.7M / 2.5M / 13.4M |
| window length | median 206 tokens, 95th percentile 650, max 1,024 |

**Label imbalance (training windows, 20K sample):**

| Field | Distribution |
|---|---|
| meter | 4/4 85%, 3/4 6%, 2/4 4%, other 3%, 6/8 2% |
| mood | happy 45%, calm 19%, epic 17%, energetic 9%, dreamy 4% |
| genre | electronic 40%, pop 34%, classical 14%, rock 7%, soundtrack 3% |
| bars | 8 bars 72%, 16 bars 28% |

Key, instrument and tempo are almost always present: tempo is NULL in 0.8% of windows.
The imbalance is a data-quality issue to address, e.g. with attribute-balanced sampling,
and to report: rare combinations such as a sad 3/4 melody are under-represented.

### 3.2 Parser data

- **MidiCaps captions with label cleaning.** A metadata attribute becomes a target only
  if the caption mentions it (keyword tables in `configs/keywords.yaml`); otherwise the
  target is null. The instrument target is the first instrument the caption names.
- **Template queries.** 14 templates (`configs/parser_templates.yaml`) filled with
  sampled attribute words. They are shorter and more casual than MidiCaps captions.

**Built (2026-11-10):**
- train: 30,000 rows (20,000 captions + 10,000 templates);
- val and test: 1,500 rows each (1,000 + 500);
- training-target coverage after cleaning: instrument 83%, key 78%, genre 78%,
  tempo 78%, mood 75%, meter 66%, bars 5% (only templates mention bars).

## 4. Models

### 4.1 Composer (student-implemented Transformer)

**Architecture.** GPT-style decoder: token embedding + learned positional embedding;
8 pre-LayerNorm blocks with 8-head causal self-attention and a 2048-wide GELU MLP;
final LayerNorm; LM head tied to the token embedding. d_model is 512 and the context is
1,024 tokens, about 25M parameters.

**Optional extensions:**
- a KV cache (about 30-50x faster generation on CPU);
- RoPE.

**Training objective.** Next-token cross-entropy on music tokens only: prompt targets are
masked.

**Augmentation:**
- Random transposition in [-5, +6] semitones (p = 0.8). Pitch tokens *and* the KEY token
  are shifted together.
- Attribute dropout: each control token becomes NULL with p = 0.15, and the whole prompt
  with p = 0.1.

Dropout enables partial descriptions and classifier-free guidance:

```
logits = uncond + s * (cond - uncond)
```

**Optimisation:**
- AdamW (betas 0.9 / 0.95, weight decay 0.1 on matrices only);
- learning rate 3e-4 with 1,000 warm-up steps and cosine decay to 10%;
- batch 24 x 2 accumulation steps;
- bf16 autocast on the RTX 4070, gradient clipping at 1.0;
- early stopping on validation loss.

### 4.2 Baselines (same decoder, same grammar)

| Name | What it is |
|---|---|
| uniform | all tokens equally likely; shows what the grammar and key bias alone produce |
| bigram | first-order Markov chain over tokens, fitted on the training split |
| GRU | 3 x 1152 GRU language model (about 24M parameters), same data and training loop |
| real data | the test windows themselves, giving reference statistics |

### 4.3 Parser

**Model.** Qwen2.5-0.5B-Instruct (Apache-2.0) with LoRA on all attention and MLP
projections: r = 16, alpha = 32, dropout 0.05, about 1% of the parameters trainable.

**Training.** Supervised fine-tuning on chat-formatted examples with the loss on the
JSON answer only. The adapter with the best validation field accuracy is kept.

**Baselines:**
- the keyword parser;
- zero-shot prompting (schema in the system prompt);
- few-shot prompting (5 worked examples).

The 1.5B model is a drop-in alternative. Avoid the 3B model: its licence is not
Apache-2.0.

## 5. Generation details

- **Grammar** (`model/generate.py`):
  - the sequence follows SEP → Bar → TimeSig → (Position → Pitch → Velocity → Duration)* → Bar … → EOS;
  - positions strictly increase within a bar and stay below its length;
  - the time signature is locked to the requested meter; an unspecified meter is limited
    to 4/4, 3/4, 2/4 or 6/8;
  - EOS is allowed only after the requested number of bars, and Bar only before it.
- **Key bias.** Subtract `key_bias` from the logits of out-of-key pitches (natural and
  harmonic minor allowed).
- **Sampling.** Temperature, top-k and top-p, implemented by the student in
  `sampling_filters.py`. Greedy decoding when temperature is 0.
- **Batching.** Several versions per request are generated as one batch; with CFG the
  unconditional prompts join the same batch.

## 6. Evaluation

| Question | Metric | Script |
|---|---|---|
| Does the parser understand the text? | per-field accuracy where the gold label exists, hallucination rate where it does not, exact match, JSON validity, latency; by source (captions / templates) | `eval/eval_parser.py` |
| Does the music follow the attributes? | key match (and up to the relative key), share of in-scale notes, meter match, bar match | `eval/run_eval.py` |
| Does it look like real music? | note density, pitch range, pitch-class entropy, repetition of 4-note patterns, empty-bar rate, against the real-data reference | `eval/run_eval.py` |
| Does it fit the description? | CLAP text-audio similarity on rendered audio (as in Text2midi); a small listening test | `eval/run_eval.py --clap` |
| Is it fast? | seconds per piece, ms per parse | both scripts |

**Planned ablations** (one change at a time, on the same prompts and seeds):
1. CFG scale 1.0 / 1.5 / 3.0;
2. key bias 0 / 1 / 2;
3. grammar on / off;
4. transposition augmentation on / off;
5. attribute dropout on / off;
6. key labels from metadata vs detected;
7. Transformer vs GRU at a similar parameter count.

Hyperparameter sweeps on validation loss: learning rate, model width / depth, dropout.

## 7. Run-time estimates (RTX 4070 laptop GPU, 8 GB)

| Step | Estimate |
|---|---|
| download MidiCaps (2 GB) | 5–20 min |
| prepare-data (168K files, CPU) | measured: 133 s with 31 worker processes |
| composer, Transformer base | first usable model in about 1 h; full run 2–4 h |
| parser LoRA, 0.5B, 2 epochs over 30K rows | 30–60 min |
| evaluation (200 pieces per system) | a few minutes per system |

## 8. Mapping to the CS372 rubric (planning aid)

| Work | Rubric item |
|---|---|
| Transformer composer written and trained by the student | Defined and trained a custom neural network architecture (5) |
| Melody generation system | Implemented music generation model (7) |
| Parser LM output → composer input → renderer | Multi-stage ML pipeline (7) |
| Qwen + LoRA on the instruction-formatted parser data | Fine-tuned transformer LM (7) + PEFT (+3) + SFT (+3) |
| Temperature / top-k / top-p, grammar, CFG | Custom generation with a sampling strategy (5) |
| Transposition + attribute dropout, with ablation | Data augmentation with evidence (5) |
| Lead-track extraction, label cleaning, window filtering, with evidence | Preprocessing for data quality (7) |
| Transformer vs GRU vs bigram vs uniform | Compared architectures / approaches (7) |
| CFG, grammar, key bias, augmentation, label source | Ablation study (7) |
| Zero-shot vs few-shot vs LoRA parser | In-context learning (5) |
| FastAPI + web UI with rate limiting, validation, logging, error handling | Web application (10), production-grade deployment (10) |
| Failure cases by attribute and genre | Error analysis (7) |
| Hyperparameter sweeps | Hyperparameter tuning (5) |

Supporting 3-point items: train/val/test split, training curves, DataLoader, LR
schedule, LayerNorm, mixed precision and gradient clipping, GPU training, baselines,
three metrics, inference time.

## 9. Known limitations

- Melodies only: no accompaniment or chords (the `piano` reduction mode is prepared but
  not the default).
- Lyrics are reserved in the schema (`lyrics: null`) but not implemented.
- Mood and genre labels come from audio classifiers run on MidiCaps renders; they are
  noisy.
- MidiCaps captions are formulaic, so the template queries and a human-written test set
  matter for judging the parser.
