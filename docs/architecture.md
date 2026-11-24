# Architecture

## Pipeline

```
"a gentle piano waltz in A minor, slow and a little sad"
   | parser: Qwen2.5-0.5B + LoRA (baselines: keyword rules, zero-shot, few-shot)
   v
attributes {key: A minor, tempo_bpm: 70, meter: 3/4, instruments: [piano], mood: sad, ...}
   | prompt: BOS KEY_A_minor METER_3/4 TEMPO_slow INST1_piano ... SEP
   v
composer: decoder-only Transformer (baselines: GRU, bigram, uniform)
   | constrained decoding: grammar mask, key bias, classifier-free guidance, sampling
   v
REMI tokens -> MIDI (symusic) -> WAV (SoundFont) and piano roll -> web app
```

The two stages are trained and evaluated separately, and the user can see and override
what the parser understood. MuseCoco uses the same split; Text2midi is the end-to-end
alternative.

| Module | Role |
|---|---|
| `attributes.py` | attribute schema, vocabularies, MidiCaps label mapping |
| `tokenizer.py` | control tokens plus miditok REMI tokens in one vocabulary (421 melody, 504 band) |
| `music/` | melody and multi-track window extraction, key finding |
| `data/` | download, preparation, PyTorch dataset with augmentation |
| `model/` | Transformer (student), reference Transformer, GRU, sampling filters, constrained generation |
| `diffusion/` | piano-roll encoding, U-Net, DDPM training, DDIM sampling and editing |
| `understand/` | rule parser, prompts, LM parser, parser data, LoRA training |
| `train/`, `eval/`, `render/` | training loop, metrics and baselines, audio and piano rolls |
| `pipeline.py`, `app/`, `web/` | end-to-end use, FastAPI server, frontend |

## Data

All music comes from MidiCaps: 168,385 Lakh MIDI files with LLM-written captions and
automatically extracted labels. MidiCaps' own test files form the test split, and 2% of
the rest (by a hash of the path) the validation split.

**Melody windows.** The most melody-like track is chosen (high, mostly monophonic, many
notes) and reduced to one note per onset (skyline). The track is cut into
non-overlapping 8- or 16-bar windows. Key, tempo, mood and genre come from the metadata;
meter, length and instrument come from the window. Result: 452,863 / 9,598 / 51,180
windows and 119M training tokens.

**Band windows.** Tracks are merged by instrument family (at most 6 notes per onset), and
the four busiest families plus drums are kept and cut into 8-bar windows. The labels list
the families and drums actually present. Result: 654,645 / 13,915 / 73,791 windows and
625M training tokens. Windows over 2,048 tokens (6.6%) are dropped.

**Parser data.** 30,000 training rows: 20,000 MidiCaps captions and 10,000 short template
queries. A label is kept only if the caption mentions it. Validation and test have
1,500 rows each.

The labels are noisy. Keys are computed on whole songs, so real windows match their label
only 40–52% of the time. Mood and genre come from audio classifiers with low confidence.
The data is also imbalanced: 85% is in 4/4 and 45% is tagged happy.

## Models

**Composer.** A GPT-style decoder: token and learned position embeddings, 8 pre-norm
blocks (8 heads, d_model 512, MLP 2048), LM head tied to the embedding. The melody model
has a 1,024-token context (26.0M parameters), the band model 2,048 (26.5M). The
student's `transformer.py` and the reference `transformer_ref.py` share one interface and
one test file.

Training uses next-token cross-entropy on the music tokens only, with two augmentations:
transposition by -5 to +6 semitones (the KEY token moves too), and attribute dropout
(each control token becomes NULL with p = 0.15, the whole prompt with p = 0.1), which
makes classifier-free guidance possible. Optimisation: AdamW, learning rate 3e-4 with
warm-up and cosine decay, 40,000 steps of 48 sequences, bf16, length-bucketed batches.

**Baselines.** A 3 x 1152 GRU (24.4M) on the same data and loop, a bigram Markov chain,
and a uniform model. All of them decode with the same grammar.

**Diffusion composer.** It generates a whole 8-bar 4/4 piece at once as a piano roll of
shape 9 x 128 x 96: onset and sustain channels for four instrument slots plus a drum
channel, 128 sixteenth-note frames, 96 pitches. A U-Net (26.6M; 64 to 256 channels,
self-attention at the coarsest level) predicts the noise. It is conditioned on the
same control tokens, which scale and shift every residual block. Training uses a cosine
schedule with 1,000 steps on 570K 4/4 windows. Sampling uses 50 DDIM steps with guidance.
At every step, cells outside the allowed instruments and ranges are switched off, and
cells the user wants to keep are reset, which is how it rewrites chosen bars or writes
new parts around a kept one.

**Parser.** Qwen2.5-0.5B-Instruct with LoRA (r = 16, alpha = 32) on all attention and MLP
projections, 8.8M trainable parameters. It is fine-tuned with the loss on the JSON answer
only. Before parsing, stretched or slightly misspelt keywords are repaired ("sloooow",
"guita").

## Generation

- Grammar: Bar, TimeSig, then Position / (Program) / Pitch / Duration events. Positions
  increase within a bar, the meter is locked to the request, and the bar count is exact.
- Band mode: only the requested families play, each within its pitch range, and every
  requested part must play before the end.
- Rest limit: bar 1 has notes and at most one silent bar follows another.
- Key bias: out-of-key pitches get a logit penalty.
- Guidance: `logits = uncond + s * (cond - uncond)`, with s = 1.5 by default.
- Sampling: temperature, top-k, top-p. The app samples at temperature 1.0 without top-p,
  the setting whose statistics are closest to real music.

## Evaluation

| Question | Metrics |
|---|---|
| Does the parser understand the text? | accuracy per field where a label exists, hallucination where it does not, exact match, latency |
| Does the music follow the attributes? | key match, in-scale notes, meter and bar match, instrument precision and recall, drums match |
| Does it look like real music? | notes per bar, notes per onset, pitch-class entropy, 4-gram repetition, empty bars, compared with real test windows |
| Does it fit the description? | optional CLAP text-audio score, blind listening pages |

Scripts: `eval/run_eval.py`, `eval/eval_parser.py`, `eval/listening.py`.

## Student-implemented parts

`model/transformer.py` and `model/sampling_filters.py` are skeletons with TODO markers.
Their tests skip until each function works:

```bash
pixi run pytest tests/test_transformer.py tests/test_sampling_filters.py -q -rs
```

`configs/composer_base.yaml` trains the student Transformer into
`models/composer/transformer-base/`. To use it, set `composer_checkpoint` to
`models/composer/transformer-base/best.pt` in `configs/app.yaml`. For band mode, copy
`configs/composer_ensemble.yaml` and set `type: transformer`.

## Limitations

- MidiCaps is mostly amateur MIDI with machine labels, and the models inherit its
  quality.
- Band pieces are 8 bars long and have no velocity tokens. The diffusion model only
  handles 4/4.
- Audio is a General MIDI SoundFont rendering, so it sounds synthetic.
- The parser test labels share the rule parser's keyword tables; a human-written test set
  is missing.
- Lyrics are not implemented.
