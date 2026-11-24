# tunesmith

tunesmith writes original music from a short English description. A fine-tuned
Qwen2.5-0.5B reads the request, and Transformers trained from scratch on MidiCaps
compose a melody or a small band arrangement as MIDI, played back in the browser.

## What it Does

You type something like "a gentle piano waltz in A minor, slow and a little sad". A
LoRA-tuned Qwen2.5-0.5B turns it into attributes (key, tempo, meter, instruments, drums,
mood, genre, length), and a decoder-only Transformer writes REMI music tokens conditioned
on them: either one melody line or up to four instruments plus drums. A token grammar
keeps every output valid, and classifier-free guidance and an in-key bias make the music
follow the request more closely. The web app shows what it understood, lets you change
it, plays the result on a piano roll and gives you the MIDI file. An experimental
diffusion model (a U-Net over 8-bar piano rolls) can also rewrite chosen bars of a piece.
Each part is compared with baselines: keyword rules and zero/few-shot prompting for the
parser, and random, bigram and GRU composers for the music.

## Quick Start

On Windows, double-click `start_tunesmith.bat`. It installs pixi and the environment,
downloads the models on the first run and opens the app in the browser. Elsewhere:

```bash
pixi install
pixi run download-models --all   # trained models, Qwen base model, SoundFont
pixi run app                     # http://127.0.0.1:8000
```

With an NVIDIA GPU, use `pixi install -e cuda` and `pixi run -e cuda app`; parsing and
generation are much faster. From the command line:

```bash
pixi run generate "a gentle piano waltz in A minor, slow" --n 2
pixi run generate "an upbeat pop song with piano, electric guitar and bass, with drums" --mode ensemble
```

Training and evaluation are covered in [SETUP.md](SETUP.md).

## Video Links

- Demo (3–5 min): `videos/demo.mp4` (not recorded yet)
- Technical walkthrough (5–10 min): `videos/walkthrough.mp4` (not recorded yet)

## Evaluation

Every composer generates through the same constrained decoder for 200 MidiCaps test
prompts (attributes from the metadata, seed 0, temperature 1.0, no top-p unless noted).
The metrics are defined in [docs/architecture.md](docs/architecture.md#evaluation).

**Melody, 16 bars**

| System | key match | in-scale notes | pitch-class entropy | 4-gram repetition | notes / bar | empty bars | val. perplexity |
|---|---|---|---|---|---|---|---|
| real test melodies | 0.40 | 0.93 | 2.38 | 0.27 | 6.8 | 0.11 | — |
| uniform random | 0.02 | 0.62 | 3.41 | 0.00 | 3.1 | 0.03 | — |
| bigram | 0.08 | 0.67 | 3.27 | 0.00 | 2.4 | 0.12 | 12.58 |
| GRU (24.4M) | 0.29 | 0.92 | 2.36 | 0.17 | 5.1 | 0.14 | 2.33 |
| GRU, guidance 1.5 | 0.45 | 0.96 | 2.37 | 0.15 | 5.1 | 0.17 | 2.33 |
| Transformer (26.0M) | 0.32 | 0.92 | 2.36 | 0.29 | 4.7 | 0.18 | 2.13 |
| Transformer, guidance 1.5 | 0.50 | 0.96 | 2.37 | 0.24 | 4.4 | 0.22 | 2.13 |
| Transformer, guidance 1.5, key bias, rest limit (app) | 0.53 | 0.99 | 2.37 | 0.25 | 5.3 | 0.06 | 2.13 |
| same, temperature 0.9, top-p 0.92 | 0.42 | 1.00 | 1.83 | 0.75 | 7.1 | 0.02 | 2.13 |
| Transformer, student implementation | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

At the same size and number of steps, the Transformer reaches lower perplexity than the
GRU and repeats motifs about as often as real melodies do (0.29 vs 0.27; GRU 0.17).
Guidance raises key match for both. Real melodies agree with their MidiCaps key label only
40% of the time, because the label describes the whole song, so this metric tops out
around 0.5. Two decoding fixes came from listening to the app. The rest limit (notes in
bar 1, at most one silent bar in a row) cuts empty bars from 0.22 to 0.06. The old web
defaults with top-p truncation made melodies monotonous (repetition 0.75), so the app now
samples without it. Both models were trained for 40,000 steps of 48 sequences on an RTX
4070 laptop GPU: 76 minutes for the Transformer, 5.1 hours for the GRU.

**Band, 8 bars**

| System | instrument recall | instrument precision | drums match | key match | notes per onset | notes / bar | 4-gram repetition | bars match | val. perplexity |
|---|---|---|---|---|---|---|---|---|---|
| real test windows | 1.00 | 1.00 | 1.00 | 0.52 | 2.85 | 22.8 | 0.20 | — | — |
| uniform random | 1.00 | 1.00 | 1.00 | 0.02 | 2.11 | 5.8 | 0.00 | 1.00 | — |
| bigram | 0.99 | 1.00 | 1.00 | 0.06 | 2.95 | 8.8 | 0.00 | 1.00 | 15.97 |
| Transformer (26.5M) | 0.98 | 1.00 | 0.995 | 0.50 | 2.97 | 22.5 | 0.13 | 0.99 | 1.72 |
| Transformer, guidance 1.5 | 1.00 | 1.00 | 1.00 | 0.60 | 2.84 | 21.8 | 0.14 | 1.00 | 1.72 |
| Transformer, guidance 1.5, no grammar | 1.00 | 0.97 | 1.00 | 0.62 | 2.84 | 21.3 | 0.12 | 0.75 | 1.72 |
| Diffusion U-Net (26.6M), guidance 2.0 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | — |

With the grammar switched off, the model still plays the requested instruments and
follows the drums request, so it learned instrument control on its own. The grammar
mainly fixes the bar count (0.75 to 1.00). Density and polyphony match the real windows.
Training took 6.5 hours (40,000 steps, about three passes over 625M tokens).

**Parser** (1,500 test rows: 1,000 held-out MidiCaps captions, 500 template queries)

| Parser | field accuracy | exact match | exact match, captions | hallucination | instruments F1 | ms / query |
|---|---|---|---|---|---|---|
| keyword rules | 0.945 | 0.561 | 0.372 | 0.102 | 1.00* | 1.3 |
| Qwen2.5-0.5B, zero-shot | 0.519 | 0.000 | 0.000 | 0.667 | 0.30 | 392 |
| Qwen2.5-0.5B, five examples | 0.626 | 0.029 | 0.004 | 0.282 | 0.37 | 359 |
| Qwen2.5-0.5B + LoRA (r=16) | 0.953 | 0.755 | 0.632 | 0.016 | 1.00 | 646 |

\* The test labels were cleaned with the same keyword tables the rule parser uses, so the
rules get instruments right by construction and are favoured elsewhere too.

Prompting alone does not work at this model size. LoRA fine-tuning (8.8M trainable
parameters, 2.6 hours) makes the 0.5B model the most accurate parser, with almost no
invented fields, and the app uses it.

`pixi run listening` builds blind listening pages that mix real test excerpts with
generated pieces; the app links to them at the bottom of the page.

## Individual Contributions

Solo project.
