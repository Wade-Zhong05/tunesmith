# Attribution

## AI tools

Most of the code and documentation was written with Claude Code (Claude Opus 5.5,
Anthropic), following designs discussed with the student. Each file written this way
says so in its module docstring. OpenAI Codex set up the repository, ran the GRU training
and packaged the model release.

| Part | Files | Tool, date |
|---|---|---|
| Framework: data pipeline, tokenizer, GRU, constrained generation, parser, evaluation, app, tests | `src/tunesmith/**` except the student files, `configs/`, `tests/`, `web/`, `pyproject.toml`, `requirements.txt`, the documentation | Claude Code, 2026-11-10 |
| Skeletons of the student files: signatures, docstrings, TODO markers | `model/transformer.py`, `model/sampling_filters.py` | Claude Code, 2026-11-10 |
| Reference Transformer, so a Transformer could be trained before the student's version was done | `model/transformer_ref.py`, `configs/composer_transformer_ref.yaml` | Claude Code, 2026-11-14 |
| Ensemble (multi-instrument) mode | `music/ensemble.py`; ensemble parts of the tokenizer, grammar, metrics, rendering, pipeline, parser and UI; `tests/test_ensemble.py` | Claude Code, 2026-11-22 |
| Diffusion composer | `diffusion/`, `configs/diffusion_*.yaml`, `tests/test_diffusion.py` | Claude Code, 2026-11-23 |
| Web UI redesign, demo piece, listening-test pages | `web/`, `app/make_demo.py`, `eval/listening.py` | Claude Code, 2026-11-23 |
| Reference sampling filters (used until the student's are finished), tinysoundfont rendering, rest limit in the grammar, spelling repair in the rule parser | `model/sampling_filters_ref.py`, `render/render.py`, `model/generate.py`, `understand/rules.py` | Claude Code, 2026-11-23 |
| Repository setup and `.gitignore` | | Codex, 2026-11-10 |
| GRU training run | `models/composer/gru-base` | Codex, 2026-11-10 |
| Model release, download script, missing-model notice | `download_models.py`, `configs/model_release.yaml`, `tests/test_download_models.py`, `web/` | Codex, 2026-11-23 |

Claude Code also ran the reference Transformer, ensemble, diffusion and LoRA training
runs and the evaluations.

Written by the student: the method bodies in `model/transformer.py` and
`model/sampling_filters.py` (to be completed).

Problems found by running and testing the generated code, and fixed:

- miditok counts a 6/8 bar in eighth notes (48 positions, not 24). A 6/8 round-trip test
  exposed the wrong bar length in the grammar.
- Melodies rang past the last bar (19–20 bars measured for 16 requested). Notes are now
  clipped at the final bar line.
- Requests without a meter could produce rare time signatures. They are now limited to
  4/4, 3/4, 2/4 and 6/8.
- `build_model` passed `vocab_size` twice to the Transformer config.
- The first full data run crashed on the meter label "other". Malformed rows now skip
  the file instead of stopping the run.
- Unconstrained band requests used up to 22 instruments. Requests that name none are
  capped at four.
- The melody model sometimes opened with 9–11 silent bars, which led to the rest limit.
- Temperature and top-p were silently ignored while the student's filters were TODO,
  which led to the reference filters.

## Datasets

| Dataset | Use | Licence |
|---|---|---|
| MidiCaps (Melechovsky, Roy, Herremans, ISMIR 2024), `amaai-lab/MidiCaps` | MIDI files, captions, attribute labels | CC BY-SA 4.0 |
| Lakh MIDI dataset (Raffel, 2016) | the MIDI files inside MidiCaps | CC BY 4.0 |

The released composer weights and bigram baselines are trained on MidiCaps and shared
under CC BY-SA 4.0.

## Pretrained models and assets

| Asset | Use | Licence |
|---|---|---|
| Qwen2.5-0.5B-Instruct (Qwen team, Alibaba) | parser base model, fine-tuned with LoRA | Apache-2.0 |
| laion/larger_clap_music_and_speech | optional CLAP evaluation | Apache-2.0 |
| MuseScore General SoundFont (S. Christian Collins) | audio rendering | MIT |
| Instrument Sans | web typeface, `web/fonts/` | SIL Open Font License 1.1 |

## Libraries

PyTorch, miditok, symusic, tinysoundfont, Hugging Face transformers, PEFT,
huggingface_hub, accelerate, FastAPI, Uvicorn, pydantic, NumPy, Matplotlib, PyYAML,
tqdm, pytest; pixi for environments.

## Methods from prior work

- REMI tokens: Huang & Yang, Pop Music Transformer (ACM MM 2020).
- Text to attributes to music: MuseCoco (Lu et al., 2023). End-to-end comparison and
  metrics: Text2midi (Bhandari et al., AAAI 2025).
- Multi-track tokens with program markers: Multitrack Music Transformer (Dong et al.,
  ICASSP 2023).
- Classifier-free guidance: Ho & Salimans (2022). Nucleus sampling: Holtzman et al.
  (ICLR 2020). LoRA: Hu et al. (ICLR 2022).
- Key finding: Krumhansl–Schmuckler algorithm with Krumhansl & Kessler profiles (1982).
- Diffusion: DDPM (Ho et al., NeurIPS 2020), cosine schedule (Nichol & Dhariwal, ICML
  2021), U-Net with scale-shift conditioning (Dhariwal & Nichol, NeurIPS 2021; Ronneberger
  et al., MICCAI 2015), DDIM (Song et al., ICLR 2021), inpainting by replacement (Song et
  al., ICLR 2021; Lugmayr et al., CVPR 2022), onset and sustain piano rolls (Polyffusion,
  Min et al., ISMIR 2023). No code was copied from these projects.
