# data/

Nothing here is committed except this file; everything is downloaded or generated.

| Folder | Created by | Contents |
|---|---|---|
| `raw/midicaps/` | `pixi run download-data` | MidiCaps `train.json` (captions + metadata, 428 MB) and `lmd_full/` (168K MIDI files from the 1.6 GB archive) |
| `processed/midicaps/` | `pixi run prepare-data` | `tokens_{split}.bin`, `index_{split}.npy`, `files_{split}.jsonl`, `meta.json` (see `src/tunesmith/data/prepare.py`) |
| `parser/` | `pixi run make-parser-data` | `{train,val,test}.jsonl` with `text`, `target`, `source`; `stats.json` |

MidiCaps: Melechovsky, Roy and Herremans, "MidiCaps: A Large-scale MIDI Dataset with Text
Captions", ISMIR 2024 (CC BY-SA 4.0), built on the Lakh MIDI dataset (CC BY 4.0).
