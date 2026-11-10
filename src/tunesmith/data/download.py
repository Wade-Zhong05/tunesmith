"""Download MidiCaps (Melechovsky et al., ISMIR 2024) from the Hugging Face Hub.

    pixi run download-data                  # metadata (428 MB) + MIDI archive (1.6 GB), extracted
    pixi run download-data --metadata-only  # only train.json

Files go to data/raw/midicaps/ (git-ignored). The archive is extracted with Python's
"data" tar filter, which rejects absolute paths, ".." and links.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from ..paths import resolve

REPO_ID = "amaai-lab/MidiCaps"
FILES = ("train.json", "midicaps.tar.gz")


def download(out_dir: Path, metadata_only: bool = False, extract: bool = True) -> None:
    from huggingface_hub import hf_hub_download

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES[:1] if metadata_only else FILES:
        print(f"downloading {REPO_ID}/{name} ...", flush=True)
        path = hf_hub_download(repo_id=REPO_ID, filename=name, repo_type="dataset", local_dir=str(out_dir))
        print(f"  -> {path}")
    archive = out_dir / "midicaps.tar.gz"
    marker = out_dir / ".extracted"
    if extract and archive.exists() and not marker.exists():
        print("extracting midicaps.tar.gz (about 168K MIDI files) ...", flush=True)
        with tarfile.open(archive) as tar:
            tar.extractall(out_dir, filter="data")
        marker.write_text("ok\n")
        print("  done")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/raw/midicaps")
    ap.add_argument("--metadata-only", action="store_true")
    ap.add_argument("--no-extract", action="store_true")
    args = ap.parse_args(argv)
    download(resolve(args.out), args.metadata_only, not args.no_extract)


if __name__ == "__main__":
    main()
