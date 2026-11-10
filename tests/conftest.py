"""Shared fixtures: a tokenizer and a small prepared synthetic dataset.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

import pytest

from tunesmith.data.prepare import prepare
from tunesmith.data.synthetic import generate_dataset
from tunesmith.paths import load_yaml
from tunesmith.tokenizer import TuneTokenizer


@pytest.fixture(scope="session")
def tok() -> TuneTokenizer:
    return TuneTokenizer()


@pytest.fixture(scope="session")
def synth_data(tmp_path_factory):
    """60 synthetic pieces, prepared into token files. Returns (raw_dir, processed_dir)."""
    root = tmp_path_factory.mktemp("synth")
    meta = generate_dataset(root / "raw", n=60, seed=1)
    cfg = load_yaml("configs/data.yaml")
    cfg.update(metadata=str(meta), midi_root=str(root / "raw"), out_dir=str(root / "processed"),
               workers=1, val_fraction=0.2)
    prepare(cfg)
    return root / "raw", root / "processed"
