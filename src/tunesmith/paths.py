"""Project paths and small config helpers.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def _find_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "src" / "tunesmith").exists():
            return parent
    return Path.cwd()


ROOT = _find_root()
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"


def resolve(path: str | Path) -> Path:
    """Resolve a path relative to the project root unless it is absolute."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(resolve(path), encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_json(obj: Any, path: str | Path) -> None:
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    with open(resolve(path), encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str | Path):
    with open(resolve(path), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows, path: str | Path) -> int:
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
