"""Download the parser's base model (default Qwen/Qwen2.5-0.5B-Instruct, Apache-2.0).

    pixi run download-parser-base
    pixi run download-parser-base --repo Qwen/Qwen2.5-1.5B-Instruct --out models/parser/base/Qwen2.5-1.5B-Instruct

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

import argparse

from ..paths import resolve


def main(argv: list[str] | None = None) -> None:
    from huggingface_hub import snapshot_download

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default="models/parser/base/Qwen2.5-0.5B-Instruct")
    args = ap.parse_args(argv)
    path = snapshot_download(repo_id=args.repo, local_dir=str(resolve(args.out)),
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt", "vocab.json"])
    print(f"downloaded {args.repo} -> {path}")


if __name__ == "__main__":
    main()
