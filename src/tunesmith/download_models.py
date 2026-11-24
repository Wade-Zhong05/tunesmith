"""Download the trained models listed in configs/model_release.yaml.

    pixi run download-models --all

The manifest uses JSON-compatible YAML so this downloader needs only the standard
library. Release packaging and downloader by OpenAI Codex, 2026-11-23.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
CHUNK_SIZE = 1 << 20


def verify_file(path: Path, expected: dict) -> bool:
    """Return False for a missing file and raise on unexpected existing contents."""
    if not path.exists():
        return False
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != expected["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected['sha256']}, got {digest}")
    size = path.stat().st_size
    if size != expected["size"]:
        raise ValueError(f"Size mismatch for {path}: expected {expected['size']}, got {size}")
    return True


def relative_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts
            or "\\" in name or ":" in name or path == PurePosixPath(".")):
        raise ValueError(f"Unsafe release path: {name!r}")
    return path


def model_path(root: Path, name: str) -> Path:
    relative = relative_path(name)
    if relative.parts[0] != "models":
        raise ValueError(f"Release targets must be inside models/: {name!r}")
    target = root.joinpath(*relative.parts)
    if not target.resolve().is_relative_to((root / "models").resolve()):
        raise ValueError(f"Release target escapes models/: {name!r}")
    return target


def download_file(url: str, target: Path, expected: dict) -> None:
    label = expected.get("file", target.name)
    if verify_file(target, expected):
        print(f"Already verified: {label}", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    downloaded, reported = 0, 0
    print(f"Downloading {label} ({expected['size']:,} bytes)", flush=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".", suffix=".part", delete=False) as output:
        part = Path(output.name)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "tunesmith-model-downloader"})
        with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as output:
            while block := response.read(CHUNK_SIZE):
                downloaded += len(block)
                if downloaded > expected["size"]:
                    raise ValueError(f"Size mismatch for {target.name}: download exceeds {expected['size']} bytes")
                output.write(block)
                if downloaded - reported >= 8 * CHUNK_SIZE or downloaded == expected["size"]:
                    print(f"  {label}: {downloaded:,}/{expected['size']:,} bytes", flush=True)
                    reported = downloaded
        verify_file(part, expected)
        part.replace(target)
    finally:
        part.unlink(missing_ok=True)


def extracted_files(root: Path, asset: dict) -> dict[Path, dict]:
    if not asset.get("members"):
        raise ValueError(f"Archive has no member checksums: {asset['file']}")
    files = {}
    for name, expected in asset["members"].items():
        member = relative_path(name)
        target = model_path(root, (PurePosixPath(asset["target"]) / member).as_posix())
        files[target] = expected
    return files


def extract_archive(archive: Path, root: Path, asset: dict) -> None:
    target = model_path(root, asset["target"])
    files = extracted_files(root, asset)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".extract-") as temporary:
        staging = Path(temporary)
        with zipfile.ZipFile(archive) as source:
            members = source.infolist()
            for member in members:
                relative_path(member.filename)
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError(f"Archive contains a symbolic link: {member.filename}")
            regular = [member for member in members if not member.is_dir()]
            names = [member.filename for member in regular]
            if len(set(names)) != len(names) or set(names) != set(asset["members"]):
                raise ValueError(f"Archive members differ from the manifest: {asset['file']}")
            for member in regular:
                expected = asset["members"][member.filename]
                if member.file_size != expected["size"]:
                    raise ValueError(f"Size mismatch for archive member {member.filename}")
                staged = staging.joinpath(*relative_path(member.filename).parts)
                staged.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as incoming, staged.open("wb") as output:
                    shutil.copyfileobj(incoming, output, length=CHUNK_SIZE)
                verify_file(staged, expected)
        # Every staged file has been checked before any destination is changed.
        for destination in files:
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging.joinpath(*destination.relative_to(target).parts).replace(destination)
    print(f"Extracted and verified: {asset['target']}", flush=True)


def download_models(root: Path = ROOT, *, base_url: str | None = None, with_training_runs: bool = False) -> None:
    root = Path(root)
    manifest_path = root / "configs" / "model_release.yaml"
    with manifest_path.open(encoding="utf-8") as source:
        manifest = json.load(source)
    base = (base_url or manifest["base_url"]).rstrip("/") + "/"
    if urllib.parse.urlsplit(base).scheme not in ("https", "http", "file"):
        raise ValueError("The base URL must use https://, http:// or file://")
    print(f"Model release: {manifest['tag']}", flush=True)
    for asset in manifest["assets"]:
        if asset.get("optional") and not with_training_runs:
            continue
        name = asset["file"]
        if relative_path(name).name != name:
            raise ValueError(f"Asset file must be a plain filename: {name!r}")
        target = model_path(root, asset["target"])
        url = urllib.parse.urljoin(base, urllib.parse.quote(name))
        if asset.get("extract"):
            files = extracted_files(root, asset)
            verified = [verify_file(path, expected) for path, expected in files.items()]
            if all(verified):
                print(f"Already verified: {asset['target']}", flush=True)
                continue
            cache = model_path(root, f"models/.downloads/{name}")
            download_file(url, cache, asset)
            extract_archive(cache, root, asset)
        else:
            download_file(url, target, asset)
    print("Trained model files are ready. Restart the app to load them.", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--with-parser-base", action="store_true", help="Also fetch Qwen from its original source")
    parser.add_argument("--with-soundfont", action="store_true", help="Also fetch the MuseScore General SoundFont")
    parser.add_argument("--all", action="store_true", help="Fetch models, the parser base and the SoundFont")
    parser.add_argument("--with-training-runs", action="store_true", help="Also extract optional training records")
    parser.add_argument("--base-url", help="Use an alternative release URL or a file:// mirror")
    args = parser.parse_args(argv)
    try:
        download_models(base_url=args.base_url, with_training_runs=args.with_training_runs)
        modules = []
        if args.all or args.with_parser_base:
            modules.append("tunesmith.understand.download_base")
        if args.all or args.with_soundfont:
            modules.append("tunesmith.render.fetch_soundfont")
        for module in modules:
            subprocess.run([sys.executable, "-m", module], cwd=ROOT, check=True)
    except (OSError, ValueError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
        parser.exit(1, f"download-models: {error}\n")


if __name__ == "__main__":
    main()
