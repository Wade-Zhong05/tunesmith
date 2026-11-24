"""Release downloads, checksum validation and archive installation on local mirrors.

Written by OpenAI Codex, 2026-11-23.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from tunesmith import download_models as dm


def checksum(data: bytes) -> dict:
    return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


def make_release(tmp_path: Path, *, archive: bool = False, member: str = "adapter_config.json"):
    mirror, root = tmp_path / "mirror", tmp_path / "clone"
    mirror.mkdir()
    (root / "configs").mkdir(parents=True)
    if archive:
        source = mirror / "adapter.zip"
        with zipfile.ZipFile(source, "w") as output:
            output.writestr(member, b'{"r": 16}')
            output.writestr("tokenizer/nested.json", b"{}")
        asset = {"file": source.name, "target": "models/parser/test", "extract": True,
                 "members": {member: checksum(b'{"r": 16}'), "tokenizer/nested.json": checksum(b"{}")}}
    else:
        source = mirror / "model.pt"
        source.write_bytes(b"fake-model")
        asset = {"file": source.name, "target": "models/composer/test/best.pt"}
    asset.update(checksum(source.read_bytes()))
    manifest = {"tag": "test", "base_url": mirror.as_uri() + "/", "assets": [asset]}
    (root / "configs/model_release.yaml").write_text(json.dumps(manifest), encoding="ascii")
    return root, mirror, source, asset


def test_file_download_and_verified_skip(tmp_path, monkeypatch, capsys):
    root, mirror, _, _ = make_release(tmp_path)
    dm.download_models(root, base_url=mirror.as_uri())
    target = root / "models/composer/test/best.pt"
    assert target.read_bytes() == b"fake-model"
    modified = target.stat().st_mtime_ns
    monkeypatch.setattr(dm.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("Verified files must skip HTTP"))
    dm.download_models(root)
    assert target.stat().st_mtime_ns == modified
    assert "Already verified" in capsys.readouterr().out


def test_bad_download_is_rejected_and_can_be_retried(tmp_path):
    root, _, source, _ = make_release(tmp_path)
    source.write_bytes(b"wrong-data")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        dm.download_models(root)
    assert not (root / "models/composer/test/best.pt").exists()
    assert not list((root / "models").rglob("*.part"))
    source.write_bytes(b"fake-model")
    dm.download_models(root)
    assert (root / "models/composer/test/best.pt").read_bytes() == b"fake-model"


def test_existing_mismatch_fails_loudly(tmp_path):
    root, _, _, _ = make_release(tmp_path)
    dm.download_models(root)
    target = root / "models/composer/test/best.pt"
    target.write_bytes(b"wrong-data")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        dm.download_models(root)
    assert target.read_bytes() == b"wrong-data"


def test_archive_extraction_and_skip(tmp_path, monkeypatch):
    root, _, _, _ = make_release(tmp_path, archive=True)
    dm.download_models(root)
    target = root / "models/parser/test"
    assert (target / "adapter_config.json").read_bytes() == b'{"r": 16}'
    assert (target / "tokenizer/nested.json").read_bytes() == b"{}"
    monkeypatch.setattr(dm.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("Verified members must skip HTTP"))
    dm.download_models(root)
    # A partial installation is completed from the verified archive cache.
    (target / "tokenizer/nested.json").unlink()
    dm.download_models(root)
    assert (target / "tokenizer/nested.json").read_bytes() == b"{}"


def test_archive_rejects_path_traversal(tmp_path):
    root, _, _, _ = make_release(tmp_path, archive=True, member="../escape.json")
    with pytest.raises(ValueError, match="Unsafe release path"):
        dm.download_models(root)
    assert not (root / "models/parser/escape.json").exists()


def test_archive_checks_extracted_hashes_before_installing(tmp_path):
    root, _, _, asset = make_release(tmp_path, archive=True)
    asset["members"]["adapter_config.json"] = checksum(b"incorrect")
    path = root / "configs/model_release.yaml"
    manifest = json.loads(path.read_text())
    manifest["assets"] = [asset]
    path.write_text(json.dumps(manifest), encoding="ascii")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        dm.download_models(root)
    assert not (root / "models/parser/test/adapter_config.json").exists()


def test_optional_training_records(tmp_path):
    root, _, _, asset = make_release(tmp_path)
    path = root / "configs/model_release.yaml"
    manifest = json.loads(path.read_text())
    asset["optional"] = True
    manifest["assets"] = [asset]
    path.write_text(json.dumps(manifest), encoding="ascii")
    dm.download_models(root)
    assert not (root / asset["target"]).exists()
    dm.download_models(root, with_training_runs=True)
    assert (root / asset["target"]).is_file()


@pytest.mark.parametrize("flags,modules", [
    ([], []),
    (["--with-parser-base"], ["tunesmith.understand.download_base"]),
    (["--with-soundfont"], ["tunesmith.render.fetch_soundfont"]),
    (["--all"], ["tunesmith.understand.download_base", "tunesmith.render.fetch_soundfont"]),
])
def test_extra_download_flags(monkeypatch, flags, modules):
    downloads, commands = [], []
    monkeypatch.setattr(dm, "download_models", lambda **kw: downloads.append(kw))
    monkeypatch.setattr(dm.subprocess, "run", lambda cmd, **kw: commands.append((cmd, kw)))
    dm.main([*flags, "--base-url", "file:///mirror/", "--with-training-runs"])
    assert downloads == [{"base_url": "file:///mirror/", "with_training_runs": True}]
    assert [command[0][-1] for command in commands] == modules
    assert all(options == {"cwd": dm.ROOT, "check": True} for _, options in commands)
