"""Web API and end-to-end pipeline (no trained model needed: random/bigram baselines and a tiny GRU).

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tunesmith.app.server import create_app
from tunesmith.paths import load_yaml
from tunesmith.render.render import render_wav


def app_config(tmp_path, **kw):
    cfg = load_yaml("configs/app.yaml")
    cfg.update(parser="rules", composer_checkpoint=str(tmp_path / "missing.pt"), fallback_composer="random",
               output_dir=str(tmp_path / "out"), device="cpu", rate_limit_per_minute=5, **kw)
    cfg["render"]["soundfont"] = None
    return cfg


def test_api_end_to_end(tmp_path):
    client = TestClient(create_app(app_config(tmp_path)))
    h = client.get("/api/health").json()
    assert h["ok"] and not h["trained"] and "random" in h["composer"]
    assert "A minor" in client.get("/api/options").json()["key"]
    parsed = client.post("/api/parse", json={"text": "a calm flute melody in D major"}).json()["attributes"]
    assert parsed["instrument"] == "flute" and parsed["key"] == "D major"
    r = client.post("/api/generate", json={"text": "a sad cello melody, 3/4, 8 bars", "n": 2, "seed": 1,
                                            "overrides": {"key": "E minor"}})
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["attributes"]["key"] == "E minor" and res["attributes"]["meter"] == "3/4"
    assert len(res["pieces"]) == 2
    for p in res["pieces"]:
        assert p["metrics"]["bars"] == 8 and p["notes"]["meter"] == "3/4"
        assert client.get(p["midi_url"]).status_code == 200
        assert client.get(p["wav_url"]).content[:4] == b"RIFF"
    assert client.get("/").status_code == 200 and b"tunesmith" in client.get("/").content


def test_api_errors_and_rate_limit(tmp_path):
    client = TestClient(create_app(app_config(tmp_path)))
    assert client.post("/api/generate", json={"text": "x", "mode": "lyrics"}).status_code == 400
    assert client.post("/api/generate", json={"text": "a" * 600}).status_code == 422
    assert client.post("/api/generate", json={"text": "x", "n": 9}).status_code == 422
    codes = [client.post("/api/generate", json={"text": "short piano", "n": 1}).status_code for _ in range(6)]
    assert 429 in codes


def test_numpy_render(tmp_path, tok):
    from tunesmith.attributes import MusicAttributes
    from tunesmith.eval.baselines import UniformComposer
    from tunesmith.model.generate import generate

    a = MusicAttributes(bars=2, instrument="flute", tempo_bpm=120)
    score = tok.decode_for(generate(UniformComposer(tok.vocab_size), tok, [a], seed=0)[0], a)
    assert render_wav(score, tmp_path / "x.wav", soundfont=None) == "numpy"
    assert (tmp_path / "x.wav").stat().st_size > 1000


@pytest.mark.slow
def test_smoke_pipeline_with_trained_gru(tmp_path):
    from tunesmith.smoke import run

    res = run("gru", steps=10, root=str(tmp_path / "smoke"))
    assert res["trained"] and len(res["pieces"]) == 2
