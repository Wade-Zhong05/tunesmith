"""Web app: FastAPI backend + static single-page frontend (web/).

    pixi run app                 # http://127.0.0.1:8000
    pixi run app --port 8080 --config configs/app.yaml

Endpoints:
    GET  /api/health    model status
    GET  /api/options   allowed attribute values (for the override menus)
    POST /api/parse     {"text"} -> attributes
    POST /api/generate  {"text", "overrides", "n", "seed", sampling options} -> pieces
    GET  /files/<name>  generated MIDI / WAV

Production touches: input validation, a per-client rate limit, one model lock, request
logging to outputs/app/app.log, and errors returned as JSON instead of stack traces.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import defaultdict, deque
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..attributes import BARS, GENRES, INSTRUMENTS, KEYS, METERS, MOODS, TEMPO_BINS
from ..paths import ROOT, load_yaml, resolve

log = logging.getLogger("tunesmith.app")


class ParseRequest(BaseModel):
    text: str = Field("", max_length=2000)


class GenerateRequest(BaseModel):
    text: str = Field("", max_length=2000)
    mode: Literal["instrumental", "lyrics"] = "instrumental"
    overrides: dict[str, Any] = Field(default_factory=dict)
    n: int = Field(1, ge=1, le=4)
    seed: Optional[int] = Field(None, ge=0, le=2**31 - 1)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(None, ge=0, le=400)
    top_p: Optional[float] = Field(None, gt=0.0, le=1.0)
    cfg_scale: Optional[float] = Field(None, ge=0.0, le=5.0)
    key_bias: Optional[float] = Field(None, ge=0.0, le=10.0)


class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[str, deque] = defaultdict(deque)

    def allow(self, client: str) -> bool:
        now = time.time()
        q = self.hits[client]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            return False
        q.append(now)
        return True


def create_app(config: str | dict[str, Any] = "configs/app.yaml", smith=None) -> FastAPI:
    cfg = load_yaml(config) if isinstance(config, str) else dict(config)
    out_dir = resolve(cfg.get("output_dir", "outputs/app"))
    out_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(out_dir / "app.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    if not any(isinstance(h, logging.FileHandler) for h in log.handlers):
        log.addHandler(handler)
    log.setLevel(logging.INFO)

    app = FastAPI(title="tunesmith", version="0.1.0")
    limiter = RateLimiter(int(cfg.get("rate_limit_per_minute", 30)))
    max_chars = int(cfg.get("max_text_chars", 500))
    state: dict[str, Any] = {"smith": smith}

    def get_smith():
        if state["smith"] is None:
            from ..pipeline import TuneSmith

            state["smith"] = TuneSmith(cfg)
        return state["smith"]

    @app.exception_handler(Exception)
    async def on_error(request: Request, exc: Exception):  # never leak stack traces to the client
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"error": "internal error", "detail": type(exc).__name__})

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        s = get_smith()
        return {"ok": True, "parser": s.parser_name, "composer": s.composer_name, "trained": s.trained,
                "device": str(s.device), "warnings": s.warnings}

    @app.get("/api/options")
    def options() -> dict[str, Any]:
        return {"key": list(KEYS), "meter": list(METERS[:4]), "instrument": list(INSTRUMENTS), "mood": list(MOODS),
                "genre": list(GENRES), "bars": list(BARS),
                "tempo": [{"name": n, "bpm": rep} for n, _, _, rep in TEMPO_BINS]}

    @app.post("/api/parse")
    def parse(req: ParseRequest) -> dict[str, Any]:
        if len(req.text) > max_chars:
            raise HTTPException(422, f"text longer than {max_chars} characters")
        return {"attributes": get_smith().parse(req.text).model_dump()}

    @app.post("/api/generate")
    def generate(req: GenerateRequest, request: Request) -> dict[str, Any]:
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client):
            raise HTTPException(429, "too many requests, try again in a minute")
        if len(req.text) > max_chars:
            raise HTTPException(422, f"text longer than {max_chars} characters")
        if req.mode == "lyrics":
            raise HTTPException(400, "lyrics mode is coming later; the instrumental version comes first")
        t0 = time.time()
        try:
            res = get_smith().compose(req.text, overrides=req.overrides, n=req.n, seed=req.seed,
                                      temperature=req.temperature, top_k=req.top_k, top_p=req.top_p,
                                      cfg_scale=req.cfg_scale, key_bias=req.key_bias)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        for p in res["pieces"]:
            p["midi_url"] = f"/files/{p['midi']}"
            if "wav" in p:
                p["wav_url"] = f"/files/{p['wav']}"
        log.info("generate client=%s chars=%d n=%d attrs=%s %.2fs", client, len(req.text), req.n,
                 {k: v for k, v in res["attributes"].items() if v is not None}, time.time() - t0)
        return res

    app.mount("/files", StaticFiles(directory=str(out_dir)), name="files")
    app.mount("/", StaticFiles(directory=str(ROOT / "web"), html=True), name="web")
    return app


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/app.yaml")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    uvicorn.run(create_app(args.config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
