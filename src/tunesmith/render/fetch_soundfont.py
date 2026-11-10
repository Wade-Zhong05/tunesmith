"""Download the MuseScore General SoundFont (SF3) used for realistic playback.

    pixi run fetch-soundfont      # -> models/soundfonts/MuseScore_General.sf3 (git-ignored)

MuseScore General by S. Christian Collins, MIT licence. Without it the numpy
synthesizer is used.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request

from ..paths import resolve

URL = "https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf3"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--out", default="models/soundfonts/MuseScore_General.sf3")
    args = ap.parse_args(argv)
    out = resolve(args.out)
    if out.exists():
        print(f"already present: {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    print(f"downloading {args.url}")
    with urllib.request.urlopen(args.url, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.replace(out)
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
