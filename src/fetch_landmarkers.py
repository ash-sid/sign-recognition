"""
src/fetch_landmarkers.py

Downloads the two landmarker models the web app loads and pins them by hash
in data/extractor_manifest.json, so that src/extract_landmarks.py runs the
same models the browser does and can prove it.

The URLs are read out of web/src/tracking.ts rather than written down a
second time here. A second copy would be free to drift, and the first sign of
drift would be a model trained on one extractor and deployed behind another,
which produces plausible landmarks either way.

The first run records each file's sha256. Every later run re-downloads and
compares against the record, and refuses to overwrite it on a mismatch: the
URLs name a version, but nothing stops the file behind one from changing, and
a changed file means the browser is already loading something other than what
was measured.

Run: uv run python src\\fetch_landmarkers.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACKING_TS = ROOT / "web" / "src" / "tracking.ts"
MANIFEST = ROOT / "data" / "extractor_manifest.json"
MODELS_DIR = ROOT / "models" / "landmarkers"

CONSTANTS = {"hand": "HAND_MODEL", "pose": "POSE_MODEL"}


def read_browser_pins(source: str) -> tuple[str, dict[str, str]]:
    version = re.search(r'const TASKS_VERSION = "([^"]+)";', source)
    if not version:
        raise SystemExit(f"TASKS_VERSION not found in {TRACKING_TS}")
    urls = {}
    for role, name in CONSTANTS.items():
        match = re.search(rf'const {name} =\s*"(https://[^"]+\.task)";', source)
        if not match:
            raise SystemExit(f"{name} not found in {TRACKING_TS}")
        urls[role] = match.group(1)
    return version.group(1), urls


def download(url: str, path: Path) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(url, timeout=60) as response, open(temporary, "wb") as out:
        for block in iter(lambda: response.read(1 << 20), b""):
            digest.update(block)
            out.write(block)
            size += len(block)
    os.replace(temporary, path)
    return digest.hexdigest(), size


def main() -> None:
    version, urls = read_browser_pins(TRACKING_TS.read_text(encoding="utf-8"))
    recorded = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else None

    models = {}
    for role, url in urls.items():
        file = url.rsplit("/", 1)[1]
        sha, size = download(url, MODELS_DIR / file)
        models[role] = {"url": url, "file": file, "sha256": sha, "bytes": size}
        print(f"{role:5s} {file}  {size} bytes  sha256 {sha}")

    manifest = {
        "source": "web/src/tracking.ts",
        "tasks_vision_version": version,
        "models": models,
    }
    if recorded is None:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"pinned in {MANIFEST}")
    elif recorded == manifest:
        print(f"matches {MANIFEST}")
    else:
        print(f"MISMATCH against {MANIFEST}; not overwritten.")
        print("recorded:", json.dumps(recorded, indent=2))
        print("fetched: ", json.dumps(manifest, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
