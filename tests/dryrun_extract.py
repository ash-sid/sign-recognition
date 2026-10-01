"""
tests/dryrun_extract.py

Dry run for src/extract_landmarks.py. Builds synthetic videos and frame
directories, drives the real extractor over them as a subprocess with the
real landmarker models, and checks the properties an overnight run over tens
of thousands of videos depends on and would not reveal until morning:

  - --limit selects the same inputs every time, in sorted order;
  - outputs have the documented keys, shapes and dtypes, one frame per
    decoded frame, with the documented timestamps;
  - nothing detected is written as NaN, never as zero;
  - a second run skips what exists without rewriting it, and a video that
    cannot be decoded is recorded as failed without stopping the run;
  - no partially written file survives;
  - a run with a different configuration is refused rather than mixed into
    an existing directory;
  - a model file that does not match its pinned hash is refused;
  - several worker processes start and finish, which on Windows exercises the
    guard that stops workers re-importing the script without end;
  - frame sequences carry the hash of exactly the pixels decoded from each PNG.

The synthetic images contain no people, so no hands are expected. What is
checked is the plumbing around the landmarkers, not the landmarkers.

Needs the models fetched first: uv run python src\\fetch_landmarkers.py
Run: uv run python tests\\dryrun_extract.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "extract_landmarks.py"
MANIFEST = ROOT / "data" / "extractor_manifest.json"
FRAMES = 40
FPS = 25.0
TIMEOUT = 600
CHECKS: list[tuple[str, bool]] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    ok = bool(condition)
    CHECKS.append((name, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail and not ok else ''}")
    return ok


def run(*args: str) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=TIMEOUT,
    )
    return result


def show(result: subprocess.CompletedProcess) -> str:
    return f"exit {result.returncode}\n--- stdout\n{result.stdout[-2000:]}\n--- stderr\n{result.stderr[-2000:]}"


def synthetic_frames(seed: int, width: int = 320, height: int = 240) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(FRAMES):
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        x = 20 + i * 5
        cv2.rectangle(frame, (x, 80), (x + 40, 140), (200, 180, 120), -1)
        frame[:10] = rng.integers(0, 255, (10, width, 3), dtype=np.uint8)
        frames.append(frame)
    return frames


def write_video(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (320, 240))
    if not writer.isOpened():
        raise SystemExit(f"OpenCV cannot write {path}; the dry run needs an mp4 writer")
    for frame in synthetic_frames(seed):
        writer.write(frame)
    writer.release()


def write_png_dir(path: Path, seed: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(synthetic_frames(seed)):
        cv2.imwrite(str(path / f"frame_{i:04d}.png"), frame)


def main() -> None:
    if not MANIFEST.exists():
        raise SystemExit("models not pinned yet; run: uv run python src\\fetch_landmarkers.py")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        videos = tmp / "videos"
        write_video(videos / "a" / "001.mp4", 1)
        write_video(videos / "a" / "002.mp4", 2)
        write_video(videos / "b" / "003.mp4", 3)
        (videos / "b" / "zz_corrupt.mp4").write_bytes(b"not a video" * 100)
        out = tmp / "out"

        # --- limit and output format ------------------------------------------
        first = run(str(videos), "--out", str(out), "--limit", "2")
        if not check("limited run succeeds", first.returncode == 0, show(first)):
            return report()
        produced = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.npz"))
        check("--limit takes the first inputs in sorted order",
              produced == ["a/001.npz", "a/002.npz"], str(produced))

        with np.load(out / "a" / "001.npz", allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        check("documented keys, and no pixel hashes for video",
              set(arrays) == {"hands", "pose", "handedness", "handedness_score", "detection_index",
                              "timestamps", "fps", "size"}, str(set(arrays)))
        t = arrays["timestamps"].shape[0]
        check("one output frame per decoded frame", t == FRAMES, f"{t} frames")
        check("shapes and dtypes",
              arrays["hands"].shape == (t, 42, 3) and arrays["hands"].dtype == np.float32
              and arrays["pose"].shape == (t, 33, 3) and arrays["pose"].dtype == np.float32
              and arrays["handedness"].shape == (t, 2) and arrays["handedness"].dtype == np.int8
              and arrays["handedness_score"].shape == (t, 2) and arrays["handedness_score"].dtype == np.float32
              and arrays["detection_index"].shape == (t, 2) and arrays["detection_index"].dtype == np.int8)
        expected = np.array([round(i * 1000 / float(arrays["fps"])) for i in range(t)])
        check("timestamps are round(i * 1000 / fps)", np.array_equal(arrays["timestamps"], expected))
        check("size is width, height", tuple(arrays["size"]) == (320, 240), str(arrays["size"]))
        check("undetected hands are NaN, not zero",
              np.isnan(arrays["hands"]).all() and not (arrays["hands"] == 0).any())
        check("no hand means handedness -1, score NaN, detection index -1",
              (arrays["handedness"] == -1).all() and np.isnan(arrays["handedness_score"]).all()
              and (arrays["detection_index"] == -1).all())

        # --- resume, failure isolation, atomicity -------------------------------
        before = {p: p.stat().st_mtime_ns for p in out.rglob("*.npz")}
        second = run(str(videos), "--out", str(out), "--workers", "2")
        check("a run with a failed input exits non-zero", second.returncode == 1, show(second))
        check("existing outputs are skipped, not rewritten",
              all(p.stat().st_mtime_ns == m for p, m in before.items()))
        check("the remaining good video is extracted with two workers", (out / "b" / "003.npz").exists())
        check("the corrupt video has no output", not (out / "b" / "zz_corrupt.npz").exists())
        check("no partial files survive", not list(out.rglob("*.partial")))
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        check("manifest records both runs", len(manifest["runs"]) == 2)
        check("manifest names the failed input",
              any("zz_corrupt" in f["source"] for f in manifest["runs"][-1]["failed"]))
        log_lines = (out / "extract.log").read_text(encoding="utf-8").splitlines()
        check("log has one line per attempted input", len(log_lines) == 4, f"{len(log_lines)} lines")

        # --- configuration guard ----------------------------------------------
        count = len(list(out.rglob("*.npz")))
        refused = run(str(videos), "--out", str(out), "--max-width", "160")
        check("a different configuration is refused", refused.returncode != 0 and "max_width" in refused.stdout + refused.stderr,
              show(refused))
        check("the refused run wrote nothing", len(list(out.rglob("*.npz"))) == count)

        # --- model pin -------------------------------------------------------
        pins = json.loads(MANIFEST.read_text(encoding="utf-8"))
        pins["models"]["hand"]["sha256"] = "0" * 64
        bad_manifest = tmp / "bad_manifest.json"
        bad_manifest.write_text(json.dumps(pins), encoding="utf-8")
        pinned = run(str(videos), "--out", str(tmp / "pinned"), "--manifest", str(bad_manifest))
        check("a model that does not match its pin is refused",
              pinned.returncode != 0 and "pins" in pinned.stdout + pinned.stderr, show(pinned))

        # --- frame sequences ------------------------------------------------------
        write_png_dir(tmp / "frames" / "clipA", 4)
        write_png_dir(tmp / "frames" / "clipB", 5)
        no_fps = run(str(tmp / "frames" / "clipA"), "--frames", "--out", str(tmp / "fout"))
        check("--frames without --fps is refused", no_fps.returncode != 0)
        framed = run(str(tmp / "frames" / "clipA"), str(tmp / "frames" / "clipB"),
                     "--frames", "--fps", "30", "--out", str(tmp / "fout"), "--workers", "2")
        if check("frame sequences extract", framed.returncode == 0, show(framed)):
            with np.load(tmp / "fout" / "clipA.npz", allow_pickle=False) as data:
                hashes = [str(h) for h in data["pixel_sha256"]]
                stamps = data["timestamps"]
            direct = []
            for png in sorted((tmp / "frames" / "clipA").glob("*.png")):
                rgb = cv2.cvtColor(cv2.imread(str(png)), cv2.COLOR_BGR2RGB)
                direct.append(hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest())
            check("pixel hashes are of the RGB bytes of each PNG", hashes == direct)
            check("frame timestamps use --fps", np.array_equal(stamps, [round(i * 1000 / 30) for i in range(FRAMES)]))

    report()


def report() -> None:
    failed = [name for name, ok in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)} passed, {len(failed)} failed")
    if failed:
        print("FAILURES:", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
