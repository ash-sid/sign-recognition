"""
src/extract_landmarks.py

Runs the browser's landmarkers over recorded video and writes one compressed
array file per input, so that a model can be trained on the landmarks the
deployed page will actually produce.

The training data a model sees and the input it is deployed on have to come
from the same extractor. Landmarks from a different MediaPipe pipeline carry
their own conventions for which hand is tracked and when, and no amount of
preprocessing can reconcile two sets of conventions that were never measured
against each other. This module therefore loads the identical `.task` files
the web app loads -- pinned by hash in data/extractor_manifest.json, which
src/fetch_landmarkers.py fills from the URLs in web/src/tracking.ts -- and
assigns hands to slots with a port of the browser's rule (src/tracking.py).

Each output holds a superset of what the current contract uses: both hands
in slot order, and all 33 pose landmarks rather than the eight the contract
keeps. A later contract can then choose different landmarks without
re-extracting. Anything not detected is NaN, never zero; zero is reserved by
the contract for a hand the sign does not use, which is a decision made over
a whole sequence and not something an extractor can know.

    hands        (T, 42, 3) float32   left slot then right slot
    pose         (T, 33, 3) float32
    handedness   (T, 2)     int8      label the landmarker reported for the
                                      hand placed in each slot: 0 Left,
                                      1 Right, -1 no hand
    timestamps   (T,)       int64     milliseconds passed to the landmarkers
    fps          ()         float64
    size         (2,)       int64     width, height of the image landmarked
    pixel_sha256 (T,)       str       frame sequences only; see --frames

Timestamps are round(i * 1000 / fps) for frame i. Video mode in both
landmarkers carries tracking state and smoothing from one frame to the next,
so the timestamps are part of the input, and the browser's seam page uses
the same rule. A fresh pair of landmarkers is created for every input so that
no tracking state carries from the end of one video into the start of the
next.

Every output directory has a manifest.json recording the configuration that
produced it. A run whose configuration differs from the one already recorded
there is refused: a cache holding outputs from two extractor configurations
is the mixed-extractor problem this module exists to prevent, and nothing in
the arrays themselves would reveal it.

Resumable: an input whose output already exists is skipped. Outputs are
written to a temporary file and renamed into place, so an interrupted run
cannot leave a truncated file that the next run would then skip as done.

Run:
    uv run python src\\extract_landmarks.py D:\\asl-citizen\\videos --out data\\landmarks\\asl_citizen --workers 6
    uv run python src\\extract_landmarks.py @sample.txt --out data\\landmarks\\sample
    uv run python src\\extract_landmarks.py data\\seam\\clip1 --frames --fps 30 --out data\\landmarks\\seam --workers 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracking  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "extractor_manifest.json"
DEFAULT_MODELS = ROOT / "models" / "landmarkers"
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}
FRAME_SUFFIX = ".png"

# The options the browser passes. Anything left at MediaPipe's default in
# web/src/tracking.ts is left at the same default here.
NUM_HANDS = 2
NUM_POSES = 1

HANDEDNESS_CODE = {"Left": 0, "Right": 1}


@dataclass(frozen=True)
class Config:
    """Everything that decides what an output contains. Two runs with
    different values of any field must not write into one directory."""

    hand_model_sha256: str
    pose_model_sha256: str
    mediapipe_version: str
    delegate: str
    num_hands: int
    num_poses: int
    max_width: int | None
    frames: bool
    frames_fps: float | None


# --- inputs -------------------------------------------------------------------

def discover(inputs: list[str], frames: bool) -> list[tuple[Path, Path]]:
    """(source, path relative to its input root) for every input to process,
    sorted so --limit selects the same inputs on every run."""
    found: list[tuple[Path, Path]] = []
    for raw in inputs:
        path = Path(raw)
        if frames:
            if not path.is_dir():
                raise SystemExit(f"--frames expects directories of {FRAME_SUFFIX} files: {path}")
            found.append((path, Path(path.name)))
        elif path.is_file():
            found.append((path, Path(path.name)))
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in VIDEO_SUFFIXES:
                    found.append((child, child.relative_to(path)))
        else:
            raise SystemExit(f"no such file or directory: {path}")
    return sorted(found, key=lambda item: str(item[1]))


def output_path(out_dir: Path, relative: Path) -> Path:
    return out_dir / relative.with_suffix(".npz")


# --- frames -------------------------------------------------------------------

def _fit_width(rgb: np.ndarray, max_width: int | None) -> np.ndarray:
    height, width = rgb.shape[:2]
    if max_width is None or width <= max_width:
        return rgb
    new_height = round(height * max_width / width)
    return cv2.resize(rgb, (max_width, new_height), interpolation=cv2.INTER_AREA)


def video_frames(path: Path, max_width: int | None):
    """Yield RGB frames from a video, and return its container frame rate."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise RuntimeError(f"container reports no usable frame rate ({fps})")
        yield fps
        while True:
            ok, bgr = capture.read()
            if not ok:
                return
            yield _fit_width(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), max_width)
    finally:
        capture.release()


def sequence_frames(path: Path, fps: float, max_width: int | None):
    """Yield RGB frames from a directory of PNG files, in filename order."""
    files = sorted(p for p in path.iterdir() if p.suffix.lower() == FRAME_SUFFIX)
    if not files:
        raise RuntimeError(f"no {FRAME_SUFFIX} files in {path}")
    yield fps
    for file in files:
        bgr = cv2.imread(str(file), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"could not read {file}")
        yield _fit_width(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), max_width)


# --- landmarking ----------------------------------------------------------------

_WORKER: dict = {}


def _init_worker(config: Config, hand_model: str, pose_model: str) -> None:
    """Per-process setup. Module-level so that Windows, which starts workers
    by re-importing this file, can find it."""
    cv2.setNumThreads(1)
    _WORKER.update(config=config, hand_model=hand_model, pose_model=pose_model)


def _landmarkers(config: Config, hand_model: str, pose_model: str):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    delegate = getattr(BaseOptions.Delegate, config.delegate)
    hands = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=hand_model, delegate=delegate),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=config.num_hands,
        )
    )
    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=pose_model, delegate=delegate),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=config.num_poses,
        )
    )
    return mp, hands, pose


def _as_array(landmarks) -> np.ndarray:
    return np.array([(p.x, p.y, p.z) for p in landmarks], dtype=np.float32)


def landmark_sequence(frames, config: Config, hand_model: str, pose_model: str) -> dict:
    """Landmark every frame of one input with a fresh pair of landmarkers."""
    fps = next(frames)
    mp, hand_landmarker, pose_landmarker = _landmarkers(config, hand_model, pose_model)
    hands_out, pose_out, handed_out, stamps, hashes = [], [], [], [], []
    size = None
    try:
        for i, rgb in enumerate(frames):
            rgb = np.ascontiguousarray(rgb)
            if size is None:
                size = (rgb.shape[1], rgb.shape[0])
            elif size != (rgb.shape[1], rgb.shape[0]):
                raise RuntimeError(f"frame {i} is {rgb.shape[1]}x{rgb.shape[0]}, not {size[0]}x{size[1]}")
            if config.frames:
                hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())

            timestamp = round(i * 1000 / fps)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # Pose first, then hands, as the browser does. Each landmarker
            # keeps its own state, so the order is not expected to matter,
            # but there is no reason to find out.
            pose_result = pose_landmarker.detect_for_video(image, timestamp)
            hand_result = hand_landmarker.detect_for_video(image, timestamp)

            pose_frame = np.full((tracking.POSE_LANDMARKS, 3), np.nan, dtype=np.float32)
            pose_landmarks = None
            if pose_result.pose_landmarks:
                pose_landmarks = _as_array(pose_result.pose_landmarks[0])
                pose_frame[:] = pose_landmarks

            detected = [_as_array(h) for h in hand_result.hand_landmarks]
            labels = [c[0].category_name if c else "" for c in hand_result.handedness]
            slots = tracking.assign_slots(detected, pose_landmarks, labels)

            hand_frame = np.full((tracking.HANDS_END, 3), np.nan, dtype=np.float32)
            handed = np.full(2, -1, dtype=np.int8)
            for hand, label, slot in zip(detected, labels, slots):
                hand_frame[slot : slot + tracking.HAND_LANDMARKS] = hand
                handed[0 if slot == tracking.LEFT_HAND_START else 1] = HANDEDNESS_CODE.get(label, -1)

            hands_out.append(hand_frame)
            pose_out.append(pose_frame)
            handed_out.append(handed)
            stamps.append(timestamp)
    finally:
        hand_landmarker.close()
        pose_landmarker.close()

    if not stamps:
        raise RuntimeError("no frames decoded")
    arrays = {
        "hands": np.stack(hands_out),
        "pose": np.stack(pose_out),
        "handedness": np.stack(handed_out),
        "timestamps": np.asarray(stamps, dtype=np.int64),
        "fps": np.float64(fps),
        "size": np.asarray(size, dtype=np.int64),
    }
    if config.frames:
        arrays["pixel_sha256"] = np.asarray(hashes)
    return arrays


def write_atomic(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def process_one(job: tuple[str, str]) -> dict:
    """Worker entry point: landmark one input and write its output."""
    source, destination = Path(job[0]), Path(job[1])
    config: Config = _WORKER["config"]
    started = time.perf_counter()
    try:
        if config.frames:
            frames = sequence_frames(source, config.frames_fps, config.max_width)
        else:
            frames = video_frames(source, config.max_width)
        arrays = landmark_sequence(frames, config, _WORKER["hand_model"], _WORKER["pose_model"])
        write_atomic(destination, arrays)
        return {
            "source": str(source),
            "status": "ok",
            "frames": int(arrays["timestamps"].shape[0]),
            "seconds": time.perf_counter() - started,
        }
    except Exception as error:  # one bad video must not end an overnight run
        return {
            "source": str(source),
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "trace": traceback.format_exc(limit=3),
        }


# --- provenance -----------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_models(manifest_path: Path, models_dir: Path) -> tuple[dict, Path, Path]:
    """The pinned model files, refusing to proceed if either differs from
    its recorded hash."""
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} not found; run src\\fetch_landmarkers.py first")
    pins = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = {}
    for role in ("hand", "pose"):
        entry = pins["models"][role]
        path = models_dir / entry["file"]
        if not path.exists():
            raise SystemExit(f"{path} not found; run src\\fetch_landmarkers.py first")
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise SystemExit(
                f"{path} has sha256 {actual}, but {manifest_path} pins {entry['sha256']}. "
                "The file on disk is not the model the manifest describes."
            )
        paths[role] = path
    return pins, paths["hand"], paths["pose"]


def check_output_config(out_dir: Path, config: Config) -> dict:
    """Load the directory's manifest, refusing if it records a different
    configuration from this run's."""
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        return {"config": asdict(config), "runs": []}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["config"] != asdict(config):
        differing = sorted(
            key for key in asdict(config) if manifest["config"].get(key) != asdict(config)[key]
        )
        raise SystemExit(
            f"{out_dir} holds outputs from a different extractor configuration "
            f"(differs in: {', '.join(differing)}). Write to a new directory."
        )
    return manifest


def write_manifest(out_dir: Path, manifest: dict) -> None:
    path = out_dir / "manifest.json"
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[1], fromfile_prefix_chars="@"
    )
    parser.add_argument("inputs", nargs="+", help="video files, directories of videos, "
                        "or with --frames directories of PNG frames; @file reads one per line")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", action="store_true",
                        help="each input is a directory of PNG frames forming one sequence")
    parser.add_argument("--fps", type=float, default=None,
                        help="frame rate of --frames sequences (required with --frames)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None,
                        help="consider only the first N inputs in sorted order")
    parser.add_argument("--max-width", type=int, default=None,
                        help="downscale wider frames to this width before landmarking")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    args = parser.parse_args()

    if args.frames and not args.fps:
        parser.error("--frames needs --fps")
    if not args.frames and args.fps:
        parser.error("--fps applies only to --frames; videos use their container frame rate")

    import mediapipe

    pins, hand_model, pose_model = verified_models(args.manifest, args.models)
    config = Config(
        hand_model_sha256=pins["models"]["hand"]["sha256"],
        pose_model_sha256=pins["models"]["pose"]["sha256"],
        mediapipe_version=mediapipe.__version__,
        delegate="CPU",
        num_hands=NUM_HANDS,
        num_poses=NUM_POSES,
        max_width=args.max_width,
        frames=args.frames,
        frames_fps=args.fps,
    )
    if config.mediapipe_version != pins.get("tasks_vision_version"):
        print(f"note: mediapipe {config.mediapipe_version} here, "
              f"@mediapipe/tasks-vision {pins.get('tasks_vision_version')} in the browser")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = check_output_config(args.out, config)
    # Recorded before anything is extracted, so that an interrupted first run
    # still leaves the directory marked with the configuration it holds.
    write_manifest(args.out, manifest)

    found = discover(args.inputs, args.frames)
    if args.limit is not None:
        found = found[: args.limit]
    jobs = [(str(src), str(output_path(args.out, rel))) for src, rel in found]
    todo = [job for job in jobs if not Path(job[1]).exists()]
    print(f"{len(jobs)} inputs, {len(jobs) - len(todo)} already extracted, {len(todo)} to do")

    log_path = args.out / "extract.log"
    started = time.time()
    done = failed = 0
    failures = []
    initargs = (config, str(hand_model), str(pose_model))
    with open(log_path, "a", encoding="utf-8") as log:
        if args.workers > 1:
            pool = multiprocessing.Pool(args.workers, initializer=_init_worker, initargs=initargs)
            results = pool.imap_unordered(process_one, todo)
        else:
            pool = None
            _init_worker(*initargs)
            results = map(process_one, todo)
        try:
            for result in results:
                log.write(json.dumps(result, ensure_ascii=True) + "\n")
                log.flush()
                if result["status"] == "ok":
                    done += 1
                else:
                    failed += 1
                    failures.append({"source": result["source"], "error": result["error"]})
                finished = done + failed
                if finished % 50 == 0 or finished == len(todo):
                    elapsed = time.time() - started
                    rate = finished / elapsed if elapsed else 0.0
                    print(f"{finished}/{len(todo)}  failed {failed}  {rate:.2f}/s")
        finally:
            if pool is not None:
                pool.close()
                pool.join()

    manifest["runs"].append({
        "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        "seconds": round(time.time() - started, 1),
        "extractor_sha256": sha256_file(Path(__file__)),
        "tasks_vision_version": pins.get("tasks_vision_version"),
        "inputs": len(jobs),
        "extracted": done,
        "failed": failures,
    })
    write_manifest(args.out, manifest)
    print(f"extracted {done}, failed {failed}; log {log_path}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
