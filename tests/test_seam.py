"""
tests/test_seam.py

Compares what the browser's landmarkers and src/extract_landmarks.py produce
from identical frames, and measures how each candidate slot rule behaves on
both. This is the seam the preprocessing parity fixture does not cover: that
fixture starts from a landmark array, and this is where the array comes from.

Inputs are the JSON files web/seam.html downloads and the .npz files the
extractor writes with --frames over the same PNG directories. Both hold the
landmarkers' raw detections -- hands in detection order with handedness label
and score, and the full pose -- as well as the slotted layout each side built.
Everything here is computed from those files, so it reproduces from a clone
without a browser, a camera or the frames.

Required, and any failure exits non-zero:

  - every frame's RGB hash and timestamp agree, so both sides landmarked the
    same pixels with the same tracking input;
  - src/tracking.py, applied to each side's raw detections, reproduces that
    side's slotted output on every frame. For the browser this is the port
    checked against the original on real detections, not only on the
    hand-written fixture.

Measured and reported:

  - hand detection: hands are paired across the two sides by mean distance
    over all 21 landmarks, never by wrist alone, since two hands held close
    together have close wrists. Pairs under MATCH_PX are the same hand; pairs
    between MATCH_PX and DIFFERENT_PX are counted as ambiguous and excluded,
    rather than guessed either way;
  - landmark discrepancy on paired hands, and on the pose split into
    landmarks inside and outside the frame, since the pose model reports
    positions for body parts the camera cannot see;
  - for each candidate slot rule, how often a hand changes slot between
    consecutive frames of one run while staying the same hand, and how often
    the two sides put the same hand in different slots.

The slot rules are evaluated, not asserted. They are a design choice whose
effect this measures. The first four decide frame by frame, or carry the
previous frame's decision forward; the last two are
tracking.assign_sequence_slots, which links hands into tracks and gives each
track one slot, run over a whole clip as for training videos and over the
trailing window the live page would see. For those two, flips within a run
come only from tracks breaking, and from a window's vote changing as it
slides.

Run:
    uv run python tests\\test_seam.py (Get-ChildItem tests\\fixtures\\seam\\seam_*.json).FullName --python tests\\fixtures\\seam
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import tracking  # noqa: E402

# Same-hand pairing thresholds, in pixels of mean distance over 21 landmarks.
# Chosen from the distribution between runtimes on the first recorded clips,
# which is dense below 60 px and empty between 80 and 120.
MATCH_PX = 60.0
DIFFERENT_PX = 120.0
# How far a hand may move between consecutive frames and still be taken for
# the same hand by the rules that keep a hand in its previous slot.
CONTINUITY_PX = 60.0

LABELS = {0: "Left", 1: "Right"}
SLOT_NAME = {tracking.LEFT_HAND_START: "L", tracking.RIGHT_HAND_START: "R"}


@dataclass
class Hand:
    landmarks: np.ndarray  # (21, 3), normalized
    label: str
    score: float


@dataclass
class Frame:
    hands: list[Hand]
    pose: np.ndarray | None  # (33, 3), normalized
    slotted: np.ndarray  # (42, 3) as the side itself laid out the hands


@dataclass
class Side:
    name: str
    frames: list[Frame]
    timestamps: np.ndarray
    sha256: list[str]
    size: tuple[int, int]


# --- loading --------------------------------------------------------------------

def _array(values, shape) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], dtype=np.float64).reshape(shape)


def load_browser(path: Path) -> tuple[Side, dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if "hands" not in doc["frames"][0]:
        raise SystemExit(f"{path} has no raw detections; record it again with the current web/seam.html")
    frames = []
    for f in doc["frames"]:
        hands = [
            Hand(_array(h["landmarks"], (tracking.HAND_LANDMARKS, 3)), h["handedness"],
                 np.nan if h["score"] is None else float(h["score"]))
            for h in f["hands"]
        ]
        pose = None if f["pose"] is None else _array(f["pose"], (tracking.POSE_LANDMARKS, 3))
        layout = _array(f["landmarks"], (tracking.CONTRACT_LANDMARKS, 3))
        frames.append(Frame(hands, pose, layout[: tracking.HANDS_END]))
    side = Side(
        f"browser {doc['delegate']}",
        frames,
        np.array([f["timestamp"] for f in doc["frames"]], dtype=np.int64),
        [f["sha256"] for f in doc["frames"]],
        tuple(doc["size"]),
    )
    return side, doc


def load_python(path: Path) -> Side:
    with np.load(path, allow_pickle=False) as data:
        if "detection_index" not in data.files:
            raise SystemExit(f"{path} has no detection order; extract it again with the current extractor")
        hands = data["hands"].astype(np.float64)
        pose = data["pose"].astype(np.float64)
        handed, scores, order = data["handedness"], data["handedness_score"], data["detection_index"]
        frames = []
        for t in range(hands.shape[0]):
            found = []
            for column, start in enumerate((tracking.LEFT_HAND_START, tracking.RIGHT_HAND_START)):
                if order[t, column] >= 0:
                    found.append((int(order[t, column]), Hand(
                        hands[t, start : start + tracking.HAND_LANDMARKS],
                        LABELS.get(int(handed[t, column]), ""),
                        float(scores[t, column]),
                    )))
            found.sort(key=lambda item: item[0])
            frame_pose = None if np.isnan(pose[t]).all() else pose[t]
            frames.append(Frame([h for _, h in found], frame_pose, hands[t]))
        return Side(
            "extractor",
            frames,
            data["timestamps"],
            [str(h) for h in data["pixel_sha256"]],
            tuple(int(v) for v in data["size"]),
        )


# --- geometry -------------------------------------------------------------------

def pixel_distance(a: np.ndarray, b: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Per-landmark 2D distance in pixels between two (N, 3) normalized arrays."""
    return np.hypot((a[:, 0] - b[:, 0]) * size[0], (a[:, 1] - b[:, 1]) * size[1])


def hand_distance(a: Hand, b: Hand, size) -> float:
    return float(np.nanmean(pixel_distance(a.landmarks, b.landmarks, size)))


def pair(first: list[Hand], second: list[Hand], size) -> list[tuple[int, int, float]]:
    """Pair hands across two lists by the assignment with least total
    distance, over as many pairs as the shorter list allows."""
    if not first or not second:
        return []
    best, best_cost = [], np.inf
    if len(first) <= len(second):
        for chosen in itertools.permutations(range(len(second)), len(first)):
            pairs = [(i, j, hand_distance(first[i], second[j], size)) for i, j in enumerate(chosen)]
            cost = sum(d for _, _, d in pairs)
            if cost < best_cost:
                best, best_cost = pairs, cost
    else:
        flipped = pair(second, first, size)
        best = [(i, j, d) for j, i, d in flipped]
    return best


# --- slot rules -------------------------------------------------------------------

def rule_nearest_wrist(frame: Frame) -> list[str]:
    """The current rule: src/tracking.py, which the browser also runs."""
    slots = tracking.assign_slots(
        [h.landmarks for h in frame.hands], frame.pose, [h.label for h in frame.hands]
    )
    return [SLOT_NAME[s] for s in slots]


def rule_handedness(frame: Frame) -> list[str]:
    """The landmarker's own label, inverted for an unmirrored image. When both
    hands carry the same label, the less confident one takes the other slot."""
    slots = ["R" if h.label == "Left" else "L" for h in frame.hands]
    if len(slots) == 2 and slots[0] == slots[1]:
        a, b = frame.hands
        weaker = 1 if not (b.score > a.score) else 0
        slots[weaker] = "R" if slots[weaker] == "L" else "L"
    return slots


def keep_previous(fallback):
    """A hand that stays within CONTINUITY_PX of a hand in the previous frame
    keeps that hand's slot; a hand with no such predecessor takes a free slot
    if its partner was kept, and otherwise the fallback rule decides."""

    def run(frames: list[Frame], size) -> list[list[str]]:
        out: list[list[str]] = []
        previous: list[tuple[Hand, str]] = []
        for frame in frames:
            if not frame.hands:
                out.append([])
                previous = []
                continue
            slots: list[str | None] = [None] * len(frame.hands)
            if previous:
                for i, j, d in pair(frame.hands, [h for h, _ in previous], size):
                    if d < CONTINUITY_PX:
                        slots[i] = previous[j][1]
            if all(s is None for s in slots):
                decided = fallback(frame)
            else:
                taken = {s for s in slots if s is not None}
                free = [s for s in ("L", "R") if s not in taken]
                decided = [s if s is not None else free.pop(0) for s in slots]
            out.append(decided)
            previous = list(zip(frame.hands, decided))
        return out

    return run


def stateless(rule):
    def run(frames: list[Frame], size) -> list[list[str]]:
        return [rule(f) if f.hands else [] for f in frames]

    return run


def _sequence_slots(frames: list[Frame], size) -> list[list[str]]:
    slots = tracking.assign_sequence_slots(
        [[h.landmarks for h in f.hands] for f in frames],
        [f.pose for f in frames],
        size[1] / size[0],
    )
    return [[SLOT_NAME[s] for s in frame] for frame in slots]


def track_vote(frames: list[Frame], size) -> list[list[str]]:
    """tracking.assign_sequence_slots over the whole clip, as for a training
    video."""
    return _sequence_slots(frames, size)


def track_vote_window(width: int):
    """tracking.assign_sequence_slots over the `width` frames ending at each
    frame, keeping that frame's slots: what a live window would show for its
    newest frame."""

    def run(frames: list[Frame], size) -> list[list[str]]:
        return [_sequence_slots(frames[max(0, t - width + 1) : t + 1], size)[-1] for t in range(len(frames))]

    return run


WINDOW = 48

RULES = {
    "nearest pose wrist (current)": stateless(rule_nearest_wrist),
    "handedness label": stateless(rule_handedness),
    "keep previous, else nearest wrist": keep_previous(rule_nearest_wrist),
    "keep previous, else handedness": keep_previous(rule_handedness),
    "track vote, whole sequence": track_vote,
    f"track vote, {WINDOW}-frame window": track_vote_window(WINDOW),
}


def flips(frames: list[Frame], slots: list[list[str]], size) -> int:
    """Hands that change slot between consecutive frames while remaining,
    by CONTINUITY_PX, the same hand."""
    count = 0
    for t in range(1, len(frames)):
        for i, j, d in pair(frames[t].hands, frames[t - 1].hands, size):
            if d < CONTINUITY_PX and slots[t][i] != slots[t - 1][j]:
                count += 1
    return count


# --- comparison -------------------------------------------------------------------

def reproduces_slots(side: Side) -> list[int]:
    """Frames where src/tracking.py, given this side's raw detections, does
    not lay the hands out exactly as this side did."""
    bad = []
    for t, frame in enumerate(side.frames):
        expected = np.full((tracking.HANDS_END, 3), np.nan)
        slots = tracking.assign_slots(
            [h.landmarks for h in frame.hands], frame.pose, [h.label for h in frame.hands]
        )
        for hand, start in zip(frame.hands, slots):
            expected[start : start + tracking.HAND_LANDMARKS] = hand.landmarks
        if not np.array_equal(expected, frame.slotted, equal_nan=True):
            bad.append(t)
    return bad


def summary(values: np.ndarray) -> str:
    values = values[~np.isnan(values)]
    if values.size == 0:
        return "nothing to compare"
    p50, p99 = np.percentile(values, [50, 99])
    return f"median {p50:6.2f}  p99 {p99:6.2f}  max {values.max():6.2f} px"


def compare(browser_path: Path, python_dir: Path, totals: dict) -> list[str]:
    browser, doc = load_browser(browser_path)
    clip = doc["clip"]
    python_path = python_dir / f"{clip}.npz"
    print(f"\n== {clip}: {browser_path.name} against {python_path.name}")
    print(f"   tasks-vision {doc['tasks_vision_version']}, {doc['user_agent']}")
    if not python_path.exists():
        return [f"{clip}: {python_path} not found"]
    python = load_python(python_path)
    size = python.size

    n = len(browser.frames)
    if n != len(python.frames):
        return [f"{clip}: browser has {n} frames, extractor {len(python.frames)}"]
    mismatched = [t for t in range(n) if browser.sha256[t] != python.sha256[t]]
    if mismatched:
        return [f"{clip}: pixel hashes differ on {len(mismatched)} frames (first {mismatched[:5]})"]
    failures = []
    if not np.array_equal(browser.timestamps, python.timestamps):
        failures.append(f"{clip}: timestamps differ")
    for side in (browser, python):
        bad = reproduces_slots(side)
        if bad:
            failures.append(f"{clip}: src/tracking.py does not reproduce {side.name} slots on frames {bad[:10]}")
    print(f"   {n} frames; pixels and timestamps agree; "
          f"src/tracking.py reproduces both sides' slots" if not failures else f"   {n} frames")

    # Hand detection and landmark agreement.
    same, ambiguous, only_b, only_p = 0, 0, 0, 0
    label_agree = 0
    errors, pairs_by_frame = [], []
    for fb, fp in zip(browser.frames, python.frames):
        paired = pair(fb.hands, fp.hands, size)
        kept = []
        for i, j, d in paired:
            if d < MATCH_PX:
                same += 1
                kept.append((i, j))
                label_agree += fb.hands[i].label == fp.hands[j].label
                errors.append(pixel_distance(fb.hands[i].landmarks, fp.hands[j].landmarks, size))
            elif d < DIFFERENT_PX:
                ambiguous += 1
        pairs_by_frame.append(kept)
        paired_b = {i for i, _, d in paired if d < DIFFERENT_PX}
        paired_p = {j for _, j, d in paired if d < DIFFERENT_PX}
        only_b += len(fb.hands) - len(paired_b)
        only_p += len(fp.hands) - len(paired_p)
    print(f"   hands: {same} same on both sides, {ambiguous} ambiguous, "
          f"browser only {only_b}, extractor only {only_p}; labels agree on {label_agree}/{same}")
    hand_px = np.concatenate(errors) if errors else np.array([])
    print(f"   hand landmarks       {summary(hand_px)}")

    inside, outside = [], []
    for fb, fp in zip(browser.frames, python.frames):
        if fb.pose is None or fp.pose is None:
            continue
        d = pixel_distance(fb.pose, fp.pose, size)
        visible = np.all((fb.pose[:, :2] >= 0) & (fb.pose[:, :2] <= 1)
                         & (fp.pose[:, :2] >= 0) & (fp.pose[:, :2] <= 1), axis=1)
        inside.append(d[visible])
        outside.append(d[~visible])
    print(f"   pose inside frame    {summary(np.concatenate(inside) if inside else np.array([]))}")
    print(f"   pose outside frame   {summary(np.concatenate(outside) if outside else np.array([]))}")

    # Slot rules.
    print(f"   {'slot rule':36s} {'flips browser':>13s} {'flips extractor':>15s} {'same hand, other slot':>22s}")
    key = doc["delegate"]
    for name, run in RULES.items():
        slots_b, slots_p = run(browser.frames, size), run(python.frames, size)
        flip_b = flips(browser.frames, slots_b, size)
        flip_p = flips(python.frames, slots_p, size)
        split = sum(slots_b[t][i] != slots_p[t][j] for t, kept in enumerate(pairs_by_frame) for i, j in kept)
        print(f"   {name:36s} {flip_b:13d} {flip_p:15d} {split:14d} of {same:<5d}")
        total = totals.setdefault(key, {}).setdefault(name, [0, 0, 0, 0])
        total[0] += flip_b
        total[1] += flip_p
        total[2] += split
        total[3] += same
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare browser and extractor landmarks.")
    parser.add_argument("browser", nargs="+", type=Path, help="JSON files from web/seam.html")
    parser.add_argument("--python", type=Path, required=True,
                        help="directory of .npz files from extract_landmarks.py --frames")
    args = parser.parse_args()

    failures: list[str] = []
    totals: dict = {}
    for path in args.browser:
        failures += compare(path, args.python, totals)

    print("\n== all clips")
    for delegate, rules in totals.items():
        print(f"   browser {delegate}")
        for name, (flip_b, flip_p, split, same) in rules.items():
            print(f"   {name:36s} {flip_b:13d} {flip_p:15d} {split:14d} of {same:<5d}")

    print()
    if failures:
        for failure in failures:
            print("FAIL ", failure)
        sys.exit(1)
    print(f"PASS  {len(args.browser)} comparisons: same pixels, same timestamps, "
          "slots reproduced by src/tracking.py on both sides")


if __name__ == "__main__":
    main()
