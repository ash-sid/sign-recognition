"""
src/make_sequence_fixture.py

Writes web/tests/fixtures/sequence_slots_generated.json: sequences of hand
detections built around the decisions in tracking.assign_sequence_slots()
that rounding can tip -- steps between frames within a hair of the continuity
distance, hands near the midpoint between the pose wrists so their margins
sum to almost nothing, frames without a pose, frames without hands -- with
the slots this implementation assigns to each.

The hand-written fixture says what the rule should do. This one says what the
Python implementation does on inputs where the last bit matters, so that the
browser port is checked against it exactly and not only on cases with a wide
margin.

Random sequences rarely land close enough to a threshold for arithmetic to
decide it, so some cases are found by search instead: inputs where computing
the continuity distance, or the wrist margin, in float32 rather than in
doubles reverses the decision. A port that narrowed its arithmetic -- typed
float32 buffers are the easy way to do it in a browser -- fails on those.
Differences of a single double-precision ulp, such as hypot against sqrt,
cannot be found this way in reasonable time; both implementations avoid them
by writing the arithmetic out identically instead. Inputs are rounded to float32 before the expected slots are computed,
as the landmarker's are, and are written in their shortest float32 form;
both tests round what they read back to float32, so both sides start from
the same numbers.

Deterministic: the same seed writes the same file.

Run: uv run python src\\make_sequence_fixture.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import tracking

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "tests" / "fixtures" / "sequence_slots_generated.json"
SEED = 20260930
SEQUENCES = 30
SEARCHED = 6
ASPECTS = (480 / 640, 416 / 640, 720 / 1280, 1.0)


def f32(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def short(array: np.ndarray) -> list[float]:
    """float32 values written in their shortest round-tripping form. Both
    tests round what they read back to float32, which recovers these values
    exactly, so the file need not carry seventeen digits per number."""
    return [float(str(v)) for v in f32(array).ravel()]


def _distance32(a: np.ndarray, b: np.ndarray, aspect: float) -> float:
    """tracking._mean_distance computed in float32 throughout."""
    total = np.float32(0)
    for p, q in zip(a, b):
        dx = np.float32(p[0]) - np.float32(q[0])
        dy = (np.float32(p[1]) - np.float32(q[1])) * np.float32(aspect)
        total = np.float32(total + np.sqrt(np.float32(dx * dx + dy * dy)))
    return float(total / np.float32(len(a)))


def _margin_sign32(wrist, left, right, aspect: float) -> bool:
    """Whether tracking._wrist_margin, computed in float32, is >= 0."""
    w, l, r, k = (np.float32(v) for v in (0, 0, 0, aspect))
    lx = np.float32(wrist[0]) - np.float32(left[0])
    ly = (np.float32(wrist[1]) - np.float32(left[1])) * k
    rx = np.float32(wrist[0]) - np.float32(right[0])
    ry = (np.float32(wrist[1]) - np.float32(right[1])) * k
    return bool(np.float32(rx * rx + ry * ry) - np.float32(lx * lx + ly * ly) >= 0)


def _pose(left, right) -> dict:
    return {"left": short(left), "right": short(right)}


def searched_cases(rng, template) -> list[dict]:
    """Two-frame sequences whose linking, and one-frame sequences whose slot,
    float32 arithmetic would decide the other way."""
    names = {tracking.LEFT_HAND_START: "left", tracking.RIGHT_HAND_START: "right"}
    found = []
    aspect = 480 / 640
    # Linking: frame 0 sits on the left wrist; frame 1, alone, would vote
    # right, because its pose wrists are swapped. Linked, the stronger frame 0
    # carries both frames left; unlinked, frame 1 goes right.
    left_wrist, right_wrist = np.array([0.35, 0.6]), np.array([0.65, 0.6])
    tries = 0
    while sum(c["name"].startswith("searched continuity") for c in found) < SEARCHED:
        tries += 1
        start = left_wrist + rng.normal(0, 0.005, 2)
        angle = rng.uniform(0, 2 * np.pi)
        step = tracking.CONTINUITY * (1 + rng.uniform(-3e-7, 3e-7))
        end = start + step * np.array([np.cos(angle), np.sin(angle) / aspect])
        a = f32(np.column_stack([start[0] + template[:, 0], start[1] + template[:, 1], np.zeros(21)]))
        b = f32(np.column_stack([end[0] + template[:, 0], end[1] + template[:, 1], np.zeros(21)]))
        a[tracking.HAND_WRIST, :2], b[tracking.HAND_WRIST, :2] = f32(start), f32(end)
        exact = tracking._mean_distance(b.tolist(), a.tolist(), aspect)
        if (exact < tracking.CONTINUITY) == (_distance32(b, a, aspect) < tracking.CONTINUITY):
            continue
        frames = [
            {"hands": [short(a)], "pose_wrists": _pose(left_wrist, right_wrist)},
            {"hands": [short(b)], "pose_wrists": _pose(right_wrist, left_wrist)},
        ]
        found.append(_case(f"searched continuity {len(found):02d}", aspect, frames, names))
    print(f"continuity: {tries} tries for {SEARCHED} cases")

    tries = 0
    while len(found) < 2 * SEARCHED:
        tries += 1
        left = left_wrist + rng.normal(0, 0.02, 2)
        right = right_wrist + rng.normal(0, 0.02, 2)
        mid = (left + right) / 2
        normal = np.array([-(right - left)[1] * aspect, (right - left)[0] / aspect])
        wrist = mid + normal * rng.uniform(-1, 1) + (right - left) * rng.uniform(-2e-7, 2e-7)
        hand = f32(np.column_stack([wrist[0] + template[:, 0], wrist[1] + template[:, 1], np.zeros(21)]))
        hand[tracking.HAND_WRIST, :2] = f32(wrist)
        l32, r32 = f32(left), f32(right)
        pose = np.zeros((tracking.POSE_LANDMARKS, 3), dtype=np.float32)
        pose[tracking.POSE_LEFT_WRIST, :2], pose[tracking.POSE_RIGHT_WRIST, :2] = l32, r32
        exact = tracking._wrist_margin(hand.tolist(), pose.tolist(), aspect) >= 0
        if exact == _margin_sign32(hand[tracking.HAND_WRIST], l32, r32, aspect):
            continue
        frames = [{"hands": [short(hand)], "pose_wrists": _pose(left, right)}]
        found.append(_case(f"searched margin {len(found) - SEARCHED:02d}", aspect, frames, names))
    print(f"margin: {tries} tries for {SEARCHED} cases")
    return found


def _case(name: str, aspect: float, frames: list[dict], names: dict) -> dict:
    """A case with the slots this implementation assigns, computed from the
    values exactly as written to the file."""
    hands = [[f32(h).reshape(tracking.HAND_LANDMARKS, 3) for h in f["hands"]] for f in frames]
    poses = []
    for f in frames:
        p = np.zeros((tracking.POSE_LANDMARKS, 3), dtype=np.float32)
        p[tracking.POSE_LEFT_WRIST, :2] = f["pose_wrists"]["left"]
        p[tracking.POSE_RIGHT_WRIST, :2] = f["pose_wrists"]["right"]
        poses.append(p)
    slots = tracking.assign_sequence_slots(hands, poses, aspect)
    return {"name": name, "aspect": aspect, "frames": frames,
            "expected": [[names[s] for s in frame] for frame in slots]}


def main() -> None:
    rng = np.random.default_rng(SEED)
    # One hand shape, translated and lightly perturbed, so that a hand's mean
    # landmark distance to its own previous position is close to the length
    # of the step it took.
    template = rng.normal(0, 0.03, (tracking.HAND_LANDMARKS, 2))
    template[tracking.HAND_WRIST] = 0.0

    cases = []
    near_continuity = 0
    for n in range(SEQUENCES):
        aspect = float(ASPECTS[n % len(ASPECTS)])
        length = int(rng.integers(3, 9))
        count = int(rng.integers(1, 3))
        left_wrist = np.array([0.35, 0.6]) + rng.normal(0, 0.02, 2)
        right_wrist = np.array([0.65, 0.6]) + rng.normal(0, 0.02, 2)
        middle = (left_wrist + right_wrist) / 2
        wrists = [middle + rng.normal(0, 0.03, 2) for _ in range(count)]
        frames = []
        for t in range(length):
            roll = rng.random()
            if roll < 0.08:
                frames.append({"hands": [], "pose_wrists": None})
                continue
            hands = []
            for k in range(count):
                if t > 0:
                    # Step either well inside the continuity distance or
                    # within a few float32 ulps of it, in a random direction
                    # corrected for the aspect ratio.
                    angle = rng.uniform(0, 2 * np.pi)
                    if rng.random() < 0.5:
                        length_step = tracking.CONTINUITY * (1 + rng.choice([-2e-7, -5e-8, 0.0, 5e-8, 2e-7]))
                        near_continuity += 1
                    else:
                        length_step = tracking.CONTINUITY * rng.uniform(0.1, 0.6)
                    wrists[k] = wrists[k] + length_step * np.array([np.cos(angle), np.sin(angle) / aspect])
                points = np.column_stack([
                    wrists[k][0] + template[:, 0],
                    wrists[k][1] + template[:, 1],
                    np.zeros(tracking.HAND_LANDMARKS),
                ])
                points[tracking.HAND_WRIST, :2] = wrists[k]
                hands.append(f32(points))
            if count == 2 and rng.random() < 0.3:
                hands.reverse()
            pose = None if roll < 0.18 else {
                "left": short(left_wrist + rng.normal(0, 0.01, 2)),
                "right": short(right_wrist + rng.normal(0, 0.01, 2)),
            }
            frames.append({"hands": [short(h) for h in hands], "pose_wrists": pose})

        poses = []
        for frame in frames:
            if frame["pose_wrists"] is None:
                poses.append(None)
                continue
            p = np.zeros((tracking.POSE_LANDMARKS, 3), dtype=np.float32)
            p[tracking.POSE_LEFT_WRIST, :2] = frame["pose_wrists"]["left"]
            p[tracking.POSE_RIGHT_WRIST, :2] = frame["pose_wrists"]["right"]
            poses.append(p)
        slots = tracking.assign_sequence_slots(
            [[f32(h).reshape(tracking.HAND_LANDMARKS, 3) for h in frame["hands"]] for frame in frames],
            poses,
            aspect,
        )
        names = {tracking.LEFT_HAND_START: "left", tracking.RIGHT_HAND_START: "right"}
        cases.append({
            "name": f"generated {n:02d}",
            "aspect": aspect,
            "frames": frames,
            "expected": [[names[s] for s in frame] for frame in slots],
        })

    cases += searched_cases(rng, template)

    doc = {
        "description": "Generated by src/make_sequence_fixture.py; expected slots are what "
                       "src/tracking.py assigns. Hands are 21 x 3 landmarks, flattened.",
        "seed": SEED,
        "cases": cases,
    }
    OUT.write_text(json.dumps(doc) + "\n", encoding="utf-8")
    print(f"{len(cases)} sequences, {near_continuity} steps within float32 rounding of the "
          f"continuity distance; wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} kB)")


if __name__ == "__main__":
    main()
