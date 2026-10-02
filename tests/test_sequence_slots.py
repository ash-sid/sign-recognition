"""
tests/test_sequence_slots.py

Checks src/tracking.py's assign_sequence_slots() against two fixtures that
web/tests/sequenceSlots.test.ts checks the browser port against too:

  - web/tests/fixtures/sequence_slots.json, hand-written cases whose expected
    slots were reasoned out by hand, each exercising one documented behaviour;
  - web/tests/fixtures/sequence_slots_generated.json, sequences built by
    src/make_sequence_fixture.py around near-ties, with the slots this
    implementation produced when the file was written. Here that checks the
    implementation has not changed since; there it checks the port agrees
    bit for bit on inputs where rounding decides the answer.

Coordinates are rounded to float32 on the way in, as the landmarker's are.

Run: uv run python tests\\test_sequence_slots.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import tracking  # noqa: E402

FIXTURES = ROOT / "web" / "tests" / "fixtures"
NAMES = {tracking.LEFT_HAND_START: "left", tracking.RIGHT_HAND_START: "right"}
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail and not cond else ''}")


def hand(values):
    if len(values) == 2:
        return np.tile(np.array([values[0], values[1], 0.0], dtype=np.float32), (tracking.HAND_LANDMARKS, 1))
    return np.asarray(values, dtype=np.float32).reshape(tracking.HAND_LANDMARKS, 3)


def pose(wrists):
    if wrists is None:
        return None
    out = np.zeros((tracking.POSE_LANDMARKS, 3), dtype=np.float32)
    out[tracking.POSE_LEFT_WRIST, :2] = wrists["left"]
    out[tracking.POSE_RIGHT_WRIST, :2] = wrists["right"]
    return out


def run(case):
    slots = tracking.assign_sequence_slots(
        [[hand(h) for h in f["hands"]] for f in case["frames"]],
        [pose(f["pose_wrists"]) for f in case["frames"]],
        case["aspect"],
    )
    return [[NAMES[s] for s in frame] for frame in slots]


for fixture in ("sequence_slots.json", "sequence_slots_generated.json"):
    path = FIXTURES / fixture
    if not path.exists():
        check(f"{fixture} exists", False, "run src\\make_sequence_fixture.py")
        continue
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    check(f"{fixture} has cases", len(cases) > 0)
    for case in cases:
        got = run(case)
        check(f"{fixture}: {case['name']}", got == case["expected"], f"got {got}")

print()
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILURES:", FAIL)
    sys.exit(1)
