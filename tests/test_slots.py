"""
tests/test_slots.py

Checks src/tracking.py's slot assignment against the hand-written cases in
web/tests/fixtures/slots.json. web/tests/slots.test.ts checks the browser's
implementation against the same file, so the two agree with each other by
each agreeing with cases neither of them produced.

A slot error does not fail anywhere else. Both hands still arrive, the
tensor has the right shape, and the only symptom is a model trained on one
convention and fed another.

Run: uv run python tests\\test_slots.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import tracking  # noqa: E402

FIXTURE = ROOT / "web" / "tests" / "fixtures" / "slots.json"
SLOT_NAMES = {tracking.LEFT_HAND_START: "left", tracking.RIGHT_HAND_START: "right"}
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")


# Coordinates are rounded to float32 on the way in, as the landmarker's are,
# and as web/tests/slots.test.ts does with Math.fround. The arithmetic after
# that is in doubles on both sides.
def hand_from_wrist(xy):
    hand = np.full((tracking.HAND_LANDMARKS, 3), np.nan, dtype=np.float32)
    hand[tracking.HAND_WRIST] = (xy[0], xy[1], 0.0)
    return hand


def pose_from_wrists(wrists):
    if wrists is None:
        return None
    pose = np.full((tracking.POSE_LANDMARKS, 3), np.nan, dtype=np.float32)
    pose[tracking.POSE_LEFT_WRIST] = (*wrists["left"], 0.0)
    pose[tracking.POSE_RIGHT_WRIST] = (*wrists["right"], 0.0)
    return pose


cases = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
check("fixture has cases", len(cases) > 0, f"{len(cases)} cases")

for case in cases:
    hands = [hand_from_wrist(xy) for xy in case["hands"]]
    slots = tracking.assign_slots(hands, pose_from_wrists(case["pose_wrists"]), case["handedness"])
    got = [SLOT_NAMES[s] for s in slots]
    check(case["name"], got == case["expected"], f"got {got}, expected {case['expected']}")

# The layout helper keeps the contract's pose subset in the contract's order.
hands = np.arange(2 * 42 * 3, dtype=np.float32).reshape(2, 42, 3)
pose = np.arange(2 * 33 * 3, dtype=np.float32).reshape(2, 33, 3) + 1000
layout = tracking.to_contract_layout(hands, pose)
check("contract layout is (T, 50, 3)", layout.shape == (2, 50, 3))
check("hands come first, unchanged", np.array_equal(layout[:, :42], hands))
check("pose subset in contract order",
      np.array_equal(layout[:, 42:], pose[:, [11, 12, 13, 14, 15, 16, 23, 24]]))

print()
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILURES:", FAIL)
    sys.exit(1)
