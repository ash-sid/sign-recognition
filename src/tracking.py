"""
src/tracking.py

Lays one frame of landmarker output out in the order the preprocessing
contract specifies. Port of the slot logic in web/src/tracking.ts; the two
must decide identically, because landmarks extracted here train the model
that the browser feeds.

Which hand goes in which slot is decided by proximity to the pose wrists,
not by the handedness the landmarker reports. The landmarker assumes a
mirrored image and neither the browser nor this module mirrors what it hands
over, so the reported label means the opposite hand. Handedness is used only
when there is no pose, and there it is inverted.

Distances are computed on Python floats, which are IEEE doubles, exactly as
the browser computes them on JavaScript numbers. Both start from the same
float32 landmark values, so the costs agree bit for bit and a tie resolves
the same way on both sides. Doing this arithmetic in float32 arrays would
not, and a near-tie would then land in different slots for no visible
reason.

Held to web/src/tracking.ts by a shared hand-written fixture,
web/tests/fixtures/slots.json, checked by tests/test_slots.py here and by
web/tests/slots.test.ts there.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

HAND_LANDMARKS = 21
POSE_LANDMARKS = 33
NUM_COORDS = 3

LEFT_HAND_START = 0
RIGHT_HAND_START = HAND_LANDMARKS
HANDS_END = 2 * HAND_LANDMARKS

# Pose landmarks the contract stores, in its order: left and right shoulder,
# elbow, wrist, then hip.
POSE_SOURCE_INDICES = (11, 12, 13, 14, 15, 16, 23, 24)
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16
HAND_WRIST = 0

CONTRACT_LANDMARKS = HANDS_END + len(POSE_SOURCE_INDICES)


def _squared_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return dx * dx + dy * dy


def assign_slots(
    hands: Sequence[np.ndarray],
    pose: np.ndarray | None,
    handedness: Sequence[str],
) -> list[int]:
    """Slot start index for each detected hand, in detection order.

    `hands` holds one (21, 3) array per detected hand, `pose` the (33, 3)
    pose landmarks or None, `handedness` the landmarker's label per hand.

    With two hands, both possible assignments are costed and the cheaper
    taken, rather than matching each hand to its own nearest wrist.
    Independent matching sends both hands to one slot whenever the arms
    cross. A tie goes to the straight assignment, and a single hand
    equidistant from both wrists goes left, as in the browser.
    """
    if len(hands) == 0:
        return []

    if pose is not None:
        left = pose[POSE_LEFT_WRIST]
        right = pose[POSE_RIGHT_WRIST]
        cost = [
            (
                _squared_distance(hand[HAND_WRIST], left),
                _squared_distance(hand[HAND_WRIST], right),
            )
            for hand in hands
        ]
        if len(hands) == 1:
            return [LEFT_HAND_START if cost[0][0] <= cost[0][1] else RIGHT_HAND_START]
        straight = cost[0][0] + cost[1][1]
        crossed = cost[0][1] + cost[1][0]
        if straight <= crossed:
            return [LEFT_HAND_START, RIGHT_HAND_START]
        return [RIGHT_HAND_START, LEFT_HAND_START]

    # No pose to match against: the reported handedness assumes a mirrored
    # image, and this one is not mirrored, so it names the opposite hand.
    slots = [RIGHT_HAND_START if label == "Left" else LEFT_HAND_START for label in handedness]
    if len(slots) == 2 and slots[0] == slots[1]:
        slots[1] = RIGHT_HAND_START if slots[0] == LEFT_HAND_START else LEFT_HAND_START
    return slots


def to_contract_layout(hands: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """(T, 42, 3) slotted hands and (T, 33, 3) pose to the contract's (T, 50, 3).

    The extractor keeps every pose landmark so that a later contract can
    choose a different subset without re-extracting. This is the subset the
    current contract uses, in its order.
    """
    if hands.shape[1:] != (HANDS_END, NUM_COORDS):
        raise ValueError(f"expected hands (T, {HANDS_END}, 3), got {hands.shape}")
    if pose.shape[1:] != (POSE_LANDMARKS, NUM_COORDS):
        raise ValueError(f"expected pose (T, {POSE_LANDMARKS}, 3), got {pose.shape}")
    return np.concatenate([hands, pose[:, list(POSE_SOURCE_INDICES)]], axis=1)
