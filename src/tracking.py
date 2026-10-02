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

assign_sequence_slots() decides slots for a whole sequence at once instead of
frame by frame; see its docstring for why. It is held to its TypeScript port
by web/tests/fixtures/sequence_slots.json, written by hand, and by
web/tests/fixtures/sequence_slots_generated.json, written by
src/make_sequence_fixture.py from this implementation.
"""
from __future__ import annotations

import math
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


# --- sequence-level slot assignment --------------------------------------------

# Largest mean landmark distance, in image widths, at which a hand in one frame
# is taken to be the same hand as one in the frame before. 60 pixels at the
# 640-pixel width the live page captures.
CONTINUITY = 60 / 640


def _mean_distance(a: list, b: list, aspect: float) -> float:
    """Mean distance over corresponding landmarks of two hands, in image
    widths. `aspect` is height over width, so that a vertical step counts the
    same as a horizontal one of equal length on screen. Summed in landmark
    order with sqrt rather than hypot, so the browser port can reproduce it
    exactly."""
    total = 0.0
    for p, q in zip(a, b):
        dx = p[0] - q[0]
        dy = (p[1] - q[1]) * aspect
        total += math.sqrt(dx * dx + dy * dy)
    return total / len(a)


def _pair(current: list, previous: list, aspect: float) -> list[tuple[int, int, float]]:
    """Pair hands in this frame with hands in the previous one by the
    assignment with least total distance. At most two hands on either side."""
    n, m = len(current), len(previous)
    if n == 0 or m == 0:
        return []
    if n > 2 or m > 2:
        raise ValueError("at most two hands per frame")
    d = [[_mean_distance(c, p, aspect) for p in previous] for c in current]
    if n == 1 and m == 1:
        return [(0, 0, d[0][0])]
    if n == 1:
        j = 0 if d[0][0] <= d[0][1] else 1
        return [(0, j, d[0][j])]
    if m == 1:
        i = 0 if d[0][0] <= d[1][0] else 1
        return [(i, 0, d[i][0])]
    if d[0][0] + d[1][1] <= d[0][1] + d[1][0]:
        return [(0, 0, d[0][0]), (1, 1, d[1][1])]
    return [(0, 1, d[0][1]), (1, 0, d[1][0])]


def _wrist_margin(hand: list, pose: list | None, aspect: float) -> float:
    """How much nearer this hand's wrist is to the left pose wrist than to the
    right, from +1 (on the left wrist) to -1 (on the right), 0 without a pose."""
    if pose is None:
        return 0.0
    lx = hand[HAND_WRIST][0] - pose[POSE_LEFT_WRIST][0]
    ly = (hand[HAND_WRIST][1] - pose[POSE_LEFT_WRIST][1]) * aspect
    rx = hand[HAND_WRIST][0] - pose[POSE_RIGHT_WRIST][0]
    ry = (hand[HAND_WRIST][1] - pose[POSE_RIGHT_WRIST][1]) * aspect
    left = lx * lx + ly * ly
    right = rx * rx + ry * ry
    if left + right == 0:
        return 0.0
    return (right - left) / (right + left)


def assign_sequence_slots(
    hands: Sequence[Sequence[np.ndarray]],
    poses: Sequence[np.ndarray | None],
    aspect: float,
) -> list[list[int]]:
    """Slot start index for every detected hand in every frame of a sequence.

    `hands[t]` holds frame t's detected hands as (21, 3) arrays in detection
    order, `poses[t]` its (33, 3) pose or None, and `aspect` is image height
    over width.

    Frame-by-frame matching to the nearer pose wrist moves a stationary hand
    between slots whenever the pose wrists are closer together than their own
    error, which is routine when the arms cross: the pose model's wrists
    disagree by tens of pixels between runtimes. A hand that alternates slots
    leaves both slots mostly present with gaps, and preprocessing then
    interpolates across the gaps, inventing motion in both.

    So hands are first linked from frame to frame into tracks -- a hand within
    CONTINUITY of a hand in the previous frame continues its track, and a frame
    with no hands ends every track -- and each track then takes one slot for
    its whole length, by the sign of its summed wrist margin. Many noisy
    comparisons average out where one does not. Where two tracks that share a
    frame voted for the same slot, the one with the smaller summed margin takes
    the other slot, the later-starting one on a tie; if a chain of such
    corrections still leaves two hands in one slot in some frame, the same
    comparison separates them in that frame alone.

    Called on a whole video for training data, and on the frames of one window
    in the browser, so each tensor the model sees has one decision per hand.
    Arithmetic is on Python floats in a fixed order, as in the browser port.
    """
    frames = [[np.asarray(h).tolist() for h in frame] for frame in hands]
    pose_lists = [None if p is None else np.asarray(p).tolist() for p in poses]

    ids: list[list[int]] = []
    total: list[float] = []
    previous: list[tuple[list, int]] = []
    for frame, pose in zip(frames, pose_lists):
        current: list[int | None] = [None] * len(frame)
        for i, j, d in _pair(frame, [h for h, _ in previous], aspect):
            if d < CONTINUITY:
                current[i] = previous[j][1]
        for i in range(len(frame)):
            if current[i] is None:
                current[i] = len(total)
                total.append(0.0)
        for hand, track in zip(frame, current):
            total[track] += _wrist_margin(hand, pose, aspect)
        ids.append(current)
        previous = list(zip(frame, current))

    def weaker(a: int, b: int) -> int:
        if abs(total[b]) < abs(total[a]) or (abs(total[b]) == abs(total[a]) and b > a):
            return b
        return a

    def other(slot: int) -> int:
        return RIGHT_HAND_START if slot == LEFT_HAND_START else LEFT_HAND_START

    slot = [LEFT_HAND_START if t >= 0 else RIGHT_HAND_START for t in total]
    for tracks in ids:
        if len(tracks) == 2 and slot[tracks[0]] == slot[tracks[1]]:
            moved = weaker(tracks[0], tracks[1])
            slot[moved] = other(slot[moved])

    out = []
    for tracks in ids:
        slots = [slot[t] for t in tracks]
        if len(slots) == 2 and slots[0] == slots[1]:
            i = 0 if weaker(tracks[0], tracks[1]) == tracks[0] else 1
            slots[i] = other(slots[i])
        out.append(slots)
    return out
