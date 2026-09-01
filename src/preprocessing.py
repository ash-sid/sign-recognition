"""
src/preprocessing.py

Turns a raw landmark parquet sequence (long format: one row per landmark per
frame) into a fixed-shape, normalized numpy array ready for model input.

Selects a fixed subset of landmarks (both hands + an upper-body pose subset),
distinguishes "hand not used for this sign" from "tracking lost mid-sequence"
when filling missing values, normalizes for signer position/scale, and
resamples every sequence to a fixed number of frames.

The landmark selection, output ordering, and normalization formula are the
interface contract between this module and: (a) the model, (b) the
TypeScript port used in the browser demo. See reports/contract.md. Do not
change any of it here without updating that document and the TS port.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- Landmark selection -----------------------------------------------------
# MediaPipe pose landmark indices (0-32 within the "pose" type group):
# left shoulder, right shoulder, left elbow, right elbow, left wrist,
# right wrist, left hip, right hip. Enough for arm position/reach without
# pulling in legs or face. Shoulders (positions 0, 1 below) double as the
# normalization reference -- see _normalize.
POSE_INDICES = [11, 12, 13, 14, 15, 16, 23, 24]

# Both hands, full 21-point MediaPipe hand topology.
HAND_INDICES = list(range(21))

# Fixed output ordering: left_hand (21) -> right_hand (21) -> pose (8) = 50.
# Face/lips landmarks are deliberately excluded: recognition here is meant
# to rely on hand shape/motion and arm position, not facial grammar or other
# non-manual markers (see README non-goals). Keeping face out of the
# baseline avoids quietly relying on a signal the project isn't scoped to use.
LANDMARK_GROUPS: list[tuple[str, list[int]]] = [
    ("left_hand", HAND_INDICES),
    ("right_hand", HAND_INDICES),
    ("pose", POSE_INDICES),
]
NUM_LANDMARKS = sum(len(idxs) for _, idxs in LANDMARK_GROUPS)  # 50
NUM_COORDS = 3  # x, y, z

# Fixed sequence length after resampling, chosen from the observed length
# distribution (see reports/data-notes.md: median ~22-23 frames, p95 ~125-135,
# long tail beyond that). Update here and in reports/contract.md together if
# it changes.
TARGET_LEN = 70

# A hand with at least this fraction of NaN frames is treated as "not used
# for this sign" (zero-filled) rather than "tracking lost partway through"
# (interpolated). Missing-hand data in this dataset is systematic for
# one-handed signs, not just tracking noise -- see reports/data-notes.md
# for per-hand usage rates.
UNUSED_HAND_NAN_THRESHOLD = 0.95

# Sequences with fewer valid frames than this are considered too degenerate
# to use (near-total tracking failure, not a genuinely short sign).
MIN_USABLE_FRAMES = 4


def is_usable_sequence(df: pd.DataFrame) -> bool:
    """Cheap pre-filter: does this sequence have enough real frames to bother
    processing? Call before process_sequence when scanning many sequences."""
    return df["frame"].nunique() >= MIN_USABLE_FRAMES


def is_both_hands_unused(df: pd.DataFrame) -> bool:
    """Cheap pre-filter: are both hands >= UNUSED_HAND_NAN_THRESHOLD NaN for
    this sequence? Every sign in this dataset uses at least one hand, so a
    sequence tripping this is very likely a tracking failure (bad framing,
    occlusion) rather than a genuine no-manual-signal case -- see
    reports/data-notes.md for the full-dataset rate. Call before
    process_sequence when scanning many sequences; safe to use as a training
    filter independently of is_usable_sequence."""
    for hand in ("left_hand", "right_hand"):
        sub = df[df["type"] == hand]
        nan_frac = sub["x"].isna().mean() if len(sub) else 1.0
        if nan_frac < UNUSED_HAND_NAN_THRESHOLD:
            return False
    return True


def process_sequence(
    df: pd.DataFrame, target_len: int = TARGET_LEN, normalize: bool = True
) -> np.ndarray:
    """Raw long-format landmark dataframe for one sequence -> fixed-shape
    array of shape (target_len, NUM_LANDMARKS, NUM_COORDS), normalized and
    with missing values filled. Landmark order matches LANDMARK_GROUPS.

    normalize=False skips the shoulder-relative normalization step and
    leaves coordinates in the raw MediaPipe frame. This exists only to
    measure how much normalization contributes to accuracy; it is not part
    of the interface contract in reports/contract.md, and nothing outside
    that measurement (the exported model, the browser port) should use it.
    Note that raw x/y live in roughly [0, 1] rather than being centered on
    the signer, so 0.0 is a plausible real coordinate rather than an
    unambiguous "hand not used" marker."""
    arr, group_ranges = _extract(df)
    arr, unused_hands = _fill_missing(arr, group_ranges)
    if normalize:
        arr = _normalize(arr, group_ranges)
    # Re-zero unused hands *after* normalization: normalization subtracts a
    # shared shoulder-center from every landmark, which would otherwise shift
    # the "hand not used" placeholder away from zero and turn "absent" into
    # a fake coordinate.
    for hand in unused_hands:
        start, end = group_ranges[hand]
        arr[:, start:end, :] = 0.0
    arr = _resample(arr, target_len)
    return arr.astype(np.float32)


def flatten(arr: np.ndarray) -> np.ndarray:
    """(T, NUM_LANDMARKS, NUM_COORDS) -> (T, NUM_LANDMARKS * NUM_COORDS)."""
    return arr.reshape(arr.shape[0], -1)


def mean_pool(arr: np.ndarray) -> np.ndarray:
    """(T, NUM_LANDMARKS, NUM_COORDS) -> (NUM_LANDMARKS * NUM_COORDS,).
    Collapses the time axis by averaging -- useful for simple non-temporal
    baselines (e.g. logistic regression), not the sequence models."""
    return flatten(arr).mean(axis=0)


# --- internals ---------------------------------------------------------------

def _extract(df: pd.DataFrame) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    frames = np.sort(df["frame"].unique())
    frame_pos = {f: i for i, f in enumerate(frames)}
    n_frames = len(frames)

    out = np.full((n_frames, NUM_LANDMARKS, NUM_COORDS), np.nan, dtype=np.float64)
    group_ranges: dict[str, tuple[int, int]] = {}
    col = 0
    for name, indices in LANDMARK_GROUPS:
        sub = df[(df["type"] == name) & (df["landmark_index"].isin(indices))]
        if len(sub):
            idx_pos = {idx: p for p, idx in enumerate(indices)}
            f_pos = sub["frame"].map(frame_pos).to_numpy()
            l_pos = sub["landmark_index"].map(idx_pos).to_numpy() + col
            out[f_pos, l_pos, 0] = sub["x"].to_numpy()
            out[f_pos, l_pos, 1] = sub["y"].to_numpy()
            out[f_pos, l_pos, 2] = sub["z"].to_numpy()
        group_ranges[name] = (col, col + len(indices))
        col += len(indices)
    return out, group_ranges


def _interpolate_time(block: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs along the time axis, per landmark/coord.
    Leading/trailing NaNs get the nearest valid value (np.interp's default
    extrapolation behavior)."""
    T = block.shape[0]
    out = block.copy()
    t = np.arange(T)
    for j in range(block.shape[1]):
        for c in range(block.shape[2]):
            col = block[:, j, c]
            valid = ~np.isnan(col)
            if not valid.any():
                out[:, j, c] = 0.0
            elif not valid.all():
                out[:, j, c] = np.interp(t, t[valid], col[valid])
    return out


def _fill_missing(
    arr: np.ndarray, group_ranges: dict[str, tuple[int, int]]
) -> tuple[np.ndarray, list[str]]:
    """Returns the filled array plus the list of hands ('left_hand',
    'right_hand') judged unused for this sign. Unused hands are left as a
    placeholder here (exact value doesn't matter -- normalization treats
    each landmark independently) and must be re-zeroed by the caller after
    normalization; see process_sequence."""
    arr = arr.copy()
    unused_hands: list[str] = []
    for hand in ("left_hand", "right_hand"):
        start, end = group_ranges[hand]
        block = arr[:, start:end, :]
        nan_frac = np.isnan(block[..., 0]).mean() if block.size else 1.0
        if nan_frac >= UNUSED_HAND_NAN_THRESHOLD:
            arr[:, start:end, :] = 0.0  # placeholder; re-zeroed post-normalize
            unused_hands.append(hand)
        else:
            arr[:, start:end, :] = _interpolate_time(block)  # tracking dropout
    pose_start, pose_end = group_ranges["pose"]
    arr[:, pose_start:pose_end, :] = _interpolate_time(arr[:, pose_start:pose_end, :])
    return arr, unused_hands


def _normalize(arr: np.ndarray, group_ranges: dict[str, tuple[int, int]]) -> np.ndarray:
    """Per-frame translation + scale normalization using the shoulders as
    reference: center on the shoulder midpoint, scale by shoulder width.
    Makes the representation invariant to the signer's distance from and
    position relative to the camera."""
    pose_start, _ = group_ranges["pose"]
    l_shoulder = arr[:, pose_start + 0, :2]  # POSE_INDICES[0] = 11 = L shoulder
    r_shoulder = arr[:, pose_start + 1, :2]  # POSE_INDICES[1] = 12 = R shoulder
    center = (l_shoulder + r_shoulder) / 2.0
    scale = np.linalg.norm(l_shoulder - r_shoulder, axis=1, keepdims=True)
    scale = np.clip(scale, 1e-3, None)  # guard degenerate/occluded frames

    out = arr.copy()
    out[..., 0] = (arr[..., 0] - center[:, 0:1]) / scale
    out[..., 1] = (arr[..., 1] - center[:, 1:2]) / scale
    out[..., 2] = arr[..., 2] / scale  # same scale factor, no z translation
    return out


def _resample(arr: np.ndarray, target_len: int) -> np.ndarray:
    T = arr.shape[0]
    if T == target_len:
        return arr
    src_t = np.linspace(0.0, 1.0, T)
    dst_t = np.linspace(0.0, 1.0, target_len)
    out = np.empty((target_len, arr.shape[1], arr.shape[2]), dtype=arr.dtype)
    for j in range(arr.shape[1]):
        for c in range(arr.shape[2]):
            out[:, j, c] = np.interp(dst_t, src_t, arr[:, j, c])
    return out
