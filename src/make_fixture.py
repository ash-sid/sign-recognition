"""
src/make_fixture.py

Builds the fixture that pins the TypeScript preprocessing port to this
implementation, and writes it to web/tests/fixtures/.

Two implementations of the same preprocessing will agree on the easy cases
and diverge on the ones nobody thought about: a hand that vanishes for three
frames, a landmark that is never tracked at all, shoulders that momentarily
coincide. A divergence there does not raise anything. It produces a slightly
wrong tensor, the classifier answers confidently, and the demo looks like a
weaker model rather than a broken pipeline. The only way to notice is to fix
the inputs and compare the outputs numerically.

Each case records the array the browser assembles from a landmarker, the
tensor this module produces from it, that tensor mirrored, and the logits the
exported graph returns for it. The port is checked against all four, so a
mismatch says which stage moved.

Two sources of cases:

- Synthetic ones are written by hand to hit each documented branch, including
  the ones real sequences rarely contain. They need no dataset, so the parity
  test runs from a clone.
- Dataset ones are drawn from the test split when data/raw is present. They
  cover what real tracking actually does, which is not always what a synthetic
  case predicts it does.

The array a browser builds and the array read out of a parquet file are the
same object, so the fixture stores that array rather than the file it came
from. Extraction from the dataset's long format is not part of what a
consumer implements; everything downstream of it is.

Coordinates are stored as raw little-endian float32 next to a JSON manifest.
Absent landmarks are NaN, which JSON cannot represent, and a text encoding of
a float is a second thing that has to round-trip exactly before the comparison
means anything.

Run: uv run python src\\make_fixture.py --run-name abl_hands_aug
     uv run python src\\make_fixture.py --run-name abl_hands_aug --dataset-cases 0
Requires models/<run-name>.onnx. Uses data/raw and data/splits.json when
present. Writes web/tests/fixtures/parity.json and parity.bin.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

import preprocessing as pp

MODELS = Path("models")
RAW = Path("data/raw")
SPLITS = Path("data/splits.json")
FIXTURE_DIR = Path("web/tests/fixtures")

INPUT_NAME = "landmarks"

# Fixed seed for the synthetic cases. They are a specification, not a sample:
# regenerating the fixture must not change what the port is being held to,
# or a failure becomes indistinguishable from a reroll.
SEED = 0

# Frame counts for the synthetic cases, chosen to sit either side of the
# resampler's equal-length shortcut so both the interpolating and the
# pass-through paths are covered.
SHORT_FRAMES = 23
EXACT_FRAMES = pp.TARGET_LEN
LONG_FRAMES = 142

MIRROR_CASE_NAMES = ("both_hands", "left_hand_unused", "dataset_00")


def _hands_end() -> int:
    return 2 * len(pp.HAND_INDICES)


def mirror_permutation(num_landmarks: int) -> list[int]:
    """Index permutation swapping left-side and right-side landmarks.

    Matches src/augment.py. Reflecting coordinates alone is not enough: after
    a flip what was the left hand is physically on the right, so the two hand
    blocks trade places along with each left/right pose pair."""
    n = len(pp.HAND_INDICES)
    order = list(range(num_landmarks))
    order[0:n] = list(range(n, 2 * n))
    order[n : 2 * n] = list(range(0, n))
    for i in range(_hands_end(), num_landmarks - 1, 2):
        order[i], order[i + 1] = order[i + 1], order[i]
    return order


def mirror(arr: np.ndarray) -> np.ndarray:
    """Reflect one (T, L, 3) sequence about the body midline."""
    perm = mirror_permutation(arr.shape[1])
    out = arr[:, perm, :].copy()
    out[..., 0] = -out[..., 0]
    return out


# --- synthetic cases ---------------------------------------------------------

def _skeleton(n_frames: int, rng: np.random.Generator) -> np.ndarray:
    """A plausible tracked signer: shoulders roughly level near the middle of
    the frame, arms hanging off them, hands orbiting in front of the chest.
    The numbers only have to be geometrically sane -- the point is to drive
    every branch, not to look like any particular sign."""
    arr = np.full((n_frames, pp.NUM_LANDMARKS, pp.NUM_COORDS), np.nan, dtype=np.float64)
    t = np.linspace(0.0, 1.0, n_frames)
    sway = 0.01 * np.sin(2 * np.pi * t)

    pose_start = 2 * len(pp.HAND_INDICES)
    shoulder_y = 0.34 + sway
    arr[:, pose_start + 0, :] = np.stack([0.38 + sway, shoulder_y, np.zeros(n_frames)], 1)
    arr[:, pose_start + 1, :] = np.stack([0.62 + sway, shoulder_y, np.zeros(n_frames)], 1)
    arr[:, pose_start + 2, :] = np.stack([0.34 + sway, shoulder_y + 0.14, -0.02 * np.ones(n_frames)], 1)
    arr[:, pose_start + 3, :] = np.stack([0.66 + sway, shoulder_y + 0.14, -0.02 * np.ones(n_frames)], 1)

    for side, sign in ((0, -1.0), (1, 1.0)):
        angle = 2 * np.pi * t + side * np.pi / 3
        wrist_x = 0.50 + sign * (0.08 + 0.03 * np.cos(angle))
        wrist_y = 0.52 + 0.05 * np.sin(angle)
        wrist_z = 0.02 * np.sin(angle)
        arr[:, pose_start + 4 + side, :] = np.stack([wrist_x, wrist_y, wrist_z], 1)
        arr[:, pose_start + 6 + side, :] = np.stack(
            [0.42 + sign * 0.08 + sway, 0.78 + np.zeros(n_frames), np.zeros(n_frames)], 1
        )

        block = side * len(pp.HAND_INDICES)
        for j in range(len(pp.HAND_INDICES)):
            spread = (j - 10) / 60.0
            arr[:, block + j, 0] = wrist_x + sign * spread + 0.004 * rng.standard_normal(n_frames)
            arr[:, block + j, 1] = wrist_y - abs(spread) + 0.004 * rng.standard_normal(n_frames)
            arr[:, block + j, 2] = wrist_z + spread / 4.0

    return arr


def _hand_slice(side: str) -> slice:
    n = len(pp.HAND_INDICES)
    return slice(0, n) if side == "left" else slice(n, 2 * n)


def synthetic_cases() -> list[tuple[str, str, np.ndarray]]:
    """(name, description, raw frames) for each hand-written case."""
    rng = np.random.default_rng(SEED)
    cases: list[tuple[str, str, np.ndarray]] = []

    both = _skeleton(SHORT_FRAMES, rng)
    cases.append(("both_hands", "both hands tracked throughout, fewer frames than the target", both))

    for side in ("left", "right"):
        arr = _skeleton(SHORT_FRAMES, rng)
        arr[:, _hand_slice(side), :] = np.nan
        cases.append(
            (
                f"{side}_hand_unused",
                f"{side} hand absent for the whole sequence, so it is zeroed rather than interpolated",
                arr,
            )
        )

    arr = _skeleton(EXACT_FRAMES, rng)
    arr[10:14, _hand_slice("right"), :] = np.nan
    cases.append(
        (
            "dropout_midway",
            "right hand lost for four frames in the middle, interpolated across the gap, "
            "and already at the target length so the resampler passes it through",
            arr,
        )
    )

    arr = _skeleton(SHORT_FRAMES, rng)
    arr[:3, _hand_slice("left"), :] = np.nan
    arr[-2:, _hand_slice("left"), :] = np.nan
    cases.append(
        (
            "dropout_at_edges",
            "left hand missing at the start and end, filled with the nearest tracked frame",
            arr,
        )
    )

    arr = _skeleton(LONG_FRAMES, rng)
    arr[:, 7, :] = np.nan
    cases.append(
        (
            "landmark_never_tracked",
            "one finger landmark never tracked while its hand is otherwise present, "
            "and more frames than the target",
            arr,
        )
    )

    arr = _skeleton(SHORT_FRAMES, rng)
    pose_start = 2 * len(pp.HAND_INDICES)
    arr[5:9, pose_start + 0, :2] = arr[5:9, pose_start + 1, :2]
    cases.append(
        (
            "shoulders_coincide",
            "shoulders at the same point for four frames, where the scale clip is the only "
            "thing between the sequence and a division by zero",
            arr,
        )
    )

    arr = _skeleton(SHORT_FRAMES, rng)
    arr[4:11, pose_start : pose_start + len(pp.POSE_INDICES), :] = np.nan
    cases.append(
        (
            "pose_dropout",
            "pose lost for seven frames, interpolated before it is used as the normalization "
            "reference",
            arr,
        )
    )

    arr = _skeleton(SHORT_FRAMES, rng)
    arr[:, : _hands_end(), :] = np.nan
    cases.append(
        (
            "both_hands_unused",
            "neither hand tracked, which the dataset treats as a failed recording and a live "
            "demo should decline to answer rather than classify",
            arr,
        )
    )

    return cases


# --- dataset cases -----------------------------------------------------------

def dataset_cases(limit: int) -> list[tuple[str, str, np.ndarray]]:
    if limit <= 0:
        return []
    train_csv = RAW / "train.csv"
    if not train_csv.exists() or not SPLITS.exists():
        print(f"  {RAW} or {SPLITS} not present; writing synthetic cases only.")
        return []

    with open(SPLITS, encoding="utf-8") as f:
        test_ids = json.load(f)["test"]
    catalogue = pd.read_csv(train_csv)
    subset = catalogue[catalogue["participant_id"].isin(test_ids)].sort_values("path")

    cases: list[tuple[str, str, np.ndarray]] = []
    # Spread the picks across the split rather than taking the first few, which
    # would all come from one signer and one stretch of the vocabulary.
    stride = max(1, len(subset) // (limit * 4))
    for _, row in subset.iloc[::stride].iterrows():
        if len(cases) >= limit:
            break
        df = pd.read_parquet(RAW / row["path"])
        if not pp.is_usable_sequence(df) or pp.is_both_hands_unused(df):
            continue  # dropped from every split, so not a model input
        frames, _ = pp._extract(df)
        cases.append(
            (
                f"dataset_{len(cases):02d}",
                f"recorded sequence, sign '{row['sign']}', {frames.shape[0]} frames",
                frames,
            )
        )
    if len(cases) < limit:
        print(f"  only {len(cases)} usable recorded sequences found, wanted {limit}.")
    return cases


# --- assembly ----------------------------------------------------------------

def expected_outputs(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run the reference implementation over a raw frame array.

    process_sequence takes the dataset's long format, so the array is put back
    into it. Round-tripping rather than calling the internals keeps the fixture
    pinned to the function everything else in the repository calls, instead of
    to a private path through it that could drift from the public one."""
    df = _to_long_format(frames)
    processed = pp.process_sequence(df)
    round_tripped, _ = pp._extract(df)
    both_nan = np.isnan(frames) & np.isnan(round_tripped)
    if not np.all(both_nan | (frames == round_tripped)):
        raise SystemExit(
            "Long-format round trip changed the frame array; the fixture would be "
            "recording something other than what it stores as the input."
        )
    return processed, mirror(processed).astype(np.float32)


def _to_long_format(frames: np.ndarray) -> pd.DataFrame:
    n_frames = frames.shape[0]
    types: list[str] = []
    indices: list[int] = []
    for name, group in pp.LANDMARK_GROUPS:
        types.extend([name] * len(group))
        indices.extend(group)
    return pd.DataFrame(
        {
            "frame": np.repeat(np.arange(n_frames), pp.NUM_LANDMARKS),
            "type": types * n_frames,
            "landmark_index": indices * n_frames,
            "x": frames[..., 0].reshape(-1),
            "y": frames[..., 1].reshape(-1),
            "z": frames[..., 2].reshape(-1),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--dataset-cases",
        type=int,
        default=4,
        help="how many recorded sequences to include, if the raw dataset is present",
    )
    args = parser.parse_args()

    graph_path = MODELS / f"{args.run_name}.onnx"
    if not graph_path.exists():
        raise SystemExit(f"Missing exported graph {graph_path}.")
    runtime = ort.InferenceSession(str(graph_path), providers=["CPUExecutionProvider"])

    cases = synthetic_cases() + dataset_cases(args.dataset_cases)
    print(f"Building {len(cases)} cases against {graph_path}")

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    blob = bytearray()
    manifest: list[dict[str, object]] = []

    def store(arr: np.ndarray) -> dict[str, object]:
        payload = np.ascontiguousarray(arr, dtype="<f4").tobytes()
        # Offsets count float32 elements, not bytes. A reader that gets the
        # unit wrong lands mid-value and fails loudly on the first case rather
        # than quietly comparing against a shifted window of the blob.
        entry = {"offset": len(blob) // 4, "shape": list(arr.shape)}
        blob.extend(payload)
        return entry

    for name, description, frames in cases:
        # The fixture stores coordinates as float32, so the expected output has
        # to be computed from the float32 values a reader will see rather than
        # from the doubles they were built as. Otherwise the two sides start
        # from inputs that differ in the last bits and the comparison can only
        # ever be approximate, which is the kind of parity test that passes
        # while something is wrong.
        frames = frames.astype(np.float32).astype(np.float64)
        processed, mirrored = expected_outputs(frames)
        logits = runtime.run(None, {INPUT_NAME: processed[None].astype(np.float32)})[0][0]
        entry: dict[str, object] = {
            "name": name,
            "description": description,
            "frames": int(frames.shape[0]),
            "input": store(frames.astype(np.float32)),
            "expected": store(processed),
            "logits": store(logits),
        }
        if name in MIRROR_CASE_NAMES:
            entry["mirrored"] = store(mirrored)
        manifest.append(entry)
        print(f"  {name:24s} {frames.shape[0]:4d} frames  {description}")

    bin_path = FIXTURE_DIR / "parity.bin"
    with open(bin_path, "wb") as f:
        f.write(blob)

    json_path = FIXTURE_DIR / "parity.json"
    document = {
        "note": (
            "Generated by src/make_fixture.py. Coordinates are little-endian float32 in "
            "parity.bin; offsets count float32 elements from the start of that file. "
            "Absent landmarks are NaN."
        ),
        "graph": graph_path.name,
        "target_len": pp.TARGET_LEN,
        "num_landmarks": pp.NUM_LANDMARKS,
        "num_coords": pp.NUM_COORDS,
        "unused_hand_nan_threshold": pp.UNUSED_HAND_NAN_THRESHOLD,
        "cases": manifest,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=1, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {json_path} and {bin_path} ({len(blob) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
