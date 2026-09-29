"""
src/animate_sign.py

Renders recorded landmark sequences for one sign as an animated GIF, so a
sign can be watched as the dataset's signers performed it. The dataset ships
landmarks rather than video, so this is the only view of a performance there
is.

Two views:
  default        the frames as recorded, at their real length, with gaps
                 where tracking dropped out. This is what the signer did.
  --model-input  the (TARGET_LEN, 50, 3) tensor process_sequence produces:
                 shoulder-normalized, gaps filled, resampled. This is what
                 the model is given.

The two differ in ways that matter when comparing a live attempt against the
data: the default view shows tempo and tracking quality, the model view shows
neither, because resampling removes the first and filling hides the second.

Run:
  uv run python src\\animate_sign.py --sign finish
  uv run python src\\animate_sign.py --sign finish --model-input
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import PillowWriter

import preprocessing

RAW = Path("data/raw")
SPLITS = Path("data/splits.json")
DEFAULT_OUT_DIR = Path("reports")

# MediaPipe's 21-point hand topology, as bone segments between landmark
# indices. Drawing these rather than a point cloud is what makes a handshape
# readable at a glance.
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

# Segments between the eight retained pose points, in the order
# preprocessing.POSE_INDICES selects them: 0/1 shoulders, 2/3 elbows,
# 4/5 wrists, 6/7 hips.
POSE_BONES = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5), (6, 7), (0, 6), (1, 7)]

HAND_BLOCKS = {"left_hand": (0, 21), "right_hand": (21, 42)}
POSE_BLOCK = (42, 50)


def load_index() -> pd.DataFrame:
    """The dataset's sequence index: one row per recorded sequence."""
    return pd.read_csv(RAW / "train.csv")


def signers_for(split: str) -> list[int] | None:
    """Participant ids belonging to a split, or None for every signer."""
    if split == "all":
        return None
    with open(SPLITS, encoding="utf-8") as handle:
        return json.load(handle)[split]


def recorded_frames(df: pd.DataFrame) -> np.ndarray:
    """Long-format landmark rows -> (frames, 50, 3) in recorded order, with
    NaN left in place. No filling, no resampling, no normalization: this is
    the sequence as it sits in the file."""
    frames = np.sort(df["frame"].unique())
    out = np.full((len(frames), preprocessing.NUM_LANDMARKS, preprocessing.NUM_COORDS), np.nan)
    position = {frame: i for i, frame in enumerate(frames)}
    start = 0
    for group, indices in preprocessing.LANDMARK_GROUPS:
        wanted = {index: start + offset for offset, index in enumerate(indices)}
        rows = df[df["type"] == group]
        for frame, index, x, y, z in zip(
            rows["frame"], rows["landmark_index"], rows["x"], rows["y"], rows["z"]
        ):
            column = wanted.get(index)
            if column is not None:
                out[position[frame], column] = (x, y, z)
        start += len(indices)
    return out


def tracked_fraction(arr: np.ndarray, block: tuple[int, int]) -> float:
    """Fraction of frames in which any landmark of a block was tracked."""
    start, end = block
    present = ~np.isnan(arr[:, start:end, 0])
    return float(present.any(axis=1).mean())


def draw_frame(ax, frame: np.ndarray, invert_y: bool) -> None:
    """Draw one frame's hands and arms into a cleared axis."""
    ax.clear()
    ax.set_aspect("equal")
    ax.axis("off")
    sign = -1.0 if invert_y else 1.0

    for block, colour in ((HAND_BLOCKS["left_hand"], "tab:blue"),
                          (HAND_BLOCKS["right_hand"], "tab:orange")):
        start, end = block
        # A hand not used for this sign is set to exactly zero by
        # process_sequence. Zero is a real coordinate once normalized -- the
        # shoulder midpoint -- so drawing it would put a collapsed hand in the
        # middle of the chest rather than showing its absence.
        if np.all(frame[start:end] == 0.0):
            continue
        for a, b in HAND_BONES:
            pa, pb = frame[start + a], frame[start + b]
            if not (np.isnan(pa[0]) or np.isnan(pb[0])):
                ax.plot([pa[0], pb[0]], [sign * pa[1], sign * pb[1]], color=colour, linewidth=1.4)

    start, _ = POSE_BLOCK
    for a, b in POSE_BONES:
        pa, pb = frame[start + a], frame[start + b]
        if not (np.isnan(pa[0]) or np.isnan(pb[0])):
            ax.plot([pa[0], pb[0]], [sign * pa[1], sign * pb[1]], color="tab:gray", linewidth=1.2)


def axis_limits(sequences: list[np.ndarray], invert_y: bool) -> tuple[tuple[float, float], tuple[float, float]]:
    """One set of limits shared by every panel, so panels stay comparable and
    nothing appears to move because its axis rescaled underneath it."""
    points = np.concatenate([seq.reshape(-1, preprocessing.NUM_COORDS) for seq in sequences])
    xs, ys = points[:, 0], points[:, 1]
    if invert_y:
        ys = -ys
    pad = 0.05
    x_span = np.nanmax(xs) - np.nanmin(xs)
    y_span = np.nanmax(ys) - np.nanmin(ys)
    return (
        (float(np.nanmin(xs) - pad * x_span), float(np.nanmax(xs) + pad * x_span)),
        (float(np.nanmin(ys) - pad * y_span), float(np.nanmax(ys) + pad * y_span)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sign", required=True, help="sign label to render")
    parser.add_argument("--count", type=int, default=4, help="sequences to render side by side")
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test", "all"],
        help="which signers to draw from; train is what shaped the model",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for choosing which sequences")
    parser.add_argument("--fps", type=int, default=12, help="playback rate of the output")
    parser.add_argument(
        "--model-input",
        action="store_true",
        help="render the processed tensor the model consumes instead of the recorded frames",
    )
    parser.add_argument("--out", type=Path, default=None, help="output GIF path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = load_index()

    signers = signers_for(args.split)
    rows = index[index["sign"] == args.sign]
    if signers is not None:
        rows = rows[rows["participant_id"].isin(signers)]
    if rows.empty:
        raise SystemExit(f"No sequences for sign '{args.sign}' among {args.split} signers.")

    chosen = rows.sample(n=min(args.count, len(rows)), random_state=args.seed)

    sequences, captions = [], []
    for _, row in chosen.iterrows():
        df = pd.read_parquet(RAW / row["path"])
        recorded = recorded_frames(df)
        left = tracked_fraction(recorded, HAND_BLOCKS["left_hand"])
        right = tracked_fraction(recorded, HAND_BLOCKS["right_hand"])
        sequences.append(preprocessing.process_sequence(df) if args.model_input else recorded)
        captions.append(
            f"signer {row['participant_id']}\n"
            f"{len(recorded)} frames | L {left:.0%} R {right:.0%} tracked"
        )
        print(
            f"signer {row['participant_id']:>6}  {len(recorded):>4} frames  "
            f"left hand tracked {left:6.1%}  right hand tracked {right:6.1%}"
        )

    # Recorded coordinates put y downward in image space; the normalized ones
    # are already shoulder-relative and keep their own sign.
    invert_y = not args.model_input
    xlim, ylim = axis_limits(sequences, invert_y)
    longest = max(len(seq) for seq in sequences)

    fig, axes = plt.subplots(1, len(sequences), figsize=(3.2 * len(sequences), 3.8))
    axes = np.atleast_1d(axes)
    view = "model input" if args.model_input else "as recorded"
    fig.suptitle(f"'{args.sign}' - {args.split} signers - {view}")

    out = args.out or DEFAULT_OUT_DIR / f"sign_{args.sign}_{args.split}{'_model' if args.model_input else ''}.gif"
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = PillowWriter(fps=args.fps)
    with writer.saving(fig, str(out), dpi=110):
        for step in range(longest):
            for ax, seq, caption in zip(axes, sequences, captions):
                # Sequences differ in length, so shorter ones hold their last
                # frame rather than disappearing part way through.
                draw_frame(ax, seq[min(step, len(seq) - 1)], invert_y)
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
                ax.set_title(caption, fontsize=8)
            writer.grab_frame()

    plt.close(fig)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
