"""
src/data_notes.py

Summarizes the raw dataset: signer count, sign count, sequence length
distribution, missing-frame rate, per-signer sign coverage, and per-hand
usage rate. Writes reports/data-notes.md.

Runs over the full dataset (all ~94k sequences), not a sample -- this is
the authoritative version of these stats, used to confirm the resample
length and missing-hand-handling decisions baked into src/preprocessing.py.

Per-file reads are parallelized across processes (see main()): reading
~94k small parquet files one at a time is bottlenecked by per-file
open/parse overhead rather than raw disk throughput, so a process pool
gets much better use of both a multi-core CPU and an NVMe SSD's ability to
service many reads concurrently. Must run under `if __name__ == "__main__"`
for this to work correctly, since Windows (unlike Linux) starts worker
processes by re-importing this module rather than forking.

Run once after downloading the dataset into data/raw/.
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import preprocessing as pp

RAW = Path("data/raw")
REPORTS = Path("reports")

# Same threshold as src/preprocessing.py -- keep these in sync if either changes.
UNUSED_HAND_THRESHOLD = 0.95

# Leave one logical core free for the OS/other work; every other core gets
# a worker. Override by setting this directly if you want to tune it.
MAX_WORKERS = max(1, (os.cpu_count() or 4) - 1)
CHUNKSIZE = 64  # sequences handed to each worker per round-trip; reduces IPC overhead


def _scan_one(path: str) -> dict:
    """Worker: read one parquet file, return its length/missing-frame/
    per-hand stats. Runs in a separate process -- must not depend on any
    module-level mutable state, only the constants above."""
    df = pd.read_parquet(path)
    n_frames = df["frame"].nunique()
    missing = df["x"].isna().mean()

    hand_nan = {}
    for hand in ("left_hand", "right_hand"):
        sub = df[df["type"] == hand]
        hand_nan[hand] = sub["x"].isna().mean() if len(sub) else 1.0

    return {
        "n_frames": n_frames,
        "missing_frac": missing,
        "left_nan": hand_nan["left_hand"],
        "right_nan": hand_nan["right_hand"],
    }


def fmt_hand_stats(hand_label: str, unused_count: int, n_sequences: int, dropouts: np.ndarray) -> str:
    unused_pct = 100 * unused_count / n_sequences
    dropout_line = f"mean {dropouts.mean():.3f}, max {dropouts.max():.3f}" if len(dropouts) else "n/a"
    return (
        f"- {hand_label}: unused (>= {UNUSED_HAND_THRESHOLD:.0%} NaN) in "
        f"{unused_count} sequences ({unused_pct:.1f}%); among sequences where it "
        f"IS used, per-sequence NaN rate: {dropout_line}"
    )


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    train = pd.read_csv(RAW / "train.csv")
    with open(RAW / "sign_to_prediction_index_map.json") as f:
        sign_map = json.load(f)

    n_signers = train["participant_id"].nunique()
    n_signs = train["sign"].nunique()
    n_sequences = len(train)

    paths = [str(RAW / p) for p in train["path"]]

    print(f"Scanning {n_sequences} sequences with {MAX_WORKERS} worker processes...")
    lengths = []
    missing_frac = []
    hand_unused_count = {"left_hand": 0, "right_hand": 0}
    hand_dropout_frac = {"left_hand": [], "right_hand": []}
    both_hands_unused_count = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i, result in enumerate(executor.map(_scan_one, paths, chunksize=CHUNKSIZE), start=1):
            if i % 5000 == 0:
                print(f"  {i}/{n_sequences} sequences processed...")
            lengths.append(result["n_frames"])
            missing_frac.append(result["missing_frac"])

            unused_this_seq = 0
            for hand, nan_key in (("left_hand", "left_nan"), ("right_hand", "right_nan")):
                nan_frac = result[nan_key]
                if nan_frac >= UNUSED_HAND_THRESHOLD:
                    hand_unused_count[hand] += 1
                    unused_this_seq += 1
                else:
                    hand_dropout_frac[hand].append(nan_frac)
            if unused_this_seq == 2:
                both_hands_unused_count += 1

    lengths = np.array(lengths)
    missing_frac = np.array(missing_frac)
    both_hands_unused_pct = 100 * both_hands_unused_count / n_sequences

    # Per-signer sign coverage: does every signer cover most of the vocabulary?
    # Relevant to make_split.py -- if coverage is patchy, holding out any small
    # group of signers risks losing signs from train entirely.
    per_signer_coverage = train.groupby("participant_id")["sign"].nunique()

    left_line = fmt_hand_stats("left_hand", hand_unused_count["left_hand"], n_sequences, np.array(hand_dropout_frac["left_hand"]))
    right_line = fmt_hand_stats("right_hand", hand_unused_count["right_hand"], n_sequences, np.array(hand_dropout_frac["right_hand"]))

    report = f"""# Data notes

Generated from the full dataset ({n_sequences} sequences).

## Counts
- Signers (participant_id): {n_signers}
- Distinct signs: {n_signs} (sign_to_prediction_index_map.json has {len(sign_map)} entries)
- Total sequences: {n_sequences}

## Sequence length (frames), full dataset
- min: {lengths.min()}
- max: {lengths.max()}
- mean: {lengths.mean():.1f}
- median: {np.median(lengths):.1f}
- p95: {np.percentile(lengths, 95):.1f}
- p99: {np.percentile(lengths, 99):.1f}

## Missing-frame rate (fraction of NaN x-coords per sequence), full dataset
- mean: {missing_frac.mean():.3f}
- max: {missing_frac.max():.3f}

## Per-hand usage
{left_line}
{right_line}
- both hands unused (>= {UNUSED_HAND_THRESHOLD:.0%} NaN on both): {both_hands_unused_count} sequences ({both_hands_unused_pct:.1f}%). Every sign uses at least one hand, so these are very likely tracking failures (bad framing, occlusion) rather than genuine no-manual-signal cases -- consider dropping them rather than training on an all-zero-hands input.

## Per-signer sign coverage (of {n_signs} total signs)
- min: {per_signer_coverage.min()}
- max: {per_signer_coverage.max()}
- mean: {per_signer_coverage.mean():.1f}
- signers below 90% coverage: {(per_signer_coverage < 0.9 * n_signs).sum()}

## Schema
- train.csv: path, participant_id, sequence_id, sign
- train_landmark_files/<participant_id>/<sequence_id>.parquet: frame, row_id, type, landmark_index, x, y, z
- type in {{face, pose, left_hand, right_hand}}; 543 landmarks per frame total

## Follow-ups
- Confirm TARGET_LEN in src/preprocessing.py (currently {pp.TARGET_LEN}) against
  the p95/p99 above -- longer covers more of the tail but wastes compute on
  the many short sequences.
- If any signer is well below full coverage, make_split.py's search should
  still find a workable partition (it optimizes for this), but check its
  printed coverage number before trusting the split.
- Review the both-hands-unused count above; if it's non-trivial, add a filter
  step before training rather than feeding those sequences in as-is.
"""

    (REPORTS / "data-notes.md").write_text(report)
    print(report)
    print(f"\nWritten to {REPORTS / 'data-notes.md'}")


if __name__ == "__main__":
    main()
