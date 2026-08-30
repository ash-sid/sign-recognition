"""
src/data_notes.py

Summarizes the raw dataset: signer count, sign count, sequence length
distribution, missing-frame rate, per-signer sign coverage, and per-hand
usage rate. Writes reports/data-notes.md.

Runs over the full dataset (all ~94k sequences), not a sample -- this is
the authoritative version of these stats, used to confirm the resample
length and missing-hand-handling decisions baked into src/preprocessing.py.
Slower than a sampled pass (reads every parquet file); expect several
minutes depending on disk speed.

Run once after downloading the dataset into data/raw/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import preprocessing as pp

RAW = Path("data/raw")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

train = pd.read_csv(RAW / "train.csv")
with open(RAW / "sign_to_prediction_index_map.json") as f:
    sign_map = json.load(f)

n_signers = train["participant_id"].nunique()
n_signs = train["sign"].nunique()
n_sequences = len(train)

lengths = []
missing_frac = []
# Per-hand: fraction of sequences where that hand is ~entirely NaN (i.e.
# "not used for this sign" per the Session 1 finding), vs used but with some
# tracking dropout. Same 0.95 threshold as src/preprocessing.py -- keep
# these in sync if either changes.
UNUSED_HAND_THRESHOLD = 0.95
hand_unused_count = {"left_hand": 0, "right_hand": 0}
hand_dropout_frac = {"left_hand": [], "right_hand": []}  # nan frac for hands that ARE used

for i, row in enumerate(train.itertuples(index=False), start=1):
    if i % 5000 == 0:
        print(f"  {i}/{n_sequences} sequences processed...")
    df = pd.read_parquet(RAW / row.path)
    n_frames = df["frame"].nunique()
    lengths.append(n_frames)
    missing_frac.append(df["x"].isna().mean())

    for hand in ("left_hand", "right_hand"):
        sub = df[df["type"] == hand]
        nan_frac = sub["x"].isna().mean() if len(sub) else 1.0
        if nan_frac >= UNUSED_HAND_THRESHOLD:
            hand_unused_count[hand] += 1
        else:
            hand_dropout_frac[hand].append(nan_frac)

lengths = np.array(lengths)
missing_frac = np.array(missing_frac)

# Per-signer sign coverage: does every signer cover most of the vocabulary?
# Relevant to make_split.py -- if coverage is patchy, holding out any small
# group of signers risks losing signs from train entirely.
per_signer_coverage = train.groupby("participant_id")["sign"].nunique()

def fmt_hand_stats(hand: str) -> str:
    unused = hand_unused_count[hand]
    unused_pct = 100 * unused / n_sequences
    dropouts = np.array(hand_dropout_frac[hand])
    if len(dropouts):
        dropout_line = f"mean {dropouts.mean():.3f}, max {dropouts.max():.3f}"
    else:
        dropout_line = "n/a"
    return (
        f"- {hand}: unused (>= {UNUSED_HAND_THRESHOLD:.0%} NaN) in "
        f"{unused} sequences ({unused_pct:.1f}%); among sequences where it "
        f"IS used, per-sequence NaN rate: {dropout_line}"
    )

report = f"""# Data notes

Generated from the full dataset ({n_sequences} sequences). Supersedes the
Session 1 sampled pass.

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
{fmt_hand_stats("left_hand")}
{fmt_hand_stats("right_hand")}

## Per-signer sign coverage (of {n_signs} total signs)
- min: {per_signer_coverage.min()}
- max: {per_signer_coverage.max()}
- mean: {per_signer_coverage.mean():.1f}
- signers below 90% coverage: {(per_signer_coverage < 0.9 * n_signs).sum()}

## Schema
- train.csv: path, participant_id, sequence_id, sign
- train_landmark_files/<participant_id>/<sequence_id>.parquet: frame, row_id, type, landmark_index, x, y, z
- type in {{face, pose, left_hand, right_hand}}; 543 landmarks per frame total

## Follow-ups for Session 2 decisions
- Confirm TARGET_LEN in src/preprocessing.py (currently {pp.TARGET_LEN}) against
  the p95/p99 above -- longer covers more of the tail but wastes compute on
  the many short sequences.
- If any signer is well below full coverage, make_split.py's search should
  still find a workable partition (it optimizes for this), but check its
  printed coverage number before trusting the split.
"""

(REPORTS / "data-notes.md").write_text(report)
print(report)
print(f"\nWritten to {REPORTS / 'data-notes.md'}")
