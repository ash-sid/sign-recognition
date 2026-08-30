"""
src/data_notes.py

Summarizes the raw dataset: signer count, sign count, sequence length
distribution, and missing-frame rate. Writes reports/data-notes.md.

Run once after downloading the dataset into data/raw/.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("data/raw")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

train = pd.read_csv(RAW / "train.csv")
with open(RAW / "sign_to_prediction_index_map.json") as f:
    sign_map = json.load(f)

n_signers = train["participant_id"].nunique()
n_signs = train["sign"].nunique()
n_sequences = len(train)

# Reading all ~94k parquet files is slow, so this samples a subset for the
# length/missing-frame stats. Re-run over the full dataset before finalizing
# the preprocessing pipeline (sequence length cutoff, imputation strategy).
sample = train.sample(n=min(300, len(train)), random_state=0)

lengths = []
missing_frac = []
for _, row in sample.iterrows():
    df = pd.read_parquet(RAW / row["path"])
    n_frames = df["frame"].nunique()
    lengths.append(n_frames)
    # fraction of (frame, landmark) rows that are NaN in x
    missing_frac.append(df["x"].isna().mean())

lengths = np.array(lengths)
missing_frac = np.array(missing_frac)

report = f"""# Data notes

Generated from a sample of {len(sample)} sequences. Re-run over the full
dataset before finalizing preprocessing decisions (sequence length cutoff,
imputation strategy).

## Counts
- Signers (participant_id): {n_signers}
- Distinct signs: {n_signs} (sign_to_prediction_index_map.json has {len(sign_map)} entries)
- Total sequences: {n_sequences}

## Sequence length (frames), sample of {len(sample)}
- min: {lengths.min()}
- max: {lengths.max()}
- mean: {lengths.mean():.1f}
- median: {np.median(lengths):.1f}
- p95: {np.percentile(lengths, 95):.1f}

## Missing-frame rate (fraction of NaN x-coords per sequence), sample of {len(sample)}
- mean: {missing_frac.mean():.3f}
- max: {missing_frac.max():.3f}

## Schema
- train.csv: path, participant_id, sequence_id, sign
- train_landmark_files/<participant_id>/<sequence_id>.parquet: frame, row_id, type, landmark_index, x, y, z
- type in {{face, pose, left_hand, right_hand}}; 543 landmarks per frame total

## Open follow-ups
- Confirm per-signer sequence counts are not wildly imbalanced (affects split strategy)
- Decide fixed sequence length for resampling based on the distribution above
"""

(REPORTS / "data-notes.md").write_text(report)
print(report)
print(f"\nWritten to {REPORTS / 'data-notes.md'}")
