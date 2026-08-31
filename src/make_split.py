"""
src/make_split.py

Builds a signer-independent train/val/test split by holding out whole
participants, not individual sequences -- the single most important
correctness decision in this project (a model that has seen a signer's hand
shape/style in training will look artificially good on that signer at test
time). With only 21 signers in this dataset, the held-out groups are coarse:
a handful of participants each for val and test.

Because the pool is small, a naive random hold-out risks leaving some signs
absent from val/test (or, worse, absent from train). This script searches a
number of random 3/3/15-style partitions and keeps the one that maximizes
the minimum sign-vocabulary coverage across all three splits.

Writes data/splits.json: {"train": [...], "val": [...], "test": [...]} of
participant_id values. Run once; re-run only with a deliberate decision to
change the split, since changing it silently invalidates any results that
were compared against the old one.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
DATA = Path("data")

N_VAL_SIGNERS = 3
N_TEST_SIGNERS = 3
N_SEARCH_TRIALS = 2000
SEED = 0


def coverage(train_df: pd.DataFrame, split_df: pd.DataFrame) -> float:
    """Fraction of split_df's distinct signs that also appear in train_df."""
    train_signs = set(train_df["sign"].unique())
    split_signs = set(split_df["sign"].unique())
    if not split_signs:
        return 0.0
    return len(split_signs & train_signs) / len(split_signs)


def main() -> None:
    train = pd.read_csv(RAW / "train.csv")
    participants = sorted(train["participant_id"].unique())
    n = len(participants)
    print(f"{n} signers, {train['sign'].nunique()} distinct signs")

    if n < N_VAL_SIGNERS + N_TEST_SIGNERS + 1:
        raise SystemExit(
            f"Only {n} signers available, need at least "
            f"{N_VAL_SIGNERS + N_TEST_SIGNERS + 1} for this split shape."
        )

    rng = np.random.default_rng(SEED)
    best = None  # (min_coverage, val_ids, test_ids, train_ids)

    for _ in range(N_SEARCH_TRIALS):
        shuffled = rng.permutation(participants)
        val_ids = sorted(shuffled[:N_VAL_SIGNERS].tolist())
        test_ids = sorted(shuffled[N_VAL_SIGNERS : N_VAL_SIGNERS + N_TEST_SIGNERS].tolist())
        train_ids = sorted(shuffled[N_VAL_SIGNERS + N_TEST_SIGNERS :].tolist())

        train_df = train[train["participant_id"].isin(train_ids)]
        val_df = train[train["participant_id"].isin(val_ids)]
        test_df = train[train["participant_id"].isin(test_ids)]

        cov = min(coverage(train_df, val_df), coverage(train_df, test_df))
        if best is None or cov > best[0]:
            best = (cov, val_ids, test_ids, train_ids)
            if cov == 1.0:
                break  # can't do better than full coverage

    min_cov, val_ids, test_ids, train_ids = best
    print(f"Best partition found: min val/test sign coverage by train = {min_cov:.4f}")
    if min_cov < 0.95:
        print(
            "WARNING: coverage below 95% -- some signs in val/test have "
            "little or no training data for that sign. Consider whether "
            "N_VAL_SIGNERS/N_TEST_SIGNERS need to shrink, or whether this "
            "is acceptable given the current vocabulary size."
        )

    train_df = train[train["participant_id"].isin(train_ids)]
    val_df = train[train["participant_id"].isin(val_ids)]
    test_df = train[train["participant_id"].isin(test_ids)]
    print(f"train: {len(train_ids)} signers, {len(train_df)} sequences, {train_df['sign'].nunique()} signs")
    print(f"val:   {len(val_ids)} signers, {len(val_df)} sequences, {val_df['sign'].nunique()} signs")
    print(f"test:  {len(test_ids)} signers, {len(test_df)} sequences, {test_df['sign'].nunique()} signs")

    DATA.mkdir(exist_ok=True)
    out = {"train": train_ids, "val": val_ids, "test": test_ids}
    (DATA / "splits.json").write_text(json.dumps(out, indent=2))
    print(f"Written to {DATA / 'splits.json'}")


if __name__ == "__main__":
    main()
