"""
src/baseline.py

Two dumb baselines against the signer-independent split in data/splits.json:
1. Majority class -- always predict the most common training sign.
2. Logistic regression on mean-pooled, preprocessed landmarks.

These exist to give every later model something meaningful to beat. A model
that can't clear the logistic-regression number isn't learning temporal
structure worth having.

The same two sequence-level filters the temporal models apply are applied
here -- degenerate sequences under preprocessing.MIN_USABLE_FRAMES, and
sequences where neither hand is tracked -- so that every model in
reports/results.md is scored on the same set of test sequences and the
numbers are directly comparable.

Per-file reads are parallelized across processes (see load_split_features):
reading tens of thousands of small parquet files one at a time is
bottlenecked by per-file open/parse overhead rather than raw disk
throughput, so a process pool gets much better use of both a multi-core
CPU and an NVMe SSD's ability to service many reads concurrently.

Run: uv run python src\\baseline.py
Requires data/splits.json (run src\\make_split.py first).
Writes the baselines section of reports/results.md.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.preprocessing import LabelEncoder

import preprocessing as pp
import report

RAW = Path("data/raw")
DATA = Path("data")
REPORTS = Path("reports")

# Leave one logical core free for the OS/other work; every other core gets
# a worker. Override by setting this directly if you want to tune it.
MAX_WORKERS = max(1, (os.cpu_count() or 4) - 1)
CHUNKSIZE = 64  # sequences handed to each worker per round-trip; reduces IPC overhead


def _process_one(path: str) -> tuple[np.ndarray | None, str | None]:
    """Worker: read and preprocess one parquet file, mean-pool it into a
    feature vector. Returns (None, reason) for sequences filtered out;
    reason is 'too_short' or 'both_hands_unused'. Runs in a separate
    process -- must not depend on any module-level mutable state."""
    df = pd.read_parquet(path)
    if not pp.is_usable_sequence(df):
        return None, "too_short"
    if pp.is_both_hands_unused(df):
        return None, "both_hands_unused"
    return pp.mean_pool(pp.process_sequence(df)), None


def load_split_features(train_csv: pd.DataFrame, ids: list) -> tuple[np.ndarray, np.ndarray]:
    subset = train_csv[train_csv["participant_id"].isin(ids)]
    paths = [str(RAW / p) for p in subset["path"]]
    signs = subset["sign"].tolist()

    feats, labels = [], []
    skip_counts = {"too_short": 0, "both_hands_unused": 0}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for (feat, reason), sign in zip(executor.map(_process_one, paths, chunksize=CHUNKSIZE), signs):
            if feat is None:
                skip_counts[reason] += 1
                continue
            feats.append(feat)
            labels.append(sign)
    if sum(skip_counts.values()):
        print(
            f"  skipped {skip_counts['too_short']} too-short "
            f"(< {pp.MIN_USABLE_FRAMES} frames), "
            f"{skip_counts['both_hands_unused']} both-hands-unused"
        )
    return np.stack(feats), np.array(labels)


def main() -> None:
    splits = json.loads((DATA / "splits.json").read_text(encoding="utf-8"))
    train_csv = pd.read_csv(RAW / "train.csv")

    print("Loading train split...")
    X_train, y_train_raw = load_split_features(train_csv, splits["train"])
    print("Loading val split...")
    X_val, y_val_raw = load_split_features(train_csv, splits["val"])
    print("Loading test split...")
    X_test, y_test_raw = load_split_features(train_csv, splits["test"])

    print(f"train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

    le = LabelEncoder()
    le.fit(train_csv["sign"].unique())  # fit on the full vocabulary, not just train split
    y_train = le.transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    y_test = le.transform(y_test_raw)
    n_classes = len(le.classes_)

    # top_k_accuracy_score compares against the full vocabulary, so it needs
    # every sign to survive into the training split. With hundreds of
    # sequences per sign this is never close, but a filter that removed one
    # entirely would otherwise surface as an opaque shape mismatch deep
    # inside scikit-learn rather than as the data problem it is.
    missing = sorted(set(range(n_classes)) - set(np.unique(y_train).tolist()))
    if missing:
        raise SystemExit(
            f"{len(missing)} sign(s) have no training sequences left after filtering, "
            f"e.g. {[le.classes_[i] for i in missing[:5]]}. The sequence filters are "
            f"too aggressive for this split."
        )

    # --- Baseline 1: majority class -----------------------------------------
    majority_class = np.bincount(y_train).argmax()
    maj_val_acc = accuracy_score(y_val, np.full_like(y_val, majority_class))
    maj_test_acc = accuracy_score(y_test, np.full_like(y_test, majority_class))

    # --- Baseline 2: logistic regression on mean-pooled landmarks ----------
    t0 = time.time()
    # lbfgs is multinomial (softmax) by default for multi-class problems as
    # of scikit-learn >= 1.7 -- the old multi_class="multinomial" kwarg was
    # removed, not just deprecated, so don't pass it. n_jobs has no effect
    # on lbfgs (single-threaded solver) and is deprecated as of 1.8.
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)
    fit_seconds = time.time() - t0

    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)
    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)

    val_proba = clf.predict_proba(X_val)
    test_proba = clf.predict_proba(X_test)
    present_labels = np.arange(n_classes)
    val_top5 = top_k_accuracy_score(y_val, val_proba, k=5, labels=present_labels)
    test_top5 = top_k_accuracy_score(y_test, test_proba, k=5, labels=present_labels)

    section = f"""# Results

## Split (data/splits.json)
- train signers: {len(splits['train'])} ({X_train.shape[0]} sequences)
- val signers: {len(splits['val'])} ({X_val.shape[0]} sequences)
- test signers: {len(splits['test'])} ({X_test.shape[0]} sequences)
- vocabulary: {n_classes} signs

## Baselines

| Model | Val top-1 | Test top-1 | Val top-5 | Test top-5 |
|---|---|---|---|---|
| Majority class | {maj_val_acc:.4f} | {maj_test_acc:.4f} | — | — |
| Logistic regression (mean-pooled landmarks) | {val_acc:.4f} | {test_acc:.4f} | {val_top5:.4f} | {test_top5:.4f} |

Logistic regression fit time: {fit_seconds:.1f}s.

Any subsequent temporal model needs to clear the logistic-regression test
top-1 number above to be worth the added complexity over mean-pooling.
"""
    report.update_section(REPORTS / "results.md", "baselines", section)
    print(section)
    print(f"Baselines section written to {REPORTS / 'results.md'}")


if __name__ == "__main__":
    main()
