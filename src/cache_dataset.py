"""
src/cache_dataset.py

Preprocesses every sequence in each split once and writes the result to
data/cache/ as .npy arrays, so that repeated training runs read a handful
of large files instead of ~94k small parquet ones.

Preprocessing the full dataset takes minutes and produces the same arrays
every time; a run that varies only the model or the training schedule has
no reason to redo it. Comparing several configurations means that cost
would otherwise be paid once per configuration.

Two cache variants exist, selected by --no-normalize, because the
normalization step happens inside preprocessing and cannot be undone
afterwards. Everything else a run might vary (which landmarks the model
sees, augmentation) is a transform on the cached array and needs no
separate cache.

Per-file reads are parallelized across processes, same approach as
src/baseline.py and src/train.py.

Alongside the arrays, each run writes data/cache/meta_<split>.csv: one row
per kept sequence recording which signer produced it, how many frames it had
before resampling, and how much of each hand was missing. None of that
survives into the fixed-shape array, and all of it is needed to ask which
kinds of sequence a model gets wrong. --metadata-only writes just that file,
skipping the preprocessing and leaving existing arrays untouched, and checks
the result against them.

Run: uv run python src\\cache_dataset.py
     uv run python src\\cache_dataset.py --no-normalize
     uv run python src\\cache_dataset.py --metadata-only
Requires data/splits.json (run src\\make_split.py first).
Writes data/cache/<variant>_<split>_{X,y}.npy, data/cache/<variant>.json and
data/cache/meta_<split>.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

import preprocessing as pp

RAW = Path("data/raw")
DATA = Path("data")
CACHE = DATA / "cache"

MAX_WORKERS = max(1, (os.cpu_count() or 4) - 1)
CHUNKSIZE = 64

SPLITS = ("train", "val", "test")


def variant_name(normalize: bool) -> str:
    return "normalized" if normalize else "raw"


def cache_paths(variant: str, split: str) -> tuple[Path, Path]:
    return CACHE / f"{variant}_{split}_X.npy", CACHE / f"{variant}_{split}_y.npy"


def _sequence_meta(df: pd.DataFrame) -> dict[str, float]:
    """Per-sequence properties that the cached arrays cannot express.

    The frame count is the length *before* resampling, which the fixed-length
    output has by definition thrown away. The per-hand NaN fractions are
    computed exactly as preprocessing.is_both_hands_unused computes them, so
    thresholding one of them here reproduces that function's per-hand verdict
    rather than approximating it."""
    meta: dict[str, float] = {"n_frames": int(df["frame"].nunique())}
    for hand in ("left_hand", "right_hand"):
        sub = df[df["type"] == hand]
        meta[f"{hand}_nan_frac"] = float(sub["x"].isna().mean()) if len(sub) else 1.0
    return meta


def _process_one(
    path: str, normalize: bool, metadata_only: bool
) -> tuple[np.ndarray | None, str | None, dict[str, float]]:
    """Worker: read one parquet file, decide whether it survives the two
    sequence filters, and preprocess it into a full
    (T, NUM_LANDMARKS, NUM_COORDS) sequence. Returns (None, reason, meta)
    for sequences filtered out.

    metadata_only skips the preprocessing itself and returns (None, None,
    meta) for a sequence that passed both filters. Neither filter looks at
    the preprocessed array -- both read the raw dataframe -- so the set of
    sequences kept is identical either way, which is what lets the metadata
    be lined up against arrays cached by an earlier run.

    Runs in a separate process -- must not depend on any module-level
    mutable state."""
    df = pd.read_parquet(path)
    meta = _sequence_meta(df)
    if not pp.is_usable_sequence(df):
        return None, "too_short", meta
    if pp.is_both_hands_unused(df):
        return None, "both_hands_unused", meta
    if metadata_only:
        return None, None, meta
    return pp.process_sequence(df, normalize=normalize), None, meta


def build_split(
    train_csv: pd.DataFrame,
    ids: list,
    split_name: str,
    normalize: bool,
    metadata_only: bool = False,
) -> tuple[np.ndarray | None, np.ndarray, dict[str, int], list[dict[str, object]]]:
    """Preprocess one split's sequences in dataset order, dropping the ones
    the two filters reject.

    Returns the stacked array (None when metadata_only), the kept labels,
    the skip counts, and one metadata row per kept sequence. The three
    per-sequence outputs are in the same order and describe the same
    sequences, which is the property the metadata is for."""
    subset = train_csv[train_csv["participant_id"].isin(ids)]
    paths = [str(RAW / p) for p in subset["path"]]
    signs = subset["sign"].tolist()
    participants = subset["participant_id"].tolist()
    sequence_ids = subset["sequence_id"].tolist()

    # Preallocated and trimmed rather than stacked from a list: a list of
    # ~67k arrays plus the stacked copy holds the whole split in memory
    # twice at the moment of stacking.
    out = (
        None
        if metadata_only
        else np.empty((len(paths), pp.TARGET_LEN, pp.NUM_LANDMARKS, pp.NUM_COORDS), dtype=np.float32)
    )
    labels: list[str] = []
    meta_rows: list[dict[str, object]] = []
    skip_counts = {"too_short": 0, "both_hands_unused": 0}
    kept = 0

    worker = partial(_process_one, normalize=normalize, metadata_only=metadata_only)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = executor.map(worker, paths, chunksize=CHUNKSIZE)
        for (arr, reason, meta), sign, participant, sequence_id in zip(
            results, signs, participants, sequence_ids
        ):
            if reason is not None:
                skip_counts[reason] += 1
                continue
            if out is not None:
                out[kept] = arr
            labels.append(sign)
            meta_rows.append(
                {
                    "participant_id": participant,
                    "sequence_id": sequence_id,
                    "sign": sign,
                    "n_frames": meta["n_frames"],
                    "left_hand_nan_frac": round(meta["left_hand_nan_frac"], 6),
                    "right_hand_nan_frac": round(meta["right_hand_nan_frac"], 6),
                    "left_hand_unused": int(
                        meta["left_hand_nan_frac"] >= pp.UNUSED_HAND_NAN_THRESHOLD
                    ),
                    "right_hand_unused": int(
                        meta["right_hand_nan_frac"] >= pp.UNUSED_HAND_NAN_THRESHOLD
                    ),
                }
            )
            kept += 1

    print(
        f"  {split_name}: kept {kept}, skipped {skip_counts['too_short']} too-short "
        f"(< {pp.MIN_USABLE_FRAMES} frames), "
        f"{skip_counts['both_hands_unused']} both-hands-unused"
    )
    return (None if out is None else out[:kept]), np.array(labels), skip_counts, meta_rows


META_FIELDS = [
    "participant_id",
    "sequence_id",
    "sign",
    "n_frames",
    "left_hand_nan_frac",
    "right_hand_nan_frac",
    "left_hand_unused",
    "right_hand_unused",
]


def meta_path(split: str) -> Path:
    """Metadata is a property of the raw sequence and of which filters it
    passed, neither of which depends on normalization, so one file serves
    both cache variants."""
    return CACHE / f"meta_{split}.csv"


def write_meta(split: str, meta_rows: list[dict[str, object]]) -> Path:
    path = meta_path(split)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS)
        writer.writeheader()
        writer.writerows(meta_rows)
    return path


def verify_alignment(split: str, meta_rows: list[dict[str, object]]) -> None:
    """Check the metadata lines up with every cached label array on disk.

    The metadata is only useful because row i of meta_<split>.csv describes
    row i of <variant>_<split>_X.npy. That holds because both come from the
    same filtered pass over the same dataset order, but 'it should hold' is
    not a check: a silent misalignment would attribute every prediction to
    the wrong signer while looking entirely normal. Comparing the label
    columns element-by-element is a direct test of the ordering, over every
    sequence in the split."""
    meta_labels = np.array([row["sign"] for row in meta_rows])
    checked = 0
    for variant in ("normalized", "raw"):
        _, y_path = cache_paths(variant, split)
        if not y_path.exists():
            continue
        cached = np.load(y_path, allow_pickle=False)
        if len(cached) != len(meta_labels):
            raise SystemExit(
                f"{split}: metadata has {len(meta_labels)} rows but {y_path.name} has "
                f"{len(cached)}. The cached arrays were built from a different filter "
                f"or a different dataset; rebuild them before using the metadata."
            )
        mismatches = int((cached != meta_labels).sum())
        if mismatches:
            first = int(np.flatnonzero(cached != meta_labels)[0])
            raise SystemExit(
                f"{split}: metadata labels disagree with {y_path.name} at {mismatches} "
                f"of {len(cached)} rows, first at row {first} "
                f"({meta_labels[first]!r} vs {cached[first]!r}). Row order does not match, "
                f"so the metadata cannot be joined to predictions."
            )
        print(f"  {split}: {len(cached)} labels match {y_path.name} exactly")
        checked += 1
    if not checked:
        print(f"  {split}: no cached label array present, alignment not verified")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess and cache each split as .npy arrays.")
    p.add_argument(
        "--no-normalize",
        action="store_true",
        help="skip shoulder-relative normalization; caches the raw-coordinate variant",
    )
    p.add_argument(
        "--metadata-only",
        action="store_true",
        help="write only the per-sequence metadata, leaving the cached arrays untouched",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    normalize = not args.no_normalize
    variant = variant_name(normalize)

    splits = json.loads((DATA / "splits.json").read_text(encoding="utf-8"))
    train_csv = pd.read_csv(RAW / "train.csv")

    CACHE.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"variant": variant, "normalize": normalize, "splits": {}}
    t_start = time.time()

    for split in SPLITS:
        action = "Scanning" if args.metadata_only else "Preprocessing"
        suffix = "" if args.metadata_only else f" ({variant})"
        print(f"{action} {split} split{suffix}...")
        x, y, skip_counts, meta_rows = build_split(
            train_csv, splits[split], split, normalize, metadata_only=args.metadata_only
        )
        print(f"  wrote {write_meta(split, meta_rows)} ({len(meta_rows)} rows)")

        if args.metadata_only:
            verify_alignment(split, meta_rows)
            continue

        x_path, y_path = cache_paths(variant, split)
        np.save(x_path, x)
        np.save(y_path, y)
        manifest["splits"][split] = {
            "sequences": int(x.shape[0]),
            "shape": list(x.shape),
            "signers": splits[split],
            "skipped": skip_counts,
        }
        print(f"  wrote {x_path} {x.shape} ({x.nbytes / 1e9:.2f} GB)")

    if args.metadata_only:
        print(f"Metadata written for {len(SPLITS)} splits ({time.time() - t_start:.1f}s total)")
        return

    manifest["seconds"] = round(time.time() - t_start, 1)
    (CACHE / f"{variant}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Cache manifest written to {CACHE / f'{variant}.json'} ({manifest['seconds']}s total)")


if __name__ == "__main__":
    main()
