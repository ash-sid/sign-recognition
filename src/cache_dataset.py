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

Run: uv run python src\\cache_dataset.py
     uv run python src\\cache_dataset.py --no-normalize
Requires data/splits.json (run src\\make_split.py first).
Writes data/cache/<variant>_<split>_{X,y}.npy and data/cache/<variant>.json.
"""
from __future__ import annotations

import argparse
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


def _process_one(path: str, normalize: bool) -> tuple[np.ndarray | None, str | None]:
    """Worker: read and preprocess one parquet file, keeping the full
    (T, NUM_LANDMARKS, NUM_COORDS) sequence. Returns (None, reason) for
    sequences filtered out. Runs in a separate process -- must not depend
    on any module-level mutable state."""
    df = pd.read_parquet(path)
    if not pp.is_usable_sequence(df):
        return None, "too_short"
    if pp.is_both_hands_unused(df):
        return None, "both_hands_unused"
    return pp.process_sequence(df, normalize=normalize), None


def build_split(
    train_csv: pd.DataFrame, ids: list, split_name: str, normalize: bool
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    subset = train_csv[train_csv["participant_id"].isin(ids)]
    paths = [str(RAW / p) for p in subset["path"]]
    signs = subset["sign"].tolist()

    # Preallocated and trimmed rather than stacked from a list: a list of
    # ~67k arrays plus the stacked copy holds the whole split in memory
    # twice at the moment of stacking.
    out = np.empty((len(paths), pp.TARGET_LEN, pp.NUM_LANDMARKS, pp.NUM_COORDS), dtype=np.float32)
    labels: list[str] = []
    skip_counts = {"too_short": 0, "both_hands_unused": 0}
    kept = 0

    worker = partial(_process_one, normalize=normalize)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for (arr, reason), sign in zip(executor.map(worker, paths, chunksize=CHUNKSIZE), signs):
            if arr is None:
                skip_counts[reason] += 1
                continue
            out[kept] = arr
            labels.append(sign)
            kept += 1

    print(
        f"  {split_name}: kept {kept}, skipped {skip_counts['too_short']} too-short "
        f"(< {pp.MIN_USABLE_FRAMES} frames), "
        f"{skip_counts['both_hands_unused']} both-hands-unused"
    )
    return out[:kept], np.array(labels), skip_counts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess and cache each split as .npy arrays.")
    p.add_argument(
        "--no-normalize",
        action="store_true",
        help="skip shoulder-relative normalization; caches the raw-coordinate variant",
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
        print(f"Preprocessing {split} split ({variant})...")
        x, y, skip_counts = build_split(train_csv, splits[split], split, normalize)
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

    manifest["seconds"] = round(time.time() - t_start, 1)
    (CACHE / f"{variant}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Cache manifest written to {CACHE / f'{variant}.json'} ({manifest['seconds']}s total)")


if __name__ == "__main__":
    main()
