"""
src/train.py

Trains a temporal model (GRU or 1D-CNN) over preprocessed landmark
sequences and evaluates it against the signer-independent split in
data/splits.json. This is the first model with real temporal structure --
the logistic-regression baseline in reports/results.md (mean-pooled
landmarks, no time axis) is the number it needs to beat.

Per-file reads are parallelized across processes (see load_split), same
approach as src/baseline.py: reading tens of thousands of small parquet
files one at a time is bottlenecked by per-file open/parse overhead rather
than raw disk throughput, so a process pool gets much better use of both a
multi-core CPU and an NVMe SSD's ability to service many reads
concurrently. Unlike baseline.py, sequences are kept in full (T, 50, 3)
form rather than mean-pooled, so a full split held in memory is a few GB
(~4GB for all three splits combined) -- fine on a modern machine, but
revisit with on-disk caching or a streaming Dataset if that becomes a
problem.

Two sequence-level filters are applied during loading, matching
reports/data-notes.md's recommendations:
1. Degenerate sequences (< preprocessing.MIN_USABLE_FRAMES real frames) --
   same filter baseline.py already applies.
2. Both-hands-unused sequences (preprocessing.is_both_hands_unused) --
   every sign in this dataset uses at least one hand, so these are very
   likely tracking failures rather than genuine signal; applied to all
   three splits, not just train, consistent with how baseline.py already
   drops degenerate sequences from all three splits rather than only train.

Run: uv run python src\\train.py --model gru
Requires data/splits.json (run src\\make_split.py first).
Writes models/v1_<model>.pt (checkpoint) and reports/train_log_v1_<model>.csv
(per-epoch metrics), and updates the "Model v1" section of reports/results.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

import preprocessing as pp

RAW = Path("data/raw")
DATA = Path("data")
REPORTS = Path("reports")
MODELS = Path("models")

MAX_WORKERS = max(1, (os.cpu_count() or 4) - 1)
CHUNKSIZE = 64

# Fixed CNN channel/kernel config -- not exposed as CLI args since the
# immediate goal is comparing GRU vs 1D-CNN as architectures, not tuning
# either one in depth (that's later work). Defined once here so build_model
# and the results.md report can't drift out of sync with each other.
CNN_CHANNELS = (128, 128, 256)
CNN_KERNEL_SIZE = 5

# Markers used to replace just one model's section of reports/results.md on
# re-run, without disturbing the baseline table or other models' sections.
# Keyed by model type so running --model gru then --model cnn appends a
# second section instead of overwriting the first.
def _section_markers(model_type: str) -> tuple[str, str]:
    tag = model_type.upper()
    return f"<!-- MODEL_V1_{tag}_START -->", f"<!-- MODEL_V1_{tag}_END -->"


# --- Data loading -------------------------------------------------------

def _process_one(path: str) -> tuple[np.ndarray | None, str | None]:
    """Worker: read and preprocess one parquet file, keeping the full
    (T, 50, 3) sequence (no mean-pooling, unlike baseline.py's worker).
    Returns (None, reason) for sequences filtered out; reason is
    'too_short' or 'both_hands_unused'. Runs in a separate process -- must
    not depend on any module-level mutable state."""
    df = pd.read_parquet(path)
    if not pp.is_usable_sequence(df):
        return None, "too_short"
    if pp.is_both_hands_unused(df):
        return None, "both_hands_unused"
    return pp.process_sequence(df), None


def load_split(train_csv: pd.DataFrame, ids: list, split_name: str) -> tuple[np.ndarray, np.ndarray]:
    subset = train_csv[train_csv["participant_id"].isin(ids)]
    paths = [str(RAW / p) for p in subset["path"]]
    signs = subset["sign"].tolist()

    arrs, labels = [], []
    skip_counts = {"too_short": 0, "both_hands_unused": 0}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for (arr, reason), sign in zip(executor.map(_process_one, paths, chunksize=CHUNKSIZE), signs):
            if arr is None:
                skip_counts[reason] += 1
                continue
            arrs.append(arr)
            labels.append(sign)
    total_skipped = sum(skip_counts.values())
    if total_skipped:
        print(
            f"  {split_name}: skipped {skip_counts['too_short']} too-short "
            f"(< {pp.MIN_USABLE_FRAMES} frames), "
            f"{skip_counts['both_hands_unused']} both-hands-unused"
        )
    return np.stack(arrs), np.array(labels)


# --- Models ---------------------------------------------------------------

class GRUClassifier(nn.Module):
    """GRU over the flattened per-frame landmark vector (T, 150). Uses the
    final layer's last hidden state as the sequence representation -- for
    this dataset's short/variable-length signs (resampled to a fixed
    TARGET_LEN), a single final state is a reasonable summary without the
    added complexity of attention pooling."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)  # h_n: (num_layers, B, hidden_size)
        last = self.dropout(h_n[-1])
        return self.fc(last)


class CNNClassifier(nn.Module):
    """1D-CNN over the time axis, channels = flattened landmark features.
    Three conv/pool blocks followed by global average pooling, so the
    classifier doesn't depend on the exact post-pooling sequence length."""

    def __init__(self, input_size: int, num_classes: int, channels: tuple[int, ...], kernel_size: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = input_size
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T) for Conv1d
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


def build_model(args: argparse.Namespace, num_classes: int) -> nn.Module:
    input_size = pp.NUM_LANDMARKS * pp.NUM_COORDS
    if args.model == "gru":
        return GRUClassifier(input_size, args.hidden_size, args.num_layers, num_classes, args.dropout)
    return CNNClassifier(input_size, num_classes, channels=CNN_CHANNELS, kernel_size=CNN_KERNEL_SIZE, dropout=args.dropout)


# --- Eval -------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct1 = correct5 = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        top5 = logits.topk(5, dim=1).indices
        correct1 += (top5[:, 0] == yb).sum().item()
        correct5 += (top5 == yb.unsqueeze(1)).any(dim=1).sum().item()
        total += yb.size(0)
    return correct1 / total, correct5 / total


# --- Reporting ----------------------------------------------------------

def update_results_md(model_type: str, section_text: str) -> None:
    """Replace this model's section of reports/results.md (between its
    marker comments) if present, else append it. Leaves the baseline table
    and any other model's section untouched either way."""
    start, end = _section_markers(model_type)
    path = REPORTS / "results.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    block = f"{start}\n{section_text}\n{end}"

    if start in existing and end in existing:
        pre = existing.split(start)[0]
        post = existing.split(end)[1]
        new_content = pre + block + post
    elif existing:
        sep = "\n" if existing.endswith("\n") else "\n\n"
        new_content = existing + sep + block + "\n"
    else:
        new_content = block + "\n"

    REPORTS.mkdir(exist_ok=True)
    path.write_text(new_content, encoding="utf-8")


# --- Main -----------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the first temporal model (GRU or 1D-CNN) over landmark sequences.")
    p.add_argument("--model", choices=["gru", "cnn"], default="gru")
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10, help="stop if val top-1 doesn't improve for this many epochs")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    splits = json.loads((DATA / "splits.json").read_text(encoding="utf-8"))
    train_csv = pd.read_csv(RAW / "train.csv")

    print("Loading train split...")
    X_train, y_train_raw = load_split(train_csv, splits["train"], "train")
    print("Loading val split...")
    X_val, y_val_raw = load_split(train_csv, splits["val"], "val")
    print("Loading test split...")
    X_test, y_test_raw = load_split(train_csv, splits["test"], "test")
    print(f"train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

    le = LabelEncoder()
    le.fit(train_csv["sign"].unique())  # fit on the full vocabulary, matching baseline.py
    y_train = le.transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    y_test = le.transform(y_test_raw)
    num_classes = len(le.classes_)

    # (N, T, NUM_LANDMARKS, NUM_COORDS) -> (N, T, NUM_LANDMARKS * NUM_COORDS).
    # Equivalent to applying pp.flatten() per-sequence, but pp.flatten() is
    # written for a single (T, NUM_LANDMARKS, NUM_COORDS) sequence -- this
    # reshape does the same collapse across the whole batch at once.
    train_x = torch.from_numpy(X_train.reshape(X_train.shape[0], X_train.shape[1], -1))
    val_x = torch.from_numpy(X_val.reshape(X_val.shape[0], X_val.shape[1], -1))
    test_x = torch.from_numpy(X_test.reshape(X_test.shape[0], X_test.shape[1], -1))

    train_ds = TensorDataset(train_x, torch.from_numpy(y_train).long())
    val_ds = TensorDataset(val_x, torch.from_numpy(y_val).long())
    test_ds = TensorDataset(test_x, torch.from_numpy(y_test).long())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = build_model(args, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    ckpt_path = MODELS / f"v1_{args.model}.pt"
    log_path = REPORTS / f"train_log_v1_{args.model}.csv"

    best_val_top1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    t_start = time.time()

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_top1", "val_top5", "epoch_seconds"])

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()
            total_loss = 0.0
            n_batches = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            train_loss = total_loss / n_batches

            val_top1, val_top5 = evaluate(model, val_loader, device)
            epoch_seconds = time.time() - t0
            writer.writerow([epoch, f"{train_loss:.4f}", f"{val_top1:.4f}", f"{val_top5:.4f}", f"{epoch_seconds:.1f}"])
            f.flush()
            print(
                f"epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}  "
                f"val_top1={val_top1:.4f}  val_top5={val_top5:.4f}  ({epoch_seconds:.1f}s)"
            )

            if val_top1 > best_val_top1:
                best_val_top1 = val_top1
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_type": args.model,
                        "hidden_size": args.hidden_size,
                        "num_layers": args.num_layers,
                        "dropout": args.dropout,
                        "num_classes": num_classes,
                        "label_classes": le.classes_.tolist(),
                        "seed": args.seed,
                        "epoch": epoch,
                        "val_top1": val_top1,
                    },
                    ckpt_path,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    print(f"no val_top1 improvement for {args.patience} epochs, stopping early")
                    break

    total_seconds = time.time() - t_start

    # Reload best checkpoint (not necessarily the last epoch) before test eval.
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_top1, test_top5 = evaluate(model, test_loader, device)

    print(f"best epoch: {best_epoch}  val_top1={best_val_top1:.4f}")
    print(f"test_top1={test_top1:.4f}  test_top5={test_top5:.4f}")

    baseline_test_top1 = 0.2671
    beat_baseline = test_top1 > baseline_test_top1
    verdict = "beats" if beat_baseline else "does NOT beat"

    if args.model == "gru":
        arch_desc = f"GRU, hidden_size={args.hidden_size}, num_layers={args.num_layers}, dropout={args.dropout}"
    else:
        arch_desc = (
            f"1D-CNN, channels={CNN_CHANNELS}, kernel_size={CNN_KERNEL_SIZE}, dropout={args.dropout}"
        )

    section = f"""## Model v1 ({args.model})

- Architecture: {arch_desc}
- Seed: {args.seed}
- Best epoch: {best_epoch} / {args.epochs} (early stop patience={args.patience})
- Total train time: {total_seconds:.0f}s

| Split | Top-1 | Top-5 |
|---|---|---|
| Val | {best_val_top1:.4f} | — |
| Test | {test_top1:.4f} | {test_top5:.4f} |

This {verdict} the logistic-regression baseline (test top-1 {baseline_test_top1:.4f}).
"""
    update_results_md(args.model, section)
    print(f"Checkpoint written to {ckpt_path}")
    print(f"Per-epoch log written to {log_path}")
    print(f"reports/results.md updated ({verdict} baseline)")


if __name__ == "__main__":
    main()
