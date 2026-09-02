"""
src/train.py

Trains a temporal model over preprocessed landmark sequences and evaluates
it against the signer-independent split in data/splits.json. Three
architectures are available (--model): a GRU, a 1D-CNN, and a small
transformer encoder.

Reads the preprocessed arrays written by src/cache_dataset.py rather than
walking the parquet files itself, so comparing many configurations doesn't
repeat the same minutes-long preprocessing pass each time. Run
src\\cache_dataset.py first; --no-normalize selects the un-normalized cache
variant.

Every run appends a row to reports/ablations.csv keyed by --run-name, and
re-renders that file as a table in reports/results.md. Re-running a name
replaces its row rather than adding a duplicate, and the row records the
full configuration, so two rows can always be compared on what actually
differs between them.

Input and training variants, all defaulting to off so that a bare
`--model gru` reproduces the plain configuration exactly:
  --landmarks hands   drop the eight pose landmarks from the model input.
                      Normalization is unaffected: it happens during
                      preprocessing and has already used the shoulders by
                      the time this slice is taken.
  --no-normalize      read the un-normalized cache variant.
  --augment           mirror / time-warp / jitter each training batch.
                      Training data only; validation and test are never
                      augmented.
  --shuffle-time      randomly permute frames within every sequence, on
                      all three splits. A diagnostic rather than an
                      augmentation: it measures how much of the task
                      survives with temporal order removed entirely.

Run: uv run python src\\train.py --model cnn --run-name v1_cnn
Writes models/<run-name>.pt, reports/train_log_<run-name>.csv, and updates
reports/ablations.csv and reports/results.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

import augment
import cache_dataset as cache
import preprocessing as pp
import report

DATA = Path("data")
RAW = DATA / "raw"
REPORTS = Path("reports")
MODELS = Path("models")

RESULTS_MD = REPORTS / "results.md"
RUNS_CSV = REPORTS / "ablations.csv"

# Landmark positions 0-41 are the two hands; 42-49 are the pose points.
HANDS_END = 2 * len(pp.HAND_INDICES)

# Test top-1 of the mean-pooled logistic regression, for the note under the
# results table, so the table is readable without cross-referencing the
# baselines section above it.
BASELINE_TEST_TOP1 = 0.2671

CNN_CHANNELS = (128, 128, 256)
CNN_KERNEL_SIZE = 5

# Fields recorded for every run. Configuration comes before outcome, so a
# diff between two rows shows what was varied before it shows what changed
# as a result.
RUN_FIELDS = [
    "run_name",
    "model",
    "landmarks",
    "normalized",
    "augment",
    "shuffle_time",
    "aug_mirror_p",
    "aug_warp",
    "aug_jitter",
    "gru_pool",
    "optimizer",
    "lr_schedule",
    "lr",
    "dropout",
    "seed",
    "epochs_run",
    "best_epoch",
    "val_top1",
    "test_top1",
    "test_top5",
    "train_seconds",
]
CONFIG_FIELDS = RUN_FIELDS[1:15]


# --- Data -------------------------------------------------------------------

def load_cached_split(variant: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    x_path, y_path = cache.cache_paths(variant, split)
    if not x_path.exists():
        flag = "" if variant == "normalized" else " --no-normalize"
        raise SystemExit(
            f"Missing {x_path}. Build it first with:\n"
            f"  uv run python src\\cache_dataset.py{flag}"
        )
    return np.load(x_path), np.load(y_path)


def select_landmarks(x: np.ndarray, landmarks: str) -> np.ndarray:
    """Slice the landmark axis down to the active set."""
    return x if landmarks == "hands_pose" else x[:, :, :HANDS_END, :]


# --- Models -----------------------------------------------------------------

class GRUClassifier(nn.Module):
    """GRU over the flattened per-frame landmark vector.

    pool="last" uses the final layer's last hidden state as the sequence
    representation; pool="mean" averages the top layer's outputs over time
    instead. The distinction matters here because every sequence is
    resampled to the same length regardless of the underlying sign's
    duration, which tends to push a short sign's informative frames away
    from the end of the sequence."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        pool: str = "last",
    ):
        super().__init__()
        self.pool = pool
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
        outputs, h_n = self.gru(x)  # h_n: (num_layers, B, hidden_size)
        summary = outputs.mean(dim=1) if self.pool == "mean" else h_n[-1]
        return self.fc(self.dropout(summary))


class CNNClassifier(nn.Module):
    """1D-CNN over the time axis, channels = flattened landmark features.
    Three conv/pool blocks followed by global average pooling, so the
    classifier doesn't depend on the exact post-pooling sequence length."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        channels: tuple[int, ...],
        kernel_size: int,
        dropout: float,
    ):
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


class TransformerClassifier(nn.Module):
    """Small transformer encoder over the per-frame landmark vector.

    Every sequence is resampled to the same fixed length upstream, so there
    is no padding and therefore no attention mask -- each frame attends to
    all the others unconditionally. Position is supplied by a learned
    per-frame embedding.

    Pre-norm blocks (norm_first) are used because they train stably at a
    constant learning rate, where post-norm blocks generally need a warmup
    schedule to avoid diverging early. Time is collapsed by mean pooling,
    matching the CNN's global average pooling, so that comparing the two
    isolates the encoder instead of also changing how the sequence gets
    summarized."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_len: int,
    ):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x) + self.pos[:, : x.shape[1]]
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(self.dropout(x))


def build_model(args: argparse.Namespace, input_size: int, num_classes: int) -> nn.Module:
    if args.model == "gru":
        return GRUClassifier(
            input_size, args.hidden_size, args.num_layers, num_classes, args.dropout, pool=args.gru_pool
        )
    if args.model == "cnn":
        return CNNClassifier(
            input_size, num_classes, channels=CNN_CHANNELS, kernel_size=CNN_KERNEL_SIZE, dropout=args.dropout
        )
    return TransformerClassifier(
        input_size,
        num_classes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.ff_dim,
        dropout=args.dropout,
        max_len=pp.TARGET_LEN,
    )


def describe_architecture(args: argparse.Namespace) -> str:
    if args.model == "gru":
        return (
            f"GRU, hidden_size={args.hidden_size}, num_layers={args.num_layers}, "
            f"pool={args.gru_pool}, dropout={args.dropout}"
        )
    if args.model == "cnn":
        return f"1D-CNN, channels={CNN_CHANNELS}, kernel_size={CNN_KERNEL_SIZE}, dropout={args.dropout}"
    return (
        f"Transformer, d_model={args.d_model}, nhead={args.nhead}, layers={args.num_layers}, "
        f"ff_dim={args.ff_dim}, dropout={args.dropout}"
    )


# --- Train / eval -----------------------------------------------------------

def to_model_input(xb: torch.Tensor, shuffle: bool) -> torch.Tensor:
    """(B, T, L, 3) -> (B, T, L*3), optionally destroying frame order first."""
    if shuffle:
        xb = augment.shuffle_time(xb)
    return xb.reshape(xb.shape[0], xb.shape[1], -1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, shuffle: bool) -> tuple[float, float]:
    model.eval()
    correct1 = correct5 = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(to_model_input(xb, shuffle))
        top5 = logits.topk(5, dim=1).indices
        correct1 += (top5[:, 0] == yb).sum().item()
        correct5 += (top5 == yb.unsqueeze(1)).any(dim=1).sum().item()
        total += yb.size(0)
    return correct1 / total, correct5 / total


# --- Run record -------------------------------------------------------------

def read_runs() -> list[dict[str, str]]:
    if not RUNS_CSV.exists():
        return []
    with open(RUNS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_runs(rows: list[dict[str, str]]) -> None:
    REPORTS.mkdir(exist_ok=True)
    with open(RUNS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def record_run(row: dict[str, str]) -> list[dict[str, str]]:
    """Insert or replace this run's row, keyed by run name.

    A name reused with a different configuration is almost always a
    mistake: it overwrites a result that the replacing row no longer
    describes. It warns rather than failing, since by the time this is
    reached the run has already happened and its numbers are worth
    keeping."""
    rows = read_runs()
    for i, existing in enumerate(rows):
        if existing["run_name"] != row["run_name"]:
            continue
        changed = [f for f in CONFIG_FIELDS if existing.get(f) != row[f]]
        if changed:
            print(
                f"WARNING: run name '{row['run_name']}' was already recorded with a different "
                f"configuration ({', '.join(changed)}). Replacing it; the earlier result is lost."
            )
        rows[i] = row
        break
    else:
        rows.append(row)
    write_runs(rows)
    return rows


def row_notes(row: dict[str, str]) -> str:
    """Short description of anything this run varied that the table's own
    columns don't show, so no two rows can look identical while having
    produced different numbers."""
    notes = []
    if row.get("shuffle_time") == "true":
        notes.append("frames shuffled")
    if row.get("augment") == "true":
        # A blank strength means the run predates these columns and used the
        # defaults, so only an explicit zero counts as a disabled transform.
        enabled = [
            label
            for label, field in (("mirror", "aug_mirror_p"), ("warp", "aug_warp"), ("jitter", "aug_jitter"))
            if row.get(field) != "0"
        ]
        if len(enabled) < 3:
            notes.append(f"{'/'.join(enabled) if enabled else 'none'} only")
    if row.get("gru_pool") == "mean":
        notes.append("mean pooling")
    if row.get("optimizer") != "adam":
        notes.append(row["optimizer"])
    if row.get("lr_schedule") != "none":
        notes.append(f"{row['lr_schedule']} LR")
    return ", ".join(notes) if notes else "—"


def render_runs_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| Run | Model | Landmarks | Normalized | Augment | Notes | Seed | Best epoch "
        "| Val top-1 | Test top-1 | Test top-5 | Train s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['run_name']} | {r['model']} | {r['landmarks']} | {r['normalized']} "
            f"| {r['augment']} | {row_notes(r)} | {r['seed']} | {r['best_epoch']} | {r['val_top1']} "
            f"| {r['test_top1']} | {r['test_top5']} | {r['train_seconds']} |"
        )
    table = "\n".join(lines)
    return (
        "## Models\n\n"
        f"{table}\n\n"
        "One row per training run. `Landmarks` is either both hands plus the eight pose\n"
        "points (`hands_pose`) or the hands alone (`hands`); `Normalized` is whether the\n"
        "shoulder-relative normalization was applied during preprocessing; `Augment` is\n"
        "whether training batches were mirrored, time-warped and jittered. `Notes` covers\n"
        "anything else a run varied. The full per-run configuration, including fields not\n"
        f"shown here, is in {RUNS_CSV.as_posix()}.\n\n"
        "A model has to clear the mean-pooled logistic regression's test top-1 of "
        f"{BASELINE_TEST_TOP1:.4f}\nto be worth its added complexity over ignoring time entirely.\n"
    )


# --- Main -------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a temporal model over cached landmark sequences.")
    p.add_argument("--model", choices=["gru", "cnn", "transformer"], default="gru")
    p.add_argument("--run-name", default=None, help="identifier for this run; defaults to v1_<model>")
    p.add_argument("--landmarks", choices=["hands_pose", "hands"], default="hands_pose")
    p.add_argument("--no-normalize", action="store_true", help="use the un-normalized cache variant")
    p.add_argument("--augment", action="store_true", help="mirror/warp/jitter training batches")
    p.add_argument("--shuffle-time", action="store_true", help="permute frames on all splits (diagnostic)")
    p.add_argument("--aug-mirror-p", type=float, default=0.5, help="probability of mirroring a training sample")
    p.add_argument("--aug-warp", type=float, default=0.2, help="maximum time-warp strength; 0 disables")
    p.add_argument("--aug-jitter", type=float, default=0.01, help="coordinate noise standard deviation; 0 disables")
    p.add_argument("--gru-pool", choices=["last", "mean"], default="last")
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--ff-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    p.add_argument("--lr-schedule", choices=["none", "cosine"], default="none")
    p.add_argument("--patience", type=int, default=10, help="stop if val top-1 doesn't improve for this many epochs")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.augment and args.no_normalize:
        raise SystemExit(
            "--augment cannot be combined with --no-normalize: the mirror transform "
            "reflects about x = 0, which is the body midline only in normalized "
            "coordinates. Raw coordinates would need a different reflection."
        )
    if args.run_name is None:
        args.run_name = f"v1_{args.model}"
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    normalize = not args.no_normalize
    variant = cache.variant_name(normalize)
    print(f"run: {args.run_name}  cache: {variant}  landmarks: {args.landmarks}")

    x_train, y_train_raw = load_cached_split(variant, "train")
    x_val, y_val_raw = load_cached_split(variant, "val")
    x_test, y_test_raw = load_cached_split(variant, "test")

    x_train = select_landmarks(x_train, args.landmarks)
    x_val = select_landmarks(x_val, args.landmarks)
    x_test = select_landmarks(x_test, args.landmarks)
    print(f"train: {x_train.shape}, val: {x_val.shape}, test: {x_test.shape}")

    vocabulary = json.loads((RAW / "sign_to_prediction_index_map.json").read_text(encoding="utf-8"))
    le = LabelEncoder()
    le.fit(sorted(vocabulary))  # fit on the full vocabulary, not just the training split
    num_classes = len(le.classes_)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(le.transform(y_train_raw)).long())
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(le.transform(y_val_raw)).long())
    test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(le.transform(y_test_raw)).long())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    num_landmarks = x_train.shape[2]
    input_size = num_landmarks * pp.NUM_COORDS
    model = build_model(args, input_size, num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{describe_architecture(args)}  ({n_params / 1e6:.2f}M parameters)")

    optimizer_cls = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        if args.lr_schedule == "cosine"
        else None
    )
    criterion = nn.CrossEntropyLoss()

    mirror_perm = augment.mirror_permutation(num_landmarks, device) if args.augment else None

    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    ckpt_path = MODELS / f"{args.run_name}.pt"
    log_path = REPORTS / f"train_log_{args.run_name}.csv"

    best_val_top1 = -1.0
    best_epoch = -1
    epochs_run = 0
    epochs_without_improvement = 0
    t_start = time.time()

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_top1", "val_top5", "lr", "epoch_seconds"])

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            epochs_run = epoch
            model.train()
            total_loss = 0.0
            n_batches = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                if args.augment:
                    xb = augment.augment_batch(
                        xb, mirror_perm, args.aug_mirror_p, args.aug_warp, args.aug_jitter
                    )
                optimizer.zero_grad()
                loss = criterion(model(to_model_input(xb, args.shuffle_time)), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            train_loss = total_loss / n_batches

            current_lr = optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                scheduler.step()

            val_top1, val_top5 = evaluate(model, val_loader, device, args.shuffle_time)
            epoch_seconds = time.time() - t0
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.4f}",
                    f"{val_top1:.4f}",
                    f"{val_top5:.4f}",
                    f"{current_lr:.6f}",
                    f"{epoch_seconds:.1f}",
                ]
            )
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
                        "run_name": args.run_name,
                        "model_type": args.model,
                        "config": vars(args),
                        "num_landmarks": num_landmarks,
                        "input_size": input_size,
                        "num_classes": num_classes,
                        "label_classes": le.classes_.tolist(),
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

    # Reload the best checkpoint (not necessarily the last epoch) before test eval.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_top1, test_top5 = evaluate(model, test_loader, device, args.shuffle_time)

    print(f"best epoch: {best_epoch}  val_top1={best_val_top1:.4f}")
    print(f"test_top1={test_top1:.4f}  test_top5={test_top5:.4f}")

    row = {
        "run_name": args.run_name,
        "model": args.model,
        "landmarks": args.landmarks,
        "normalized": str(normalize).lower(),
        "augment": str(args.augment).lower(),
        "shuffle_time": str(args.shuffle_time).lower(),
        "aug_mirror_p": f"{args.aug_mirror_p:g}" if args.augment else "",
        "aug_warp": f"{args.aug_warp:g}" if args.augment else "",
        "aug_jitter": f"{args.aug_jitter:g}" if args.augment else "",
        "gru_pool": args.gru_pool if args.model == "gru" else "",
        "optimizer": args.optimizer,
        "lr_schedule": args.lr_schedule,
        "lr": f"{args.lr:g}",
        "dropout": f"{args.dropout:g}",
        "seed": str(args.seed),
        "epochs_run": str(epochs_run),
        "best_epoch": str(best_epoch),
        "val_top1": f"{best_val_top1:.4f}",
        "test_top1": f"{test_top1:.4f}",
        "test_top5": f"{test_top5:.4f}",
        "train_seconds": f"{total_seconds:.0f}",
    }
    rows = record_run(row)
    report.update_section(RESULTS_MD, "runs", render_runs_table(rows))

    verdict = "beats" if test_top1 > BASELINE_TEST_TOP1 else "does NOT beat"
    print(f"Checkpoint written to {ckpt_path}")
    print(f"Per-epoch log written to {log_path}")
    print(f"{RUNS_CSV} and {RESULTS_MD} updated ({verdict} the logistic-regression baseline)")


if __name__ == "__main__":
    main()
