"""
src/evaluate.py

Scores a trained checkpoint on one split and writes a per-sequence record
of what it predicted, to reports/preds_<run-name>_<split>.csv.

Accuracy alone says how often a model is right; it says nothing about
which sequences it is wrong on. Every downstream analysis needs the latter,
so the predictions are written out once and read back as a file rather than
recomputed. That also means the analysis reproduces from the repository
without a GPU, without the raw dataset, and without retraining anything.

The input configuration is read from the checkpoint, not from flags. A run
trained on hands only must be scored on hands only, and a run trained on
un-normalized coordinates must be scored against the matching cache
variant; supplying either by hand is an easy way to quietly measure the
wrong thing.

--mirror reflects every sequence about the body midline before scoring, and
writes to a separate file. Mirroring is a valid re-performance of a sign by
someone of the opposite handedness, so a model that has not keyed on
handedness should score roughly the same either way, and one that has
should not. Comparing the two files measures that directly, over the whole
split, instead of inferring it from the handful of signers a split happens
to contain.

Run: uv run python src\\evaluate.py --run-name abl_hands_aug --split test
     uv run python src\\evaluate.py --run-name abl_hands_aug --split test --mirror
Requires the cache built by src\\cache_dataset.py and models/<run-name>.pt.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import augment
import cache_dataset as cache
import train as tr

REPORTS = Path("reports")
MODELS = Path("models")

# Number of ranked predictions written per sequence. Five so that the file
# supports top-5 accuracy without a second pass over the model.
TOP_K = 5

PRED_FIELDS = [
    "row",
    "true",
    "pred_1",
    "pred_2",
    "pred_3",
    "pred_4",
    "pred_5",
    "true_rank",
    "top1_prob",
    "true_prob",
]


def load_checkpoint(run_name: str, device: torch.device):
    """Rebuild a trained model from its checkpoint.

    The checkpoint stores the full argument namespace the run was trained
    with, so the architecture is reconstructed from the run's own record
    rather than from defaults that may since have changed."""
    path = MODELS / f"{run_name}.pt"
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Checkpoints for runs outside the kept set are not stored; "
            f"retrain it with the configuration recorded in {tr.RUNS_CSV.as_posix()}."
        )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = argparse.Namespace(**ckpt["config"])
    model = tr.build_model(config, ckpt["input_size"], ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt, config


def load_split(config: argparse.Namespace, split: str) -> tuple[np.ndarray, np.ndarray]:
    variant = cache.variant_name(not config.no_normalize)
    x, y = tr.load_cached_split(variant, split)
    return tr.select_landmarks(x, config.landmarks), y


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mirror: bool,
    num_landmarks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns the top-K predicted class indices, the rank the true class
    was given (1 = correct), and the top-1 and true-class probabilities."""
    perm = augment.mirror_permutation(num_landmarks, device) if mirror else None
    top_k_out: list[np.ndarray] = []
    ranks_out: list[np.ndarray] = []
    probs_out: list[np.ndarray] = []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if perm is not None:
            # p=1.0: mirror the whole batch, not a random subset. This is a
            # measurement, so every sequence has to get the same treatment.
            xb = augment.mirror(xb, perm, p=1.0)
        logits = model(tr.to_model_input(xb, shuffle=False))
        probs = logits.softmax(dim=1)

        top_k = logits.topk(TOP_K, dim=1).indices
        # Rank of the true class: how many classes scored strictly higher,
        # plus one. Defined for every sequence, unlike a top-5 hit flag, so
        # top-1 and top-5 both fall out of it and near-misses stay visible.
        true_logit = logits.gather(1, yb.unsqueeze(1))
        rank = (logits > true_logit).sum(dim=1) + 1

        top_k_out.append(top_k.cpu().numpy())
        ranks_out.append(rank.cpu().numpy())
        probs_out.append(
            torch.stack([probs.max(dim=1).values, probs.gather(1, yb.unsqueeze(1)).squeeze(1)], dim=1)
            .cpu()
            .numpy()
        )

    return np.concatenate(top_k_out), np.concatenate(ranks_out), np.concatenate(probs_out)


def write_predictions(
    path: Path, labels: list[str], y_true: np.ndarray, top_k: np.ndarray, ranks: np.ndarray, probs: np.ndarray
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(PRED_FIELDS)
        for i in range(len(y_true)):
            writer.writerow(
                [i, labels[y_true[i]]]
                + [labels[k] for k in top_k[i]]
                + [int(ranks[i]), f"{probs[i, 0]:.4f}", f"{probs[i, 1]:.4f}"]
            )


def recorded_scores(run_name: str) -> dict[str, str] | None:
    """The accuracy this run recorded when it was trained, if available."""
    for row in tr.read_runs():
        if row["run_name"] == run_name:
            return row
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score a checkpoint and write per-sequence predictions.")
    p.add_argument("--run-name", required=True, help="checkpoint to score, from models/<run-name>.pt")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument(
        "--mirror",
        action="store_true",
        help="reflect every sequence about the body midline before scoring",
    )
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, ckpt, config = load_checkpoint(args.run_name, device)
    x, y_raw = load_split(config, args.split)

    labels = ckpt["label_classes"]
    label_to_index = {name: i for i, name in enumerate(labels)}
    y = np.array([label_to_index[s] for s in y_raw])

    print(f"run: {args.run_name}  split: {args.split}  landmarks: {config.landmarks}")
    print(f"{tr.describe_architecture(config)}  input {x.shape}")
    if args.mirror:
        print("scoring mirrored input")

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y).long()), batch_size=args.batch_size
    )
    top_k, ranks, probs = predict(model, loader, device, args.mirror, x.shape[2])

    top1 = float((ranks == 1).mean())
    top5 = float((ranks <= TOP_K).mean())
    print(f"top-1: {top1:.4f}  top-5: {top5:.4f}  ({len(ranks)} sequences)")

    REPORTS.mkdir(exist_ok=True)
    suffix = "_mirrored" if args.mirror else ""
    out_path = REPORTS / f"preds_{args.run_name}_{args.split}{suffix}.csv"
    write_predictions(out_path, labels, y, top_k, ranks, probs)
    print(f"Per-sequence predictions written to {out_path}")

    # Scoring an unmirrored split should reproduce the number this run
    # recorded when it trained. A mismatch means the checkpoint, the cache
    # or the preprocessing has moved since, which would invalidate every
    # comparison in the results table, so it is worth failing loudly on.
    recorded = recorded_scores(args.run_name)
    if args.mirror or recorded is None or args.split not in ("val", "test"):
        return
    key = "val_top1" if args.split == "val" else "test_top1"
    expected = float(recorded[key])
    if abs(expected - top1) > 5e-5:
        raise SystemExit(
            f"Measured {args.split} top-1 {top1:.4f} but {tr.RUNS_CSV.as_posix()} records "
            f"{expected:.4f} for this run. The checkpoint, the cache or the preprocessing "
            f"has changed since it was trained; results are not comparable until this is "
            f"explained."
        )
    print(f"Reproduces the recorded {key} of {expected:.4f}")


if __name__ == "__main__":
    main()
