"""
src/onnx_parity.py

Scores an exported graph over a split and compares it, prediction by
prediction, against the predictions the checkpoint already made.

Aggregate accuracy is the wrong instrument for this. Two models can
agree on a percentage while disagreeing about which sequences they get
right, and that has already happened once in this project: a mirrored
probe moved 4.6% of individual predictions while accuracy shifted by
0.01pp. An export or a quantization pass that changed which sequences
are recognised would be invisible to an accuracy check and obvious to
this one.

Three comparisons, answering three different questions:

  The module on CPU   Whether the export preserves every prediction.
                      This is the parity gate, and it is measured against
                      the module on the same device the graph runs on,
                      because the stored predictions were produced on the
                      GPU and CPU and GPU disagree with each other by
                      more than an export does. Comparing across devices
                      would charge the export for a difference it did not
                      cause.
  The float32 graph   For a quantized graph, what quantization changed,
                      isolated from both the device and the runtime by
                      comparing against the float32 export scored the
                      same way.
  Stored predictions  Reported rather than enforced, because the stored
                      file carries a device difference this script cannot
                      remove without rescoring it.
  Disagreement margin The gap between the top two logits on every
                      sequence whose prediction moved. A prediction that
                      flips on a margin near the runtimes' own numerical
                      noise was a coin toss either way; one that flips on
                      a wide margin means the model changed. A count of
                      changed predictions cannot tell those apart, so the
                      count is never reported without them.

Writes predictions in the same format and through the same writer as the
checkpoint's own scoring, so the two files are directly comparable and
cannot drift in layout.

Run: uv run python src\\onnx_parity.py --run-name abl_hands_aug --variant fp32
     uv run python src\\onnx_parity.py --run-name abl_hands_aug --variant int8_dynamic
     uv run python src\\onnx_parity.py --run-name abl_hands_aug --variant int8_static
Requires the graph from src\\export_onnx.py or src\\quantize_onnx.py, the
cache built by src\\cache_dataset.py, and reports/preds_<run-name>_<split>.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

import cache_dataset as cache
import error_analysis as ea
import evaluate
import export_onnx as ex
import train as tr

REPORTS = Path("reports")
MODELS = Path("models")

TOP_K = evaluate.TOP_K

# Any graph reaching this script carries its own weights; the floor is set
# below the smallest quantized artifact rather than the float32 one.
MIN_GRAPH_BYTES = 200_000

# Batch size for scoring. Large enough that per-call overhead is not the
# dominant cost, small enough that the logits for one batch are a modest
# allocation. Prediction outputs do not depend on it.
SCORING_BATCH = 256


def model_path(run_name: str, variant: str) -> Path:
    return MODELS / (f"{run_name}.onnx" if variant == "fp32" else f"{run_name}_{variant}.onnx")


def output_path(run_name: str, variant: str, split: str) -> Path:
    return REPORTS / f"preds_{run_name}_onnx_{variant}_{split}.csv"


def reference_path(run_name: str, split: str) -> Path:
    return REPORTS / f"preds_{run_name}_{split}.csv"


def onnx_logits(path: Path, x: np.ndarray, batch: int) -> np.ndarray:
    runtime = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = [
        runtime.run([ex.OUTPUT_NAME], {ex.INPUT_NAME: x[i : i + batch]})[0]
        for i in range(0, len(x), batch)
    ]
    return np.concatenate(out).astype(np.float32)


def torch_logits(run_name: str, x: np.ndarray, config: argparse.Namespace, batch: int) -> np.ndarray:
    """The module's own logits on the CPU, for the same input tensor.

    Scored on the CPU rather than the GPU deliberately: comparing an ONNX
    CPU graph against CUDA would mix two differences together and leave
    neither measurable."""
    model, _, _ = evaluate.load_checkpoint(run_name, torch.device("cpu"))
    wrapped = ex.ContractInput(model, config.landmarks).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(wrapped(torch.from_numpy(x[i : i + batch])).numpy())
    return np.concatenate(out).astype(np.float32)


def rank_and_probability(logits: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-K class indices, the true class's rank, and two probabilities.

    Mirrors how the checkpoint's own scoring derives these, so that a
    difference between the two files is a difference between the models
    rather than between two ways of reading the same logits."""
    order = np.argsort(-logits, axis=1, kind="stable")[:, :TOP_K]
    true_logit = logits[np.arange(len(y)), y][:, None]
    rank = (logits > true_logit).sum(axis=1) + 1

    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    probs = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    paired = np.stack([probs.max(axis=1), probs[np.arange(len(y)), y]], axis=1)
    return order, rank, paired


def top_two_margin(logits: np.ndarray) -> np.ndarray:
    """Gap between the highest and second-highest logit per sequence."""
    partitioned = np.partition(logits, -2, axis=1)
    return partitioned[:, -1] - partitioned[:, -2]


def describe_margins(margins: np.ndarray) -> str:
    return (
        f"min {margins.min():.3e}, median {np.median(margins):.3e}, max {margins.max():.3e}"
    )


def compare(
    baseline_name: str,
    predicted: np.ndarray,
    baseline: np.ndarray,
    truth: np.ndarray,
    margins: np.ndarray,
    subject: str = "This graph",
) -> int:
    """Report how two scorings of the same sequences differ.

    Prints the changed count with the margins beside it, never alone: a
    prediction that moves at a margin near the numerical noise between
    two implementations was undetermined either way, and one that moves
    at a wide margin means the models disagree. The count cannot
    distinguish them."""
    moved = predicted != baseline
    changed = int(moved.sum())
    print(f"\n{subject} against {baseline_name}: {changed} of {len(truth)} changed ({changed / len(truth):.2%})")
    lost, gained, p_value = ea.mcnemar(baseline == truth, predicted == truth)
    print(f"  {lost} correct lost, {gained} gained, exact binomial p = {p_value:.3g}")
    if changed:
        print(f"  margin where changed:   {describe_margins(margins[moved])}")
        print(f"  margin where unchanged: {describe_margins(margins[~moved])}")
    return changed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare an exported graph against stored predictions.")
    p.add_argument("--run-name", required=True)
    p.add_argument(
        "--variant",
        default="fp32",
        help="fp32, or the suffix of a quantized graph such as int8_dynamic",
    )
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--batch-size", type=int, default=SCORING_BATCH)
    p.add_argument(
        "--no-torch-reference",
        dest="torch_reference",
        action="store_false",
        help="skip the logit-space comparison against the module on the CPU",
    )
    p.add_argument(
        "--allow-changes",
        action="store_true",
        help="report changed predictions for a float32 graph instead of failing on them",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    graph = model_path(args.run_name, args.variant)
    if not graph.exists():
        raise SystemExit(
            f"Missing {graph}. Build it first with src\\export_onnx.py or src\\quantize_onnx.py."
        )
    ex.check_self_contained(graph, min_bytes=MIN_GRAPH_BYTES)

    reference_file = reference_path(args.run_name, args.split)
    if not reference_file.exists():
        raise SystemExit(
            f"Missing {reference_file}. Score the checkpoint first with:\n"
            f"  uv run python src\\evaluate.py --run-name {args.run_name} --split {args.split}"
        )

    _, ckpt, config = evaluate.load_checkpoint(args.run_name, torch.device("cpu"))
    labels = ckpt["label_classes"]
    label_to_index = {name: i for i, name in enumerate(labels)}

    # The graph slices the landmark subset internally, so it is fed the
    # full documented tensor. Slicing here as well would silently drop the
    # part of the graph this is meant to be testing.
    variant = cache.variant_name(not config.no_normalize)
    x, y_raw = tr.load_cached_split(variant, args.split)
    x = np.ascontiguousarray(x, dtype=np.float32)
    y = np.array([label_to_index[s] for s in y_raw])

    reference = pd.read_csv(reference_file)
    if len(reference) != len(y):
        raise SystemExit(
            f"{reference_file} has {len(reference)} rows but the {args.split} split has {len(y)}. "
            f"The stored predictions and the cache describe different data."
        )
    if list(reference["true"]) != [labels[i] for i in y]:
        raise SystemExit(
            f"{reference_file} does not line up with the cache row for row. Comparing them would "
            f"attribute each prediction to the wrong sequence."
        )

    print(f"run: {args.run_name}  variant: {args.variant}  split: {args.split}  {len(y)} sequences")
    print(f"graph: {graph} ({graph.stat().st_size / 1e6:.2f} MB)")

    logits = onnx_logits(graph, x, args.batch_size)
    order, rank, probs = rank_and_probability(logits, y)

    out_file = output_path(args.run_name, args.variant, args.split)
    evaluate.write_predictions(out_file, labels, y, order, rank, probs)
    print(f"Per-sequence predictions written to {out_file}")

    top1 = float((rank == 1).mean())
    top5 = float((rank <= TOP_K).mean())
    reference_top1 = float((reference["true"] == reference["pred_1"]).mean())
    reference_top5 = float((reference["true_rank"] <= TOP_K).mean())
    print(
        f"top-1: {top1:.4f} against {reference_top1:.4f} stored "
        f"({(top1 - reference_top1) * 100:+.2f}pp)"
    )
    print(
        f"top-5: {top5:.4f} against {reference_top5:.4f} stored "
        f"({(top5 - reference_top5) * 100:+.2f}pp)"
    )

    truth = reference["true"].to_numpy()
    predicted = np.array([labels[i] for i in order[:, 0]])
    margins = top_two_margin(logits)

    compare("stored predictions", predicted, reference["pred_1"].to_numpy(), truth, margins)

    module_predictions = None
    if args.torch_reference:
        module_logits = torch_logits(args.run_name, x, config, args.batch_size)
        difference = float(np.abs(logits - module_logits).max())
        scale = float(np.abs(module_logits).max())
        print(
            f"\nAgainst the module on the CPU: {difference:.3e} absolute, "
            f"{difference / scale:.3e} relative to a maximum logit magnitude of {scale:.2f}"
        )
        module_predictions = np.array(
            [labels[i] for i in np.argsort(-module_logits, axis=1, kind="stable")[:, 0]]
        )
        compare("the module on the CPU", predicted, module_predictions, truth, margins)
        compare("stored predictions", module_predictions, reference["pred_1"].to_numpy(), truth,
                top_two_margin(module_logits), subject="The module on the CPU")

    if args.variant != "fp32":
        float_file = output_path(args.run_name, "fp32", args.split)
        if float_file.exists():
            float_predictions = pd.read_csv(float_file)["pred_1"].to_numpy()
            compare("the float32 export", predicted, float_predictions, truth, margins)
        else:
            print(
                f"\nNo {float_file} to compare against; quantization cannot be separated from "
                f"the device difference without it."
            )

    # The parity gate. Measured against the module on the same device
    # rather than against the stored file, so a difference the GPU
    # introduced is not attributed to the export.
    if args.variant == "fp32" and module_predictions is not None and not args.allow_changes:
        changed = int((predicted != module_predictions).sum())
        if changed:
            raise SystemExit(
                f"The float32 export changed {changed} of {len(y)} predictions relative to the "
                f"module on the same device. Export is supposed to preserve every one of them."
            )

    # A float32 graph should also land on the accuracy the run recorded
    # when it trained. Passing the per-prediction check but missing this
    # would mean the stored predictions themselves had drifted.
    recorded = evaluate.recorded_scores(args.run_name)
    if args.variant == "fp32" and recorded is not None and args.split in ("val", "test"):
        key = "val_top1" if args.split == "val" else "test_top1"
        expected = float(recorded[key])
        if abs(expected - top1) > 5e-5:
            raise SystemExit(
                f"Measured {args.split} top-1 {top1:.4f} but {tr.RUNS_CSV.as_posix()} records "
                f"{expected:.4f} for this run."
            )
        print(f"Reproduces the recorded {key} of {expected:.4f}")


if __name__ == "__main__":
    main()
