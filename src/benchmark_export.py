"""
src/benchmark_export.py

Collects what the exported and quantized graphs cost against what they
are worth, into one table, and writes it into reports/results.md.

Accuracy is read from the per-sequence prediction files rather than
recomputed. Those files are the record the parity comparison already
produced, and taking accuracy from anywhere else would let this table
and that comparison disagree about the same model. Reading the predicted
label directly also sidesteps a trap the parity work turned up: a
quantized model can score two classes identically, and any accuracy
derived from ranks rather than from the predicted label counts those
ties in its own favour.

Size is reported raw and compressed. The browser downloads whatever the
server sends, and servers send these compressed, so the raw figure
overstates what the network actually carries.

Load time is separated from inference time because they answer different
questions. Inference happens on every frame; loading happens once, and
for a page a visitor waits on, once is the number that shows up as a
delay. Both are measured with one thread, matching how the existing
latency figure was taken, so the numbers here sit on the same scale as
that one rather than on a more flattering one.

A caution about what the timings mean: they are measured through this
runtime on this machine. The demo runs a different build on different
hardware, so these compare the variants against each other, which is
what the choice between them needs, and do not predict what a browser
will do.

Run: uv run python src\\benchmark_export.py --run-name abl_hands_aug
Requires the graphs, and the prediction files written by src\\onnx_parity.py.
Writes a section of reports/results.md.
"""
from __future__ import annotations

import argparse
import gzip
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

import error_analysis as ea
import evaluate
import export_onnx as ex
import latency as lat
import onnx_parity as parity
import preprocessing as pp
import quantize_onnx as qz
import report

RESULTS_MD = Path("reports") / "results.md"

# Variants in the order they should appear, float32 first so every other
# row is read as a departure from it.
DEFAULT_VARIANTS = (
    "fp32",
    "int8_dynamic",
    "int8_dynamic_conv_only",
    "int8_static",
    "int8_static_conv_only",
)

WARMUP_ITERATIONS = 30
MEASURED_ITERATIONS = 300
LOAD_ITERATIONS = 20

# Frames per second the live demo aims to sustain, shared with the
# existing latency measurement so both express the budget the same way.
TARGET_FPS = lat.TARGET_FPS


def single_thread_options() -> ort.SessionOptions:
    """One thread, to match how the existing latency figure was taken.

    Left alone the runtime spreads one inference over every core, which a
    browser will not do. One thread is both the conservative bound and
    the comparable one."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    # Errors only. This graph's shape metadata provokes a harmless warning
    # every time a runtime is built over it, and this script builds well
    # over a hundred of them; left on, the measurements are unreadable
    # among the repeats. The export and parity scripts build one runtime
    # each and do not suppress it, so the warning is still visible where
    # seeing it once is useful.
    options.log_severity_level = 3
    return options


def compressed_size(path: Path) -> int:
    """Bytes after compression, which is what a server sends."""
    return len(gzip.compress(path.read_bytes(), compresslevel=9))


def time_load(path: Path, iterations: int) -> dict[str, float]:
    """Milliseconds to construct a runtime over the graph.

    The file is in the operating system's cache after the first read, so
    this measures parsing and graph preparation rather than a cold start
    from disk. That is the honest thing to measure here anyway: a browser
    fetches the bytes over the network, a cost this machine cannot
    stand in for, and then pays this preparation cost on top."""
    options = single_thread_options()
    for _ in range(2):
        ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        timings.append((time.perf_counter() - start) * 1000.0)
    return lat.summarize(timings)


def time_inference(path: Path, iterations: int, warmup: int) -> dict[str, float]:
    """Milliseconds per single-sequence call, steady state.

    Batch size one: live inference classifies one sequence as it
    completes, and a figure taken on a large batch measures throughput,
    which is a different and much kinder quantity."""
    runtime = ort.InferenceSession(
        str(path), sess_options=single_thread_options(), providers=["CPUExecutionProvider"]
    )
    sample = {ex.INPUT_NAME: ex.sample_input(1)}
    outputs = [ex.OUTPUT_NAME]
    for _ in range(warmup):
        runtime.run(outputs, sample)
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        runtime.run(outputs, sample)
        timings.append((time.perf_counter() - start) * 1000.0)
    return lat.summarize(timings)


def score(predictions: pd.DataFrame) -> tuple[float, float]:
    """Top-1 and top-5 from the predicted labels themselves."""
    columns = [f"pred_{i}" for i in range(1, evaluate.TOP_K + 1)]
    top1 = float((predictions["true"] == predictions["pred_1"]).mean())
    top5 = float(predictions[columns].eq(predictions["true"], axis=0).any(axis=1).mean())
    return top1, top5


def measure(run_name: str, variant: str, split: str, args: argparse.Namespace) -> dict[str, object]:
    graph = parity.model_path(run_name, variant)
    predictions_file = parity.output_path(run_name, variant, split)
    if not graph.exists():
        raise SystemExit(f"Missing {graph}.")
    if not predictions_file.exists():
        raise SystemExit(
            f"Missing {predictions_file}. Score it first with:\n"
            f"  uv run python src\\onnx_parity.py --run-name {run_name} --variant {variant}"
        )
    predictions = pd.read_csv(predictions_file)
    top1, top5 = score(predictions)
    return {
        "variant": variant,
        "bytes": graph.stat().st_size,
        "gzipped": compressed_size(graph),
        "load": time_load(graph, args.load_iterations),
        "inference": time_inference(graph, args.iterations, args.warmup),
        "top1": top1,
        "top5": top5,
        "predictions": predictions,
        "provenance": qz.read_provenance(graph),
    }


def render(run_name: str, split: str, rows: list[dict[str, object]], args: argparse.Namespace) -> str:
    baseline = rows[0]
    baseline_predictions = baseline["predictions"]
    truth = baseline_predictions["true"].to_numpy()
    baseline_correct = (truth == baseline_predictions["pred_1"].to_numpy())
    budget = 1000.0 / TARGET_FPS

    out: list[str] = ["## Export and quantization", ""]
    out.append(
        f"`{run_name}` exported to ONNX and quantized, scored on the {split} split "
        f"({len(truth)} sequences). Sizes are of the graph file, which carries its own weights. "
        f"Timings are single-threaded on {lat.cpu_name()}: {args.iterations} calls at batch size 1 "
        f"after {args.warmup} discarded, and {args.load_iterations} runtime constructions."
    )
    out.append("")
    out.append(
        "| Variant | Size | Gzipped | Load | Inference (median) | Inference (p95) | "
        "Top-1 | Top-5 | Predictions changed | p |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        predicted = row["predictions"]["pred_1"].to_numpy()
        if row is baseline:
            changed_text, p_text = "—", "—"
        else:
            changed = int((predicted != baseline_predictions["pred_1"].to_numpy()).sum())
            _, _, p_value = ea.mcnemar(baseline_correct, predicted == truth)
            changed_text = f"{changed} ({changed / len(truth):.2%})"
            p_text = f"{p_value:.3g}"
        out.append(
            f"| `{row['variant']}` | {row['bytes'] / 1e6:.2f} MB | {row['gzipped'] / 1e6:.2f} MB "
            f"| {row['load']['median']:.1f} ms | {row['inference']['median']:.2f} ms "
            f"| {row['inference']['p95']:.2f} ms | {row['top1']:.2%} | {row['top5']:.2%} "
            f"| {changed_text} | {p_text} |"
        )
    out.append("")
    out.append(
        f"Predictions changed counts how many of the {len(truth)} individual top-1 predictions "
        f"differ from the float32 export, and p is a paired exact test over the ones each variant "
        f"gets right that the other does not. Two models can agree on accuracy while disagreeing "
        f"about which sequences they recognise, so the count is reported alongside the accuracy "
        f"rather than left to be inferred from it. Quantizing a fixed set of weights is "
        f"deterministic, so these differences are exact and repeatable rather than estimates "
        f"with run-to-run variation around them."
    )
    out.append("")
    out.append(
        f"At {TARGET_FPS} FPS the pipeline has {budget:.0f} ms per frame, and every variant here "
        f"uses under {max(r['inference']['median'] for r in rows) / budget * 100:.0f}% of it. "
        f"Inference cost is not what separates these options and quantization is not being asked "
        f"to buy speed; the columns that differ meaningfully are size and load time."
    )
    calibrated = [r for r in rows if "calibration_split" in r["provenance"]]
    if calibrated:
        sample = calibrated[0]["provenance"]
        out.append("")
        out.append(
            f"Statically quantized variants were calibrated on {sample['calibration_size']} "
            f"sequences drawn from the {sample['calibration_split']} split at seed "
            f"{sample['calibration_seed']}. Calibrating on the split a model is scored on would "
            f"fit the quantization to the evaluation and make every accuracy figure here a "
            f"measurement of the wrong thing."
        )
    out.append("")
    return "\n".join(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark exported and quantized graphs.")
    p.add_argument("--run-name", required=True)
    p.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--warmup", type=int, default=WARMUP_ITERATIONS)
    p.add_argument("--iterations", type=int, default=MEASURED_ITERATIONS)
    p.add_argument("--load-iterations", type=int, default=LOAD_ITERATIONS)
    p.add_argument("--report-path", type=Path, default=RESULTS_MD)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.variants[0] != "fp32":
        raise SystemExit(
            "The float32 export has to come first: every other row is reported as a departure "
            "from it, and the comparison is meaningless against a quantized baseline."
        )

    rows = []
    for variant in args.variants:
        row = measure(args.run_name, variant, args.split, args)
        rows.append(row)
        print(
            f"{variant:26} {row['bytes'] / 1e6:5.2f} MB  gz {row['gzipped'] / 1e6:5.2f} MB  "
            f"load {row['load']['median']:6.1f} ms  infer {row['inference']['median']:5.2f} ms  "
            f"top-1 {row['top1']:.2%}"
        )

    section = render(args.run_name, args.split, rows, args)
    report.update_section(args.report_path, "export", section)
    print(f"{args.report_path} export section updated")


if __name__ == "__main__":
    main()
