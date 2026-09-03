"""
src/latency.py

Measures how long the trained model takes to classify one sequence on the
CPU, and writes the result into reports/results.md.

The browser demo has a per-frame budget, so inference cost is a
requirement rather than a curiosity, and a guess is not a measurement.
Three things make the difference between a number and a useful number:

  Batch size one    Live inference classifies a single sequence as it
                    completes. Throughput measured on a large batch is a
                    different quantity and a much flattering one.
  Warmup discarded  The first calls pay for lazy allocation and kernel
                    selection. Including them measures start-up, not
                    steady state.
  Percentiles       Latency distributions have a long right tail. A mean
                    hides the occasional slow call, which is exactly the
                    thing a frame budget cares about.

Thread count is reported both ways. Left alone, PyTorch spreads one
inference across every core on the machine, which a browser will not do;
pinned to one thread it is a closer proxy for the deployment target and a
conservative bound.

This is a PyTorch-on-CPU figure. It is a reference point, not the number
the browser will see -- that depends on the exported runtime, and it is
only one part of a live pipeline in which landmark extraction is expected
to dominate.

Run: uv run python src\\latency.py --run-name abl_hands_aug
Requires models/<run-name>.pt.
Writes a section of reports/results.md.
"""
from __future__ import annotations

import argparse
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

import evaluate
import preprocessing as pp
import report
import train as tr

RESULTS_MD = Path("reports") / "results.md"

WARMUP_ITERATIONS = 30
MEASURED_ITERATIONS = 300

# Frames per second the live demo aims to sustain, and the resulting budget
# for everything that happens per frame.
TARGET_FPS = 20


def cpu_name() -> str:
    """Best-effort processor description, for a number that is only
    meaningful alongside the machine that produced it."""
    name = platform.processor() or ""
    try:
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return name or platform.machine() or "unknown CPU"


def time_forward(
    model: torch.nn.Module, sample: torch.Tensor, warmup: int, iterations: int
) -> list[float]:
    """Milliseconds per single-sequence forward pass, steady state.

    perf_counter rather than time: it is monotonic and has the resolution
    to measure a single millisecond-scale call without the timer itself
    being a meaningful share of the result."""
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        timings = []
        for _ in range(iterations):
            start = time.perf_counter()
            model(sample)
            timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def summarize(timings: list[float]) -> dict[str, float]:
    ordered = sorted(timings)
    return {
        "median": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, int(round(0.95 * len(ordered))) - 1)],
        "p99": ordered[min(len(ordered) - 1, int(round(0.99 * len(ordered))) - 1)],
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def measure(
    model: torch.nn.Module, sample: torch.Tensor, threads: int | None, args: argparse.Namespace
) -> dict[str, float]:
    if threads is not None:
        torch.set_num_threads(threads)
    timings = time_forward(model, sample, args.warmup, args.iterations)
    result = summarize(timings)
    result["threads"] = float(torch.get_num_threads())
    return result


def render(
    run_name: str,
    config: argparse.Namespace,
    parameters: int,
    input_shape: tuple[int, ...],
    single: dict[str, float],
    default: dict[str, float],
    iterations: int,
) -> str:
    budget = 1000.0 / TARGET_FPS
    out: list[str] = ["## Inference latency", ""]
    out.append(
        f"`{run_name}` classifying one sequence on the CPU, batch size 1, "
        f"input {tuple(input_shape)}, {parameters / 1e6:.2f}M parameters. "
        f"{iterations} timed calls after {WARMUP_ITERATIONS} discarded warmup calls, "
        f"on {cpu_name()}."
    )
    out.append("")
    out.append("| Threads | Median | p95 | p99 | Min | Max |")
    out.append("|---|---|---|---|---|---|")
    for label, result in (("1", single), (f"{int(default['threads'])} (default)", default)):
        out.append(
            f"| {label} | {result['median']:.2f} ms | {result['p95']:.2f} ms "
            f"| {result['p99']:.2f} ms | {result['min']:.2f} ms | {result['max']:.2f} ms |"
        )
    out.append("")
    out.append(
        f"Single-threaded is the number to plan against. It is the conservative bound "
        f"and the closer analogue of a browser, which will not hand one inference every "
        f"core on the machine."
    )
    out.append("")
    out.append(
        f"At {TARGET_FPS} FPS the whole pipeline has {budget:.0f} ms per frame. The model "
        f"forward pass uses {single['median'] / budget * 100:.1f}% of that single-threaded "
        f"at the median. The remainder covers landmark extraction, which is expected to "
        f"dominate, so this figure shows the classifier is not the constraint rather than "
        f"showing the pipeline will hold."
    )
    out.append("")
    out.append(
        "Measured through PyTorch on the CPU. The browser will run a different runtime "
        "on different hardware, so treat this as a reference point for the model's cost, "
        "not as a prediction of what the demo will do."
    )
    out.append("")
    return "\n".join(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure single-sequence CPU inference latency.")
    p.add_argument("--run-name", required=True)
    p.add_argument("--warmup", type=int, default=WARMUP_ITERATIONS)
    p.add_argument("--iterations", type=int, default=MEASURED_ITERATIONS)
    p.add_argument("--report-path", type=Path, default=RESULTS_MD)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    model, ckpt, config = evaluate.load_checkpoint(args.run_name, device)
    parameters = sum(p.numel() for p in model.parameters())

    # Random input: the forward pass cost depends on the shape, not the
    # values, so this avoids loading a multi-gigabyte cache to time a
    # single call.
    full = (
        np.random.default_rng(0)
        .standard_normal((1, pp.TARGET_LEN, pp.NUM_LANDMARKS, pp.NUM_COORDS))
        .astype(np.float32)
    )
    x = tr.select_landmarks(full, config.landmarks)
    # The checkpoint records the per-frame vector length the run was
    # trained with. Timing a shape the model was not built for would
    # produce a plausible number for the wrong model.
    per_frame = x.shape[2] * x.shape[3]
    if per_frame != ckpt["input_size"]:
        raise SystemExit(
            f"Built a {per_frame}-value input frame but the checkpoint was trained on "
            f"{ckpt['input_size']}. The landmark selection does not match the run."
        )
    sample = tr.to_model_input(torch.from_numpy(x), shuffle=False)

    default_threads = torch.get_num_threads()
    print(f"run: {args.run_name}  {parameters / 1e6:.2f}M parameters  input {tuple(sample.shape)}")

    single = measure(model, sample, 1, args)
    print(f"  1 thread : median {single['median']:.2f} ms  p95 {single['p95']:.2f} ms")
    default = measure(model, sample, default_threads, args)
    print(
        f"  {int(default['threads'])} threads: median {default['median']:.2f} ms  "
        f"p95 {default['p95']:.2f} ms"
    )

    section = render(
        args.run_name, config, parameters, tuple(sample.shape), single, default, args.iterations
    )
    report.update_section(args.report_path, "latency", section)
    print(f"{args.report_path} latency section updated")


if __name__ == "__main__":
    main()
