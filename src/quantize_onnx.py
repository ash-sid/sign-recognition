"""
src/quantize_onnx.py

Quantizes an exported graph to INT8, in the two modes the runtime offers.

Dynamic quantization stores weights as INT8 and computes activation
scales at run time. It needs no data. Static quantization also fixes the
activation scales ahead of time, which requires running representative
inputs through the graph first and makes the choice of those inputs part
of the result.

Calibration samples the training split. The test split is disqualified
outright: fitting quantization scales to the data the model is then
scored on would make every accuracy number downstream a measurement of
the wrong thing. Validation is defensible but already carries the weight
of model selection, so training data -- the split with nothing else
riding on it -- is used instead, subsampled at a fixed seed so the
result is reproducible.

Two settings here move the numbers, not just the file:

  reduce_range  Weights drop to 7 bits. On x86 without VNNI instructions
                the 8-bit accumulation path can saturate, and narrowing
                the weights avoids it at some cost in precision. This is
                a property of the machine doing the measuring, which need
                not be the machine that eventually runs the model, so the
                figure it produces is hardware-dependent and should be
                reported as such.
  per_channel   One scale per convolution filter rather than one for the
                whole weight tensor. Filters in the same layer can differ
                enough in magnitude that a shared scale wastes most of
                the INT8 range on the largest of them.

What each mode actually quantizes is printed rather than assumed. The
operator histogram is the only honest answer to why a quantized file is
the size it is, and the two modes do not cover the same operators.

Run: uv run python src\\quantize_onnx.py --run-name abl_hands_aug --mode dynamic
     uv run python src\\quantize_onnx.py --run-name abl_hands_aug --mode static
Requires models/<run-name>.onnx and, for static, the cache built by
src\\cache_dataset.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

import cache_dataset as cache
import evaluate
import export_onnx as ex
import train as tr

MODELS = Path("models")

CALIBRATION_SPLIT = "train"
CALIBRATION_SIZE = 512
CALIBRATION_SEED = 0
CALIBRATION_BATCH = 32

# Floor for a quantized graph that carries its own weights. Set well
# below the smallest plausible INT8 result, to catch a graph whose
# weights went to a sidecar rather than to pin a size.
MIN_QUANTIZED_BYTES = 200_000

# Integer weight types. Whether quantization actually happened is a
# question about how the weights are stored, not about which operators
# the graph contains: the two output formats express the same quantized
# model with completely different operator names, and one of them leaves
# the convolutions looking like float operators for the runtime to fuse
# later. Checking the stored dtype answers the question directly and
# gives the same answer for both.
INTEGER_DTYPES = (np.int8, np.uint8, np.int32)


class SubsampleReader(CalibrationDataReader):
    """Feeds a fixed subsample of a split through the graph once.

    The runtime consumes this exactly once and stops at the first None,
    so the batches are materialised up front rather than regenerated --
    a reader that could yield different data on a second pass would make
    the calibration silently unreproducible."""

    def __init__(self, batches: list[dict[str, np.ndarray]]):
        self._batches = iter(batches)

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._batches, None)


def calibration_batches(
    config: argparse.Namespace, split: str, size: int, seed: int, batch: int
) -> tuple[list[dict[str, np.ndarray]], int]:
    """A seeded subsample of one split, in contract-shaped batches.

    The landmark subset is deliberately not applied: the exported graph
    slices it internally, so feeding pre-sliced data would calibrate a
    different input than the one the model is given in production."""
    variant = cache.variant_name(not config.no_normalize)
    x, _ = tr.load_cached_split(variant, split)
    available = len(x)
    count = min(size, available)
    index = np.random.default_rng(seed).choice(available, size=count, replace=False)
    index.sort()  # read the memory-mapped array forwards rather than at random
    chosen = x[index].astype(np.float32)
    return (
        [{ex.INPUT_NAME: chosen[i : i + batch]} for i in range(0, count, batch)],
        available,
    )


def weight_bytes_by_kind(path: Path) -> tuple[int, int]:
    """Bytes of stored weights, split into integer and floating point.

    Quantization leaves scales, zero points and often biases in float, so
    a quantized graph is a mixture rather than wholly integer. What
    distinguishes it from an unquantized one is that the bulk of the
    weight data is integer."""
    model = onnx.load(str(path))
    integer = floating = 0
    for tensor in model.graph.initializer:
        array = numpy_helper.to_array(tensor)
        if array.dtype in INTEGER_DTYPES:
            integer += array.nbytes
        elif np.issubdtype(array.dtype, np.floating):
            floating += array.nbytes
    return integer, floating


def record_provenance(path: Path, provenance: dict[str, str]) -> None:
    """Write how this file was produced into the file itself.

    A quantized graph is otherwise indistinguishable from one calibrated
    on different data, and the difference matters enough that it should
    not depend on a filename or a note in a report staying attached to
    it."""
    model = onnx.load(str(path))
    merged = {entry.key: entry.value for entry in model.metadata_props}
    merged.update({k: str(v) for k, v in provenance.items()})
    del model.metadata_props[:]
    for key, value in sorted(merged.items()):
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save_model(model, str(path), save_as_external_data=False)


def read_provenance(path: Path) -> dict[str, str]:
    model = onnx.load(str(path), load_external_data=False)
    return {entry.key: entry.value for entry in model.metadata_props}


def output_name(run_name: str, args: argparse.Namespace) -> Path:
    parts = [run_name, "int8", args.mode]
    if args.mode == "static" and not args.per_channel:
        parts.append("per_tensor")
    if not args.reduce_range:
        parts.append("full_range")
    return MODELS / ("_".join(parts) + ".onnx")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quantize an exported graph to INT8.")
    p.add_argument("--run-name", required=True)
    p.add_argument("--mode", choices=["dynamic", "static"], required=True)
    p.add_argument("--calibration-split", choices=["train", "val"], default=CALIBRATION_SPLIT)
    p.add_argument("--calibration-size", type=int, default=CALIBRATION_SIZE)
    p.add_argument("--calibration-seed", type=int, default=CALIBRATION_SEED)
    p.add_argument("--calibration-batch", type=int, default=CALIBRATION_BATCH)
    p.add_argument(
        "--per-tensor",
        dest="per_channel",
        action="store_false",
        help="one weight scale per tensor instead of one per convolution filter",
    )
    p.add_argument(
        "--full-range",
        dest="reduce_range",
        action="store_false",
        help="keep weights at 8 bits; may saturate on x86 without VNNI",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    source = MODELS / f"{args.run_name}.onnx"
    if not source.exists():
        raise SystemExit(
            f"Missing {source}. Export it first with:\n"
            f"  uv run python src\\export_onnx.py --run-name {args.run_name}"
        )
    ex.check_self_contained(source)

    out_path = output_name(args.run_name, args)
    provenance = {
        "source_model": source.name,
        "quantization_mode": args.mode,
        "per_channel": str(args.per_channel),
        "reduce_range": str(args.reduce_range),
    }

    # Shape inference and folding before quantization. The quantizer warns
    # when this is skipped, and the graph it produces without it can quantize
    # fewer operators than it otherwise would.
    prepared = MODELS / f"{args.run_name}.prepared.onnx"
    quant_pre_process(str(source), str(prepared), skip_symbolic_shape=False)

    try:
        if args.mode == "dynamic":
            quantize_dynamic(
                str(prepared),
                str(out_path),
                weight_type=QuantType.QInt8,
                per_channel=args.per_channel,
                reduce_range=args.reduce_range,
            )
        else:
            _, _, config = evaluate.load_checkpoint(args.run_name, torch.device("cpu"))
            batches, available = calibration_batches(
                config,
                args.calibration_split,
                args.calibration_size,
                args.calibration_seed,
                args.calibration_batch,
            )
            used = sum(len(b[ex.INPUT_NAME]) for b in batches)
            print(
                f"Calibrating on {used} of {available} {args.calibration_split} sequences "
                f"(seed {args.calibration_seed}, {len(batches)} batches)"
            )
            provenance.update(
                calibration_split=args.calibration_split,
                calibration_size=used,
                calibration_seed=args.calibration_seed,
            )
            quantize_static(
                str(prepared),
                str(out_path),
                SubsampleReader(batches),
                quant_format=QuantFormat.QDQ,
                # Unsigned activations against signed weights is the
                # combination the CPU backend has fast kernels for.
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                per_channel=args.per_channel,
                reduce_range=args.reduce_range,
            )
    finally:
        prepared.unlink(missing_ok=True)

    record_provenance(out_path, provenance)
    # An INT8 graph is legitimately far smaller than the source, so the
    # floor is lowered rather than inherited.
    ex.check_self_contained(out_path, min_bytes=MIN_QUANTIZED_BYTES)

    fp32_size = source.stat().st_size
    size = out_path.stat().st_size
    print(f"Wrote {out_path} ({size / 1e6:.2f} MB, {size / fp32_size:.1%} of the {fp32_size / 1e6:.2f} MB source)")

    histogram = ex.op_histogram(out_path)
    print("Graph operators: " + ", ".join(f"{op} x{n}" for op, n in sorted(histogram.items())))

    integer_bytes, float_bytes = weight_bytes_by_kind(out_path)
    total = integer_bytes + float_bytes
    print(
        f"Stored weights: {integer_bytes / 1e6:.2f} MB integer, {float_bytes / 1e6:.2f} MB float "
        f"({integer_bytes / total:.1%} integer)"
    )
    if integer_bytes <= float_bytes:
        raise SystemExit(
            f"Most of {out_path} is still stored in floating point ({float_bytes / total:.1%}). "
            f"The pass produced a file but did not quantize the weights; the size and accuracy "
            f"of this artifact would describe nothing."
        )
    print("Recorded provenance: " + json.dumps(read_provenance(out_path), sort_keys=True))


if __name__ == "__main__":
    main()
