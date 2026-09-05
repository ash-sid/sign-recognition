"""
src/export_onnx.py

Exports a trained checkpoint to ONNX, so the classifier can run outside
Python without a second implementation of the model.

The exported graph takes the documented input tensor, not the flat
per-frame vector the PyTorch module consumes. Selecting the landmark
subset and flattening it are properties of a particular trained model,
not of preprocessing, so they belong inside the artifact that carries
the weights. A consumer then implements the documented tensor and
nothing else, and cannot pick the wrong landmark subset for the
checkpoint it happens to be running.

Batch is the only dynamic axis. Frame count and landmark count are fixed
by the preprocessing contract and baking them in lets the runtime plan
shapes once instead of on every call.

The export writes its weights to a sidecar file by default, leaving a
graph of a few tens of kilobytes that only works next to its data file.
That is a poor deployment artifact -- it is two files to serve instead of
one -- and a worse measurement, because the size of the graph file alone
says nothing about the size of the model. The weights are folded back in
here and the result is re-read from disk to confirm it, rather than
trusting an exporter default that can change between versions.

Batch normalization does not survive as an operator: the decomposition
the exporter runs folds each one into the convolution it follows. The
graph therefore has fewer operators than the module has layers while
computing the same function. If a BatchNormalization node does appear,
that folding did not happen and quantization has a different graph to
work on than this expects.

Verification here is deliberately shallow -- shapes, a valid graph, and
close outputs on random input. It catches a broken export immediately
and cheaply. It is not the parity check: agreeing on random noise says
much less than agreeing on every stored prediction, which is what
src\\onnx_parity.py measures against the real split.

Run: uv run python src\\export_onnx.py --run-name abl_hands_aug
Requires models/<run-name>.pt. Writes models/<run-name>.onnx.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

import evaluate
import preprocessing as pp
import train as tr

MODELS = Path("models")

# The opset the exporter emits. Requesting a lower one asks for a
# down-conversion that fails on this graph and silently falls back,
# leaving the model at the exporter's own version anyway. Naming that
# version here means the export either produces it or fails, instead of
# producing something other than what was asked for. Every operator in
# this graph is long-established; the only thing this number constrains
# is the minimum runtime version that will load the file.
OPSET = 18

INPUT_NAME = "landmarks"
OUTPUT_NAME = "logits"
BATCH_AXIS = "batch"

# Batch size used for the shape check and the random-input comparison.
# More than one so that a graph which silently collapses the batch axis
# fails here rather than at parity time.
CHECK_BATCH = 4

# Smallest plausible size, in bytes, for a self-contained graph carrying
# this model's weights. Well under the real figure -- this is here to
# catch a graph whose weights live somewhere else, not to pin a size.
MIN_SELF_CONTAINED_BYTES = 500_000

# Random-input agreement threshold, relative to the magnitude of the
# reference logits. Export reorders and fuses arithmetic, so bit-identical
# outputs are not expected; a difference this size is float noise, and
# anything larger is a real change in what the graph computes. Expressed
# as a ratio because an absolute difference is not interpretable without
# knowing the scale of the values it is a difference between.
LOGIT_RTOL = 1e-5


class ContractInput(nn.Module):
    """Wraps a trained classifier so the graph accepts `(B, T, 50, 3)`.

    The wrapped module expects `(B, T, L*3)` over its own landmark subset.
    This slices the documented tensor down to that subset and flattens the
    coordinate axis, which is exactly what the training and evaluation
    paths do before every forward pass."""

    def __init__(self, model: nn.Module, landmarks: str):
        super().__init__()
        self.model = model
        self.hands_only = landmarks != "hands_pose"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.hands_only:
            x = x[:, :, : tr.HANDS_END, :]
        return self.model(x.flatten(start_dim=2))


def sample_input(batch: int, seed: int = 0) -> np.ndarray:
    """A contract-shaped batch of random landmarks.

    Values are irrelevant to shape and operator checks, and using random
    input keeps this script independent of the cache, so a broken export
    is caught without a multi-gigabyte load."""
    return (
        np.random.default_rng(seed)
        .standard_normal((batch, pp.TARGET_LEN, pp.NUM_LANDMARKS, pp.NUM_COORDS))
        .astype(np.float32)
    )


def export(model: nn.Module, path: Path, batch: int) -> None:
    torch.onnx.export(
        model,
        torch.from_numpy(sample_input(batch)),
        str(path),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: BATCH_AXIS}, OUTPUT_NAME: {0: BATCH_AXIS}},
        opset_version=OPSET,
    )


def inline_weights(path: Path) -> int:
    """Fold sidecar weight data back into the graph file.

    Reading with external data resolved and writing without it produces a
    single self-contained file. The sidecar is then removed, because
    leaving a stale copy of the weights beside the model is an invitation
    to ship the wrong one. Returns the number of initializers that were
    external, so the caller can report whether this did anything."""
    model = onnx.load(str(path))
    external = sum(1 for t in model.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL)
    onnx.save_model(model, str(path), save_as_external_data=False)
    sidecar = path.with_suffix(path.suffix + ".data")
    sidecar.unlink(missing_ok=True)
    return external


def check_self_contained(path: Path, min_bytes: int = MIN_SELF_CONTAINED_BYTES) -> None:
    """Re-read the saved file and confirm it carries its own weights.

    Deliberately re-reads from disk rather than inspecting the in-memory
    model: the artifact that gets shipped and measured is the file, and
    the file is the thing that was wrong before this check existed.

    min_bytes is a floor, not a size assertion. A quantized graph is
    legitimately much smaller than this one, so callers holding a smaller
    artifact pass their own."""
    model = onnx.load(str(path), load_external_data=False)
    external = [t.name for t in model.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL]
    if external:
        raise SystemExit(
            f"{path} still refers to weights it does not contain: {', '.join(external)}. "
            f"The graph will not load without its sidecar file."
        )
    if path.with_suffix(path.suffix + ".data").exists():
        raise SystemExit(f"A stale weight sidecar remains beside {path}.")
    size = path.stat().st_size
    if size < min_bytes:
        raise SystemExit(
            f"{path} is {size} bytes, too small to hold this model's weights. "
            f"They are stored somewhere other than the graph file."
        )


def graph_opset(path: Path) -> int:
    model = onnx.load(str(path), load_external_data=False)
    for entry in model.opset_import:
        if entry.domain in ("", "ai.onnx"):
            return entry.version
    raise SystemExit(f"{path} declares no default-domain opset.")


def op_histogram(path: Path) -> Counter:
    """Counts operators in the exported graph.

    Printed because it explains the file size rather than leaving it as a
    bare number, and because it is the quickest way to see whether an
    export or a later quantization pass did what it claimed to."""
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    return Counter(node.op_type for node in model.graph.node)


def torch_logits(model: nn.Module, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy(x)).numpy()


def onnx_logits(path: Path, x: np.ndarray) -> np.ndarray:
    runtime = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return runtime.run([OUTPUT_NAME], {INPUT_NAME: x})[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a trained checkpoint to ONNX.")
    p.add_argument("--run-name", required=True, help="checkpoint to export, from models/<run-name>.pt")
    p.add_argument("--batch", type=int, default=CHECK_BATCH, help="batch size used for the export trace")
    return p.parse_args()


def main() -> None:
    # The exporter reports progress with characters Windows' default
    # console encoding cannot represent. Encoding errors raise rather than
    # degrade, so redirecting this script's output to a file, or capturing
    # it from another process, crashes inside a print statement that was
    # announcing success. Replacing unrepresentable characters keeps the
    # script usable in both cases without changing what it does.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    args = parse_args()
    device = torch.device("cpu")

    model, ckpt, config = evaluate.load_checkpoint(args.run_name, device)
    wrapped = ContractInput(model, config.landmarks).eval()
    parameters = sum(p.numel() for p in model.parameters())

    print(f"run: {args.run_name}  landmarks: {config.landmarks}  {parameters / 1e6:.2f}M parameters")
    print(f"{tr.describe_architecture(config)}")

    # The wrapper has to reproduce the shape the checkpoint was trained
    # on. Exporting a graph that feeds the model a differently-sized frame
    # would produce a working artifact for the wrong model.
    x = sample_input(args.batch)
    per_frame = (tr.HANDS_END if config.landmarks != "hands_pose" else pp.NUM_LANDMARKS) * pp.NUM_COORDS
    if per_frame != ckpt["input_size"]:
        raise SystemExit(
            f"Wrapper builds a {per_frame}-value input frame but the checkpoint was trained on "
            f"{ckpt['input_size']}. The landmark selection does not match the run."
        )

    MODELS.mkdir(exist_ok=True)
    out_path = MODELS / f"{args.run_name}.onnx"
    export(wrapped, out_path, args.batch)

    external = inline_weights(out_path)
    check_self_contained(out_path)
    print(
        f"Exported to {out_path} ({out_path.stat().st_size / 1e6:.2f} MB, "
        f"{external} weight tensors folded in from a sidecar)"
    )

    delivered = graph_opset(out_path)
    if delivered != OPSET:
        raise SystemExit(
            f"Exported graph declares opset {delivered}, not the {OPSET} this script targets. "
            f"The runtime requirement is not what the export claims."
        )
    print(f"Opset {delivered}")

    histogram = op_histogram(out_path)
    print("Graph operators: " + ", ".join(f"{op} x{n}" for op, n in sorted(histogram.items())))
    if "BatchNormalization" in histogram:
        raise SystemExit(
            "Batch normalization survived as an operator instead of folding into the "
            "convolutions. The graph is not the one downstream steps expect."
        )

    reference = torch_logits(wrapped, x)
    exported = onnx_logits(out_path, x)

    if exported.shape != reference.shape:
        raise SystemExit(
            f"Exported graph returned {exported.shape} where the module returns {reference.shape}."
        )
    if exported.shape != (args.batch, ckpt["num_classes"]):
        raise SystemExit(
            f"Exported graph returned {exported.shape}, expected "
            f"{(args.batch, ckpt['num_classes'])} for this checkpoint."
        )

    max_diff = float(np.abs(exported - reference).max())
    scale = float(np.abs(reference).max())
    relative = max_diff / scale if scale else 0.0
    agree = int((exported.argmax(axis=1) == reference.argmax(axis=1)).sum())
    print(
        f"Logit difference on random input: {max_diff:.3e} absolute, {relative:.3e} relative "
        f"to a maximum logit magnitude of {scale:.2f}"
    )
    print(f"Top-1 agreement on random input: {agree}/{args.batch}")

    if relative > LOGIT_RTOL:
        raise SystemExit(
            f"Exported graph disagrees with the module by {relative:.3e} relative, above the "
            f"{LOGIT_RTOL:.0e} tolerance. The export changed what the model computes; this is "
            f"not float noise."
        )

    # A single dynamic axis, exercised at a size the trace never saw, so a
    # graph that hard-coded the trace's batch fails here.
    other = sample_input(CHECK_BATCH + 3, seed=1)
    if onnx_logits(out_path, other).shape[0] != CHECK_BATCH + 3:
        raise SystemExit("Exported graph does not accept a batch size other than the one it was traced at.")
    print(f"Batch axis is dynamic (checked at {args.batch} and {CHECK_BATCH + 3})")


if __name__ == "__main__":
    main()
