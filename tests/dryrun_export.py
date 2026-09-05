"""Dry run for the export, quantization, parity and benchmark scripts.

Builds a small synthetic dataset shaped like the real one, trains a
throwaway checkpoint on it, and drives the real scripts over it as
subprocesses. Nothing here touches the real dataset or the real
checkpoints.

The dataset builder and the subprocess runner are shared with the
evaluation harness rather than copied. Two builders that were meant to
produce the same fixtures would eventually stop doing so, and a test
fixture that has quietly drifted is worse than no fixture at all.

Every step reports the failing process's own output, and nothing reads a
file whose producing step did not succeed. A harness that hides the first
error and then dies on a missing file several steps later costs more time
than the bug it was meant to find.

The properties asserted are the ones a passing accuracy number would
hide: that the exported graph carries its own weights instead of a
sidecar, that it accepts the documented tensor shape rather than the
model's internal one, that quantization actually reduced the stored
weights to integers, that a graph and a prediction file describing
different numbers of sequences is refused rather than compared, and that
the accuracy a prediction file reports agrees with the predictions
written in the same file.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "tests"))

import dryrun_eval as fixture  # noqa: E402

CHECKS: list[tuple[str, bool]] = []
OUTPUT_TAIL = 2000


def check(name: str, condition: bool) -> bool:
    ok = bool(condition)
    CHECKS.append((name, ok))
    return ok


def step(root: Path, label: str, *args: str, expect_success: bool = True):
    """Run one script and record whether it did what was expected.

    On an unexpected outcome the process's own output is printed
    immediately. The alternative -- recording the failure and continuing
    -- means the first real error is buried under every consequence of
    it."""
    result = fixture.run(root, *args)
    succeeded = result.returncode == 0
    if not check(label, succeeded if expect_success else not succeeded):
        print(f"\n--- {label}: {' '.join(args)} exited {result.returncode}")
        if result.stdout:
            print(result.stdout[-OUTPUT_TAIL:])
        if result.stderr:
            print(result.stderr[-OUTPUT_TAIL:])
    return result


def weight_kinds(path: Path) -> tuple[int, int]:
    """Stored weight bytes, split into integer and floating point."""
    model = onnx.load(str(path))
    integer = floating = 0
    for tensor in model.graph.initializer:
        array = onnx.numpy_helper.to_array(tensor)
        if array.dtype in (np.int8, np.uint8, np.int32):
            integer += array.nbytes
        elif np.issubdtype(array.dtype, np.floating):
            floating += array.nbytes
    return integer, floating


def metadata(path: Path) -> dict[str, str]:
    return {e.key: e.value for e in onnx.load(str(path), load_external_data=False).metadata_props}


def main() -> int:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        (root / "reports").mkdir(parents=True, exist_ok=True)
        fixture.build_dataset(root)

        if not step(root, "cache build succeeds", "cache_dataset.py").returncode == 0:
            return report()
        if step(
            root, "training a checkpoint succeeds",
            "train.py", "--model", "cnn", "--landmarks", "hands",
            "--epochs", "2", "--run-name", "dry_cnn",
        ).returncode != 0:
            return report()
        if step(
            root, "checkpoint scoring succeeds",
            "evaluate.py", "--run-name", "dry_cnn", "--split", "test",
        ).returncode != 0:
            return report()

        # --- export ---------------------------------------------------------
        exported = step(root, "export succeeds", "export_onnx.py", "--run-name", "dry_cnn")
        graph = root / "models" / "dry_cnn.onnx"
        if exported.returncode != 0 or not check("exported graph exists", graph.exists()):
            return report()

        check("no weight sidecar left beside the graph",
              not (root / "models" / "dry_cnn.onnx.data").exists())

        model = onnx.load(str(graph), load_external_data=False)
        check("graph carries its own weights",
              not [t for t in model.graph.initializer
                   if t.data_location == onnx.TensorProto.EXTERNAL])

        shape = [d.dim_param or d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
        check("input is the documented landmark tensor", shape[1:] == [70, 50, 3])
        check("batch axis is symbolic", isinstance(shape[0], str))
        check("one input and one output",
              len(model.graph.input) == 1 and len(model.graph.output) == 1)
        check("batch normalization folded away",
              not any(n.op_type == "BatchNormalization" for n in model.graph.node))
        check("landmark subset applied inside the graph",
              any(n.op_type == "Slice" for n in model.graph.node))

        integer, floating = weight_kinds(graph)
        check("exported weights are floating point", floating > 0 and integer == 0)

        # --- quantization ----------------------------------------------------
        quantized: dict[str, Path] = {}
        for mode in ("dynamic", "static"):
            ok = step(root, f"{mode} quantization succeeds",
                      "quantize_onnx.py", "--run-name", "dry_cnn", "--mode", mode).returncode == 0
            path = root / "models" / f"dry_cnn_int8_{mode}.onnx"
            if ok and check(f"{mode} graph exists", path.exists()):
                quantized[mode] = path
                q_integer, q_floating = weight_kinds(path)
                check(f"{mode} weights are mostly integer", q_integer > q_floating)
                check(f"{mode} graph is smaller than the source",
                      path.stat().st_size < graph.stat().st_size)
                check(f"{mode} records how it was produced",
                      metadata(path).get("quantization_mode") == mode)

        if "static" in quantized:
            recorded = metadata(quantized["static"])
            check("static records its calibration split",
                  recorded.get("calibration_split") == "train")
            check("static never calibrates on the split it is scored on",
                  recorded.get("calibration_split") not in ("test", "val"))

        conv_ok = step(root, "conv-only quantization succeeds",
                       "quantize_onnx.py", "--run-name", "dry_cnn",
                       "--mode", "static", "--conv-only").returncode == 0
        conv_only = root / "models" / "dry_cnn_int8_static_conv_only.onnx"
        if conv_ok and conv_only.exists() and "static" in quantized:
            check("conv-only leaves the classifier head in float",
                  weight_kinds(conv_only)[1] > weight_kinds(quantized["static"])[1])

        step(root, "quantizing an absent graph fails loudly",
             "quantize_onnx.py", "--run-name", "missing_run", "--mode", "dynamic",
             expect_success=False)

        # --- parity ----------------------------------------------------------
        parity = step(root, "float32 parity succeeds",
                      "onnx_parity.py", "--run-name", "dry_cnn", "--variant", "fp32")
        if parity.returncode == 0:
            check("float32 parity preserves every prediction",
                  "the module on the CPU: 0 of" in parity.stdout)

            preds_file = root / "reports" / "preds_dry_cnn_onnx_fp32_test.csv"
            stored_file = root / "reports" / "preds_dry_cnn_test.csv"
            if check("parity wrote a prediction file", preds_file.exists()):
                preds = pd.read_csv(preds_file)
                stored = pd.read_csv(stored_file)
                check("one prediction row per stored row", len(preds) == len(stored))
                check("prediction columns match the checkpoint's own format",
                      list(preds.columns) == list(stored.columns))
                check("true labels line up row for row",
                      list(preds["true"]) == list(stored["true"]))

        for variant in ("int8_dynamic", "int8_static"):
            if variant.replace("int8_", "") not in quantized:
                continue
            result = step(root, f"{variant} parity reports rather than fails",
                          "onnx_parity.py", "--run-name", "dry_cnn", "--variant", variant)
            if result.returncode == 0:
                check(f"{variant} parity compares against the float32 export",
                      "against the float32 export" in result.stdout)

        # Rank and predicted label come from one ordering, so a rank of 1
        # and a predicted label other than the true one cannot coexist.
        # Quantization is what makes ties real, and the two definitions
        # disagree on exactly those rows.
        for variant in ("fp32", "int8_static"):
            path = root / "reports" / f"preds_dry_cnn_onnx_{variant}_test.csv"
            if path.exists():
                d = pd.read_csv(path)
                check(f"{variant} rank agrees with the predicted label",
                      bool(((d["true_rank"] == 1) == (d["true"] == d["pred_1"])).all()))

        step(root, "parity on an absent graph fails loudly",
             "onnx_parity.py", "--run-name", "dry_cnn", "--variant", "nonexistent",
             expect_success=False)

        stored_file = root / "reports" / "preds_dry_cnn_test.csv"
        if stored_file.exists():
            full = pd.read_csv(stored_file)
            full.iloc[:-1].to_csv(stored_file, index=False)
            truncated = step(root, "a row-count mismatch is refused, not compared",
                             "onnx_parity.py", "--run-name", "dry_cnn", "--variant", "fp32",
                             expect_success=False)
            check("the mismatch is reported as a data disagreement",
                  "different data" in (truncated.stdout + truncated.stderr))
            full.to_csv(stored_file, index=False)

        # --- benchmark --------------------------------------------------------
        available = ["fp32"] + [f"int8_{m}" for m in quantized]
        bench = step(root, "benchmark succeeds",
                     "benchmark_export.py", "--run-name", "dry_cnn",
                     "--variants", *available,
                     "--iterations", "5", "--warmup", "2", "--load-iterations", "2")
        if bench.returncode == 0:
            results = (root / "reports" / "results.md").read_text(encoding="utf-8")
            check("benchmark writes its own marked section", "<!-- EXPORT_START -->" in results)
            check("benchmark leaves other sections intact", "<!-- RUNS_START -->" in results)
            check("benchmark table names every variant",
                  all(f"`{v}`" in results for v in available))

        step(root, "benchmark refuses a quantized baseline",
             "benchmark_export.py", "--run-name", "dry_cnn",
             "--variants", "int8_static", "fp32",
             "--iterations", "5", "--warmup", "2", "--load-iterations", "2",
             expect_success=False)

        step(root, "benchmark fails on a variant that was never scored",
             "benchmark_export.py", "--run-name", "dry_cnn",
             "--variants", "fp32", "int8_static_conv_only",
             "--iterations", "5", "--warmup", "2", "--load-iterations", "2",
             expect_success=False)

    return report()


def report() -> int:
    passed = sum(1 for _, ok in CHECKS if ok)
    for name, ok in CHECKS:
        if not ok:
            print(f"FAIL  {name}")
    print(f"\n{passed}/{len(CHECKS)} checks passed")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
