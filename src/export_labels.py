"""
src/export_labels.py

Writes a trained checkpoint's class ordering to models/<run-name>_labels.json,
so a consumer outside Python can turn the exported graph's logits into sign
names.

The ordering is not incidental. The classifier's output column i means
whatever the label encoder called class i at training time, and nothing in
the ONNX graph records that. Reconstructing it by sorting the vocabulary is
a guess that happens to be right for this encoder and would fail silently
for one that isn't, mislabelling every prediction while every shape and
every accuracy figure stayed correct. Reading it from the checkpoint that
produced the weights removes the guess.

Written beside the graph rather than into it. Graph metadata would tie the
two together more tightly, but it also changes the bytes of an artifact
whose size is a published figure and whose per-prediction parity has already
been measured. A sidecar keeps that file untouched.

Run: uv run python src\\export_labels.py --run-name abl_hands_aug
Requires models/<run-name>.pt. Writes models/<run-name>_labels.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

MODELS = Path("models")

# Number of classes the exported graph emits. A checkpoint carrying a
# different count would produce a label list that silently misaligns with
# the logits it is meant to name.
EXPECTED_CLASSES = 250


def load_labels(run_name: str) -> list[str]:
    path = MODELS / f"{run_name}.pt"
    if not path.exists():
        raise SystemExit(f"Missing checkpoint {path}.")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "label_classes" not in ckpt:
        raise SystemExit(f"{path} has no label_classes entry.")
    labels = list(ckpt["label_classes"])
    if ckpt.get("num_classes") not in (None, len(labels)):
        raise SystemExit(
            f"{path} records {ckpt['num_classes']} classes but lists {len(labels)} labels."
        )
    return labels


def check(labels: list[str]) -> None:
    if len(labels) != EXPECTED_CLASSES:
        raise SystemExit(
            f"Expected {EXPECTED_CLASSES} labels, got {len(labels)}. The graph's output "
            "width and this list have to agree."
        )
    if len(set(labels)) != len(labels):
        raise SystemExit("Label list contains duplicates; positions would be ambiguous.")
    if any(not isinstance(name, str) or not name for name in labels):
        raise SystemExit("Label list contains an empty or non-string entry.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    labels = load_labels(args.run_name)
    check(labels)

    out_path = MODELS / f"{args.run_name}_labels.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=1)
        f.write("\n")

    print(f"Wrote {out_path} ({len(labels)} labels)")
    print(f"  first: {labels[0]}   last: {labels[-1]}")


if __name__ == "__main__":
    main()
