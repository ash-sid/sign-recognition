# Sign Recognition

Isolated sign language recognition from webcam. MediaPipe landmark sequences
→ temporal model (PyTorch) → ONNX → runs live in-browser.

**Not ASL translation.** This recognizes individual signs in isolation. Grammar,
spatial referencing, and non-manual markers (facial grammar) are out of scope —
which is why face landmarks are deliberately excluded from the model input.

## Status

The model is trained, evaluated and exported. The browser app is next; there is no
live demo link yet.

## Results

Signer-independent split over 21 signers and 250 signs: 15 signers for training,
3 for validation, 3 held out for test (14,015 sequences). No signer appears in more
than one split.

| Model | Test top-1 | Test top-5 |
|---|---|---|
| Majority class | 0.43% | — |
| Logistic regression, mean-pooled landmarks | 26.71% | 52.80% |
| 1D-CNN, first working version | 48.69% | 75.78% |
| **1D-CNN, hands-only input, mirror augmentation** | **61.03%** | **82.59%** |

The last two rows were trained under different schedules and are not directly
comparable as a before-and-after; the gain came from a combination of protocol changes
and input/augmentation ablations, measured separately. Full breakdown in
[`reports/results.md`](reports/results.md), with every run's configuration in
[`reports/ablations.csv`](reports/ablations.csv).

A transformer encoder was tried and lost to the convolution by 7.1pp at matched
settings, overfitting on 67k sequences across 250 classes. It is recorded as a measured
negative result rather than dropped.

## What the evaluation controls for

- **A measured noise floor.** Accuracy differences below 0.7pp are not interpretable,
  established from three seeds of the reference configuration. Single-seed comparisons
  are not treated as evidence.
- **Per-prediction comparison, not just accuracy.** Two models can agree on a percentage
  while disagreeing about which sequences they get right. Every comparison here reports
  how many individual predictions moved, with a paired significance test. In one case a
  0.01pp accuracy difference concealed 4.6% prediction-level change.
- **A documented flaw the evaluation found in itself.** All three test signers happen to
  use the same hand slot, against a training majority using the other. This inflates the
  measured benefit of mirror augmentation and makes the test set harder than a
  representative one. It is not corrected, because redrawing the split would invalidate
  every recorded comparison — the tradeoff is written up rather than hidden.
- **Input quality is separated from model quality.** Roughly 10pp of the gap to perfect
  accuracy is hand-tracking failure in the source data, not model capacity: sequences
  where the hand is tracked in almost every frame score 71.2%, those missing more than
  half score 50.2%.

## Deployment

`models/abl_hands_aug.onnx` — float32, opset 18, self-contained, 1.59 MB (1.46 MB
gzipped). Takes the full `(batch, 70, 50, 3)` landmark tensor; the landmark-subset slice
and flatten are inside the graph, so a consumer implements
[`reports/contract.md`](reports/contract.md) and nothing more.

The export reproduces all 14,015 stored predictions exactly on the same device.

INT8 quantization was measured across four variants and rejected: it saves about 1.1 MB
compressed but moves 3.77% of individual predictions, and inference cost was never the
constraint — the float32 graph classifies one sequence in 0.27 ms single-threaded, half a
percent of a 50 ms frame budget. Dynamic quantization is actually *slower* than float32
on a model this small. Table and reasoning in `reports/results.md`.

## Stack

- **Training:** Python 3.11, PyTorch 2.13 (CUDA), `uv` for environment management
- **Data:** Google Isolated Sign Language Recognition (Kaggle `asl-signs`) — pre-extracted
  MediaPipe landmarks, 94,477 sequences
- **Export:** ONNX, ONNX Runtime
- **Front-end:** Vite + TypeScript, MediaPipe Tasks (JS), `onnxruntime-web`

## Layout

```
src/        preprocessing, training, evaluation, export
tests/      dry-run harnesses on synthetic data
reports/    results, preprocessing contract, per-run configs, per-sequence predictions
models/     the trained checkpoint and the exported graph
data/       split definition (raw data and cache are not committed)
```

## Running it

Requires the Kaggle dataset under `data/raw/`.

```powershell
uv pip install -r requirements.txt

uv run python src\cache_dataset.py
uv run python src\train.py --model cnn --landmarks hands --augment --lr-schedule cosine --epochs 120 --run-name abl_hands_aug
uv run python src\evaluate.py --run-name abl_hands_aug --split test
uv run python src\export_onnx.py --run-name abl_hands_aug
```

Note the CUDA build of PyTorch needs an explicit index URL; see the comment above `torch`
in `requirements.txt`. A plain install pulls the CPU-only build.

Tests run on synthetic data and need no dataset:

```powershell
uv run python tests\test_transforms.py
uv run python tests\dryrun_eval.py
uv run python tests\dryrun_export.py
```

## Limitations

- Recognition of isolated signs, not translation.
- 250 signs is a fixed vocabulary and small next to any real signing repertoire.
- All 21 signers were recorded in similar lighting, framing and distance. Real-world
  variation in those conditions is not represented in the evaluation.
- The test split is handedness-confounded (see above).
- About 10pp of headroom is hand-tracking quality in the source data rather than
  modelling.
