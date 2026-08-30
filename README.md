# Sign Recognition

Isolated sign language recognition from webcam. MediaPipe landmark sequences
→ temporal model (PyTorch) → ONNX → runs live in-browser at ≥20 FPS.

**Not ASL translation.** This project recognizes individual signs in
isolation. Grammar, spatial referencing, and non-manual markers (facial
grammar, etc.) are out of scope.

## Status

Work in progress. Data pipeline and baseline in progress — no trained model
or live demo yet.

## Stack

- Training: Python 3.11, PyTorch (CUDA)
- Data: Google - Isolated Sign Language Recognition (Kaggle, `asl-signs`)
- Export: ONNX + onnxruntime quantization
- Front-end: Vite + TypeScript, MediaPipe Tasks (JS), `onnxruntime-web`

Full write-up, results, and a link to the live demo will land here once the
model and browser app are done.
