"""
src/seam_frames.py

Decodes a recorded clip into a directory of PNG frames, the input to both
halves of the extractor seam comparison: web/seam.html landmarks the frames in
a browser, src/extract_landmarks.py --frames landmarks them here, and
tests/test_seam.py compares the two.

Frames rather than the clip itself, because the browser and OpenCV decode
video with different decoders and need not produce the same pixels. PNG is
lossless, so both sides can read identical bytes; both hash what they read,
and the comparison checks the hashes rather than assuming it.

Frames wider than --width are downscaled to it, 640 by default, the width the
live page asks the camera for.

Run: uv run python src\\seam_frames.py clip1.mov --out data\\seam\\clip1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode a clip into PNG frames.")
    parser.add_argument("clip", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty; frames from two clips must not mix")
    args.out.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(args.clip))
    if not capture.isOpened():
        raise SystemExit(f"could not open {args.clip}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    count = 0
    size = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        if width > args.width:
            frame = cv2.resize(frame, (args.width, round(height * args.width / width)),
                               interpolation=cv2.INTER_AREA)
        size = (frame.shape[1], frame.shape[0])
        if not cv2.imwrite(str(args.out / f"frame_{count:04d}.png"), frame):
            raise SystemExit(f"could not write frame {count}")
        count += 1
    capture.release()

    if count == 0:
        raise SystemExit(f"no frames decoded from {args.clip}")
    print(f"{count} frames, {size[0]}x{size[1]}, container reports {fps:.3f} per second")
    print(f"pass --fps {fps:.3f} to src\\extract_landmarks.py and the same value to web/seam.html")


if __name__ == "__main__":
    main()
