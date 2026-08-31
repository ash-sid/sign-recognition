# Preprocessing interface contract

This is the contract between the Python preprocessing/modelling side and
the export/browser side. Both `src/preprocessing.py` and the eventual
TypeScript port in `web/src/preprocessing.ts` must implement exactly this.
**Do not change any of it silently** -- a change here means re-checking
both implementations agree (a fixture-based parity test is planned once
the browser port exists) and re-running anything downstream.

## Input tensor shape

`(T, 50, 3)` where:
- `T` = 70 (`TARGET_LEN` in `src/preprocessing.py`), fixed by resampling.
  Chosen from the observed sequence-length distribution (median ~22-23,
  p95 ~135, p99 ~219 frames, per `reports/data-notes.md`) -- 70 sits
  between p90 and p95, so most sequences are covered without padding the
  common case out to the long tail. Revisit if a later model's error
  analysis points to the tail (long signs) specifically.
- `50` = number of selected landmarks (below).
- `3` = x, y, z coordinates.

Flattened form (used by the mean-pooled baseline and any model that wants a
flat feature vector): `(T, 150)`.

## Landmark selection and ordering

Fixed order, 50 landmarks total:

| Positions | Group | Source indices (within MediaPipe's per-type numbering) |
|---|---|---|
| 0-20 | `left_hand` | 0-20 (all 21 hand landmarks) |
| 21-41 | `right_hand` | 0-20 (all 21 hand landmarks) |
| 42-49 | `pose` | 11, 12, 13, 14, 15, 16, 23, 24 (L/R shoulder, L/R elbow, L/R wrist, L/R hip) |

Face/lips landmarks are **not included**. This is a deliberate choice, not
an oversight: recognition here relies on hand shape/motion and arm
position, and non-manual markers are an explicit project non-goal.

Position 42 (first pose landmark) is the left shoulder; position 43 is the
right shoulder. These two are load-bearing for normalization (below) --
don't reorder `POSE_INDICES` without updating `_normalize`.

## Missing-value handling

Two distinct cases, handled differently (missing-hand data in this dataset
is systematic for one-handed signs, not just tracking noise -- see
`reports/data-notes.md` per-hand usage stats):

1. **Hand not used for this sign** (>= 95% of that hand's frames are NaN
   for the whole sequence): the hand's landmarks are set to exactly `0.0`
   for every frame, after normalization. This is a real "absent" signal,
   not an interpolated guess.
2. **Tracking dropout** (a hand that's mostly present but has some NaN
   frames, or the pose landmarks): linearly interpolated along the time
   axis; leading/trailing gaps filled with the nearest valid value.

The threshold (95%) is `UNUSED_HAND_NAN_THRESHOLD` in `src/preprocessing.py`.
Note: a sequence where *both* hands cross this threshold is very likely a
tracking failure rather than a genuine "no manual signal" case (every sign
uses at least one hand) -- see the both-hands-unused count in
`reports/data-notes.md` and consider filtering those out before training.

## Normalization formula

Per-frame translation + scale, using the shoulders as reference:

```
center = (left_shoulder_xy + right_shoulder_xy) / 2      # per frame
scale  = || left_shoulder_xy - right_shoulder_xy ||       # per frame, clipped to >= 1e-3
x_norm = (x - center_x) / scale
y_norm = (y - center_y) / scale
z_norm = z / scale                                          # no translation on z
```

Applied to every landmark (hands and pose alike) using that frame's own
shoulder positions -- this is what makes the representation invariant to
the signer's distance from and position relative to the camera. Applied
*before* the unused-hand zeroing in case 1 above (zeroing happens last, so
"absent" stays exactly `0.0` rather than getting shifted by the shoulder
center).

## Resampling

Fixed-length output via per-landmark, per-coordinate linear interpolation
in normalized time (`np.interp` over `t in [0, 1]`), independent of the
original frame count.

## What's NOT fixed yet

- Whether to filter out both-hands-unused sequences before training (see
  Missing-value handling above).
- Vocabulary size (full 250 signs vs a curated subset) -- still open.
