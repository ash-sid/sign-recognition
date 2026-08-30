# Data notes

Generated from the full dataset (94477 sequences).

## Counts
- Signers (participant_id): 21
- Distinct signs: 250 (sign_to_prediction_index_map.json has 250 entries)
- Total sequences: 94477

## Sequence length (frames), full dataset
- min: 2
- max: 537
- mean: 37.9
- median: 22.0
- p95: 135.0
- p99: 219.0

## Missing-frame rate (fraction of NaN x-coords per sequence), full dataset
- mean: 0.058
- max: 0.933

## Per-hand usage
- left_hand: unused (>= 95% NaN) in 53005 sequences (56.1%); among sequences where it IS used, per-sequence NaN rate: mean 0.368, max 0.950
- right_hand: unused (>= 95% NaN) in 39164 sequences (41.5%); among sequences where it IS used, per-sequence NaN rate: mean 0.348, max 0.950
- both hands unused (>= 95% NaN on both): 178 sequences (0.2%). Every sign uses at least one hand, so these are very likely tracking failures (bad framing, occlusion) rather than genuine no-manual-signal cases -- consider dropping them rather than training on an all-zero-hands input.

## Per-signer sign coverage (of 250 total signs)
- min: 238
- max: 250
- mean: 249.0
- signers below 90% coverage: 0

## Schema
- train.csv: path, participant_id, sequence_id, sign
- train_landmark_files/<participant_id>/<sequence_id>.parquet: frame, row_id, type, landmark_index, x, y, z
- type in {face, pose, left_hand, right_hand}; 543 landmarks per frame total

## Follow-ups
- Confirm TARGET_LEN in src/preprocessing.py (currently 70) against
  the p95/p99 above -- longer covers more of the tail but wastes compute on
  the many short sequences.
- If any signer is well below full coverage, make_split.py's search should
  still find a workable partition (it optimizes for this), but check its
  printed coverage number before trusting the split.
- Review the both-hands-unused count above; if it's non-trivial, add a filter
  step before training rather than feeding those sequences in as-is.
