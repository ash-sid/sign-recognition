# Data notes

Generated from a sample of 300 sequences. Re-run over the full
dataset before finalizing preprocessing decisions (sequence length cutoff,
imputation strategy).

## Counts
- Signers (participant_id): 21
- Distinct signs: 250 (sign_to_prediction_index_map.json has 250 entries)
- Total sequences: 94477

## Sequence length (frames), sample of 300
- min: 6
- max: 295
- mean: 36.8
- median: 23.0
- p95: 125.3

## Missing-frame rate (fraction of NaN x-coords per sequence), sample of 300
- mean: 0.056
- max: 0.427

## Schema
- train.csv: path, participant_id, sequence_id, sign
- train_landmark_files/<participant_id>/<sequence_id>.parquet: frame, row_id, type, landmark_index, x, y, z
- type in {face, pose, left_hand, right_hand}; 543 landmarks per frame total

## Open follow-ups
- Confirm per-signer sequence counts are not wildly imbalanced (affects split strategy)
- Decide fixed sequence length for resampling based on the distribution above
