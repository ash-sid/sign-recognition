<!-- BASELINES_START -->
# Results

## Split (data/splits.json)
- train signers: 15 (67022 sequences)
- val signers: 3 (13075 sequences)
- test signers: 3 (14015 sequences)
- vocabulary: 250 signs

## Baselines

| Model | Val top-1 | Test top-1 | Val top-5 | Test top-5 |
|---|---|---|---|---|
| Majority class | 0.0041 | 0.0043 | — | — |
| Logistic regression (mean-pooled landmarks) | 0.2495 | 0.2671 | 0.5034 | 0.5280 |

Logistic regression fit time: 40.2s.

Any subsequent temporal model needs to clear the logistic-regression test
top-1 number above to be worth the added complexity over mean-pooling.

<!-- BASELINES_END -->

<!-- RUNS_START -->
## Models

| Run | Model | Landmarks | Normalized | Augment | Notes | Seed | Best epoch | Val top-1 | Test top-1 | Test top-5 | Train s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v1_gru | gru | hands_pose | true | false | — | 0 | 41 | 0.4938 | 0.4687 | 0.7244 | 99 |
| v1_cnn | cnn | hands_pose | true | false | — | 0 | 19 | 0.5162 | 0.4869 | 0.7578 | 65 |

One row per training run. `Landmarks` is either both hands plus the eight pose
points (`hands_pose`) or the hands alone (`hands`); `Normalized` is whether the
shoulder-relative normalization was applied during preprocessing; `Augment` is
whether training batches were mirrored, time-warped and jittered. `Notes` covers
anything else a run varied. The full per-run configuration, including fields not
shown here, is in reports/ablations.csv.

A model has to clear the mean-pooled logistic regression's test top-1 of 0.2671
to be worth its added complexity over ignoring time entirely.

<!-- RUNS_END -->
