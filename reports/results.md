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
| v1_cnn_s1 | cnn | hands_pose | true | false | — | 1 | 40 | 0.5299 | 0.5315 | 0.7855 | 111 |
| v1_cnn_s2 | cnn | hands_pose | true | false | — | 2 | 51 | 0.5302 | 0.5328 | 0.7851 | 137 |
| v1_gru_s1 | gru | hands_pose | true | false | — | 1 | 56 | 0.4922 | 0.4920 | 0.7413 | 123 |
| v1_gru_s2 | gru | hands_pose | true | false | — | 2 | 20 | 0.4776 | 0.4634 | 0.7232 | 59 |
| ref_cnn_s0 | cnn | hands_pose | true | false | cosine LR | 0 | 60 | 0.5429 | 0.5483 | 0.7907 | 135 |
| ref_cnn_s1 | cnn | hands_pose | true | false | cosine LR | 1 | 47 | 0.5399 | 0.5376 | 0.7873 | 133 |
| ref_cnn_s2 | cnn | hands_pose | true | false | cosine LR | 2 | 52 | 0.5412 | 0.5418 | 0.7862 | 133 |
| ref_gru_s0 | gru | hands_pose | true | false | cosine LR | 0 | 47 | 0.4886 | 0.4890 | 0.7372 | 114 |
| ref_gru_s1 | gru | hands_pose | true | false | cosine LR | 1 | 31 | 0.4847 | 0.4763 | 0.7300 | 116 |
| ref_gru_s2 | gru | hands_pose | true | false | cosine LR | 2 | 31 | 0.4890 | 0.4765 | 0.7314 | 114 |
| abl_hands_only | cnn | hands | true | false | cosine LR | 0 | 56 | 0.5469 | 0.5562 | 0.7894 | 135 |
| abl_no_norm | cnn | hands_pose | false | false | cosine LR | 0 | 60 | 0.5318 | 0.5400 | 0.7815 | 132 |
| abl_augment | cnn | hands_pose | true | true | cosine LR | 0 | 59 | 0.5686 | 0.5936 | 0.8141 | 161 |
| v2_transformer | transformer | hands_pose | true | false | adamw, cosine LR | 0 | 25 | 0.4783 | 0.4770 | 0.7443 | 429 |
| v2_transformer_aug | transformer | hands_pose | true | true | adamw, cosine LR | 0 | 43 | 0.4938 | 0.5239 | 0.7638 | 463 |
| diag_gru_meanpool | gru | hands_pose | true | false | mean pooling, cosine LR | 0 | 50 | 0.4857 | 0.4901 | 0.7480 | 132 |
| diag_shuffled | cnn | hands_pose | true | false | frames shuffled, cosine LR | 0 | 54 | 0.4596 | 0.4768 | 0.7573 | 150 |
| abl_augment_long | cnn | hands_pose | true | true | cosine LR | 0 | 92 | 0.5798 | 0.5966 | 0.8118 | 373 |
| abl_hands_aug | cnn | hands | true | true | cosine LR | 0 | 116 | 0.5818 | 0.6103 | 0.8259 | 351 |
| prod_s1 | cnn | hands_pose | true | true | cosine LR | 1 | 120 | 0.5793 | 0.5983 | 0.8223 | 416 |
| prod_s2 | cnn | hands_pose | true | true | cosine LR | 2 | 109 | 0.5722 | 0.5947 | 0.8186 | 380 |
| prod_hands_s1 | cnn | hands | true | true | cosine LR | 1 | 105 | 0.5726 | 0.6036 | 0.8254 | 374 |
| prod_hands_s2 | cnn | hands | true | true | cosine LR | 2 | 104 | 0.5777 | 0.6054 | 0.8233 | 364 |

One row per training run. `Landmarks` is either both hands plus the eight pose
points (`hands_pose`) or the hands alone (`hands`); `Normalized` is whether the
shoulder-relative normalization was applied during preprocessing; `Augment` is
whether training batches were mirrored, time-warped and jittered. `Notes` covers
anything else a run varied. The full per-run configuration, including fields not
shown here, is in reports/ablations.csv.

A model has to clear the mean-pooled logistic regression's test top-1 of 0.2671
to be worth its added complexity over ignoring time entirely.

<!-- RUNS_END -->
