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
| abl_mirror_only | cnn | hands | true | true | mirror only, cosine LR | 0 | 104 | 0.5736 | 0.6025 | 0.8225 | 307 |
| abl_warp_only | cnn | hands | true | true | warp only, cosine LR | 0 | 109 | 0.5594 | 0.5666 | 0.7955 | 314 |
| abl_jitter_only | cnn | hands | true | true | jitter only, cosine LR | 0 | 116 | 0.5507 | 0.5633 | 0.7976 | 286 |
| abl_hands_only_long | cnn | hands | true | false | cosine LR | 0 | 116 | 0.5550 | 0.5633 | 0.7932 | 299 |

One row per training run. `Landmarks` is either both hands plus the eight pose
points (`hands_pose`) or the hands alone (`hands`); `Normalized` is whether the
shoulder-relative normalization was applied during preprocessing; `Augment` is
whether training batches were mirrored, time-warped and jittered. `Notes` covers
anything else a run varied. The full per-run configuration, including fields not
shown here, is in reports/ablations.csv.

A model has to clear the mean-pooled logistic regression's test top-1 of 0.2671
to be worth its added complexity over ignoring time entirely.

<!-- RUNS_END -->
## Reading these results

### Protocol

All numbers are on the signer-independent split (15 train / 3 val / 3 test signers) in
`data/splits.json`. Test scores are on three signers the model never saw.

**Noise floor: 0.7pp.** Three seeds of the best configuration gave a standard deviation
of 0.35pp, so differences below roughly 0.7pp (2 sd) are not interpretable and are
described as "within noise" throughout.

Two training protocols appear in the table and **should not be compared across**:

- **Constant learning rate, early stopping at patience 10.** The earliest runs. Val
  top-1 is noisy enough epoch-to-epoch that early stopping fired on jitter rather than
  on exhausted progress, truncating runs at effectively random points. Across three
  seeds this produced a 4.6pp spread, and within each architecture the run that
  survived longest scored highest without exception.
- **Cosine-annealed learning rate over a fixed budget, early stopping disabled**
  (`cosine LR` in the Notes column). Every run reaches the same convergence point.
  Seed spread fell roughly fivefold, and mean accuracy rose 2.6pp.

The second protocol is the one to use. The first is retained because the earliest rows
document reproducing an earlier result.

### Landmark set and normalization

**Hands only beats hands plus pose by 0.99pp** (three seeds each, Welch t = 4.39,
df = 3.0, p ~ 0.02). Small but real. A plausible reading is that elbow and hip
positions are more signer-specific than hand shape, so including them invites
overfitting to the fifteen training signers — but that is a hypothesis, not something
these runs establish.

This does **not** mean the pose landmarker can be dropped from the inference pipeline.
Normalization uses the shoulders, so those points are still required upstream; only the
model's input is narrower.

**Removing normalization costs 0.83pp, which is within noise — and this should not be
read as "normalization is unnecessary."** Every signer in this dataset was recorded in
similar framing at a similar distance from the camera, so there is little positional
variance for normalization to remove and little for the model to gain from it. A webcam
user standing closer, further away, or off to one side is precisely the case
normalization handles and precisely the case this dataset does not contain. The result
says the evaluation lacks the variation, not that the method is useless.

### Augmentation is mirroring

Augmentation is the single largest effect in the table at +4.31pp. Decomposed against a
matched unaugmented baseline, each transform run alone:

| Transform | Test top-1 | vs unaugmented |
|---|---|---|
| Horizontal mirror | 0.6025 | +3.92pp |
| Time warp | 0.5666 | +0.33pp (within noise) |
| Landmark jitter | 0.5633 | +0.00pp (within noise) |

Mirroring alone is statistically indistinguishable from all three combined. Time warp
and landmark jitter contribute nothing measurable.

The likely mechanism is handedness. With only fifteen training signers, the model can
treat "the dominant hand occupies the right-hand slot" as though it were part of a sign,
which then fails on held-out signers whose dominant hand differs. Mirroring swaps the
hand blocks and reflects the coordinates together, removing handedness as something the
model can key on. That reframes part of the signer-generalization gap as a specific,
fixable overfitting problem rather than a diffuse one.

### Why the 1D-CNN beats the GRU

The convolutional model leads the recurrent one by 6.20pp (three seeds each,
Welch t = 11.85). Three candidate explanations were tested and rejected:

- **Learning-rate schedule.** Cosine annealing *widened* the gap rather than closing it,
  giving the CNN +2.6pp and the GRU +0.6pp.
- **Seed variance.** At three seeds per architecture under the stable protocol, t = 11.85.
- **Sequence pooling.** The GRU summarized sequences with its last hidden state while the
  CNN used global average pooling. Switching the GRU to mean pooling moved it 0.11pp.

The `frames shuffled` row explains it instead. A CNN trained and evaluated on randomly
permuted frames — unable to use temporal order at all — still reaches 0.4768, against
0.5483 with frames in order and 0.2671 for a logistic regression on mean-pooled
landmarks that discards the time axis entirely. That splits the gap: roughly 21pp comes
from seeing the *distribution* of poses in a sequence rather than their average, and
only about 7pp from their *order*.

Most of this task is order-independent. A convolutional stack followed by global average
pooling is close to a purpose-built extractor for an unordered collection of local
patterns, while a recurrent model's sequential state has comparatively little to do.

### The transformer

A four-layer transformer encoder at matched dropout reaches 0.4770, 7.1pp below the CNN,
and overfits: final training loss 0.3850 against the CNN's 0.6254, with validation
peaking at epoch 25 and declining thereafter. Augmentation recovers 4.7pp but leaves it
2.4pp short, at 3.2x the training time.

67k sequences across 250 classes is not enough data for attention to overcome a
convolution's inductive bias here. The transformer is kept in the table as a measured
negative result, not adopted.

### Best model

`abl_hands_aug` — 1D-CNN, hands-only input, mirrored/warped/jittered training batches,
120 epochs on a cosine schedule. **Test top-1 61.03%, test top-5 82.59%.**

It is also the best of its three seeds by *validation* top-1 (0.5818 against 0.5726 and
0.5777), so the selection did not use the test set.
