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

<!-- ERROR_ANALYSIS_START -->
## Error analysis

`abl_hands_aug` on the val split: 58.18% top-1 (95% CI 57.33%-59.02%), 80.85% top-5, over 13075 sequences from 3 signers.

### Which hand the data is in

| Split | Left slot | Mixed | Right slot |
|---|---|---|---|
| train | 4 | 1 | 10 |
| val | 1 | 0 | 2 |
| test | 3 | 0 | 0 |

Each signer's one-handed sequences overwhelmingly put the active hand in the same block, so the block is a stable property of a signer. The training and evaluation splits do not draw from the same mixture of them.

This describes the recording, not the signer. MediaPipe assigns left and right from the camera's point of view, so a left-handed signer and a right-handed one captured in a mirrored frame are indistinguishable here. The model only ever sees which block is filled, so the distinction does not change anything below.

### Tracking quality

| Frames missing on the tracked hand | Sequences | Top-1 | 95% CI |
|---|---|---|---|
| <10% | 2800 | 68.04% | 66.28%-69.74% |
| 10-25% | 2124 | 67.89% | 65.87%-69.84% |
| 25-50% | 3557 | 60.64% | 59.02%-62.23% |
| >50% | 4594 | 45.78% | 44.34%-47.22% |

The landmark files are not uniformly complete, and the model's input does not record how much of a sequence was interpolated rather than observed.

### Sequence length

| Frames before resampling | Sequences | Top-1 | 95% CI |
|---|---|---|---|
| <=22 | 6037 | 51.70% | 50.44%-52.96% |
| 23-70 | 5001 | 62.97% | 61.62%-64.30% |
| 71-135 | 1060 | 68.87% | 66.02%-71.58% |
| >135 | 977 | 62.13% | 59.05%-65.12% |

Sequences longer than the resampling target have frames discarded; shorter ones are interpolated up. Signing tempo varies by signer, so a length effect and a signer effect are easy to confuse -- the per-signer breakdown is in the accompanying CSV.

### One- and two-handed signs

This split has 156 sequences (1.19%) in which both hands were detected, and no sign reaches 28% two-handed sequences (median 0.0%). Signs that are two-handed when performed are not two-handed in this data: the non-dominant hand is rarely tracked. Whether the model handles two-handed signs worse cannot be answered from these landmarks, because there is almost no contrast to measure.

### Which signs collide

| Actual | Confused with | -> | <- | Combined rate |
|---|---|---|---|---|
| awake | wake | 15 | 28 | 75.4% |
| glasswindow | tooth | 23 | 1 | 49.7% |
| goose | tongue | 0 | 19 | 47.5% |
| duck | goose | 17 | 7 | 46.4% |
| lamp | shhh | 19 | 0 | 46.3% |
| hear | listen | 2 | 31 | 44.2% |
| boat | there | 12 | 10 | 43.9% |
| cereal | grass | 15 | 2 | 41.3% |
| penny | think | 14 | 6 | 37.3% |
| same | stay | 2 | 17 | 36.8% |
| chin | say | 3 | 16 | 36.8% |
| pen | pencil | 6 | 14 | 36.3% |
| ear | hear | 0 | 25 | 35.7% |
| look | see | 25 | 0 | 35.2% |
| cat | kitty | 11 | 7 | 33.9% |

Ranked by the two conditional error rates summed, so a pair appears for being mutually confusable rather than for being common. 11 of 15 are confused in both directions.

The model sees hand landmarks only. Face and lip landmarks are excluded by design, so any pair of signs distinguished mainly by mouth shape or other non-manual markers is not separable by this model in principle rather than merely in practice.

### Per-class accuracy

Across 250 signs: min 0.00%, lower quartile 45.13%, median 60.36%, upper quartile 72.88%, max 98.28%. 1 signs are never predicted correctly.

These are not ranked, deliberately. The median sign has 53 sequences in this split, so a sign scoring 60% carries a 95% interval of roughly 46.94%-72.41%. A table of the worst individual signs would mostly be reporting which classes got an unlucky draw, and would not reproduce on another test set. Per-sign numbers are in the accompanying CSV for anyone who wants them with that caveat attached.

![Confusion matrix for the least accurate signs](reports/confusion_abl_hands_aug_val.png)

<!-- ERROR_ANALYSIS_END -->

<!-- ERROR_ANALYSIS_TEST_START -->
## Error analysis

`abl_hands_aug` on the test split: 61.03% top-1 (95% CI 60.22%-61.83%), 82.59% top-5, over 14015 sequences from 3 signers.

### Which hand the data is in

| Split | Left slot | Mixed | Right slot |
|---|---|---|---|
| train | 4 | 1 | 10 |
| val | 1 | 0 | 2 |
| test | 3 | 0 | 0 |

Each signer's one-handed sequences overwhelmingly put the active hand in the same block, so the block is a stable property of a signer. The training and evaluation splits do not draw from the same mixture of them.

This describes the recording, not the signer. MediaPipe assigns left and right from the camera's point of view, so a left-handed signer and a right-handed one captured in a mirrored frame are indistinguishable here. The model only ever sees which block is filled, so the distinction does not change anything below.

Scoring the same sequences mirrored moves top-1 by +0.01pp (61.03% to 61.03%). 142 sequences are lost and 143 gained, exact binomial p = 1. 4.62% of top-1 predictions change.

Against `abl_hands_only_long`, which is identical but trained without augmentation: mirroring moves that model by +2.10pp (56.33% to 58.43%), and 36.68% of its predictions change.

Augmentation is worth +4.69pp here. Aligning the unaugmented model's input with the block it was trained on recovers 2.10pp of that, so roughly 45% of the gain is the model no longer depending on which block the data is in, and the remaining 2.60pp is augmentation acting as augmentation. Both halves are real; only the first would shrink on an evaluation split that matched the training mixture.

### Tracking quality

| Frames missing on the tracked hand | Sequences | Top-1 | 95% CI |
|---|---|---|---|
| <10% | 3224 | 71.18% | 69.60%-72.72% |
| 10-25% | 2105 | 67.32% | 65.28%-69.29% |
| 25-50% | 3533 | 63.83% | 62.23%-65.40% |
| >50% | 5153 | 50.18% | 48.82%-51.55% |

The landmark files are not uniformly complete, and the model's input does not record how much of a sequence was interpolated rather than observed.

### Sequence length

| Frames before resampling | Sequences | Top-1 | 95% CI |
|---|---|---|---|
| <=22 | 5522 | 57.12% | 55.81%-58.42% |
| 23-70 | 4947 | 64.91% | 63.57%-66.23% |
| 71-135 | 2575 | 65.01% | 63.15%-66.83% |
| >135 | 971 | 52.94% | 49.79%-56.06% |

Sequences longer than the resampling target have frames discarded; shorter ones are interpolated up. Signing tempo varies by signer, so a length effect and a signer effect are easy to confuse -- the per-signer breakdown is in the accompanying CSV.

### One- and two-handed signs

This split has 169 sequences (1.21%) in which both hands were detected, and no sign reaches 19% two-handed sequences (median 0.0%). Signs that are two-handed when performed are not two-handed in this data: the non-dominant hand is rarely tracked. Whether the model handles two-handed signs worse cannot be answered from these landmarks, because there is almost no contrast to measure.

### Which signs collide

| Actual | Confused with | -> | <- | Combined rate |
|---|---|---|---|---|
| pen | pencil | 16 | 16 | 50.8% |
| duck | tongue | 0 | 28 | 49.1% |
| lips | mouth | 8 | 20 | 45.9% |
| duck | goose | 8 | 19 | 45.5% |
| animal | have | 6 | 16 | 44.9% |
| stay | that | 16 | 9 | 42.4% |
| finger | wait | 14 | 8 | 39.0% |
| nap | sleep | 14 | 7 | 37.3% |
| glasswindow | tooth | 8 | 13 | 34.4% |
| cut | scissors | 9 | 10 | 34.0% |
| dirty | pig | 2 | 16 | 32.9% |
| cloud | rain | 5 | 13 | 32.3% |
| chin | say | 10 | 8 | 32.1% |
| awake | wake | 11 | 7 | 31.7% |
| chair | read | 14 | 1 | 29.9% |

Ranked by the two conditional error rates summed, so a pair appears for being mutually confusable rather than for being common. 14 of 15 are confused in both directions.

The model sees hand landmarks only. Face and lip landmarks are excluded by design, so any pair of signs distinguished mainly by mouth shape or other non-manual markers is not separable by this model in principle rather than merely in practice.

### Per-class accuracy

Across 250 signs: min 8.77%, lower quartile 48.09%, median 62.84%, upper quartile 73.72%, max 96.49%. 0 signs are never predicted correctly.

These are not ranked, deliberately. The median sign has 57 sequences in this split, so a sign scoring 60% carries a 95% interval of roughly 46.70%-71.38%. A table of the worst individual signs would mostly be reporting which classes got an unlucky draw, and would not reproduce on another test set. Per-sign numbers are in the accompanying CSV for anyone who wants them with that caveat attached.

![Confusion matrix for the least accurate signs](confusion_abl_hands_aug_test.png)

<!-- ERROR_ANALYSIS_TEST_END -->
