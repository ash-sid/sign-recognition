# Results

## Split (data/splits.json)
- train signers: 15 (67128 sequences)
- val signers: 3 (13096 sequences)
- test signers: 3 (14066 sequences)
- vocabulary: 250 signs

## Baselines

| Model | Val top-1 | Test top-1 | Val top-5 | Test top-5 |
|---|---|---|---|---|
| Majority class | 0.0040 | 0.0043 | — | — |
| Logistic regression (mean-pooled landmarks) | 0.2496 | 0.2671 | 0.5050 | 0.5267 |

Logistic regression fit time: 38.9s.

Any subsequent temporal model needs to clear the logistic-regression test
top-1 number above to be worth the added complexity over mean-pooling.

<!-- MODEL_V1_GRU_START -->
## Model v1 (gru)

- Architecture: GRU, hidden_size=128, num_layers=2, dropout=0.3
- Seed: 0
- Best epoch: 41 / 60 (early stop patience=10)
- Total train time: 105s

| Split | Top-1 | Top-5 |
|---|---|---|
| Val | 0.4938 | — |
| Test | 0.4687 | 0.7244 |

This beats the logistic-regression baseline (test top-1 0.2671).

<!-- MODEL_V1_GRU_END -->

<!-- MODEL_V1_CNN_START -->
## Model v1 (cnn)

- Architecture: 1D-CNN, channels=(128, 128, 256), kernel_size=5, dropout=0.3
- Seed: 0
- Best epoch: 19 / 60 (early stop patience=10)
- Total train time: 70s

| Split | Top-1 | Top-5 |
|---|---|---|
| Val | 0.5162 | — |
| Test | 0.4869 | 0.7578 |

This beats the logistic-regression baseline (test top-1 0.2671).

<!-- MODEL_V1_CNN_END -->
