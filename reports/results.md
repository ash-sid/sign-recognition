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
