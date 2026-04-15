# Results

## Executive Summary

The model is a sparse linear SVM over TF-IDF text features, vendor ID, and log-scaled amount. It is not underfitting. It does show moderate overfitting because training accuracy is near 0.99 while grouped unseen accuracy is about 0.88 to 0.89. That level is acceptable for this task, but it means row-level random splits are too optimistic.

## Why The Split Strategy Changed

- Random row splits overstate performance because repeated `itemName` patterns can leak into both train and test.
- Grouped splits by normalized `itemName` are the better primary estimate for unseen transactions.
- A fully stratified grouped split is not reliable on the full dataset because 16 labels are singletons and 21 labels have fewer than 3 rows.
- A balanced unseen benchmark is useful, but it is a stress test for class fairness, not a replacement for production-like evaluation.

## Dataset Constraints

- Records: 4894
- Unique account names: 103
- Labels with fewer than 5 rows: 34
- Labels with one unique normalized item: 18
- Missing descriptions: 31

## Natural Distribution Metrics

- Random holdout accuracy: 0.9050
- Grouped holdout accuracy: 0.8917
- Repeated grouped accuracy mean: 0.8771 +/- 0.0150
- Grouped holdout macro F1: 0.7797
- Grouped train-test gap: 0.0964

## Balanced Unseen Benchmark

- Eligible labels: 64
- Samples per class: 3
- Balanced holdout accuracy: 0.8385
- Balanced holdout macro F1: 0.8035
- Repeated balanced accuracy mean with default model: 0.8354 +/- 0.0205
- Repeated balanced macro F1 mean with default model: 0.8072
- Repeated balanced accuracy mean with `class_weight='balanced'`: 0.8531
- Repeated balanced macro F1 mean with `class_weight='balanced'`: 0.8103

## Learning Curve

| Train Size | Train Accuracy | Validation Accuracy |
| --- | ---: | ---: |
| 790 | 0.9848 | 0.5895 |
| 1580 | 0.9859 | 0.7170 |
| 2371 | 0.9850 | 0.7930 |
| 3161 | 0.9850 | 0.8474 |
| 3952 | 0.9866 | 0.8773 |

## Validation Curve

| C | Train Accuracy | Validation Accuracy |
| --- | ---: | ---: |
| 0.25 | 0.9698 | 0.8657 |
| 0.5 | 0.9807 | 0.8715 |
| 1.0 | 0.9866 | 0.8773 |
| 2.0 | 0.9888 | 0.8783 |
| 4.0 | 0.9907 | 0.8794 |

## Conclusion

- No underfitting signal: both train and validation scores are high, and validation improves as more data is added.
- Moderate overfitting signal: train accuracy stays around 0.99 while grouped validation remains lower by about 0.10.
- The primary reported metric should stay the grouped natural-distribution score. The balanced benchmark should be reported alongside it, not instead of it.
