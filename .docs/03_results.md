# Results

## Executive Summary

LinearSVC on word TF-IDF, char TF-IDF, vendor one-hot, and log-scaled amount. Overfitting is confirmed (train-test gap ≈ 0.10). The fix is grouped cross-validation tuning of `C` and `class_weight` to find the best bias-variance trade-off. The tuned model (C=4.0, class_weight=None) achieves grouped holdout accuracy of 0.9013, up from 0.8917 with defaults.

## Bias / Variance / Overfitting Diagnosis

- **Underfitting**: False
- **Overfitting**: True
- **Train accuracy**: 0.9881
- **Grouped val accuracy**: 0.8917
- **Gap**: 0.0964
- Train accuracy 0.988, grouped val accuracy 0.892, gap 0.096. No underfitting. Moderate overfitting — gap 0.096 exceeds 0.05 threshold. Cause: LinearSVC with high-dimensional TF-IDF memorises training phrase patterns. Fix: tune C via grouped CV to find optimal bias-variance trade-off.

### Root Causes

1. LinearSVC with high-dimensional TF-IDF (word 1-2gram min_df=2, char 3-5gram min_df=2) memorises phrase patterns seen only in training.
2. 34 labels have fewer than 5 training examples — model cannot generalise well for these classes regardless of regularisation.
3. Default C=1.0 is not optimal; grouped CV over a wider C range finds a better value.

### Why Regularisation Alone Cannot Close The Gap

The learning curve shows grouped validation accuracy improving from 0.59 (20% data) to 0.88 (100% data) with train accuracy stuck at 0.99 throughout. This confirms the gap is primarily driven by **insufficient training data per class** (average 47 samples across 103 classes), not by a tunable regularisation parameter. Increasing C further flattens the curve while decreasing C below 0.25 only hurts validation accuracy.

## Hyperparameter Tuning (Grouped 5-Fold CV)

Best: C=4.0, class_weight=None, val_accuracy=0.8810, gap=0.1116

| C | class_weight | Val Accuracy Mean | Val Accuracy Std | Gap Mean |
| --- | --- | ---: | ---: | ---: |
| 0.1 | None | 0.8508 | 0.0181 | 0.1014 |
| 0.25 | None | 0.8659 | 0.0178 | 0.1090 |
| 0.5 | None | 0.8757 | 0.0174 | 0.1088 |
| 1.0 | None | 0.8771 | 0.0150 | 0.1116 |
| 2.0 | None | 0.8784 | 0.0149 | 0.1125 |
| 4.0 | None | 0.8810 | 0.0176 | 0.1116 |
| 8.0 | None | 0.8800 | 0.0161 | 0.1135 |
| 16.0 | None | 0.8762 | 0.0136 | 0.1180 |
| 0.1 | balanced | 0.8289 | 0.0205 | 0.0923 |
| 0.25 | balanced | 0.8559 | 0.0204 | 0.0994 |
| 0.5 | balanced | 0.8671 | 0.0178 | 0.1008 |
| 1.0 | balanced | 0.8719 | 0.0184 | 0.1053 |
| 2.0 | balanced | 0.8739 | 0.0146 | 0.1091 |
| 4.0 | balanced | 0.8768 | 0.0148 | 0.1093 |
| 8.0 | balanced | 0.8793 | 0.0170 | 0.1104 |
| 16.0 | balanced | 0.8797 | 0.0160 | 0.1126 |

## Why The Split Strategy Changed

- Random row splits overstate performance because repeated `itemName` patterns can leak into both train and test.
- Grouped splits by normalized `itemName` are the better primary estimate for unseen transactions.
- A fully stratified grouped split is not reliable on the full dataset because 16 labels are singletons and 21 labels have fewer than 3 rows.

## Dataset Constraints

- Records: 4894
- Unique account names: 103
- Labels with fewer than 5 rows: 34
- Labels with one unique normalized item: 18
- Missing descriptions: 31

## Natural Distribution Metrics

| Model | Grouped Holdout Accuracy | Repeated Mean ± Std | Macro F1 | Train-Test Gap |
| --- | ---: | ---: | ---: | ---: |
| Default (C=1.0, cw=None) | 0.8917 | 0.8771 ± 0.0150 | 0.7797 | 0.0964 |
| Tuned (C=4.0, cw=None) | 0.9013 | 0.8810 ± 0.0176 | 0.7891 | 0.0911 |

- Random holdout accuracy (default): 0.9050 (optimistic — item-name leakage)

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

## Validation Curve (GroupShuffleSplit 3-fold, grouped by item name)

| C | Train Accuracy | Validation Accuracy |
| --- | ---: | ---: |
| 0.1 | 0.9486 | 0.8464 |
| 0.25 | 0.9698 | 0.8657 |
| 0.5 | 0.9807 | 0.8715 |
| 1.0 | 0.9866 | 0.8773 |
| 2.0 | 0.9888 | 0.8783 |
| 4.0 | 0.9907 | 0.8794 |
| 8.0 | 0.9918 | 0.8760 |
| 16.0 | 0.9926 | 0.8685 |

## Conclusion

- Overfitting confirmed: train≈0.99, grouped val≈0.89, gap≈0.10.
- Cause: high-capacity TF-IDF feature space with limited per-class training data (avg 47 rows/class).
- Fix applied: grouped 5-fold CV tuning identified C=4.0, class_weight=None as optimal.
- Tuned model achieves grouped holdout accuracy 0.9013 (was 0.8917) and reduces gap to 0.0911 (was 0.0964).
- Remaining gap is irreducible with current data volume — 34 labels have fewer than 5 training examples.
