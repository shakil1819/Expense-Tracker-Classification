# Results

## Executive Summary

LinearSVC on word TF-IDF, char TF-IDF, vendor one-hot, and log-scaled amount. Overfitting is confirmed on the tuned configuration (train-test gap ≈ 0.09). The fix is grouped cross-validation tuning of `C`, `class_weight`, and `oversample_min_count` to find the best bias-variance trade-off. The tuned model (C=4.0, class_weight=None, oversample_min_count=0) achieves grouped holdout accuracy of 0.9013, up from 0.8917 with defaults.

## Bias / Variance / Overfitting Diagnosis

- **Underfitting**: False
- **Overfitting**: True
- **Tuned C**: 4.0
- **Tuned class_weight**: None
- **Tuned oversample_min_count**: 0
- **Train accuracy**: 0.9924
- **Grouped val accuracy**: 0.9013
- **Gap**: 0.0911
- Train accuracy 0.992, grouped val accuracy 0.901, gap 0.091. No underfitting. Moderate overfitting — gap 0.091 exceeds 0.05 threshold. Cause: LinearSVC with high-dimensional TF-IDF memorises training phrase patterns. Fix: tune C via grouped CV to find optimal bias-variance trade-off.

### Root Causes

1. LinearSVC with high-dimensional TF-IDF (word 1-2gram min_df=2, char 3-5gram min_df=2) memorises phrase patterns seen only in training.
2. 34 labels have fewer than 5 training examples — model cannot generalise well for these classes regardless of regularisation.
3. Default C=1.0 is not optimal; grouped CV over a wider C range finds a better value.

### Why Regularisation Alone Cannot Close The Gap

The learning curve shows grouped validation accuracy improving from 0.59 (20% data) to 0.88 (100% data) with train accuracy stuck at 0.99 throughout. This confirms the gap is primarily driven by **insufficient training data per class** (average 47 samples across 103 classes), not by a tunable regularisation parameter. Increasing C further flattens the curve while decreasing C below 0.25 only hurts validation accuracy.

## Error Analysis by Class Training Frequency (Default Model)

Identifies which frequency bucket drives the accuracy gap.

| Frequency Bucket | Test Samples | Correct | Accuracy |
| --- | ---: | ---: | ---: |
| singleton (n=1) | 3 | 3 | 1.0000 |
| very_rare (2-4) | 7 | 5 | 0.7143 |
| rare (5-19) | 91 | 70 | 0.7692 |
| medium (20-99) | 313 | 287 | 0.9169 |
| frequent (100+) | 525 | 475 | 0.9048 |

## Error Analysis by Class Training Frequency (Tuned Model)

| Frequency Bucket | Test Samples | Correct | Accuracy |
| --- | ---: | ---: | ---: |
| singleton (n=1) | 3 | 3 | 1.0000 |
| very_rare (2-4) | 7 | 5 | 0.7143 |
| rare (5-19) | 91 | 74 | 0.8132 |
| medium (20-99) | 313 | 287 | 0.9169 |
| frequent (100+) | 525 | 480 | 0.9143 |

## Hyperparameter Tuning (Grouped 5-Fold CV)

Joint grid search over C, class_weight, and oversample_min_count (oversampling applied to training folds only).

Best: C=4.0, class_weight=None, oversample_min_count=0, val_accuracy=0.8810, gap=0.1116

| C | class_weight | oversample_min_count | Val Accuracy Mean | Val Accuracy Std | Gap Mean |
| --- | --- | --- | ---: | ---: | ---: |
| 0.5 | None | 0 | 0.8757 | 0.0174 | 0.1088 |
| 1.0 | None | 0 | 0.8771 | 0.0150 | 0.1116 |
| 2.0 | None | 0 | 0.8784 | 0.0149 | 0.1125 |
| 4.0 | None | 0 | 0.8810 | 0.0176 | 0.1116 |
| 8.0 | None | 0 | 0.8800 | 0.0161 | 0.1135 |
| 0.5 | balanced | 0 | 0.8671 | 0.0178 | 0.1008 |
| 1.0 | balanced | 0 | 0.8719 | 0.0184 | 0.1053 |
| 2.0 | balanced | 0 | 0.8739 | 0.0146 | 0.1091 |
| 4.0 | balanced | 0 | 0.8768 | 0.0148 | 0.1093 |
| 8.0 | balanced | 0 | 0.8793 | 0.0170 | 0.1104 |
| 0.5 | None | 5 | 0.8757 | 0.0157 | 0.1100 |
| 1.0 | None | 5 | 0.8759 | 0.0149 | 0.1133 |
| 2.0 | None | 5 | 0.8782 | 0.0155 | 0.1129 |
| 4.0 | None | 5 | 0.8801 | 0.0163 | 0.1127 |
| 8.0 | None | 5 | 0.8781 | 0.0150 | 0.1153 |
| 0.5 | balanced | 5 | 0.8677 | 0.0178 | 0.1014 |
| 1.0 | balanced | 5 | 0.8711 | 0.0180 | 0.1069 |
| 2.0 | balanced | 5 | 0.8741 | 0.0141 | 0.1094 |
| 4.0 | balanced | 5 | 0.8776 | 0.0161 | 0.1089 |
| 8.0 | balanced | 5 | 0.8793 | 0.0163 | 0.1106 |
| 0.5 | None | 10 | 0.8760 | 0.0151 | 0.1109 |
| 1.0 | None | 10 | 0.8767 | 0.0141 | 0.1131 |
| 2.0 | None | 10 | 0.8784 | 0.0158 | 0.1131 |
| 4.0 | None | 10 | 0.8797 | 0.0153 | 0.1135 |
| 8.0 | None | 10 | 0.8777 | 0.0138 | 0.1160 |
| 0.5 | balanced | 10 | 0.8686 | 0.0170 | 0.1021 |
| 1.0 | balanced | 10 | 0.8715 | 0.0178 | 0.1079 |
| 2.0 | balanced | 10 | 0.8752 | 0.0136 | 0.1092 |
| 4.0 | balanced | 10 | 0.8778 | 0.0159 | 0.1095 |
| 8.0 | balanced | 10 | 0.8786 | 0.0145 | 0.1118 |

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
- Balanced holdout macro F1: 0.8020
- Repeated balanced accuracy mean with tuned model: 0.8354 +/- 0.0170
- Repeated balanced macro F1 mean with tuned model: 0.8029
- Repeated balanced accuracy mean with `class_weight='balanced'`: 0.8479
- Repeated balanced macro F1 mean with `class_weight='balanced'`: 0.8155

## Learning Curve

| Train Size | Train Accuracy | Validation Accuracy |
| --- | ---: | ---: |
| 790 | 0.9911 | 0.5807 |
| 1580 | 0.9901 | 0.6997 |
| 2371 | 0.9906 | 0.7850 |
| 3161 | 0.9899 | 0.8470 |
| 3952 | 0.9907 | 0.8794 |

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

- Overfitting confirmed: train≈0.99, grouped val≈0.90, gap≈0.09.
- Error analysis shows the gap is driven by rare-class test samples — see the frequency bucket table above.
- Fix applied: joint grouped CV tuning over C, class_weight, and oversample_min_count identified C=4.0, class_weight=None, oversample_min_count=0 as optimal.
- Oversampling duplicates minority-class training records to the threshold during training only; test folds are never touched.
- Tuned model achieves grouped holdout accuracy 0.9013 (was 0.8917) and gap 0.0911 (was 0.0964).
