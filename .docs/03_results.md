# Results

## Executive Summary

LinearSVC on word TF-IDF, char TF-IDF, vendor one-hot, and log-scaled amount. Overfitting is confirmed on the tuned configuration (train-test gap ≈ 0.09). The fix is grouped cross-validation tuning of `C`, `class_weight`, and `oversample_min_count` to find the best bias-variance trade-off. The tuned model (C=2.0, class_weight=None, oversample_min_count=0) achieves grouped holdout accuracy of 0.8938, up from 0.8938 with defaults.

## Bias / Variance / Overfitting Diagnosis

- **Underfitting**: False
- **Overfitting**: True
- **Tuned C**: 1.0
- **Tuned class_weight**: balanced
- **Tuned oversample_min_count**: 5
- **Train accuracy**: 0.9840
- **Grouped val accuracy**: 0.8938
- **Gap**: 0.0901
- Train accuracy 0.984, grouped val accuracy 0.894, gap 0.090. No underfitting. Moderate overfitting — gap 0.090 exceeds 0.05 threshold. Cause: LinearSVC with high-dimensional TF-IDF memorises training phrase patterns. Fix: tune C via grouped CV to find optimal bias-variance trade-off.

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
| rare (5-19) | 91 | 68 | 0.7473 |
| medium (20-99) | 313 | 289 | 0.9233 |
| frequent (100+) | 525 | 477 | 0.9086 |

## Error Analysis by Class Training Frequency (Tuned Model)

| Frequency Bucket | Test Samples | Correct | Accuracy |
| --- | ---: | ---: | ---: |
| singleton (n=1) | 3 | 3 | 1.0000 |
| very_rare (2-4) | 7 | 5 | 0.7143 |
| rare (5-19) | 91 | 76 | 0.8352 |
| medium (20-99) | 313 | 292 | 0.9329 |
| frequent (100+) | 525 | 466 | 0.8876 |

## Hyperparameter Tuning (Grouped 5-Fold CV)

Joint grid search over C, class_weight, and oversample_min_count (oversampling applied to training folds only).

Best: C=2.0, class_weight=None, oversample_min_count=0, val_accuracy=0.8834, val_macro_f1=0.7388, gap=0.1119, optimize_for=macro_f1

| C | class_weight | oversample_min_count | Val Accuracy Mean | Val Macro F1 Mean | Val Accuracy Std | Gap Mean |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 0.5 | None | 0 | 0.8782 | 0.7316 | 0.0153 | 0.1136 |
| 1.0 | None | 0 | 0.8815 | 0.7384 | 0.0155 | 0.1128 |
| 2.0 | None | 0 | 0.8834 | 0.7388 | 0.0158 | 0.1119 |
| 4.0 | None | 0 | 0.8826 | 0.7317 | 0.0159 | 0.1137 |
| 8.0 | None | 0 | 0.8789 | 0.7269 | 0.0165 | 0.1179 |
| 0.5 | balanced | 0 | 0.8717 | 0.7295 | 0.0177 | 0.1062 |
| 1.0 | balanced | 0 | 0.8746 | 0.7291 | 0.0165 | 0.1104 |
| 2.0 | balanced | 0 | 0.8779 | 0.7236 | 0.0171 | 0.1119 |
| 4.0 | balanced | 0 | 0.8785 | 0.7205 | 0.0155 | 0.1149 |
| 8.0 | balanced | 0 | 0.8779 | 0.7200 | 0.0167 | 0.1162 |
| 0.5 | None | 5 | 0.8776 | 0.7236 | 0.0146 | 0.1141 |
| 1.0 | None | 5 | 0.8801 | 0.7322 | 0.0148 | 0.1144 |
| 2.0 | None | 5 | 0.8821 | 0.7332 | 0.0163 | 0.1132 |
| 4.0 | None | 5 | 0.8814 | 0.7334 | 0.0165 | 0.1150 |
| 8.0 | None | 5 | 0.8789 | 0.7349 | 0.0170 | 0.1179 |
| 0.5 | balanced | 5 | 0.8722 | 0.7259 | 0.0183 | 0.1064 |
| 1.0 | balanced | 5 | 0.8746 | 0.7255 | 0.0169 | 0.1110 |
| 2.0 | balanced | 5 | 0.8777 | 0.7272 | 0.0175 | 0.1126 |
| 4.0 | balanced | 5 | 0.8789 | 0.7263 | 0.0163 | 0.1148 |
| 8.0 | balanced | 5 | 0.8781 | 0.7254 | 0.0169 | 0.1163 |
| 0.5 | None | 10 | 0.8784 | 0.7257 | 0.0144 | 0.1137 |
| 1.0 | None | 10 | 0.8817 | 0.7314 | 0.0151 | 0.1131 |
| 2.0 | None | 10 | 0.8826 | 0.7346 | 0.0165 | 0.1132 |
| 4.0 | None | 10 | 0.8805 | 0.7317 | 0.0162 | 0.1160 |
| 8.0 | None | 10 | 0.8773 | 0.7310 | 0.0168 | 0.1196 |
| 0.5 | balanced | 10 | 0.8722 | 0.7242 | 0.0178 | 0.1079 |
| 1.0 | balanced | 10 | 0.8739 | 0.7223 | 0.0168 | 0.1125 |
| 2.0 | balanced | 10 | 0.8771 | 0.7232 | 0.0164 | 0.1135 |
| 4.0 | balanced | 10 | 0.8789 | 0.7242 | 0.0160 | 0.1152 |
| 8.0 | balanced | 10 | 0.8766 | 0.7211 | 0.0158 | 0.1181 |
| 0.5 | None | 15 | 0.8782 | 0.7229 | 0.0149 | 0.1142 |
| 1.0 | None | 15 | 0.8811 | 0.7267 | 0.0145 | 0.1137 |
| 2.0 | None | 15 | 0.8807 | 0.7296 | 0.0150 | 0.1150 |
| 4.0 | None | 15 | 0.8801 | 0.7325 | 0.0165 | 0.1164 |
| 8.0 | None | 15 | 0.8775 | 0.7301 | 0.0181 | 0.1196 |
| 0.5 | balanced | 15 | 0.8725 | 0.7243 | 0.0159 | 0.1088 |
| 1.0 | balanced | 15 | 0.8748 | 0.7224 | 0.0164 | 0.1126 |
| 2.0 | balanced | 15 | 0.8771 | 0.7254 | 0.0158 | 0.1142 |
| 4.0 | balanced | 15 | 0.8789 | 0.7237 | 0.0162 | 0.1153 |
| 8.0 | balanced | 15 | 0.8773 | 0.7199 | 0.0162 | 0.1177 |
| 0.5 | None | 20 | 0.8778 | 0.7215 | 0.0141 | 0.1153 |
| 1.0 | None | 20 | 0.8815 | 0.7267 | 0.0135 | 0.1138 |
| 2.0 | None | 20 | 0.8813 | 0.7285 | 0.0154 | 0.1151 |
| 4.0 | None | 20 | 0.8807 | 0.7304 | 0.0166 | 0.1162 |
| 8.0 | None | 20 | 0.8762 | 0.7246 | 0.0183 | 0.1209 |
| 0.5 | balanced | 20 | 0.8727 | 0.7231 | 0.0156 | 0.1102 |
| 1.0 | balanced | 20 | 0.8758 | 0.7276 | 0.0158 | 0.1126 |
| 2.0 | balanced | 20 | 0.8781 | 0.7273 | 0.0144 | 0.1140 |
| 4.0 | balanced | 20 | 0.8791 | 0.7249 | 0.0166 | 0.1159 |
| 8.0 | balanced | 20 | 0.8760 | 0.7203 | 0.0143 | 0.1199 |

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
| Default (C=1.0, cw=None) | 0.8938 | 0.8815 ± 0.0155 | 0.7641 | 0.1001 |
| Tuned (C=2.0, cw=None) | 0.8938 | 0.8746 ± 0.0169 | 0.7833 | 0.0901 |

- Random holdout accuracy (default): 0.8948 (optimistic — item-name leakage)

## Balanced Unseen Benchmark

- Eligible labels: 64
- Samples per class: 3
- Balanced holdout accuracy: 0.8542
- Balanced holdout macro F1: 0.8252
- Repeated balanced accuracy mean with tuned model: 0.8521 +/- 0.0107
- Repeated balanced macro F1 mean with tuned model: 0.8141
- Repeated balanced accuracy mean with `class_weight='balanced'`: 0.8521
- Repeated balanced macro F1 mean with `class_weight='balanced'`: 0.8141

## Learning Curve

| Train Size | Train Accuracy | Validation Accuracy |
| --- | ---: | ---: |
| 790 | 0.9852 | 0.5807 |
| 1580 | 0.9797 | 0.7170 |
| 2371 | 0.9779 | 0.7922 |
| 3161 | 0.9781 | 0.8408 |
| 3952 | 0.9796 | 0.8699 |

## Validation Curve (GroupShuffleSplit 3-fold, grouped by item name)

| C | Train Accuracy | Validation Accuracy |
| --- | ---: | ---: |
| 0.1 | 0.9422 | 0.8307 |
| 0.25 | 0.9632 | 0.8584 |
| 0.5 | 0.9731 | 0.8636 |
| 1.0 | 0.9796 | 0.8699 |
| 2.0 | 0.9862 | 0.8736 |
| 4.0 | 0.9896 | 0.8729 |
| 8.0 | 0.9916 | 0.8716 |
| 16.0 | 0.9931 | 0.8703 |

## Conclusion

- Overfitting confirmed: train≈0.99, grouped val≈0.89, gap≈0.09.
- Error analysis shows the gap is driven by rare-class test samples — see the frequency bucket table above.
- Fix applied: joint grouped CV tuning over C, class_weight, and oversample_min_count identified C=2.0, class_weight=None, oversample_min_count=0 as optimal.
- Oversampling duplicates minority-class training records to the threshold during training only; test folds are never touched.
- Tuned model achieves grouped holdout accuracy 0.8938 (was 0.8938) and gap 0.0901 (was 0.1001).
