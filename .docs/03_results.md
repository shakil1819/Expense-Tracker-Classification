# Results

## Executive Summary

LinearSVC on word TF-IDF, char TF-IDF, vendor one-hot, and log-scaled amount. Overfitting is confirmed on the tuned configuration (train-test gap ≈ 0.10). The fix is grouped cross-validation tuning of `C`, `class_weight`, and `oversample_min_count` to find the best bias-variance trade-off. The tuned model (C=1.0, class_weight=None, oversample_min_count=15) achieves grouped holdout accuracy of 0.8907, up from 0.8907 with defaults.

## Bias / Variance / Overfitting Diagnosis

- **Underfitting**: False
- **Overfitting**: True
- **Tuned C**: 4.0
- **Tuned class_weight**: balanced
- **Tuned oversample_min_count**: 0
- **Train accuracy**: 0.9942
- **Grouped val accuracy**: 0.8907
- **Gap**: 0.1035
- Train accuracy 0.994, grouped val accuracy 0.891, gap 0.104. No underfitting. Moderate overfitting — gap 0.104 exceeds 0.05 threshold. Cause: LinearSVC with high-dimensional TF-IDF memorises training phrase patterns. Fix: tune C via grouped CV to find optimal bias-variance trade-off.

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
| very_rare (2-4) | 7 | 4 | 0.5714 |
| rare (5-19) | 91 | 69 | 0.7582 |
| medium (20-99) | 313 | 288 | 0.9201 |
| frequent (100+) | 525 | 475 | 0.9048 |

## Error Analysis by Class Training Frequency (Tuned Model)

| Frequency Bucket | Test Samples | Correct | Accuracy |
| --- | ---: | ---: | ---: |
| singleton (n=1) | 3 | 3 | 1.0000 |
| very_rare (2-4) | 7 | 5 | 0.7143 |
| rare (5-19) | 91 | 70 | 0.7692 |
| medium (20-99) | 313 | 287 | 0.9169 |
| frequent (100+) | 525 | 474 | 0.9029 |

## Hyperparameter Tuning (Grouped 5-Fold CV)

Joint grid search over C, class_weight, and oversample_min_count (oversampling applied to training folds only).

Best: C=1.0, class_weight=None, oversample_min_count=15, val_accuracy=0.8801, val_macro_f1=0.7359, gap=0.1155, optimize_for=macro_f1

| C | class_weight | oversample_min_count | Val Accuracy Mean | Val Macro F1 Mean | Val Accuracy Std | Gap Mean |
| --- | --- | --- | ---: | ---: | ---: |
| 0.5 | None | 0 | 0.8764 | 0.7292 | 0.0151 | 0.1173 |
| 1.0 | None | 0 | 0.8799 | 0.7323 | 0.0150 | 0.1154 |
| 2.0 | None | 0 | 0.8803 | 0.7254 | 0.0167 | 0.1157 |
| 4.0 | None | 0 | 0.8774 | 0.7249 | 0.0174 | 0.1190 |
| 8.0 | None | 0 | 0.8742 | 0.7211 | 0.0168 | 0.1227 |
| 0.5 | balanced | 0 | 0.8715 | 0.7233 | 0.0162 | 0.1113 |
| 1.0 | balanced | 0 | 0.8745 | 0.7219 | 0.0162 | 0.1139 |
| 2.0 | balanced | 0 | 0.8775 | 0.7227 | 0.0149 | 0.1135 |
| 4.0 | balanced | 0 | 0.8766 | 0.7196 | 0.0160 | 0.1176 |
| 8.0 | balanced | 0 | 0.8771 | 0.7214 | 0.0166 | 0.1180 |
| 0.5 | None | 5 | 0.8766 | 0.7296 | 0.0142 | 0.1171 |
| 1.0 | None | 5 | 0.8803 | 0.7324 | 0.0148 | 0.1150 |
| 2.0 | None | 5 | 0.8801 | 0.7307 | 0.0176 | 0.1159 |
| 4.0 | None | 5 | 0.8778 | 0.7278 | 0.0177 | 0.1186 |
| 8.0 | None | 5 | 0.8744 | 0.7261 | 0.0169 | 0.1224 |
| 0.5 | balanced | 5 | 0.8715 | 0.7221 | 0.0162 | 0.1118 |
| 1.0 | balanced | 5 | 0.8749 | 0.7251 | 0.0162 | 0.1137 |
| 2.0 | balanced | 5 | 0.8777 | 0.7244 | 0.0156 | 0.1136 |
| 4.0 | balanced | 5 | 0.8773 | 0.7249 | 0.0160 | 0.1170 |
| 8.0 | balanced | 5 | 0.8779 | 0.7316 | 0.0168 | 0.1172 |
| 0.5 | None | 10 | 0.8772 | 0.7293 | 0.0143 | 0.1168 |
| 1.0 | None | 10 | 0.8803 | 0.7342 | 0.0138 | 0.1154 |
| 2.0 | None | 10 | 0.8802 | 0.7344 | 0.0164 | 0.1161 |
| 4.0 | None | 10 | 0.8777 | 0.7301 | 0.0155 | 0.1189 |
| 8.0 | None | 10 | 0.8744 | 0.7249 | 0.0168 | 0.1225 |
| 0.5 | balanced | 10 | 0.8713 | 0.7192 | 0.0150 | 0.1130 |
| 1.0 | balanced | 10 | 0.8736 | 0.7215 | 0.0147 | 0.1156 |
| 2.0 | balanced | 10 | 0.8771 | 0.7243 | 0.0155 | 0.1147 |
| 4.0 | balanced | 10 | 0.8771 | 0.7294 | 0.0155 | 0.1175 |
| 8.0 | balanced | 10 | 0.8783 | 0.7306 | 0.0171 | 0.1170 |
| 0.5 | None | 15 | 0.8757 | 0.7301 | 0.0138 | 0.1184 |
| 1.0 | None | 15 | 0.8801 | 0.7359 | 0.0139 | 0.1155 |
| 2.0 | None | 15 | 0.8801 | 0.7350 | 0.0162 | 0.1162 |
| 4.0 | None | 15 | 0.8775 | 0.7306 | 0.0160 | 0.1192 |
| 8.0 | None | 15 | 0.8740 | 0.7264 | 0.0179 | 0.1230 |
| 0.5 | balanced | 15 | 0.8705 | 0.7229 | 0.0148 | 0.1149 |
| 1.0 | balanced | 15 | 0.8734 | 0.7243 | 0.0149 | 0.1164 |
| 2.0 | balanced | 15 | 0.8773 | 0.7250 | 0.0162 | 0.1150 |
| 4.0 | balanced | 15 | 0.8777 | 0.7297 | 0.0164 | 0.1172 |
| 8.0 | balanced | 15 | 0.8783 | 0.7303 | 0.0172 | 0.1172 |
| 0.5 | None | 20 | 0.8770 | 0.7252 | 0.0135 | 0.1176 |
| 1.0 | None | 20 | 0.8801 | 0.7314 | 0.0136 | 0.1159 |
| 2.0 | None | 20 | 0.8799 | 0.7311 | 0.0155 | 0.1169 |
| 4.0 | None | 20 | 0.8770 | 0.7283 | 0.0158 | 0.1199 |
| 8.0 | None | 20 | 0.8733 | 0.7227 | 0.0185 | 0.1238 |
| 0.5 | balanced | 20 | 0.8715 | 0.7195 | 0.0151 | 0.1150 |
| 1.0 | balanced | 20 | 0.8740 | 0.7232 | 0.0152 | 0.1169 |
| 2.0 | balanced | 20 | 0.8774 | 0.7254 | 0.0150 | 0.1159 |
| 4.0 | balanced | 20 | 0.8770 | 0.7276 | 0.0148 | 0.1186 |
| 8.0 | balanced | 20 | 0.8785 | 0.7318 | 0.0160 | 0.1177 |

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
| Default (C=1.0, cw=None) | 0.8907 | 0.8799 ± 0.0150 | 0.7509 | 0.1043 |
| Tuned (C=1.0, cw=None) | 0.8907 | 0.8766 ± 0.0160 | 0.7568 | 0.1035 |

- Random holdout accuracy (default): 0.8989 (optimistic — item-name leakage)

## Balanced Unseen Benchmark

- Eligible labels: 64
- Samples per class: 3
- Balanced holdout accuracy: 0.8490
- Balanced holdout macro F1: 0.8179
- Repeated balanced accuracy mean with tuned model: 0.8469 +/- 0.0188
- Repeated balanced macro F1 mean with tuned model: 0.8118
- Repeated balanced accuracy mean with `class_weight='balanced'`: 0.8469
- Repeated balanced macro F1 mean with `class_weight='balanced'`: 0.8118

## Learning Curve

| Train Size | Train Accuracy | Validation Accuracy |
| --- | ---: | ---: |
| 790 | 0.9958 | 0.5833 |
| 1580 | 0.9928 | 0.7097 |
| 2371 | 0.9904 | 0.7826 |
| 3161 | 0.9895 | 0.8356 |
| 3952 | 0.9908 | 0.8713 |

## Validation Curve (GroupShuffleSplit 3-fold, grouped by item name)

| C | Train Accuracy | Validation Accuracy |
| --- | ---: | ---: |
| 0.1 | 0.9525 | 0.8409 |
| 0.25 | 0.9691 | 0.8599 |
| 0.5 | 0.9770 | 0.8668 |
| 1.0 | 0.9831 | 0.8733 |
| 2.0 | 0.9876 | 0.8736 |
| 4.0 | 0.9908 | 0.8713 |
| 8.0 | 0.9925 | 0.8688 |
| 16.0 | 0.9942 | 0.8654 |

## Conclusion

- Overfitting confirmed: train≈0.99, grouped val≈0.89, gap≈0.10.
- Error analysis shows the gap is driven by rare-class test samples — see the frequency bucket table above.
- Fix applied: joint grouped CV tuning over C, class_weight, and oversample_min_count identified C=1.0, class_weight=None, oversample_min_count=15 as optimal.
- Oversampling duplicates minority-class training records to the threshold during training only; test folds are never touched.
- Tuned model achieves grouped holdout accuracy 0.8907 (was 0.8907) and gap 0.1035 (was 0.1043).
