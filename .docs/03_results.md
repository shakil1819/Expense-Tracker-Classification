# Results

## Executive Summary

The final model is a linear SVM over word n-grams, character n-grams, vendor ID, and log-scaled amount. It comfortably clears the 85% target on both a standard shuffled split and a stricter grouped split that keeps normalized `itemName` values out of both train and test.

## Data Analysis

- Records: 4894
- Unique vendors: 337
- Unique account names: 103
- Labels with fewer than 10 rows: 43
- Singleton labels: 16
- Missing descriptions: 31
- Amount range: -15195.0 to 161838000.0
- Exact item-name lookup baseline accuracy: 0.7426

## Validation

- Repeated shuffled holdout accuracy: 0.9075 +/- 0.0067
- Repeated shuffled holdout macro F1: 0.7868
- Repeated grouped holdout accuracy: 0.8771 +/- 0.0150
- Repeated grouped holdout macro F1: 0.7339
- Single shuffled holdout accuracy: 0.9050
- Single grouped holdout accuracy: 0.8917

## Strongest Labels

| Label | F1 | Precision | Recall | Support |
| --- | ---: | ---: | ---: | ---: |
| 511301 Display COGS | 1.000 | 1.000 | 1.000 | 15 |
| 511102 External Commission | 1.000 | 1.000 | 1.000 | 10 |
| 617202 Legal Expenses | 1.000 | 1.000 | 1.000 | 8 |
| 223001 Salaries Payable | 1.000 | 1.000 | 1.000 | 6 |
| 232101 Lease payable | 1.000 | 1.000 | 1.000 | 6 |
| 611101 Cloud server - AWS | 1.000 | 1.000 | 1.000 | 6 |
| 131020 Unbilled receivables | 1.000 | 1.000 | 1.000 | 5 |
| 617101 Customer Support - Others | 1.000 | 1.000 | 1.000 | 5 |
| 619202 Cleaning | 1.000 | 1.000 | 1.000 | 5 |
| 617103 Subcontractors/Outsource | 0.983 | 0.967 | 1.000 | 29 |

## Weakest Labels

| Label | F1 | Precision | Recall | Support |
| --- | ---: | ---: | ---: | ---: |
| 134004 Prepaid Subscription | 0.400 | 0.500 | 0.333 | 6 |
| 134001 Prepaid Operating Expense | 0.716 | 0.725 | 0.707 | 41 |
| 612016 Collateral | 0.769 | 0.833 | 0.714 | 7 |
| 619209 Others | 0.791 | 0.944 | 0.680 | 25 |
| 132098 IC Clearing account - Paid on Behalf | 0.800 | 0.870 | 0.741 | 27 |
| 134002 Prepaid Insurance | 0.800 | 1.000 | 0.667 | 6 |
| 612001 Paid Social | 0.875 | 0.933 | 0.824 | 17 |
| 614306 Other Employee Expenses | 0.875 | 1.000 | 0.778 | 9 |
| 619207 Utilities | 0.880 | 0.846 | 0.917 | 12 |
| 619201 Equipment Expense | 0.885 | 0.885 | 0.885 | 26 |

## Discussion

- The dominant signal is transaction text. Vendor ID is useful when the same supplier consistently maps to one account.
- Grouped validation is lower than shuffled validation, which is expected because it removes repeated `itemName` leakage.
- Rare labels remain the main weakness. Many classes have too little support for stable estimates, so macro F1 is the better caution metric than overall accuracy alone.
