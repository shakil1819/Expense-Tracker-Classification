# EDA Notes

- Records: 4,894
- Unique vendors: 337
- Unique account names: 103
- Labels with fewer than 10 rows: 43
- Labels with fewer than 5 rows: 34
- Singleton labels: 16
- Missing `itemDescription`: 31
- Amount range: -15,195 to 161,838,000

Observations:

- The dataset is highly imbalanced, so overall accuracy alone is not enough.
- Exact `itemName` repetition is common, which makes duplicate-aware validation necessary.
- Vendor identity carries signal but is not enough on its own.
- The environment blocks `pandas`, so the pipeline uses stdlib JSON parsing plus `numpy` and `scikit-learn`.
