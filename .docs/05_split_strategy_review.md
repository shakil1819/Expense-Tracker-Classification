# Split Strategy Review

## Key Findings

- The current model is not underfitting. Training accuracy is near 0.99 and grouped unseen accuracy stays around 0.88 to 0.89.
- There is moderate overfitting. The grouped train-test gap is about 0.10, and the balanced unseen train-test gap is about 0.15.
- Random row-level splitting is too optimistic for this dataset because repeated `itemName` patterns can appear in both train and test.

## Why A Balanced Test Set Is Not Enough

- A balanced test set is useful to measure how evenly the model handles classes.
- A balanced test set is not the same as a real-life estimate unless production traffic is also balanced.
- The repo now reports both:
  - grouped natural-distribution metrics for production-like performance
  - balanced unseen metrics for minority-class stress testing

## Recommended Reporting

1. Report grouped natural-distribution accuracy as the primary score.
2. Report macro F1 alongside accuracy because class imbalance is severe.
3. Report the balanced unseen benchmark separately to show rare-class behavior.
4. Do not rely on random row splits as the headline result.
