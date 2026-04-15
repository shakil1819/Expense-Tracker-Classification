# Plan

1. Inspect the dataset and quantify label imbalance, vendor concentration, and missing fields.
2. Build a lean text-classification pipeline in Python without over-engineering.
3. Validate on both shuffled and grouped splits so duplicate `itemName` values do not inflate the estimate.
4. Persist reproducible outputs and keep the git history specific to each task.
