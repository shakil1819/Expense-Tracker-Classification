# Peakflo Expense Classifier

Python solution for the Peakflo take-home task. The code now lives under `src/` and is split into `feature_engineering_pipeline.py`, `training_pipeline.py`, and `inference_pipeline.py`.

## Run

```powershell
. .\.venv\Scripts\Activate.ps1
uv run main.py
```

This writes:

- `artifacts/evaluation_summary.json`
- `artifacts/account_classifier.joblib`
- `.docs/03_results.md`

## Test

```powershell
. .\.venv\Scripts\Activate.ps1
python -m pytest
```

## Notes

- The dataset loading and summary path now uses `pandas`.
- The primary evaluation uses grouped holdouts by normalized `itemName` to reduce duplicate-text leakage.
- The report now includes both natural-distribution metrics and a balanced unseen benchmark.
