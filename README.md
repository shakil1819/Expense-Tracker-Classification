# Peakflo Expense Classifier

Python solution for the Peakflo take-home task. The code now lives under `src/` and is split into `feature_engineering_pipeline.py`, `training_pipeline.py`, and `inference_pipeline.py`.

## Run

```powershell
python -m uv run python main.py run
```

This writes:

- `artifacts/evaluation_summary.json`
- `artifacts/account_classifier.joblib`
- `.docs/03_results.md`

## Test

```powershell
python -m uv run pytest
```

## Notes

- The implementation avoids `pandas` because compiled `pandas` wheels are blocked in this environment.
- The primary evaluation uses grouped holdouts by normalized `itemName` to reduce duplicate-text leakage.
- The report now includes both natural-distribution metrics and a balanced unseen benchmark.
