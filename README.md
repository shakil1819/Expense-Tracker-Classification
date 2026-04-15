# Peakflo Expense Classifier

Python solution for the Peakflo take-home task. The pipeline trains a multiclass classifier that predicts `accountName` from bill text, vendor, and amount.

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
- Validation includes a standard shuffled split and a stricter grouped split by normalized `itemName` to reduce duplicate-text leakage.
