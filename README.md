# Expense Account Classifier

**89.4% grouped holdout accuracy on 103 account categories - 14.6 points above the naive vendor-lookup baseline - with a validated rare-class benchmark showing 85.4% accuracy across 64 minority categories.**

---

## What It Does

Automatically classifies expense transactions to their correct accounting account using only the fields available at invoice time: item name, item description, vendor ID, and amount. No manual review. No hardcoded rules.

The model handles the two hardest real-world conditions simultaneously:

- **Class imbalance** - 103 account categories ranging from 1,179 records down to a single example
- **Duplicate leakage** - Many expense lines share exact item names across train and test; evaluation is explicitly designed to prevent this from inflating scores

---

## Results

| Evaluation Method | Accuracy | Macro F1 | Notes |
|---|---|---|---|
| Grouped holdout (80/20, item-name split) | **89.38%** | 76.4% | Primary metric - zero item-name leakage |
| Repeated grouped (5×, averaged) | **87.46%** | - | Variance-stabilized estimate |
| Random holdout (80/20) | 89.48% | 77.6% | Optimistic - included for reference only |
| **Balanced unseen holdout** | **85.42%** | **82.5%** | Stress test: 3 unseen examples × 64 rare categories |
| Baseline (vendor+item lookup) | 74.77% | - | Naive memorisation baseline |

> The grouped holdout is the honest number. It simulates production: the model sees a new expense whose item name never appeared in training and must generalise from vendor, amount, and text patterns alone.

---

## Architecture

```mermaid
flowchart LR

  subgraph FE["` **Feature Engineering Pipeline** `"]
    direction TB
    RAW["` **Raw Input**
    vendorId · itemName
    itemDescription · amount `"]
    NORM["` **Normalise**
    lowercase · strip · dedupe whitespace `"]
    FU["` **Feature Union** (sparse concat) `"]
    W["` word_tfidf
    1-2gram · min_df=2 · sublinear `"]
    I["` item_name_tfidf
    1-2gram · min_df=1 · sublinear `"]
    C["` char_tfidf
    3-5gram · min_df=1 · sublinear `"]
    V["` vendor one-hot
    337 categories `"]
    A["` amount scaler
    log + MaxAbs `"]
    B["` amount bins
    10 quantile bins · one-hot `"]

    RAW --> NORM --> FU
    FU --> W
    FU --> I
    FU --> C
    FU --> V
    FU --> A
    FU --> B
  end

  subgraph TR["` **Training Pipeline** `"]
    direction TB
    SPLIT["` **GroupShuffleSplit**
    split on normalised itemName
    no item-name leakage `"]
    OVS["` **Oversample**
    minority classes → min_count=5
    train folds only `"]
    SVC["` **LinearSVC**
    class_weight=balanced
    C=1.0 `"]
    TUNE["` **Two-Stage Tuning**
    Stage 1: grouped 5-fold CV · 50 configs
    Stage 2: balanced unseen holdout
    select by balanced macro F1 `"]
    EVAL["` **Evaluation**
    grouped holdout 89.4%
    balanced unseen 85.4% · macro F1 82.5% `"]

    SPLIT --> OVS --> SVC --> TUNE --> EVAL
  end

  subgraph INF["` **Inference Pipeline** `"]
    direction TB
    CAL["` **CalibratedPredictor**
    sigmoid calibration on frozen SVC `"]
    CONF{"` confidence
    ≥ threshold? `"}
    PRED["` **Model prediction** `"]
    FALL["` **Vendor fallback**
    majority account for vendor `"]
    OUT["` **accountName** `"]

    CAL --> CONF
    CONF -->|yes| PRED --> OUT
    CONF -->|no| FALL --> OUT
  end

  FE -->|"` sparse feature matrix `"| TR
  TR -->|"` fitted Pipeline + joblib `"| INF
```

**Model**: `LinearSVC` wrapped in a scikit-learn `Pipeline`

**Feature union** (all sparse, concatenated):

| Feature | Transformer | Signal |
|---|---|---|
| `itemName` + `itemDescription` (word 1-2gram) | TF-IDF, min_df=2, sublinear | Dominant text signal |
| `itemName` only (word 1-2gram) | TF-IDF, min_df=1, sublinear | Dedicated rare-item signal |
| Combined text (char 3-5gram) | TF-IDF, min_df=1, sublinear | Typo + partial-match robustness |
| `vendorId` | One-hot (337 categories) | Vendor → account priors |
| `itemTotalAmount` | Log-scaled + MaxAbs | Amount range signal |
| `itemTotalAmount` | Quantile bins (10), one-hot | Non-linear amount bucket signal |

**Hyperparameter selection**: two-stage tuning —
1. 50-config grouped 5-fold CV grid (C × class_weight × oversample_min_count) to identify production-viable configs (grouped accuracy floor ≥ 0.85)
2. All eligible configs re-evaluated on the balanced unseen holdout; best selected by **balanced macro F1**

> Key insight: `class_weight='balanced'` is anti-correlated with grouped CV macro F1 but optimal for rare-class balanced accuracy. Stage 1 must not rank by grouped macro F1 before Stage 2 evaluates rare-class performance.

**Production params selected**: `C=1.0`, `class_weight='balanced'`, `oversample_min_count=5`

**Inference**: `CalibratedPredictor` - sigmoid-calibrated confidence scores with vendor-majority fallback when model confidence falls below threshold.

---

## Project Structure

```
src/
  feature_engineering_pipeline.py  # Data loading, text normalisation, feature transformers
  training_pipeline.py             # Model training, evaluation, tuning, diagnostics
  inference_pipeline.py            # CalibratedPredictor, save/load, predict API
  tracker.py                       # MLflow integration: tracking, registry, tracing, one-click retrain
tests/
  test_model.py                    # Pipeline, tuning, and evaluation tests
  test_data.py                     # Data loading and normalisation tests
  test_tracker.py                  # MLflow tracker tests
artifacts/
  evaluation_summary.json          # Full metrics from last pipeline run
  account_classifier.joblib        # Serialised trained model
mlruns/                            # Local MLflow experiment store (file backend, gitignored)
.docs/
  01_plan.md                       # Initial plan
  02_eda.md                        # EDA findings
  03_results.md                    # Auto-generated results report
  04_tutorial.md                   # End-to-end tutorial
  05_mlflow.md                     # MLflow tracker usage guide
  05_split_strategy_review.md      # Validation strategy analysis
  06_rare_class_improvements.md    # Rare-class tuning decisions and rationale
```

---

## Quickstart

**Requirements**: Python 3.14+, [uv](https://github.com/astral-sh/uv)

```powershell
# Install dependencies
uv sync

# Train, evaluate, and save model
. .\.venv\Scripts\Activate.ps1
uv run main.py
```

Outputs written to:
- `artifacts/evaluation_summary.json` - full metrics
- `artifacts/account_classifier.joblib` - trained model
- `.docs/03_results.md` - formatted results report
- `data/<timestamp>/` - train/test CSV exports for each split

```powershell
# Run the test suite (27 tests)
uv run pytest
```

---

## MLflow Integration

`src/tracker.py` wraps the training pipeline with MLflow 3.x: experiment
tracking, dataset validation and versioning, a metric card view, tracing, and a
model registry with a `champion` alias.

**One-click retrain** (runs the full pipeline under a fresh MLflow run, logs
params / metrics / dataset / artifacts / model, and promotes the new version to
`champion`):

```powershell
uv run python -m src.tracker retrain
```

**Local tracking server** (SQLite backend, browsable UI at <http://127.0.0.1:5000>):

```powershell
uv run python -m src.tracker server --backend-store-uri "sqlite:///mlflow.db" --artifacts-destination ".\mlartifacts"
```

**Load the registered champion model**:

```python
from src.tracker import MLflowTracker
model = MLflowTracker().load_model("champion")
preds = model.predict(list_of_feature_dicts)
```

Full guide: `.docs/05_mlflow.md`.

What gets logged per run:

- **Params**: best `C`, `class_weight`, `oversample_min_count`, dataset shape
- **Metrics**: baseline lookup, random / grouped / grouped-tuned holdout, balanced unseen, calibrated fallback (accuracy + macro F1 + train-test gap)
- **Metric card**: tabular comparison renderable in the MLflow UI
- **Dataset**: schema-validated `mlflow.data.PandasDataset` with row count and null counts
- **Artifacts**: `evaluation_summary.json`, `.docs/03_results.md`, `account_classifier.joblib`
- **Model**: registered as `peakflo-account-classifier`, aliased `champion`
- **Trace**: `run_training_pipeline` wrapped in an `mlflow.trace` span

---

## Validation Design

Three complementary evaluation methods, each measuring a different thing:

- **Random holdout** - upper-bound optimistic estimate; same item names can appear in both train and test
- **Grouped holdout** - honest production estimate; `GroupShuffleSplit` on normalised `itemName` ensures zero item-name overlap between train and test
- **Balanced unseen holdout** - rare-class stress test; holds out exactly 3 examples per eligible label (64 labels), all item names strictly unseen in training

The gap between random (90.5%) and grouped (87.5%) repeated means - **~3 points** - directly quantifies the duplicate leakage that row-level splits miss.
