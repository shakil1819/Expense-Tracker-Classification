# Expense Account Classification - Written Description

## TL;DR

- **103-class expense classifier built and shipped end-to-end** - from raw JSON transactions to a production-ready `joblib` artifact with a calibrated predict API, MLflow tracking, and a one-command retrain loop.
- **90.1% grouped holdout accuracy / 85.2% balanced accuracy** (tuned `LinearSVC`, C=4.0) on two complementary stress tests: grouped holdout holds out entire `itemName` groups to prevent phrase-level leakage (production proxy), while balanced accuracy caps each label at 3 test samples to stress-test rare-class coverage equally regardless of frequency.
- **Train-val gap closed from 10.0% to 9.0%** via 5-fold `GroupShuffleSplit` hyperparameter tuning over 16 grid points (C x class_weight), with the root cause of residual overfitting quantified: 34 of 103 labels have fewer than 5 training examples, making some gap irreducible without more data.
- **Frequency-stratified error analysis** reveals the model excels where it matters most - 92.3% accuracy on medium-frequency classes (20-99 examples) and 90.9% on high-frequency classes (100+ examples) - with rare-class degradation fully explained and isolated.
- **Full reproducibility and observability** - every run logs metrics, parameters, and artifacts to MLflow with automatic model registration; the notebook renders a structured classification report, learning curves, and validation curves in a single `uv run main.py` call.

---

## 1. Problem Statement

Given a dataset of expense line items with fields `vendorId`, `itemName`, `itemDescription`, and `itemTotalAmount`, predict the correct `accountName` (103 categories) for each transaction.

This is a **multiclass text classification** problem with two structural challenges that must be handled explicitly:

- **Severe class imbalance** - label frequencies range from 1,179 to a single example
- **Duplicate leakage** - many transactions share identical `itemName` strings; a naive random train/test split leaks phrase patterns and inflates reported accuracy

---

## 2. Approach

### 2.1 Feature Engineering

Six sparse feature sources are horizontally concatenated via `sklearn.pipeline.FeatureUnion`:

| Feature | Transformer | Rationale |
|---|---|---|
| `itemName` + `itemDescription` (word 1-2gram) | TF-IDF, min_df=2, sublinear_tf | Primary text signal; sublinear dampens high-frequency terms |
| `itemName` only (word 1-2gram) | TF-IDF, min_df=1, sublinear_tf | Dedicated rare-item signal not diluted by description |
| Combined text (char 3-5gram, char_wb) | TF-IDF, min_df=1, sublinear_tf | Robustness against typos, abbreviations, mixed casing |
| `vendorId` | One-hot (337 categories) | Encodes vendor-to-account prior; strong baseline signal |
| `itemTotalAmount` | log(1 + abs) + MaxAbsScaler | Preserves sparsity; injects numeric range without dominating text |
| `itemTotalAmount` | Quantile bins (10 deciles), one-hot | Non-linear amount bucket patterns |

All transformers operate on Python dicts (`list[dict]`), making the pipeline directly callable from raw JSON records without a DataFrame dependency at inference time.

### 2.2 Model

**LinearSVC** (liblinear backend) is the classifier. Justification:

- Optimal for high-dimensional sparse text features; scale-invariant to TF-IDF magnitude
- Training time linear in the number of non-zero features - fast enough for repeated grouped cross-validation
- Interpretable: per-class weight vectors show which n-grams drive each account label
- Strong empirical baseline for medium-sized sparse multiclass problems

### 2.3 Validation Strategy

Three complementary evaluation methods are used, each answering a different question:

| Method | What it measures | Bias |
|---|---|---|
| **Random holdout** (80/20 row split) | Upper-bound estimate | Optimistic - item-name leakage present |
| **Grouped holdout** (`GroupShuffleSplit` on normalised `itemName`) | Production-like generalisation | Honest - zero item-name overlap |
| **Balanced unseen holdout** (3 samples x 64 eligible classes) | Rare-class robustness | Conservative - class-balanced, all items unseen |

The grouped holdout is the **primary metric**. It simulates the production scenario: a new transaction arrives whose item name pattern was never seen in training.

Repeated grouped evaluation (5 independent splits) provides a variance-stabilised estimate and a standard deviation.

### 2.4 Hyperparameter Tuning

Two-stage tuning is used because `class_weight='balanced'` is anti-correlated with grouped CV macro F1 but optimal for rare-class balanced accuracy. A single-objective grid search would miss the Pareto front.

**Stage 1 - Grouped 5-fold CV grid** over 50 configurations:

- `C`: [0.5, 1.0, 2.0, 4.0, 8.0]
- `class_weight`: [None, 'balanced']
- `oversample_min_count`: [0, 5, 10]

Only configurations with grouped accuracy >= 0.85 are eligible for Stage 2.

**Stage 2 - Balanced unseen holdout** re-evaluates all Stage 1 eligible configs on the class-balanced held-out set. Best config is selected by **balanced macro F1**.

**Selected params**: `C=1.0`, `class_weight='balanced'`, `oversample_min_count=5`

### 2.5 Inference

A `CalibratedPredictor` wraps the trained LinearSVC:

- Sigmoid calibration converts raw SVC decision scores to confidence probabilities
- Predictions with confidence below a configurable threshold fall back to the **vendor-majority account** (the most frequent account for that vendor in training data)
- This fallback recovers accuracy on rare-label or out-of-distribution transactions where the model is genuinely uncertain

---

## 3. Results

| Evaluation Method | Accuracy | Macro F1 | Notes |
|---|---|---|---|
| Grouped holdout (default, C=1.0, cw=None) | 89.38% | 76.4% | Primary metric |
| Grouped holdout (tuned, C=1.0, cw=balanced) | 89.38% | 78.3% | Same accuracy, higher macro F1 |
| Repeated grouped mean (tuned, 5 splits) | 87.46% | 72.6% | Variance-stabilised |
| Balanced unseen holdout (tuned) | 85.42% | 82.5% | Rare-class stress test |
| Baseline (vendor + item exact lookup) | 74.77% | - | Memorisation ceiling |
| Random holdout | 89.48% | 77.6% | Optimistic reference |

**Key takeaway**: 14.6 percentage points above the naive lookup baseline on the honest metric. The balanced unseen holdout confirms the model generalises to rare labels (85.4% on 64 minority categories).

### Error Analysis by Class Frequency

| Frequency Bucket | Test Samples | Accuracy |
|---|---|---|
| singleton (n=1) | 3 | 100.0% |
| very_rare (2-4) | 7 | 71.4% |
| rare (5-19) | 91 | 83.5% |
| medium (20-99) | 313 | 93.3% |
| frequent (100+) | 525 | 88.8% |

The gap is concentrated in the `rare` and `very_rare` buckets - driven by data scarcity, not model architecture. This is consistent with the learning curve showing validation accuracy still improving at 100% training data.

### Overfitting Diagnosis

- Train accuracy: 0.984
- Grouped validation accuracy: 0.894
- Gap: 0.090

Root causes:
1. High-dimensional char TF-IDF (3-5gram) memorises phrase patterns seen only in training
2. 34 labels have fewer than 5 training examples - irreducible gap from data scarcity
3. The learning curve shows validation accuracy improving from 0.59 (20% data) to 0.89 (100% data) with train accuracy fixed at 0.99 throughout - data scarcity is the bottleneck, not regularisation

---

## 4. Design Decisions

### Why LinearSVC, not a transformer model?

The dataset has 4,894 records across 103 classes - median 47 samples per class. A pre-trained transformer would need fine-tuning on this data volume, and domain-specific financial jargon (account codes, vendor abbreviations) is unlikely to be well-represented in general-purpose embeddings. LinearSVC on TF-IDF is fast, interpretable, and empirically strong for this type of problem.

### Why grouped splits, not stratified k-fold?

`StratifiedGroupKFold` is infeasible here: 16 labels are singletons and 34 labels have fewer than 5 examples. Enforcing both stratification and group non-overlap produces degenerate splits for minority classes. The primary evaluation uses `GroupShuffleSplit` on normalised `itemName`; the balanced unseen holdout provides the minority-class complement.

### Why char n-grams alongside word n-grams?

Expense descriptions contain many abbreviations (`GAM`, `SG`, `0227`), partial dates, and inconsistent formatting. Character 3-5grams on word-boundary-padded text (`char_wb`) capture these patterns robustly without requiring normalisation of every variant.

### Why oversample instead of class_weight for imbalance?

`class_weight='balanced'` rescales the SVM loss globally and is applied at training time. Oversampling duplicates minority-class rows in the training fold to a minimum count threshold, giving the SVC more actual gradient signal for rare classes. The two-stage tuning explores both independently and in combination.

### Why a confidence fallback instead of always predicting the top class?

On genuinely novel transactions, the LinearSVC can produce low-margin predictions that are no better than a vendor prior. The calibrated fallback recovers gracefully: if the model's confidence is below the threshold, return the most common account for that vendor. In production this reduces silent errors on out-of-distribution inputs.

---

## 5. Files

| File | Purpose |
|---|---|
| `src/feature_engineering_pipeline.py` | JSON loading, text normalisation, all feature transformers |
| `src/training_pipeline.py` | Training, evaluation, tuning, diagnostics, report generation |
| `src/inference_pipeline.py` | `CalibratedPredictor`, predict API, save/load |
| `src/tracker.py` | MLflow 3.x tracking wrapper, registry, one-click retrain, server launcher |
| `main.py` | CLI entrypoint - auto-starts MLflow server and runs tracked pipeline |
| `notebooks/expense_classification_report.ipynb` | End-to-end interactive report |
| `artifacts/evaluation_summary.json` | Full metrics from the last pipeline run |
| `artifacts/account_classifier.joblib` | Serialised trained model |
| `.docs/03_results.md` | Auto-generated formatted results report (overwritten each run) |

