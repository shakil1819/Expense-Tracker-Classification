# Expense Account Classification - Written Description

## Executive Summary

This project builds a multiclass expense-account classifier that predicts `accountName` from `vendorId`, `itemName`, `itemDescription`, and `itemTotalAmount`. The core solution is a sparse feature pipeline combining word-level TF-IDF, character n-grams, vendor one-hot features, and amount-based numeric features, trained with `LinearSVC`. The main challenge was not simply achieving high accuracy, but doing so under a realistic validation strategy that avoids leakage from repeated item-name patterns. Using grouped evaluation on normalised `itemName`, the final model achieved **89.38% grouped holdout accuracy**, exceeding the assignment threshold of 85%, and outperformed a naive vendor-plus-item lookup baseline by **14.6 percentage points**. A separate balanced unseen benchmark reached **85.42% accuracy** and **82.5% macro F1**, showing that the model remains effective even when rare classes are evaluated more fairly.

---

## Data Analysis

### Dataset Overview

The dataset contains **4,894 expense records** from a Singapore operation, spanning **337 vendors** and **103 account-name categories**. The target distribution is highly uneven: the largest class contains 1,179 records, while multiple labels appear only once. This immediately frames the task as an imbalanced multiclass classification problem rather than a standard balanced text-classification benchmark.

The available predictive fields are:

- `vendorId`
- `itemName`
- `itemDescription`
- `itemTotalAmount`

These fields are well-suited for a practical accounting classifier because they capture three distinct signals:

- **textual meaning** from item names and descriptions
- **vendor-specific priors** from recurring suppliers
- **numerical context** from transaction amount

### Exploratory Findings

The first major finding was that `itemName` carries very strong signal. Many transactions reuse near-identical or identical item names, which means a random row split can accidentally place the same phrase pattern in both training and test data. In this setting, a model can appear stronger than it truly is by memorising recurring text templates.

The second major finding was severe target imbalance. The distribution is long-tailed:

- **16 labels are singletons**
- **34 labels have fewer than 5 examples**
- **43 labels have fewer than 10 examples**

This matters because overall accuracy can still look strong even if minority categories are handled poorly. It also limits which validation strategies are feasible, since some labels do not have enough support to appear robustly across many grouped folds.

The third finding was that vendor identity is highly informative. Some vendors are strongly associated with one or a few account categories, making `vendorId` a useful structured feature. However, relying on vendor alone is not sufficient, because the same vendor can map to different accounts depending on the expense text and amount.

Finally, the `itemTotalAmount` field has a very wide range, from negative values up to very large positive amounts. That makes raw numeric use unstable, so amount had to be transformed before modeling.

### Data Quality Observations

The dataset was relatively clean, but a few data-quality issues still influenced the design:

- Some descriptions were missing or empty, so the pipeline had to remain robust when `itemDescription` contributed little or no information.
- Text formatting was inconsistent, including abbreviations, partial dates, punctuation variation, and mixed casing.
- Repeated item-name templates introduced leakage risk under naive splitting.
- Rare classes created evaluation instability and reduced the reliability of per-class estimates.

These observations directly shaped the feature engineering and validation strategy. In particular, text normalization, character n-grams, and grouped splitting were all responses to issues discovered during initial analysis.

### Key Insights That Informed Modeling

The exploratory phase suggested four practical modeling decisions:

1. **Use a sparse linear text model** rather than a heavier deep model, because the dataset is modest in size and strongly text-driven.
2. **Combine structured and unstructured features**, since vendor and amount add signal that text alone does not capture.
3. **Evaluate with grouped splits**, because row-level random splitting would overstate real-world performance.
4. **Measure rare-class behavior separately**, because overall accuracy alone would hide the long-tail difficulty.

---

## Methodology

### Algorithm Choice and Rationale

The final classifier is **`LinearSVC`** inside a scikit-learn pipeline. This choice was intentional.

For this problem, the input representation is high-dimensional and sparse due to TF-IDF features. Linear SVMs are a strong baseline for exactly this setting: they train efficiently, handle sparse matrices well, and often outperform more complex models when the dataset is relatively small but text-rich. They are also easier to interpret and tune than transformer-based models in a short take-home setting.

I did not choose a transformer model because the dataset has only 4,894 rows spread across 103 classes, with many labels having very few examples. A transformer would add much higher complexity and overfitting risk without clear evidence of better generalization under the grouped split that matters most.

### Feature Engineering Approach

The feature engineering pipeline combines six sparse feature blocks using `sklearn.pipeline.FeatureUnion`:

| Feature | Transformer | Purpose |
|---|---|---|
| `itemName` + `itemDescription` | Word TF-IDF, 1-2 grams, `min_df=2`, `sublinear_tf=True` | Main semantic text signal |
| `itemName` only | Word TF-IDF, 1-2 grams, `min_df=1`, `sublinear_tf=True` | Preserves short rare-item patterns |
| Combined text | Character TF-IDF, 3-5 grams, `char_wb` | Handles abbreviations, typos, formatting variation |
| `vendorId` | One-hot encoding | Captures vendor-account prior |
| `itemTotalAmount` | `log(1 + abs(amount))` + `MaxAbsScaler` | Stabilises large numeric range |
| `itemTotalAmount` | Quantile-bin one-hot encoding | Captures non-linear amount effects |

This design balances practicality and performance. Word n-grams capture human-readable meaning, character n-grams improve robustness to noisy expense text, vendor identity supplies business prior information, and amount features help distinguish categories with similar text but different transaction sizes.

Another important implementation detail is that the pipeline operates directly on Python dictionaries rather than requiring a DataFrame at inference time. This makes deployment cleaner because the saved artifact can consume raw JSON-like records directly.

### Handling Class Imbalance

Class imbalance was one of the most important technical constraints in the project. I addressed it in three ways:

1. **Class-aware tuning**: models with and without `class_weight='balanced'` were both considered.
2. **Oversampling minority classes in training folds**: rare classes were duplicated up to a minimum support threshold during model selection.
3. **Separate rare-class evaluation**: a balanced unseen benchmark was used so long-tail performance could be measured explicitly.

The final selected production configuration was:

- `C=1.0`
- `class_weight='balanced'`
- `oversample_min_count=5`

This setting gave the best balance between strong grouped performance and improved minority-class behavior.

### Validation Strategy

Validation strategy was a central part of the methodology because naive validation would have been misleading. I used three complementary evaluation views:

| Method | Purpose | Main Interpretation |
|---|---|---|
| Random holdout | Optimistic reference | Useful as a ceiling, but leakage-prone |
| Grouped holdout on normalised `itemName` | Primary business metric | Best proxy for unseen transaction patterns |
| Balanced unseen holdout | Rare-class stress test | More equal view of minority-class performance |

The **grouped holdout** is the main metric because it prevents the same normalized item-name pattern from appearing in both training and test. That makes it a more realistic estimate of production performance.

To reduce sensitivity to a single split, I also ran **repeated grouped evaluation across 5 independent splits**, producing a variance-stabilised mean estimate.

### Hyperparameter Selection

Hyperparameter tuning was done in two stages.

**Stage 1:** grouped 5-fold cross-validation over a grid of 50 configurations varying:

- `C` in `[0.5, 1.0, 2.0, 4.0, 8.0]`
- `class_weight` in `[None, 'balanced']`
- `oversample_min_count` in `[0, 5, 10]`

Only configurations achieving at least **0.85 grouped accuracy** were promoted.

**Stage 2:** all eligible configurations were re-evaluated on the balanced unseen holdout, and the final choice was made using **balanced macro F1**.

This two-stage design was important because the best configuration for overall grouped accuracy is not always the best configuration for rare-class fairness.

---

## Results

### Overall Performance

The final model exceeded the assignment target and performed strongly under multiple evaluation views.

| Evaluation Method | Accuracy | Macro F1 | Notes |
|---|---|---|---|
| Grouped holdout (default, `C=1.0`, no class weighting) | 89.38% | 76.4% | Primary leakage-safe estimate |
| Grouped holdout (tuned, `class_weight='balanced'`) | 89.38% | 78.3% | Same accuracy, better minority balance |
| Repeated grouped mean (5 splits) | 87.46% | 72.6% | More stable estimate across splits |
| Balanced unseen holdout | 85.42% | 82.5% | Rare-class stress test |
| Random holdout | 89.48% | 77.6% | Optimistic reference |
| Baseline lookup (vendor + exact item) | 74.77% | - | Naive memorisation baseline |

The strongest business-facing result is the **89.38% grouped holdout accuracy**, because it reflects the most realistic deployment scenario. Relative to the baseline lookup system, this is an improvement of **14.6 percentage points**.

### Breakdown by Category Performance

Performance was strongest on medium-frequency and high-frequency categories, where the model had enough examples to learn stable text and vendor patterns.

| Frequency Bucket | Test Samples | Accuracy |
|---|---|---|
| singleton (`n=1`) | 3 | 100.0% |
| very rare (`2-4`) | 7 | 71.4% |
| rare (`5-19`) | 91 | 83.5% |
| medium (`20-99`) | 313 | 93.3% |
| frequent (`100+`) | 525 | 88.8% |

The strongest and most reliable performance appears in the **medium-frequency** and **frequent** buckets. The weakest region is the long tail, especially labels with fewer than 5 examples. That is expected and consistent with the data scarcity discovered during EDA.

### Error Analysis

The model does not appear to be failing because of a poor algorithmic choice; instead, most residual error comes from the inherent difficulty of the long-tail label distribution.

Key overfitting diagnostics:

- Train accuracy: **0.984**
- Grouped validation accuracy: **0.894**
- Gap: **0.090**

This gap indicates moderate overfitting, but the diagnostics suggest the root cause is mostly limited data for minority labels rather than uncontrolled model complexity. The main reasons are:

1. Character n-grams can partially memorise phrase templates seen in training.
2. **34 of 103 labels** have fewer than 5 training examples.
3. Learning-curve analysis shows validation performance continues to improve as more data is added, indicating that data scarcity remains the main bottleneck.

Overall, the results show a model that is clearly production-viable for assisted classification, while still leaving room for improvement on rare categories.

---

## Discussion

### Strengths of the Approach

This approach has several practical strengths.

First, it is **accurate enough to meet the business target** under a realistic validation design. Second, it is **efficient and reproducible**: the full pipeline can be retrained in one command, tracked with MLflow, and exported as a `joblib` artifact. Third, it is **well-aligned with the data shape**: sparse text models are a strong fit for short transactional text plus structured features. Finally, it is **interpretable enough for finance operations**, where stakeholders may care about why a prediction is being made and where failures are likely to occur.

### Limitations

The biggest limitation is the long-tail class distribution. Some labels simply do not have enough examples to estimate reliable decision boundaries. That makes rare-class performance noisy and limits how much any classical supervised model can generalize.

Another limitation is that the grouped split controls for item-name leakage, but it is still possible that some business concepts remain easier in the training data than in future production traffic. In other words, the validation design is much better than a random split, but it is still an offline estimate.

Finally, the current system predicts a single label from the historical taxonomy. If the business introduces new account categories or significant vendor behavior changes, the model would need retraining and possibly a stronger fallback or human-review policy.

### Ideas for Improvement

With more time and resources, I would prioritise the following improvements:

- **Collect more examples for rare categories**, since the learning curves suggest more data would help directly.
- **Introduce hierarchical or account-code-aware modeling**, so similar account groups can share signal.
- **Add active-learning or human-in-the-loop review** for low-confidence predictions.
- **Experiment with stronger text representations**, such as lightweight sentence embeddings or domain-adapted encoders, but only under the same grouped validation discipline.
- **Calibrate confidence thresholds more explicitly by business cost**, so different error types can trigger different workflows.

### Business Considerations for Deployment

From a business perspective, this model is best positioned as a **decision-support system** rather than a fully autonomous accountant. High-confidence predictions can be auto-suggested or auto-filled, while low-confidence cases can route to finance reviewers. This would reduce manual effort without requiring full trust in every edge case.

The vendor-majority fallback is also useful operationally because it creates graceful degradation when the model is uncertain. In a live setting, I would monitor:

- prediction confidence distribution
- drift in vendor-account relationships
- error rates by category frequency
- low-confidence review rates
- retraining cadence as new labeled transactions arrive

In summary, the solution meets the assignment goal, uses a rigorous validation strategy, and is credible as a deployable finance-assist classifier. The remaining gap to stronger performance is driven less by modeling failure and more by limited minority-class data.

