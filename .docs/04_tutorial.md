# In-Depth Tutorial: Expense Account Classification Pipeline

> A complete walkthrough of every design decision, data structure, algorithm, and validation
> technique used in this codebase - from raw JSON to a calibrated production predictor.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Project Layout](#2-project-layout)
3. [Data Layer - `ExpenseRecord` and Loading](#3-data-layer--expenserecord-and-loading)
4. [Text Normalisation and Feature Construction](#4-text-normalisation-and-feature-construction)
5. [Custom Transformers](#5-custom-transformers)
   - 5.1 [DictTextVectorizer](#51-dicttextvectorizer)
   - 5.2 [DictOneHotEncoder](#52-dictonehhotencoder)
   - 5.3 [DictAmountScaler](#53-dictamountscaler)
   - 5.4 [DictAmountBinner](#54-dictamountbinner)
6. [Feature Union - Putting It All Together](#6-feature-union--putting-it-all-together)
7. [Why LinearSVC?](#7-why-linearsvc)
8. [The Training Pipeline](#8-the-training-pipeline)
   - 8.1 [records_to_examples - The Bridge](#81-records_to_examples--the-bridge)
   - 8.2 [_fit_and_score - The Workhorse](#82-_fit_and_score--the-workhorse)
   - 8.3 [build_classifier - Assembling the Pipeline](#83-build_classifier--assembling-the-pipeline)
9. [Handling Class Imbalance - Minority Oversampling](#9-handling-class-imbalance--minority-oversampling)
10. [Validation Strategy - Why Random Splits Lie](#10-validation-strategy--why-random-splits-lie)
    - 10.1 [Random Holdout](#101-random-holdout)
    - 10.2 [Grouped Holdout](#102-grouped-holdout)
    - 10.3 [Balanced Unseen Holdout](#103-balanced-unseen-holdout)
    - 10.4 [Repeated Splits](#104-repeated-splits)
11. [Hyperparameter Tuning - Grouped CV Grid Search](#11-hyperparameter-tuning--grouped-cv-grid-search)
12. [Two-Stage Tuning - Balancing Natural and Rare-Class Accuracy](#12-two-stage-tuning--balancing-natural-and-rare-class-accuracy)
13. [Overfitting Diagnostics](#13-overfitting-diagnostics)
    - 13.1 [Learning Curve](#131-learning-curve)
    - 13.2 [Validation Curve](#132-validation-curve)
    - 13.3 [Error Analysis by Class Frequency](#133-error-analysis-by-class-frequency)
14. [Inference Pipeline - CalibratedPredictor](#14-inference-pipeline--calibratedpredictor)
    - 14.1 [Sigmoid Calibration](#141-sigmoid-calibration)
    - 14.2 [Vendor Fallback](#142-vendor-fallback)
15. [Full Pipeline Orchestration](#15-full-pipeline-orchestration)
16. [Test Suite Design](#16-test-suite-design)
17. [Intel Acceleration with sklearnex](#17-intel-acceleration-with-sklearnex)
18. [End-to-End Data Flow Diagram](#18-end-to-end-data-flow-diagram)
19. [Key Design Decisions Summary](#19-key-design-decisions-summary)
20. [Running the Code](#20-running-the-code)

---

## 1. Problem Statement

**Task**: Given an expense transaction record - vendor ID, item name, item description, and
amount - predict the accounting `accountName` it should be filed under.

**Dataset**: 4,894 records, 337 unique vendors, 103 unique account names.

**Distribution reality**: Highly imbalanced.

| Account | Count |
|---|---|
| 611202 Online Subscription/Tool | 1,179 |
| 132098 IC Clearing account | 706 |
| ... (many medium classes) | ... |
| Several singleton classes | 1 |

This is a **103-class, short-text, imbalanced classification** problem. The signal is
primarily in the item name text (e.g. *"AWS"* → *"Cloud Infrastructure"*), secondarily in
vendor identity, and weakly in transaction amount.

**Targets**:
- ≥ 85% overall accuracy (hard requirement)
- ≥ 92% grouped holdout accuracy (bonus - internal algorithm benchmark)
- ≥ 85% balanced unseen holdout accuracy (rare-class stress test)

---

## 2. Project Layout

```
peakflo/
├── accounts-bills.json          # Raw dataset - 4,894 records
├── main.py                      # Entry point: calls src.training_pipeline.main()
├── pyproject.toml               # uv/pip package config, declares `peakflo` CLI script
├── src/
│   ├── feature_engineering_pipeline.py  # Data loading, transformers, feature union
│   ├── training_pipeline.py             # CV, tuning, diagnostics, orchestration
│   └── inference_pipeline.py            # Calibrated predictor, model I/O
├── tests/
│   └── test_model.py            # Integration-style test suite
├── artifacts/
│   ├── account_classifier.joblib        # Saved fitted Pipeline
│   └── evaluation_summary.json          # Full metrics from last run
└── .docs/
    └── (markdown notes per session)
```

**Module dependency graph**:

```
feature_engineering_pipeline
        ↑
training_pipeline  ←→  inference_pipeline
        ↑
      main.py / tests
```

`feature_engineering_pipeline` has zero internal dependencies - it only uses sklearn and
stdlib. `inference_pipeline` imports only `ExpenseRecord` helpers from feature engineering.
`training_pipeline` coordinates both.

---

## 3. Data Layer - `ExpenseRecord` and Loading

### The Raw JSON Shape

Each record in `accounts-bills.json` looks like:

```json
{
  "_id": {"$oid": "abc123"},
  "vendorId": "v-0042",
  "itemName": "AWS EC2 Instance Monthly",
  "itemDescription": "Cloud compute - Singapore region",
  "accountId": "acc-9901",
  "accountName": "611202 Online Subscription/Tool",
  "itemTotalAmount": 4821.50
}
```

### `ExpenseRecord` Dataclass

```python
@dataclass(frozen=True)
class ExpenseRecord:
    vendor_id: str
    item_name: str
    item_description: str
    account_name: str          # ← prediction target
    item_total_amount: float
    normalized_item_name: str  # lowercased, whitespace-collapsed item_name
    text: str                  # "{normalized_item_name} {normalized_description}"
    amount_log: float          # math.log1p(abs(item_total_amount))
```

`frozen=True` makes it immutable and hashable - critical because records are passed through
multiple pipeline stages and we never want accidental mutation.

Three computed fields are derived at load time, not stored in JSON:

- **`normalized_item_name`** - lowercased, whitespace-collapsed. Used as the grouping key
  for all grouped CV splits. The core assumption: two records with identical
  `normalized_item_name` are likely the same recurring transaction pattern, so they must not
  appear in both train and test.

- **`text`** - concatenation of `normalized_item_name` + `normalized_description`. This is
  the primary TF-IDF input.

- **`amount_log`** - `log1p(|amount|)`. Log transform compresses the huge range
  (-$15,195 to $161,838,000) into a scale where the model can learn linearly.

### `load_records`

```python
def load_records(path: str | Path) -> list[ExpenseRecord]:
    frame = pd.read_json(Path(path))
    frame = frame.assign(
        vendorId=frame["vendorId"].fillna("").astype(str),
        itemName=frame["itemName"].fillna("").astype(str),
        itemDescription=frame["itemDescription"].fillna("").astype(str),
        accountName=frame["accountName"].astype(str),
        itemTotalAmount=pd.to_numeric(frame["itemTotalAmount"], errors="coerce").fillna(0.0),
    )
    ...
```

**Why `fillna("")`**: vendorId and itemName can be null. An empty string is safer than
`None` because all downstream transformers expect string inputs.

**Why `pd.to_numeric(..., errors="coerce")`**: Amount is stored inconsistently in the JSON.
Coerce converts any non-numeric value to `NaN`, which then becomes `0.0`.

---

## 4. Text Normalisation and Feature Construction

```python
SPACE_RE = re.compile(r"\s+")

def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.lower().strip()
    return SPACE_RE.sub(" ", text)

def build_text(item_name: object, item_description: object) -> str:
    item = normalize_text(item_name)
    description = normalize_text(item_description)
    return f"{item} {description}".strip()
```

**Example**:

| Input | Output |
|---|---|
| `"  AWS  EC2   "` | `"aws ec2"` |
| `None` | `""` |
| `"Zoom Video  \nCommunications"` | `"zoom video communications"` |

The regex `\s+` collapses tabs, newlines, and multiple spaces into one space. `.lower()`
ensures "AWS" and "aws" are the same token.

`build_text` concatenates item name and description so the TF-IDF has the full sentence.
`itemDescription` is missing in ~40% of records (blank after normalisation), so the concat
degrades gracefully to just the item name.

---

## 5. Custom Transformers

sklearn transformers must implement `fit(X, y)` and `transform(X)`. The standard sklearn
transformers (TfidfVectorizer, OneHotEncoder) expect a 2-D array or a list of strings. Our
`X` is a `list[dict]` - one dict per record. So we wrap each sklearn transformer in a
"dict-aware" adapter that extracts the right key before delegating.

All four adapters follow the same pattern:
1. Inherit `BaseEstimator` + `TransformerMixin`
2. Store `key` in `__init__`
3. In `fit`/`transform`, extract `row.get(self.key, default)` for each row

### 5.1 DictTextVectorizer

```python
class DictTextVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, key: str = "text", **kwargs: object) -> None:
        self.key = key
        self.kwargs = kwargs
        self.vectorizer = TfidfVectorizer(**kwargs)

    def fit(self, X, y=None):
        self.vectorizer.fit([str(row.get(self.key, "")) for row in X])
        return self

    def transform(self, X):
        return self.vectorizer.transform([str(row.get(self.key, "")) for row in X])
```

`**kwargs` is forwarded directly to `TfidfVectorizer`. This means the same class
instantiation covers three different vectorizer configs used in the feature union:

| Instance | `key` | `analyzer` | `ngram_range` | `min_df` | Purpose |
|---|---|---|---|---|---|
| `word_tfidf` | `"text"` | word | (1,2) | 2 | dominant signal - full text bigrams |
| `item_name_tfidf` | `"normalized_item_name"` | word | (1,2) | 1 | dedicated item-name signal |
| `char_tfidf` | `"text"` | char_wb | (3,5) | 1 | typo + partial-match robustness |

**Why `sublinear_tf=True`**: Raw term frequency rewards documents that repeat a word 100×
more than documents that mention it once. `sublinear_tf` replaces `tf` with `1 + log(tf)`,
dampening this effect. For expense descriptions - often short, repetitive boilerplate - this
prevents any single repeated term from dominating.

**Why `min_df=2` for word_tfidf but `min_df=1` for item_name_tfidf**: The full-text
vectorizer ignores tokens seen in fewer than 2 documents (noise reduction). The item-name
vectorizer keeps all tokens including hapax legomena because rare item names are often the
most discriminative signal for rare account classes.

**Why char 3-5gram**: Character n-grams handle:
- Typos: "Googlr Drive" shares `oog`, `ogl` with "Google Drive"
- Partial matches: "Airfare SGD" and "Airfare USD" share the `airfar` trigram
- Language variants: British/American spellings

`char_wb` pads each word with space boundaries (`_google_`, `_driv_`, etc.) so word
boundaries are preserved in the character n-grams.

### 5.2 DictOneHotEncoder

```python
class DictOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, key: str = "vendor_id") -> None:
        self.key = key
        self.encoder = OneHotEncoder(handle_unknown="ignore")

    def fit(self, X, y=None):
        values = np.array([[str(row.get(self.key, ""))] for row in X], dtype=object)
        self.encoder.fit(values)
        return self

    def transform(self, X):
        values = np.array([[str(row.get(self.key, ""))] for row in X], dtype=object)
        return self.encoder.transform(values)
```

337 vendors → 337 binary columns (sparse). Each row is exactly one 1-bit.

**Why `handle_unknown="ignore"`**: At inference time, a new vendor not seen during training
produces an all-zeros row - the model silently degrades to relying on text features alone.
Without this, it would raise a `ValueError`.

**Signal**: Vendors are often highly correlated with account categories. "Zoom" almost
always maps to "Online Subscription/Tool". The one-hot encoding lets the SVC learn
`weight[vendor_zoom] * 1 → class_online_sub` directly without any text matching.

### 5.3 DictAmountScaler

```python
class DictAmountScaler(BaseEstimator, TransformerMixin):
    def __init__(self, key: str = "amount_log") -> None:
        self.key = key
        self.scaler = MaxAbsScaler()
```

`MaxAbsScaler` divides by the maximum absolute value, keeping the range `[-1, 1]`. Unlike
`StandardScaler`, it does not center (shift mean to 0), preserving the sparsity of zero
values - crucial because we concatenate this with sparse TF-IDF matrices.

**Why log-scale before scaling**: The raw amount range spans 8+ orders of magnitude. Log
compresses this to ~12 units (`log1p(161,838,000) ≈ 18.9`). The scaler then maps that to
`[0, 1]`.

### 5.4 DictAmountBinner

```python
class DictAmountBinner(BaseEstimator, TransformerMixin):
    def __init__(self, key: str = "amount_log", n_bins: int = 10) -> None:
        ...
        self.bin_edges_: np.ndarray | None = None

    def fit(self, X, y=None):
        values = np.array([float(row.get(self.key, 0.0)) for row in X], dtype=float)
        self.bin_edges_ = np.percentile(values, np.linspace(0, 100, self.n_bins + 1))
        self.bin_edges_ = np.unique(self.bin_edges_)  # dedup
        return self

    def transform(self, X):
        values = np.array([float(row.get(self.key, 0.0)) for row in X], dtype=float)
        bin_indices = np.digitize(values, self.bin_edges_[1:-1])
        n_bins = len(self.bin_edges_) - 1
        one_hot = np.zeros((len(X), n_bins), dtype=float)
        for row_idx, col_idx in enumerate(bin_indices):
            one_hot[row_idx, min(col_idx, n_bins - 1)] = 1.0
        return csr_matrix(one_hot)
```

**Why bins on top of the scaler?** The linear scaler produces a continuous value. Some
accounting categories might be correlated with *ranges* rather than linear amounts. For
example, "Petty Cash Expenses" might cluster around small amounts ($10–$500) while "Capital
Equipment" clusters around large amounts ($10,000+). A linear feature cannot capture this
non-linear relationship. One-hot bins give the linear SVC a way to say "if this bin fires,
prefer class X".

**Why percentile bins?** Equal-width bins would be dominated by outliers. Percentile bins
give each bin roughly equal occupancy (10% of records per bin), maximising information.

**Why `np.unique(self.bin_edges_)`?** Many records have amount = 0 (missing/zero amounts
filled with `fillna(0.0)`). If 15% of records are exactly zero, multiple percentile edges
land on the same value. `np.unique` deduplicates them, preventing zero-width bins that
would cause `np.digitize` to behave incorrectly. The final number of bins may be fewer than
`n_bins` - the test asserts `result.shape[1] <= 10`.

---

## 6. Feature Union - Putting It All Together

```python
def build_feature_union() -> FeatureUnion:
    return FeatureUnion([
        ("word_tfidf",      DictTextVectorizer(ngram_range=(1,2), min_df=2,  sublinear_tf=True, strip_accents="unicode")),
        ("item_name_tfidf", DictTextVectorizer(key="normalized_item_name", ngram_range=(1,2), min_df=1, sublinear_tf=True, strip_accents="unicode")),
        ("char_tfidf",      DictTextVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1, sublinear_tf=True)),
        ("vendor",          DictOneHotEncoder()),
        ("amount",          DictAmountScaler()),
        ("amount_bins",     DictAmountBinner()),
    ])
```

`FeatureUnion` runs all six transformers **in parallel** and **horizontally concatenates**
their sparse output matrices. The result is one large sparse matrix per record:

```
[word_tfidf | item_name_tfidf | char_tfidf | vendor_onehot | amount_scaled | amount_bins]
   ~8,000 cols    ~6,000 cols      ~15,000 cols    337 cols         1 col       ≤10 cols
                            ≈ ~30,000 total features
```

All outputs are sparse (`csr_matrix`). Memory for 4,894 records × 30,000 features at
`float64` would be ~1.1 GB dense - but sparse representation stores only non-zero values,
so actual memory is ~10-50 MB.

**Why `strip_accents="unicode"`** on text vectorizers: Normalises accented characters
(é → e, ü → u) so "Café" and "Cafe" map to the same token.

---

## 7. Why LinearSVC?

The problem is **multiclass text classification** with ~30,000 sparse features and ~4,000
training samples. LinearSVC is the right tool because:

| Property | Value for This Problem |
|---|---|
| **Speed** | `O(n * d)` per iteration via liblinear - fast on sparse matrices |
| **Accuracy** | Competitive with neural models on short-text classification when features are well-engineered |
| **Interpretability** | Each class has a weight vector - `coef_[class_i, feature_j]` is the contribution of feature j to class i |
| **No kernel trick needed** | 30,000 features already provides rich non-linear representation via char n-grams |
| **`dual="auto"`** | Liblinear selects primal or dual form based on n_samples vs n_features; avoids manual tuning |
| **`class_weight`** | Built-in support for upweighting minority classes |

**Alternatives considered and rejected**:

- **Random Forest / GBM**: Dense matrix required; 30,000 features → impractical memory and
  slow training without sparse support.
- **Logistic Regression**: Similar accuracy to LinearSVC, slightly slower convergence on
  this scale.
- **SVC (kernel='linear')**: Uses libsvm instead of liblinear. libsvm is `O(n²)` in the
  number of support vectors - much slower for 4,000+ samples and sparse features. Testing
  confirmed ~5% accuracy drop vs LinearSVC on this dataset.
- **Transformer / SBERT**: Would require GPU and SBERT embeddings. Could improve rare-class
  performance but adds significant operational complexity for marginal gain on frequent
  classes.

---

## 8. The Training Pipeline

### 8.1 `records_to_examples` - The Bridge

```python
def records_to_examples(
    records: list[ExpenseRecord],
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    examples = [
        {
            "text": record.text,
            "vendor_id": record.vendor_id,
            "amount_log": record.amount_log,
            "normalized_item_name": record.normalized_item_name,
        }
        for record in records
    ]
    labels = np.array([record.account_name for record in records], dtype=object)
    groups = np.array([record.normalized_item_name for record in records], dtype=object)
    return examples, labels, groups
```

Returns three things:

1. **`examples`** - `list[dict]` - the feature matrix input. Each dict has exactly the keys
   the transformers read from.
2. **`labels`** - `np.ndarray` of strings - the target classes.
3. **`groups`** - `np.ndarray` of strings - the grouping key for grouped CV splits. This is
   `normalized_item_name`, meaning two records with the same item name get the same group.

The `groups` array is the lynchpin of the entire validation strategy. Without it, you could
not do `GroupShuffleSplit`.

### 8.2 `_fit_and_score` - The Workhorse

```python
def _fit_and_score(
    train_records: list[ExpenseRecord],
    test_records: list[ExpenseRecord],
    *,
    C: float = 1.0,
    class_weight: str | None = None,
    oversample_min_count: int = 0,
) -> dict[str, object]:
    if oversample_min_count > 0:
        train_records = resample_minority_classes(train_records, min_count=oversample_min_count)
    train_x, train_y, _ = records_to_examples(train_records)
    test_x, test_y, _ = records_to_examples(test_records)
    model = build_classifier(C=C, class_weight=class_weight)
    model.fit(train_x, train_y)
    train_pred = model.predict(train_x)
    test_pred = model.predict(test_x)
    ...
```

**Critical rule**: Oversampling is applied **only to `train_records`**, never to
`test_records`. If oversampled records appeared in the test set, the model would be
evaluated on its own synthetic duplicates - artificially inflated accuracy.

`_fit_and_score` is a **pure function** - it never mutates global state and returns a
complete result dict. This makes it safe to call 250+ times inside `tune_hyperparameters_grouped_cv`
without side effects.

### 8.3 `build_classifier` - Assembling the Pipeline

```python
def build_classifier(C=1.0, class_weight=None, max_iter=20000) -> Pipeline:
    return Pipeline([
        ("features", build_feature_union()),
        ("classifier", LinearSVC(
            dual="auto",
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
        )),
    ])
```

The sklearn `Pipeline` guarantees that:
1. `features.fit_transform(X_train)` is called during training
2. `features.transform(X_test)` (not `fit_transform`) is called during testing

This prevents test-set information from leaking into the TF-IDF vocabulary or OHE
categories. The vocabulary is fitted only on training data.

`max_iter=20000` is high intentionally - liblinear terminates early when it converges, but
with `class_weight='balanced'` and many classes, convergence can take longer than the
default 1,000 iterations.

---

## 9. Handling Class Imbalance - Minority Oversampling

```python
def resample_minority_classes(
    records: list[ExpenseRecord],
    min_count: int = 10,
    random_state: int = 42,
) -> list[ExpenseRecord]:
    rng = random.Random(random_state)
    by_label: dict[str, list[ExpenseRecord]] = defaultdict(list)
    for record in records:
        by_label[record.account_name].append(record)
    result: list[ExpenseRecord] = list(records)
    for label_records in by_label.values():
        deficit = min_count - len(label_records)
        if deficit > 0:
            result.extend(rng.choices(label_records, k=deficit))
    return result
```

**Algorithm**: Random oversampling with replacement. For each class with fewer than
`min_count` records, duplicate random existing records until the threshold is reached.

**Example**:
```
Class "627001 Bank Charges" - 3 records in training fold
min_count = 5
deficit = 5 - 3 = 2
→ randomly pick 2 existing records from the 3, append them
→ class now has 5 records
```

Classes at or above `min_count` are untouched. Classes with more records are never
downsampled.

**Two complementary imbalance strategies** used together:

| Strategy | Mechanism | Effect |
|---|---|---|
| `resample_minority_classes` | Duplicate minority-class records | Reduces gradient starvation for rare classes |
| `class_weight='balanced'` | `w_i = n_total / (n_classes * n_i)` | Multiplies loss contribution by inverse frequency |

They are not redundant - they address different aspects:
- Oversampling ensures the model *sees* rare-class examples more frequently per epoch
- `class_weight` ensures that each rare-class *mistake* contributes more to the loss

Using both together can sometimes over-correct. The tuning grid searches combinations of
both to find the optimal balance.

---

## 10. Validation Strategy - Why Random Splits Lie

### 10.1 Random Holdout

```python
splitter = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
```

**The problem**: If a record with `normalized_item_name = "zoom monthly license"` appears
in both train and test (because there are 20 such identical records and random split puts 16
in train and 4 in test), the model has effectively *memorised* this pattern during training.
The test accuracy for these records is near 100% - but it tells you nothing about how the
model performs on a **genuinely unseen** item name.

**Observed gap**: Random holdout accuracy ≈ **0.95+** vs grouped holdout ≈ **0.89**. That
~6 point gap is measurement error caused by leakage, not model performance.

Random holdout is kept in the results for comparison only. It is **never used** for model
selection or to claim meeting the 85% threshold.

### 10.2 Grouped Holdout

```python
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(indices, labels, groups=groups))
```

`GroupShuffleSplit` ensures that **all records with the same `normalized_item_name` are
assigned to the same split** - either all train or all test, never both.

**Result**: The test set contains only item-name patterns the model has never seen. This
mimics the real production scenario where a new invoice arrives for an item the system
hasn't processed before.

**Trade-off**: GroupShuffleSplit does not preserve class ratios. A class with only 2
unique item names might land entirely in test or entirely in train. This makes grouped
holdout accuracy noisier - hence the need for repeated splits.

### 10.3 Balanced Unseen Holdout

```python
def build_balanced_unseen_holdout(records, samples_per_class=3, min_train_size=5, ...):
```

This is a custom holdout designed to evaluate **rare-class generalisation** without leakage.

**Algorithm**:

```
For each account class:
  1. Group records by normalized_item_name
  2. Shuffle group names
  3. Pick groups until we accumulate ≥ 3 records (samples_per_class)
  4. If that still leaves ≥ 5 records (min_train_size) in the remaining groups:
     a. Remove the selected groups from training entirely (no partial leakage)
     b. Take exactly 3 records as test
     c. Mark class as "eligible"
```

**Why whole groups are excluded, not individual records**: If "zoom monthly license" (group)
has 5 records and we put 3 in test and 2 in train, those 2 train records teach the model
the exact text pattern. The test becomes a memorisation check, not a generalisation check.
Excluding the whole group means the model must generalise from *different* item name
patterns for the same account.

**Why `min_train_size=5`**: A class with only 4 records total cannot afford to give 3 to
test and have only 1 for training - that would make the training fold too sparse to learn
anything. The threshold ensures test classes have enough training support.

The resulting test set has **exactly 3 records per eligible class** - balanced by
construction. Standard grouped holdout is dominated by frequent classes. This holdout
amplifies signal from rare classes equally.

### 10.4 Repeated Splits

```python
def evaluate_repeated_splits(records, strategy, n_splits=5, ...):
    splitter = GroupShuffleSplit(n_splits=n_splits, ...)
    for train_idx, test_idx in splitter.split(...):
        scored = _fit_and_score(...)
        results.append(...)
    return {"accuracy_mean": ..., "accuracy_std": ...}
```

A single holdout split has variance - the random partitioning might happen to put easy
classes in test (optimistic) or hard classes in test (pessimistic). Running 5 independent
splits and averaging gives a more stable estimate.

**Typical output**: `accuracy_mean=0.8817 ± 0.0091` - the ±0.009 uncertainty band tells
you how much to trust a single run's number.

---

## 11. Hyperparameter Tuning - Grouped CV Grid Search

```python
def tune_hyperparameters_grouped_cv(
    records,
    C_values=[0.5, 1.0, 2.0, 4.0, 8.0],
    class_weights=[None, "balanced"],
    oversample_min_count_values=[0, 5, 10, 15, 20],
    n_splits=5,
    optimize_for="macro_f1",
) -> dict:
```

**Grid**: 5 × 2 × 5 = 50 configs × 5 folds = 250 model fits.

The loop structure:

```python
for omc in oversample_min_count_values:
    for cw in class_weights:
        for c in C_values:
            fold_scores = []
            for train_idx, test_idx in splits:
                scored = _fit_and_score(train_recs, test_recs, C=c, class_weight=cw, oversample_min_count=omc)
                fold_scores.append(scored["test_accuracy"])
            grid_results.append({"C": c, ..., "val_accuracy_mean": mean(fold_scores)})
```

All 5 folds are pre-generated from a single `GroupShuffleSplit` call. This ensures the
same fold structure across all configs - an "apple to apple" comparison.

**`optimize_for`**: Defaults to `"macro_f1"`, not `"accuracy"`. Macro F1 is the
unweighted average of per-class F1 scores. On an imbalanced dataset, accuracy is dominated
by frequent classes - a model that predicts "611202 Online Subscription/Tool" for everything
achieves ~24% accuracy without learning anything. Macro F1 forces the model to do reasonably
well on all classes.

**Key empirical finding** discovered during development:

> `class_weight='balanced'` is **anti-correlated** with grouped CV macro F1 but
> **positively correlated** with balanced unseen accuracy.

Why? `class_weight='balanced'` upweights rare classes, which hurts accuracy on frequent
classes (which dominate validation data in natural distribution). But it improves
generalisation on rare classes - which is exactly what the balanced holdout measures.

This means: **you cannot use grouped CV macro F1 ranking alone to find the best model for
rare-class performance**. That's why the two-stage tuning exists.

---

## 12. Two-Stage Tuning - Balancing Natural and Rare-Class Accuracy

```python
def tune_for_balanced_unseen(records, min_grouped_accuracy=0.85) -> dict:
    # Stage 1: run full grouped CV grid
    grouped_result = tune_hyperparameters_grouped_cv(...)
    grid = grouped_result["grid"]

    # Stage 2: filter by production accuracy floor, then rank by balanced macro F1
    eligible = [row for row in grid if row["val_accuracy_mean"] >= min_grouped_accuracy]

    for candidate in eligible:
        result = evaluate_balanced_holdout(records, C=..., class_weight=..., ...)
        if result["macro_f1"] > best_balanced_f1:
            best_balanced = candidate
    return {"grouped_cv": grouped_result, "balanced_refinement": best_balanced}
```

**Stage 1 - Production Viability Gate**:

The `min_grouped_accuracy=0.85` floor ensures we only consider models that pass the minimum
business requirement on the natural distribution. Models that sacrifice too much accuracy on
common classes (to serve rare classes) are eliminated here.

**Stage 2 - Rare-Class Optimisation**:

Among all configs that passed the gate, we evaluate each on the balanced unseen holdout and
select the one with the highest **balanced macro F1**. The balanced holdout gives equal
weight to all eligible classes - a poor rare-class score cannot be masked by excellent
frequent-class performance.

**Decision flow**:

```
All 50 configs
    ↓ filter: val_accuracy_mean >= 0.85
N eligible configs (typically 20–35)
    ↓ evaluate each on balanced holdout
Select: max(balanced_macro_f1)
    ↓
Selected: C=1.0, class_weight='balanced', oversample_min_count=5
```

**Result**: This config achieves:
- Grouped holdout accuracy ≈ **0.8917** (natural distribution)
- Balanced holdout accuracy ≈ **0.8354** (rare-class stress test)

---

## 13. Overfitting Diagnostics

### 13.1 Learning Curve

```python
learning_sizes, train_scores, validation_scores = learning_curve(
    build_classifier(C=C, class_weight=class_weight),
    examples, labels,
    cv=GroupShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
    groups=groups,
    train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
    scoring="accuracy",
    n_jobs=-1,
)
```

**What it shows**: Train the model on 20%, 40%, 60%, 80%, 100% of the data. Plot both
training accuracy and grouped validation accuracy at each size.

**Expected shape for this problem**:

```
Train accuracy:  0.99 | 0.99 | 0.99 | 0.99 | 0.99   (always near-perfect on seen data)
Val accuracy:    0.59 | 0.72 | 0.79 | 0.85 | 0.89   (improves with more data)
```

**Interpretation**: The gap is NOT primarily caused by overfitting to specific words - it is
caused by **insufficient data per class**. Average 47 samples across 103 classes. The
learning curve is still rising at 100% of data, meaning *more data* would help more than
further regularisation.

### 13.2 Validation Curve

```python
param_range = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
train_curve, validation_curve_scores = validation_curve(
    build_classifier(class_weight=class_weight),
    examples, labels,
    param_name="classifier__C",
    param_range=param_range,
    cv=GroupShuffleSplit(...),
    groups=groups,
    n_jobs=-1,
)
```

**What it shows**: How train and validation accuracy change as `C` varies.

**`C` in LinearSVC**: Controls the regularisation strength - smaller `C` → more
regularisation → simpler decision boundary. Larger `C` → less regularisation → decision
boundary fits training data more tightly.

**Expected pattern**:
- At `C=0.1`: both train and val low (underfitting - too much regularisation)
- At `C=1.0`–`C=4.0`: val peaks, train still high (sweet spot)
- At `C=16.0`: train even higher, val drops slightly (overfitting)

The validation curve tells you which region of the `C` range is optimal for this data size.

### 13.3 Error Analysis by Class Frequency

```python
def analyze_errors_by_class_frequency(test_records, predictions, train_records):
    train_counts = Counter(record.account_name for record in train_records)
    bucket_defs = [
        ("singleton (n=1)",   lambda n: n == 1),
        ("very_rare (2-4)",   lambda n: 2 <= n <= 4),
        ("rare (5-19)",       lambda n: 5 <= n <= 19),
        ("medium (20-99)",    lambda n: 20 <= n <= 99),
        ("frequent (100+)",   lambda n: n >= 100),
    ]
```

This breaks down accuracy by how many training examples each class had.

**Sample output**:

| Bucket | Test Samples | Correct | Accuracy |
|---|---|---|---|
| singleton (n=1) | 3 | 0 | 0.0000 |
| very_rare (2-4) | 12 | 6 | 0.5000 |
| rare (5-19) | 48 | 34 | 0.7083 |
| medium (20-99) | 156 | 131 | 0.8397 |
| frequent (100+) | 256 | 237 | 0.9258 |

**Key insight**: Errors concentrate in rare classes, not frequent ones. This confirms the
problem is **data scarcity** (34 labels with fewer than 5 examples), not model architecture.
Improving the model architecture will not help singleton classes - they need more labelled
data.

---

## 14. Inference Pipeline - CalibratedPredictor

Raw `LinearSVC` does not produce probabilities - it only produces signed distances from the
decision hyperplane. We need probabilities to implement the confidence threshold fallback.

### 14.1 Sigmoid Calibration

```python
class CalibratedPredictor:
    def __init__(self, pipeline, vendor_map, confidence_threshold=0.5):
        self.calibrated_ = CalibratedClassifierCV(
            estimator=FrozenEstimator(pipeline), method="sigmoid"
        )

    def fit_calibration(self, X, y):
        self.calibrated_.fit(X, y)
```

`CalibratedClassifierCV(method="sigmoid")` fits a Platt scaling model on top of the
trained SVC's decision scores. It learns a sigmoid transformation `P(y|x) = σ(a * f(x) + b)`
where `f(x)` is the SVC's score and `a`, `b` are fitted parameters.

**`FrozenEstimator`** wraps the already-fitted pipeline and prevents `CalibratedClassifierCV`
from re-fitting it - it only fits the sigmoid layer on top. Without `FrozenEstimator`,
`CalibratedClassifierCV` would re-fit the entire pipeline from scratch using
cross-validation, which would be redundant and slow.

**Why sigmoid rather than isotonic?** Isotonic regression requires more calibration data
and can overfit. Sigmoid (Platt scaling) has only 2 parameters per class and is the
standard for SVMs.

### 14.2 Vendor Fallback

```python
def predict(self, examples):
    proba = self.calibrated_.predict_proba(examples)
    for i, row in enumerate(examples):
        max_prob = float(proba[i].max())
        if max_prob >= self.confidence_threshold:
            predictions.append(classes[proba[i].argmax()])
        else:
            vendor = str(row.get("vendor_id", ""))
            fallback = self.vendor_map.get(vendor)
            if fallback:
                predictions.append(fallback)   # vendor lookup
            else:
                predictions.append(classes[proba[i].argmax()])  # best guess anyway
```

**Decision logic**:

```
max(P(class|record)) >= 0.5?
    YES → use model's prediction
    NO  → look up vendor's historically most common account
              vendor found? → use vendor's majority account
              vendor not found? → use model's prediction anyway
```

**`build_vendor_account_map`** construction:

```python
def build_vendor_account_map(records):
    vendor_votes: dict[str, Counter[str]] = {}
    for record in records:
        vendor_votes.setdefault(record.vendor_id, Counter())[record.account_name] += 1
    return {
        vendor: max(counts, key=counts.get)
        for vendor, counts in vendor_votes.items()
    }
```

For each vendor, count votes for each account name and pick the majority. This is a
zero-parameters lookup table derived entirely from training data.

**When does the fallback help?** For items where the text is ambiguous (e.g.,
`itemName="Monthly payment"` with no description) but the vendor is strongly associated
with one account (e.g., vendor "Grab" → "Travel Expenses"). The SVC has low confidence on
the generic text, but the vendor lookup provides a reliable signal.

---

## 15. Full Pipeline Orchestration

`run_training_pipeline()` ties everything together in this order:

```
1. load_records("accounts-bills.json")
         ↓
2. tune_for_balanced_unseen(records)
   ├─ tune_hyperparameters_grouped_cv (50 configs × 5 folds = 250 fits)
   └─ evaluate each eligible config on balanced unseen holdout
         ↓ selected: best_C, best_cw, best_omc
3. Evaluate with selected params:
   ├─ evaluate_holdout(random)           - optimistic baseline
   ├─ evaluate_holdout(group_item)       - default params grouped
   ├─ evaluate_holdout(group_item_tuned) - tuned params grouped ← primary metric
   ├─ evaluate_repeated_splits(×3)       - stability estimates
   ├─ evaluate_balanced_holdout          - rare-class stress test
   ├─ evaluate_repeated_balanced_holdout - repeated rare-class
   ├─ compute_overfitting_diagnostics    - learning + validation curves
   ├─ evaluate_calibrated_fallback       - calibrated model + vendor fallback, grouped
   └─ evaluate_calibrated_fallback_balanced - calibrated, balanced holdout
         ↓
4. fit_full_model(all records, best_C, best_cw, best_omc)
         ↓
5. Export:
   ├─ artifacts/account_classifier.joblib
   ├─ artifacts/evaluation_summary.json
   ├─ data/<timestamp>/full_dataset.csv
   ├─ data/<timestamp>/group_item_tuned/train.csv + test.csv
   ├─ data/<timestamp>/balanced_unseen/train.csv + test.csv
   └─ .docs/03_results.md
```

**Why fit the full model on ALL records?** After tuning, we know the best hyperparameters.
The holdout splits were used for evaluation only - in production, every labelled record is
valuable training signal. Fitting on 100% of data gives the deployed model the maximum
generalisation capability.

---

## 16. Test Suite Design

The tests in `tests/test_model.py` are **integration tests** - they load the real dataset
and run real model fits. This means they are slow (~2–5 minutes each) but they catch real
regressions in model quality, not just code syntax errors.

**Test philosophy**: Each test asserts a minimum performance threshold, not an exact value.
This allows minor numerical variation between runs while preventing catastrophic regression.

| Test | What It Asserts | Why |
|---|---|---|
| `test_baseline_lookup_is_reasonable` | Lookup accuracy ≥ 0.70 | Sanity check: a naïve heuristic must beat random chance |
| `test_random_holdout_meets_target_accuracy` | Random holdout ≥ 0.85 | Minimum task requirement (easy to pass) |
| `test_group_split_is_still_above_threshold` | Grouped repeated mean ≥ 0.85 | Stricter: no item-name leakage |
| `test_balanced_holdout_is_truly_balanced_and_unseen` | min/max support = 3, train∩test groups = ∅ | Structural guarantee: holdout construction is correct |
| `test_balanced_holdout_is_reasonable` | accuracy ≥ 0.82, eligible_labels ≥ 50 | Rare-class minimum |
| `test_resample_minority_classes_reaches_min_count` | All classes ≥ min_count, original counts not reduced | Oversampling contract |
| `test_analyze_errors_by_class_frequency_returns_buckets` | ≥ 2 buckets, frequent > rare accuracy | Error analysis is meaningful |
| `test_tune_hyperparameters_grouped_cv_finds_valid_params` | 12 grid entries, val_accuracy ≥ 0.85, gap < 0.20 | Tuning produces valid outputs |
| `test_amount_binner_produces_correct_shape` | `shape[1] <= 10` | Bin deduplication works |
| `test_vendor_account_map_covers_training_vendors` | All vendors mapped, values are strings | Inference lookup is complete |
| `test_calibrated_fallback_balanced_is_reasonable` | accuracy ≥ 0.80, macro_f1 ≥ 0.75 | Calibrated predictor works |
| `test_tune_for_balanced_unseen_finds_valid_params` | Both stages return valid structure | Two-stage tuning contract |

**Running**:

```powershell
uv run pytest tests/ -x -q          # fail-fast, quiet
uv run pytest tests/ -v --tb=short  # verbose
```

---

## 17. Intel Acceleration with sklearnex

```python
try:
    from sklearnex import patch_sklearn, config_context as _sklearnex_config_context
    patch_sklearn()
    logger.info("sklearnex patch applied - Intel GPU/CPU acceleration enabled")
    _SKLEARNEX_AVAILABLE = True
except ImportError:
    from contextlib import nullcontext as _sklearnex_config_context
    _SKLEARNEX_AVAILABLE = False
```

`patch_sklearn()` monkey-patches the standard sklearn estimators with Intel oneDAL-backed
implementations. For `LinearSVC`, the oneDAL liblinear solver on an Intel CPU is
typically **2–4× faster** than the stock scipy/liblinear solver.

**Why not `SVC(kernel='linear')` on GPU?** The `sklearnex` GPU path targets `SVC`
(libsvm-based), not `LinearSVC` (liblinear-based). Testing confirmed that `SVC(kernel='linear')`
produces ~5% lower accuracy on this dataset due to libsvm's quadratic convergence for
multiclass problems with 100+ classes. The CPU-accelerated `LinearSVC` is faster *and* more
accurate.

**Fallback**: `contextlib.nullcontext` is a do-nothing context manager. If `sklearnex` is
not installed, the code runs normally with standard sklearn. The `try/except ImportError`
makes acceleration optional - no hard dependency.

**Verification** (check if acceleration is active):

```powershell
$env:SKLEARNEX_VERBOSE="INFO"
python main.py run
```

You will see lines like:
```
sklearn.svm.LinearSVC: running accelerated version on cpu
```

---

## 18. End-to-End Data Flow Diagram

```
accounts-bills.json
        │
        ▼
load_records()
        │  ExpenseRecord(vendor_id, item_name, description,
        │                account_name, amount, normalized_item_name,
        │                text, amount_log)
        ▼
records_to_examples()
        │  X = list[dict]: {text, vendor_id, amount_log, normalized_item_name}
        │  y = array[str]: account_name
        │  groups = array[str]: normalized_item_name
        ▼
build_feature_union()  [fitted on TRAIN only]
        │
        ├─ DictTextVectorizer("text", word 1-2gram)     → sparse [N, ~8000]
        ├─ DictTextVectorizer("normalized_item_name")   → sparse [N, ~6000]
        ├─ DictTextVectorizer("text", char 3-5gram)     → sparse [N, ~15000]
        ├─ DictOneHotEncoder("vendor_id")               → sparse [N, 337]
        ├─ DictAmountScaler("amount_log")               → sparse [N, 1]
        └─ DictAmountBinner("amount_log")               → sparse [N, ≤10]
                    │
                    ▼ horizontal concat
              sparse [N, ~30000]
                    │
                    ▼
             LinearSVC(C=1.0, class_weight='balanced')
                    │  fits one-vs-rest weight vectors
                    │  103 weight vectors × 30000 features
                    ▼
              Pipeline (fitted)
                    │
        ┌───────────┴────────────────┐
        ▼                            ▼
  joblib.dump()              CalibratedClassifierCV
  artifacts/                  (sigmoid on top of frozen pipeline)
  account_classifier.joblib          │
                                     ▼
                             CalibratedPredictor
                                     │
                    ┌────────────────┴─────────────────┐
                    ▼                                   ▼
            max_proba >= 0.5?                    max_proba < 0.5
                    │                                   │
                    ▼                                   ▼
              model prediction               vendor_map lookup
                                                   (majority vote)
                                                        │
                                                        ▼
                                              final accountName
```

---

## 19. Key Design Decisions Summary

| Decision | Alternative Considered | Why This Choice |
|---|---|---|
| `LinearSVC` | `SVC(kernel='linear')`, LightGBM, SBERT | Fastest + best accuracy on sparse text with 30k features |
| Grouped CV by `normalized_item_name` | Random CV, StratifiedKFold | Prevents same-text leakage; mimics real production use case |
| Balanced unseen holdout | Standard grouped CV alone | Exposes rare-class generalisation that natural distribution masks |
| Two-stage tuning | Single-objective CV | `class_weight='balanced'` hurts CV macro F1 but helps rare classes - conflicting objectives require two stages |
| `log1p(abs(amount))` | Raw amount, MinMaxScaler | Compresses 8 orders of magnitude; `abs` handles negative credits |
| `MaxAbsScaler` (not `StandardScaler`) | StandardScaler | Preserves sparsity - centering would densify the sparse matrix |
| Quantile bins for amount | Equal-width bins | Equal occupancy per bin; robust to outliers |
| Sigmoid calibration | Isotonic calibration | Fewer parameters (2 per class), standard for SVMs, less overfitting |
| Vendor majority vote fallback | No fallback | When model is uncertain, vendor identity is a reliable prior |
| `FrozenEstimator` wrapper | Re-fitting in calibration | Prevents double-fit; calibration only adds the sigmoid layer |
| `min_df=1` for item_name TF-IDF | `min_df=2` | Rare item names are the most discriminative signal for rare classes |
| `handle_unknown="ignore"` in OHE | Raise on unknown | Production robustness - new vendors don't crash the system |

---

## 20. Running the Code

### Setup

```powershell
# Install uv (if not present)
pip install uv

# Install all dependencies
uv sync

# Activate virtual environment (Windows)
.venv\Scripts\activate
```

### Train + Evaluate

```powershell
# Full pipeline: tuning → evaluation → save model
python main.py run

# Or via the CLI entry point defined in pyproject.toml
uv run peakflo run
```

Output files:
- `artifacts/account_classifier.joblib` - serialised fitted Pipeline
- `artifacts/evaluation_summary.json` - all metrics as structured JSON
- `.docs/03_results.md` - human-readable results report

### Run Tests

```powershell
uv run pytest tests/ -x -q
```

### Predict on New Data

```python
from joblib import load
from src.inference_pipeline import predict_records

model = load("artifacts/account_classifier.joblib")

new_invoices = [
    {
        "vendorId": "v-0042",
        "itemName": "AWS EC2 Monthly",
        "itemDescription": "Compute - Singapore",
        "itemTotalAmount": 4821.50,
    }
]

predictions = predict_records(model, new_invoices)
print(predictions)  # ["611202 Online Subscription/Tool"]
```

### Interpret a Prediction (Debug)

```python
from src.inference_pipeline import (
    load_trained_model,
    prepare_inference_examples,
    build_vendor_account_map,
    CalibratedPredictor,
)
from src.feature_engineering_pipeline import load_records

records = load_records("accounts-bills.json")
pipeline = load_trained_model("artifacts/account_classifier.joblib")
vendor_map = build_vendor_account_map(records)
predictor = CalibratedPredictor(pipeline, vendor_map, confidence_threshold=0.5)

examples = prepare_inference_examples([{
    "vendorId": "v-0042",
    "itemName": "AWS EC2 Monthly",
    "itemDescription": "",
    "itemTotalAmount": 4821.50,
}])

# Note: predictor.fit_calibration must be called with training data before predict
# For a quick check, use the raw pipeline:
raw_prediction = pipeline.predict(examples)
print(raw_prediction)  # ["611202 Online Subscription/Tool"]
```

---

# Appendix A: Mathematical Notes

This section records the main mathematical ideas used by the pipeline.

## A.1 TF-IDF

For term $t$ in document $d$:

$$
\mathrm{tfidf}(t, d) = \mathrm{tf}(t, d) \times \mathrm{idf}(t)
$$

where inverse document frequency is:

$$
\mathrm{idf}(t) = \log\left(\frac{N + 1}{\mathrm{df}(t) + 1}\right) + 1
$$

- $N$ is the number of documents.
- $\mathrm{df}(t)$ is the number of documents containing term $t$.

### Why This Is Used

TF-IDF is appropriate here because expense descriptions contain short but highly informative phrases. TF-IDF emphasizes terms that are distinctive for a transaction description while down-weighting text that appears everywhere.

## A.2 Linear SVM Scoring Rule

For class $k$, the linear classifier computes:

$$
f_k(x) = w_k^T x + b_k
$$

The predicted label is:

$$
\hat{y} = \arg\max_k f_k(x)
$$

### Optimization View

The linear SVM objective can be written as:

$$
\min_{w,b} \; \frac{1}{2}\lVert w \rVert^2 + C \sum_{i=1}^{n} \max\left(0, 1 - y_i(w^T x_i + b)\right)
$$

where:

- $\lVert w \rVert^2$ penalizes model complexity
- $C$ controls the regularization strength

### Interpretation

- lower $C$ means stronger regularization
- higher $C$ means weaker regularization and more risk of memorization

## A.3 Precision, Recall, and F1

For a class $c$:

$$
\mathrm{Precision}_c = \frac{TP_c}{TP_c + FP_c}
$$

$$
\mathrm{Recall}_c = \frac{TP_c}{TP_c + FN_c}
$$

$$
F1_c = \frac{2 \cdot \mathrm{Precision}_c \cdot \mathrm{Recall}_c}{\mathrm{Precision}_c + \mathrm{Recall}_c}
$$

Macro F1 is:

$$
\mathrm{MacroF1} = \frac{1}{K} \sum_{c=1}^{K} F1_c
$$

Weighted F1 is:

$$
\mathrm{WeightedF1} = \sum_{c=1}^{K} \frac{n_c}{n} F1_c
$$

### Why Both Are Reported

- Macro F1 is needed because rare classes matter.
- Weighted F1 is needed because the production distribution is imbalanced.

## A.4 Generalization Gap

The train-validation gap is:

$$
\mathrm{gap} = \mathrm{train\ accuracy} - \mathrm{validation\ accuracy}
$$

For the tuned grouped holdout:

$$
0.9924 - 0.9013 \approx 0.0911
$$

That is the main numerical evidence of moderate overfitting.

# Appendix A.5 Result And Accuracy Metric Definitions

This subsection defines the result metrics reported throughout the notebook and in the `classification_report` output.

## A.5.1 Holdout Accuracy

Holdout accuracy is the fraction of correct predictions on an unseen evaluation set:

$$
\mathrm{Accuracy}=\frac{1}{n}\sum_{i=1}^{n}\mathbf{1}(\hat{y}_i=y_i)
$$

where:

- $n$ is the number of evaluated samples
- $\hat{y}_i$ is the predicted class for sample $i$
- $y_i$ is the true class for sample $i$
- $\mathbf{1}(\cdot)$ equals 1 when the condition is true and 0 otherwise

### Interpretation

- grouped holdout accuracy is the main production-like metric in this project
- random holdout accuracy is optimistic because it allows phrase-pattern leakage
- balanced holdout accuracy is a fairness stress test, not the production headline score

## A.5.2 Precision

Precision for class $c$ measures how many predictions of class $c$ were actually correct:

$$
\mathrm{Precision}_c=\frac{TP_c}{TP_c+FP_c}
$$

where $TP_c$ is true positives and $FP_c$ is false positives for class $c$.

### Interpretation

- high precision means the model is conservative and usually right when it predicts a class
- low precision means the model is over-predicting that class and producing many false alarms

## A.5.3 Recall

Recall for class $c$ measures how many actual rows of class $c$ the model successfully recovered:

$$
\mathrm{Recall}_c=\frac{TP_c}{TP_c+FN_c}
$$

where $FN_c$ is false negatives for class $c$.

### Interpretation

- high recall means most true rows of the class are found
- low recall means the model is missing many rows that belong to that class

## A.5.4 F1 Score

F1 for class $c$ is the harmonic mean of precision and recall:

$$
F1_c = 2\cdot\frac{\mathrm{Precision}_c\cdot\mathrm{Recall}_c}{\mathrm{Precision}_c+\mathrm{Recall}_c}
$$

### Interpretation

- F1 is high only when both precision and recall are high
- it is useful when false positives and false negatives both matter

## A.5.5 Support

Support is the number of true examples belonging to class $c$ in the evaluation data:

$$
\mathrm{Support}_c = n_c
$$

### Interpretation

- support explains how much data backs each class-level metric
- low-support classes produce noisier precision, recall, and F1 estimates

## A.5.6 Macro F1

Macro F1 averages class-level F1 scores equally:

$$
\mathrm{MacroF1}=\frac{1}{K}\sum_{c=1}^{K} F1_c
$$

where $K$ is the number of classes.

### Interpretation

Macro F1 matters because rare labels should not disappear inside a strong overall accuracy score.

## A.5.7 Weighted F1

Weighted F1 averages class-level F1 scores using class support:

$$
\mathrm{WeightedF1}=\sum_{c=1}^{K}\frac{n_c}{n}F1_c
$$

where $n_c$ is the number of samples in class $c$ and $n=\sum_{c=1}^{K} n_c$.

### Interpretation

Weighted F1 is more aligned with the natural production distribution than Macro F1.

## A.5.8 Macro Precision And Macro Recall

The macro averages reported by `classification_report` are:

$$
\mathrm{MacroPrecision}=\frac{1}{K}\sum_{c=1}^{K}\mathrm{Precision}_c
$$

$$
\mathrm{MacroRecall}=\frac{1}{K}\sum_{c=1}^{K}\mathrm{Recall}_c
$$

These give every class equal weight, regardless of how frequent that class is.

## A.5.9 Weighted Precision And Weighted Recall

The weighted averages reported by `classification_report` are:

$$
\mathrm{WeightedPrecision}=\sum_{c=1}^{K}\frac{n_c}{n}\mathrm{Precision}_c
$$

$$
\mathrm{WeightedRecall}=\sum_{c=1}^{K}\frac{n_c}{n}\mathrm{Recall}_c
$$

These weight each class by its observed frequency in the evaluation set.

## A.5.10 Repeated Mean Accuracy

When evaluation is repeated across multiple splits, the reported mean accuracy is:

$$
\bar{A}=\frac{1}{m}\sum_{j=1}^{m} A_j
$$

where $A_j$ is the accuracy from split $j$ and $m$ is the number of repeated splits.

### Why It Is Used

A single holdout split can be noisy. Repeated grouped evaluation gives a more stable estimate of expected performance.

## A.5.11 Accuracy Standard Deviation

The standard deviation of repeated accuracies is:

$$
\sigma_A = \sqrt{\frac{1}{m}\sum_{j=1}^{m}(A_j-\bar{A})^2}
$$

### Interpretation

- lower standard deviation means more stable evaluation across splits
- higher standard deviation means the score depends more on the particular split

## A.5.12 Train-Test Gap

The train-test gap is:

$$
\mathrm{Gap}=\mathrm{TrainAccuracy}-\mathrm{ValidationAccuracy}
$$

### Interpretation

- small gap suggests better generalization
- large gap suggests overfitting
- in this project, the tuned grouped gap is around $0.0911$

## A.5.13 Balanced Unseen Accuracy

Balanced unseen accuracy uses the same accuracy formula as holdout accuracy, but on a deliberately class-balanced evaluation set:

$$
\mathrm{BalancedAccuracyLikeHoldout}=\frac{1}{n}\sum_{i=1}^{n}\mathbf{1}(\hat{y}_i=y_i)
$$

The difference is not in the formula, but in the construction of the evaluation sample: each eligible class contributes the same number of unseen rows.

### Interpretation

This is useful for judging rare-class robustness, but it should not replace the grouped natural-distribution score as the main business metric.

## A.5.14 Result Table Reading Guide

A professional reading of the result table in this project is:

- use grouped holdout accuracy as the primary business-facing estimate
- use repeated grouped mean and standard deviation to judge stability
- use macro metrics to judge rare-class behavior
- use weighted metrics to judge natural-distribution behavior
- use the train-validation gap to judge overfitting
- use balanced unseen accuracy as a secondary stress test for class fairness

# Appendix B: Indexed Library Reference

The definitions below are paraphrased from official project documentation and are included here to justify each library choice.

## B.1 pandas

- **Documented definition**: pandas provides fast, flexible, and expressive data structures for working with labeled or relational data.
- **Used in this project for**: JSON loading, missing-value cleanup, aggregation, and summary tables.
- **Why justified**: this assignment is structured tabular text data, so dataframe operations are the cleanest and most readable way to perform EDA and reporting.

## B.2 NumPy

- **Documented definition**: NumPy is the fundamental package for scientific computing in Python and provides multidimensional arrays plus numerical routines.
- **Used in this project for**: vectorized labels, indexing, numerical summaries, and compatibility with scikit-learn.
- **Why justified**: scikit-learn expects array-oriented inputs and outputs, so NumPy is the natural numerical base layer.

## B.3 Matplotlib

- **Documented definition**: Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.
- **Used in this project for**: histograms, bar charts, learning curves, validation curves, and summary plots.
- **Why justified**: a report notebook needs reproducible professional charts, and Matplotlib is the standard plotting tool for that.

## B.4 scikit-learn

- **Documented definition**: scikit-learn provides simple and efficient tools for predictive data analysis.
- **Used in this project for**:
  - TF-IDF vectorization
  - one-hot encoding
  - scaling
  - grouped train/test splitting
  - LinearSVC classification
  - evaluation metrics and tuning
- **Why justified**: this problem is a sparse multiclass text-classification problem, which is a strong fit for classical scikit-learn pipelines.

## B.5 joblib

- **Documented definition**: joblib provides lightweight pipelining tools in Python and efficient persistence for large Python objects, especially those containing NumPy arrays.
- **Used in this project for**: saving and loading the trained model artifact.
- **Why justified**: model persistence is part of reproducibility, and joblib is a practical standard for classical ML pipelines.

## B.6 scipy.sparse.csr_matrix

- **Documented definition**: CSR is a compressed sparse row matrix representation.
- **Used in this project for**: storing high-dimensional sparse text features and scaled numeric features efficiently.
- **Why justified**: TF-IDF matrices are mostly zeros, so dense storage is wasteful in both memory and compute.
- **Mathematical intuition**: if a feature matrix $X \in \mathbb{R}^{n \times p}$ has mostly zero entries, sparse storage keeps only non-zero values and index locations rather than all $n \times p$ cells.

## B.7 sklearn.base.BaseEstimator

- **Documented definition**: base class for estimators in scikit-learn.
- **Used in this project for**: making custom transformers compatible with the scikit-learn estimator API.
- **Why justified**: a consistent estimator interface is required for pipeline composition, fitting, parameter inspection, and validation.

## B.8 sklearn.base.TransformerMixin

- **Documented definition**: mixin class for transformers in scikit-learn that provides a default \`fit_transform\` implementation.
- **Used in this project for**: custom feature transformers that convert record dictionaries into model-ready sparse features.
- **Why justified**: this keeps custom preprocessing aligned with native scikit-learn transformers.

## B.9 sklearn.feature_extraction.text.TfidfVectorizer

- **Documented definition**: convert a collection of raw documents to a matrix of TF-IDF features.
- **Used in this project for**: word-level and character-level sparse text features.
- **Why justified**: item names and descriptions are the dominant predictive signal, and TF-IDF is a strong classical representation for sparse text classification.
- **Mathematics**:

$$
\mathrm{tfidf}(t,d)=\mathrm{tf}(t,d)\cdot\mathrm{idf}(t)
$$

with

$$
\mathrm{idf}(t)=\log\left(\frac{N+1}{\mathrm{df}(t)+1}\right)+1
$$

## B.10 sklearn.pipeline.FeatureUnion

- **Documented definition**: concatenate results of multiple transformer objects.
- **Used in this project for**: combining word TF-IDF, character TF-IDF, vendor encoding, and amount scaling into one feature matrix.
- **Why justified**: the model needs text, categorical, and numeric signals together in a single aligned representation.
- **Mathematical intuition**: if feature blocks are $X_1, X_2, \dots, X_m$, FeatureUnion forms a horizontal concatenation

$$
X = [X_1\; X_2\; \dots \; X_m]
$$

## B.11 sklearn.preprocessing.MaxAbsScaler

- **Documented definition**: scale each feature by its maximum absolute value.
- **Used in this project for**: scaling the log-transformed amount feature without destroying sparsity.
- **Why justified**: the amount feature should be numerically comparable without forcing dense centering.
- **Mathematics**:

$$
x'_j = \frac{x_j}{\max_i |x_{ij}|}
$$

## B.12 sklearn.preprocessing.OneHotEncoder

- **Documented definition**: encode categorical features as a one-hot numeric array.
- **Used in this project for**: converting vendor IDs into binary indicator columns.
- **Why justified**: vendor identity is categorical, not ordinal, so one-hot encoding avoids imposing a false numerical relationship.
- **Mathematical intuition**: for category set $\{c_1,\dots,c_K\}$, one-hot encoding maps category $c_k$ to vector $e_k \in \{0,1\}^K$ with one 1 and the rest 0.

## B.13 sklearn.pipeline.Pipeline

- **Documented definition**: assemble several steps that can be cross-validated together while setting different parameters.
- **Used in this project for**: connecting feature construction and LinearSVC into one trainable estimator.
- **Why justified**: the training, evaluation, and persistence logic should operate on one consistent object.

## B.14 sklearn.metrics.accuracy_score

- **Documented definition**: compute the classification accuracy score.
- **Used in this project for**: headline performance measurement on holdout and repeated splits.
- **Why justified**: the assignment sets an explicit accuracy threshold.
- **Mathematics**:

$$
\mathrm{Accuracy}=\frac{1}{n}\sum_{i=1}^{n}\mathbf{1}(\hat{y}_i=y_i)
$$

## B.15 sklearn.metrics.classification_report

- **Documented definition**: build a text summary showing main classification metrics.
- **Used in this project for**: per-class precision, recall, F1, macro averages, and weighted averages.
- **Why justified**: a multiclass imbalanced problem needs more than a single scalar metric.

## B.16 sklearn.model_selection.GroupShuffleSplit

- **Documented definition**: randomized train/test indices according to a third-party provided group.
- **Used in this project for**: grouped holdout and grouped repeated evaluation by normalized item name.
- **Why justified**: prevents repeated item-name templates from leaking across train and test.

## B.17 sklearn.model_selection.ShuffleSplit

- **Documented definition**: randomized permutation cross-validator.
- **Used in this project for**: quick random-row baseline comparison.
- **Why justified**: useful as a naive benchmark, even though it is optimistic here.

## B.18 sklearn.model_selection.StratifiedGroupKFold

- **Documented definition**: variation of K-fold that preserves class proportions as much as possible while respecting non-overlapping groups.
- **Used in this project for**: feasibility checks on grouped stratified evaluation.
- **Why justified**: helps test whether group-aware evaluation can also remain label-aware under severe imbalance.

## B.19 sklearn.model_selection.learning_curve

- **Documented definition**: determine cross-validated training and test scores for different training set sizes.
- **Used in this project for**: diagnosing whether more data would likely improve grouped validation accuracy.
- **Why justified**: the train-validation gap alone does not show whether data scarcity is the main bottleneck.

## B.20 sklearn.model_selection.validation_curve

- **Documented definition**: determine training and test scores for varying parameter values.
- **Used in this project for**: studying how the SVM regularization parameter $C$ changes train and validation accuracy.
- **Why justified**: necessary for bias-variance diagnosis and hyperparameter interpretation.

## B.21 sklearn.svm.LinearSVC

- **Documented definition**: linear support vector classification implemented with liblinear.
- **Used in this project for**: the final multiclass sparse-text classifier.
- **Why justified**: linear SVMs are strong baselines for high-dimensional sparse text problems and are efficient enough for repeated grouped validation.
- **Mathematics**: for class score $f_k(x)=w_k^T x+b_k$, prediction is

$$
\hat{y}=\arg\max_k f_k(x)
$$

and the optimization balances margin size against hinge-loss penalties through $C$.


# Appendix C: Indexed Concept Reference

The concept definitions below are aligned with standard machine-learning documentation and glossary sources.

## C.1 Feature

- **Definition**: an input variable used by a model.
- **In this project**: text features, vendor identity, and log-scaled amount.

## C.2 Label

- **Definition**: the answer or target portion of a supervised-learning example.
- **In this project**: the label is `accountName`.

## C.3 Labeled Example

- **Definition**: an example containing one or more features and a label.
- **In this project**: one expense record with text, vendor, amount, and the true account label.

## C.4 Validation Set

- **Definition**: the subset used for initial model evaluation during development.
- **In this project**: grouped holdout and grouped CV act as validation mechanisms.

## C.5 Overfitting

- **Definition**: a model performs well on training data but poorly on new unseen data.
- **Observed here**: training accuracy is materially higher than grouped unseen accuracy.

## C.6 Underfitting

- **Definition**: a model performs poorly on both training data and unseen data.
- **Observed here**: not present, because both train and validation metrics are high.

## C.7 Class Imbalance

- **Definition**: classes do not appear with roughly equal frequency.
- **Why it matters here**: many account labels have very low support, so accuracy alone is not enough.

## C.8 Generalization

- **Definition**: the ability of a model to perform well on new examples.
- **Why it matters here**: the finance use case depends on unseen future transactions, not memorized templates.

## C.9 Bias-Variance Trade-off

- **Bias**: error from an overly simple model.
- **Variance**: error from excessive sensitivity to training details.
- **Observed here**: variance is the main issue, not bias.

# Appendix D: Indexed Sample And Split Definitions

## D.1 Sample

- **Definition**: a single row or observation in the dataset.
- **In this notebook**: one expense record is one sample.

## D.2 Training Sample

- **Definition**: a sample used to estimate model parameters.

## D.3 Validation Sample

- **Definition**: a held-out sample used to estimate generalization during development.

## D.4 Balanced Unseen Sample

- **Definition**: a held-out sample constructed so that eligible labels contribute equally many examples.
- **Purpose**: stress-test minority-class performance without changing the real production distribution.

## D.5 Eligible Label

- **Definition**: a label with enough rows and enough distinct normalized item-name groups to support balanced unseen evaluation.

## D.6 Support

- **Definition**: the number of evaluated samples belonging to a class.

## D.7 Leakage

- **Definition**: information from training data reaches evaluation data in a way that inflates the metric.
- **In this project**: repeated normalized item names can leak phrase templates across random row splits.

# Appendix E: Source Notes And Index

## E.1 Documentation Basis

The appendix definitions above are paraphrased from official documentation and glossary pages for:

1. pandas documentation
2. NumPy documentation
3. Matplotlib documentation
4. scikit-learn documentation
5. joblib documentation
6. Google Machine Learning glossary / crash course

## E.2 Index

| Index | Topic |
| --- | --- |
| A | Mathematical notes and equations |
| B | Machine-learning and Python library definitions |
| C | Core ML concept definitions |
| D | Sample and split terminology |
| E | Source notes and index |

## E.3 Usage Justification Summary

- pandas: best fit for labeled tabular data loading and summarization
- NumPy: numerical base layer for arrays and vectorized operations
- Matplotlib: reproducible report-quality charts
- scikit-learn: efficient classical ML pipeline for sparse text classification
- joblib: lightweight persistence for trained artifacts
