# MLflow: how to run, what to run, what to see

Operator guide. Everything you need is produced by two commands: one to train, one to view.

## 0. Activate env once

```pwsh
. .\.venv\Scripts\Activate.ps1
```

## 1. Start the MLflow UI (keep running in its own terminal)

```pwsh
uv run python -m src.tracker server --backend-store-uri "sqlite:///mlflow.db" --artifacts-destination ".\mlartifacts"
```

Open <http://127.0.0.1:5000>. Leave this terminal running.

First time you open it, you see the landing page with an empty experiment list
until a run is recorded.

## 2. Point runs at that server (one-time per new shell)

```pwsh
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
```

If you skip this, runs go to the local `./mlruns` folder and the server UI
will not see them. Set the env var before `retrain`.

## 3. Trigger a full tracked training run

```pwsh
uv run python -m src.tracker retrain
```

This executes the full pipeline (load data, two-stage tuning, all holdouts,
diagnostics, save joblib) inside a single MLflow run. Expect ~15-25 minutes
on laptop hardware.

Equivalent from Python:

```python
from src.tracker import retrain
summary = retrain(run_name="manual-retrain")
```

## 4. What you see in the MLflow dashboard

Open the UI -> click **Experiments** -> `peakflo-expense-classifier` -> click
the latest run (named `run-YYYYMMDD-HHMMSS`). You get these tabs:

### Overview tab

- Run ID, start/end time, duration, status
- System tags auto-populated by MLflow: `mlflow.source.name` (entry script),
  `mlflow.source.git.commit`, `mlflow.user`
- Custom tags set by the tracker: `project=peakflo`, `component=training`,
  `trigger=retrain`, `git_commit=<short sha>`, `data_path=accounts-bills.json`

### Parameters tab

| key | example | meaning |
|---|---|---|
| `best_C` | `1.0` | LinearSVC regularisation picked by two-stage tuning |
| `best_class_weight` | `balanced` | picked by balanced-unseen refinement |
| `best_oversample_min_count` | `5` | minority resample floor |
| `optimize_for` | `macro_f1` | stage-1 ranking objective |
| `dataset_rows` | `5226` | total records |
| `dataset_labels` | `103` | unique account_name values |
| `dataset_vendors` | `337` | unique vendor ids |

### Metrics tab

Each metric is a numeric series (single-step here):

- `baseline_lookup_accuracy`
- `holdout_random_accuracy` / `_macro_f1` / `_train_test_gap`
- `holdout_group_item_accuracy` / `_macro_f1` / `_train_test_gap`
- `holdout_group_item_tuned_accuracy` / `_macro_f1` / `_train_test_gap`
- `balanced_holdout_accuracy` / `_macro_f1`
- `calibrated_fallback_balanced_accuracy` / `_macro_f1`

Use the **Chart** view to plot them across runs as you iterate.

### Artifacts tab (browsable tree)

```
metrics_card.json              <- tabular benchmark comparison (UI renders as table)
evaluation_summary.json        <- full nested summary dict from run_training_pipeline
dataset_validation.json        <- schema check, row count, null counts
reports/
  evaluation_summary.json
  03_results.md                <- auto-generated markdown report
model_joblib/
  account_classifier.joblib    <- the exact file saved by the training pipeline
model/                         <- MLflow-flavoured sklearn model
  MLmodel                      <- flavour metadata
  model.pkl
  conda.yaml                   <- conda env for reproducing the model
  python_env.yaml              <- python version + pip
  requirements.txt             <- frozen pip deps
  input_example.json           (only if signature was provided)
```

Click any JSON/MD file to preview in-browser.

### Datasets tab

Shows the input dataset logged via `mlflow.data.from_pandas`:

- name: `accounts-bills.json`
- digest (hash) so two runs on identical data share the same dataset id
- schema (column names + dtypes)
- target column: `account_name`
- context: `training`

### Tags tab

The custom tags listed in Overview plus any MLflow system tags.

### Traces tab

Because `run_training_pipeline` is wrapped in `mlflow.trace`, you get a span
for the pipeline execution with start/end timestamps. Useful for spotting
which stage blew up when a run fails.

## 5. What you see under Models -> `peakflo-account-classifier`

- Every `retrain` creates a new **version** (v1, v2, ...).
- Latest version is tagged with alias `champion`.
- Click a version -> see its source run, the MLmodel metadata, the signature
  (if any), and the artifacts tied to that version.
- Promote/demote aliases in the UI or via the API to do blue/green rollouts.

## 6. Load the registered model from Python

```python
from src.tracker import MLflowTracker

model = MLflowTracker().load_model("champion")
preds = model.predict(list_of_feature_dicts)
```

Or smoke-test from the CLI:

```pwsh
uv run python -m src.tracker load-champion
```

## 7. Compare runs

In the UI, check two or more run rows in the experiment table and click
**Compare**. You get side-by-side parameters, metrics, and a parallel-coordinates
plot. Useful after changing a hyperparameter or dataset.

## 8. Where things live on disk

- `mlflow.db` - SQLite backend with run metadata (gitignored)
- `mlartifacts/` - artifact blobs, including the joblib file and the MLmodel
  flavour directory (gitignored)
- `mlruns/` - only used if you skip the server and run against the file backend

## Configuration knobs

Defaults live in `TrackerConfig`:

| field | default | override |
|---|---|---|
| `experiment_name` | `peakflo-expense-classifier` | pass a `TrackerConfig` |
| `tracking_uri` | `file:./mlruns` | `MLFLOW_TRACKING_URI` env var |
| `registered_model_name` | `peakflo-account-classifier` | pass a `TrackerConfig` |
| `autolog` | `False` | pass `TrackerConfig(autolog=True)` |

Autolog is off because the two-stage tuning calls `model.fit` ~150 times and
would flood the run with nested noise. The metric card table is the cleaner
equivalent.

## Troubleshooting

- **Run shows up in `./mlruns` instead of the server UI** - you forgot to set
  `MLFLOW_TRACKING_URI=http://127.0.0.1:5000` in the shell that ran `retrain`.
- **`ValueError: Dataset missing required columns`** - `accounts-bills.json`
  lost one of `account_name`, `vendor_id`, `item_name`, `amount`. Fix the
  source file; validation is intentionally strict.
- **Server port already in use** - pass `--port 5050` to `src.tracker server`.

## Tests

```pwsh
uv run pytest tests/test_tracker.py -v
```

9 tests cover run context, params/metrics logging, dataset validation (ok +
two failure modes), model log/register/load round trip, and the tracing
decorator. Full repo suite: `uv run pytest` -> 27 passed.
