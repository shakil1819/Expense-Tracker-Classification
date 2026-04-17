# MLflow Integration

`src/tracker.py` adds MLflow 3.x to the Peakflo pipeline. It covers experiment
tracking, dataset validation, model registry, tracing, a metric card view, and
a one-click retrain entry.

## What was added

- `src/tracker.py`
  - `MLflowTracker` class (runs, params, metrics, artifacts, datasets, models, tracing)
  - `retrain(...)` one-click wrapper that re-runs `run_training_pipeline` under a fresh MLflow run
  - `start_tracking_server(...)` helper
  - CLI with `server`, `retrain`, `load-champion` subcommands
- `tests/test_tracker.py` (9 tests, all passing)
- Dependency: `mlflow>=3.0` (installed `mlflow==3.11.1`). Pandas relaxed to `>=2.2,<3`
  because MLflow 3.x does not yet support pandas 3.

## Why this shape

- The core pipeline (`src/training_pipeline.py`) is not modified. The tracker
  wraps it from the outside and logs everything off the returned `summary` dict.
  That keeps the training code readable and the MLflow layer optional.
- Autolog is **off** by default. The pipeline runs ~150 inner fits during
  grouped CV tuning, which would drown the run view with nested noise. Manual
  logging of a clean metric card is far more readable.
- Model signatures are not inferred because the sklearn pipeline's first step is
  a `DictTextVectorizer` that consumes `list[dict]`, which MLflow's
  pandas-schema inference cannot represent faithfully. The model still round
  trips correctly through `mlflow.sklearn.load_model`.

## Usage

### Start the tracking server

```pwsh
. .\.venv\Scripts\Activate.ps1
uv run python -m src.tracker server --backend-store-uri "sqlite:///mlflow.db" --artifacts-destination ".\mlartifacts"
```

Then open <http://127.0.0.1:5000>.

### One-click retrain (registers a new model version and promotes it to `champion`)

```pwsh
uv run python -m src.tracker retrain
```

This does:

1. Loads `accounts-bills.json` and validates schema + row count.
2. Logs the dataset with `mlflow.data.from_pandas` (versioned, hashed).
3. Wraps `run_training_pipeline` in an `mlflow.trace` span.
4. Runs the full pipeline (tuning, holdouts, balanced unseen, diagnostics).
5. Logs best params, flat metric set, metric card table, summary JSON,
   the results markdown, and the joblib artifact.
6. Logs the sklearn model and registers it under `peakflo-account-classifier`.
7. Sets alias `champion` to the latest version.

### Load the champion model

```pwsh
uv run python -m src.tracker load-champion
```

Or in code:

```python
from src.tracker import MLflowTracker
model = MLflowTracker().load_model("champion")
preds = model.predict(list_of_feature_dicts)
```

### Tracing a custom stage

```python
from src.tracker import MLflowTracker

@MLflowTracker.traced(name="feature_build")
def build_features(records):
    ...
```

## Configuration

Default tracking URI is `file:./mlruns`. Override with
`MLFLOW_TRACKING_URI=http://host:5000` in the environment, or by passing a
`TrackerConfig(tracking_uri=...)` instance.

## Test coverage

`tests/test_tracker.py` exercises:

- param / metric / dataset / validation / registry round trip
- metric card construction
- tracing decorator pass-through

Full suite: `uv run pytest` - 27 passed.
