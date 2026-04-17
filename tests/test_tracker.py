"""Tests for the MLflow tracker wrapper.

These tests run against a throwaway file-based tracking URI inside tmp_path so
no state leaks into the repo-level ``mlruns/`` directory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import mlflow

from src.tracker import (
    MLflowTracker,
    TrackerConfig,
    _build_metric_card,
    _flat_metrics_from_summary,
    _stringify,
)


@pytest.fixture
def tracker(tmp_path):
    cfg = TrackerConfig(
        experiment_name="peakflo-test",
        tracking_uri=(tmp_path / "mlruns").as_uri(),
        registered_model_name="peakflo-test-model",
        autolog=False,
    )
    return MLflowTracker(cfg)


def test_stringify_handles_primitives_and_objects():
    assert _stringify(None) == "None"
    assert _stringify(1.5) == "1.5"
    assert _stringify("x") == "x"
    assert _stringify({"a": 1}) == '{"a": 1}'


def test_build_metric_card_has_expected_rows():
    summary = {
        "baseline": {"item_vendor_lookup_accuracy": 0.7},
        "holdout": {
            "random": {"accuracy": 0.9, "macro_f1": 0.8},
            "group_item": {"accuracy": 0.89, "macro_f1": 0.75},
            "group_item_tuned": {"accuracy": 0.9, "macro_f1": 0.77},
        },
        "balanced_holdout": {"accuracy": 0.83, "macro_f1": 0.79},
        "calibrated_fallback_balanced": {"accuracy": 0.84, "macro_f1": 0.8},
    }
    rows = _build_metric_card(summary)
    benchmarks = [r["benchmark"] for r in rows]
    assert "Baseline item/vendor lookup" in benchmarks
    assert "Grouped holdout (tuned)" in benchmarks
    assert "Calibrated fallback (balanced)" in benchmarks


def test_flat_metrics_only_includes_numeric_values():
    summary = {
        "baseline": {"item_vendor_lookup_accuracy": 0.74},
        "holdout": {
            "random": {"accuracy": 0.9, "macro_f1": 0.8, "accuracy_gap": 0.05},
            "group_item": {"accuracy": None, "macro_f1": 0.7},
            "group_item_tuned": {"accuracy": 0.9, "macro_f1": 0.77, "accuracy_gap": 0.1},
        },
        "balanced_holdout": {"accuracy": 0.83, "macro_f1": 0.79},
        "calibrated_fallback_balanced": {"accuracy": 0.84, "macro_f1": 0.8},
    }
    flat = _flat_metrics_from_summary(summary)
    assert flat["baseline_lookup_accuracy"] == 0.74
    assert flat["holdout_group_item_tuned_accuracy"] == 0.9
    assert "holdout_group_item_accuracy" not in flat  # None filtered out


def test_start_run_logs_params_and_metrics(tracker):
    with tracker.start_run(run_name="unit-test") as run:
        tracker.log_params({"C": 2.0, "class_weight": None, "optimize_for": "macro_f1"})
        tracker.log_metrics({"accuracy": 0.9, "macro_f1": 0.77})

    fetched = mlflow.get_run(run.info.run_id)
    assert fetched.data.params["C"] == "2.0"
    assert fetched.data.params["class_weight"] == "None"
    assert fetched.data.metrics["accuracy"] == pytest.approx(0.9)
    assert fetched.data.metrics["macro_f1"] == pytest.approx(0.77)


def test_validate_dataset_rejects_missing_columns(tracker):
    df = pd.DataFrame({"a": [1, 2, 3]})
    with tracker.start_run(run_name="validate-fail"):
        with pytest.raises(ValueError, match="missing required columns"):
            tracker.validate_dataset(df, required_columns=["a", "b"])


def test_validate_dataset_rejects_empty_volume(tracker):
    df = pd.DataFrame({"a": [1]})
    with tracker.start_run(run_name="validate-empty"):
        with pytest.raises(ValueError, match="need >="):
            tracker.validate_dataset(df, required_columns=["a"], min_rows=10)


def test_validate_dataset_accepts_good_frame(tracker):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", None]})
    with tracker.start_run(run_name="validate-ok"):
        report = tracker.validate_dataset(df, required_columns=["a", "b"], min_rows=2)
    assert report["schema_check"] == "ok"
    assert report["row_count"] == 3
    assert report["null_counts"]["b"] == 1


def test_log_and_register_model_round_trip(tracker):
    X = pd.DataFrame({"f1": np.arange(20), "f2": np.arange(20) * 0.5})
    y = (X["f1"] > 10).astype(int)
    pipe = Pipeline([("clf", LogisticRegression(max_iter=200))]).fit(X, y)

    with tracker.start_run(run_name="register-test"):
        uri = tracker.log_and_register_model(pipe, register=True, alias="champion")
    assert uri.startswith("runs:/") or uri.startswith("models:/")

    loaded = tracker.load_model("champion")
    preds = loaded.predict(X)
    assert len(preds) == len(X)


def test_retrain_required_columns_match_expense_record_fields():
    """Regression guard: retrain's validate_dataset column list must match
    the actual ExpenseRecord schema. Caught a real bug where 'amount' was
    required but the field is 'item_total_amount'.
    """
    from dataclasses import fields

    from src.feature_engineering_pipeline import ExpenseRecord
    from src.tracker import retrain as _retrain  # noqa: F401

    # Extract the literal list from the retrain source to avoid drifting
    import inspect
    import re

    from src import tracker as tracker_mod

    src = inspect.getsource(tracker_mod.retrain)
    match = re.search(r"required_columns\s*=\s*\[([^\]]+)\]", src)
    assert match is not None, "could not find required_columns in retrain()"
    required = [c.strip().strip('"').strip("'") for c in match.group(1).split(",")]
    record_fields = {f.name for f in fields(ExpenseRecord)}
    missing = [c for c in required if c not in record_fields]
    assert not missing, f"retrain requires columns missing from ExpenseRecord: {missing}"


def test_traced_decorator_runs_function_unchanged(tracker):
    @MLflowTracker.traced(name="adder")
    def add(a, b):
        return a + b

    with tracker.start_run(run_name="trace-test"):
        assert add(2, 3) == 5
