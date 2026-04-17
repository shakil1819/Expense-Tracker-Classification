"""MLflow integration for the Peakflo expense classifier.

Provides a thin, opinionated wrapper around MLflow 3.x:

- Experiment + run management
- Param / metric / artifact / dataset logging
- Dataset validation (schema + volume)
- Metric card view (table artifact)
- Model logging and registry with an alias (champion)
- Tracing for key pipeline stages via ``MLflowTracker.traced``
- Tracking server launcher
- One-click retrain helper and CLI

The tracker is self-contained and does not modify the core training pipeline.
It wraps ``run_training_pipeline`` post-hoc, logging its summary dict to MLflow.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import mlflow
import mlflow.sklearn
import pandas as pd
from loguru import logger
from mlflow.tracking import MlflowClient


DEFAULT_EXPERIMENT = "peakflo-expense-classifier"
DEFAULT_REGISTERED_MODEL = "peakflo-account-classifier"
DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
CHAMPION_ALIAS = "champion"


@dataclass
class TrackerConfig:
    experiment_name: str = DEFAULT_EXPERIMENT
    tracking_uri: str = DEFAULT_TRACKING_URI
    registered_model_name: str = DEFAULT_REGISTERED_MODEL
    autolog: bool = False


class MLflowTracker:
    """Thin facade over MLflow for this project."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        mlflow.set_tracking_uri(self.config.tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)
        if self.config.autolog:
            mlflow.sklearn.autolog(
                log_input_examples=False,
                log_model_signatures=False,
                log_models=False,
                silent=True,
            )
        self.client = MlflowClient(tracking_uri=self.config.tracking_uri)
        logger.info(
            "MLflow tracker ready: uri={}, experiment={}, autolog={}",
            self.config.tracking_uri,
            self.config.experiment_name,
            self.config.autolog,
        )

    # -- run management ------------------------------------------------------

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        nested: bool = False,
    ) -> Iterator[Any]:
        run_name = run_name or f"run-{datetime.now():%Y%m%d-%H%M%S}"
        merged_tags = {"project": "peakflo", "component": "training"}
        if tags:
            merged_tags.update(tags)
        with mlflow.start_run(run_name=run_name, nested=nested, tags=merged_tags) as run:
            logger.info("MLflow run started: name={}, id={}", run_name, run.info.run_id)
            try:
                yield run
            finally:
                logger.info("MLflow run finished: id={}", run.info.run_id)

    # -- logging helpers -----------------------------------------------------

    def log_params(self, params: dict[str, Any]) -> None:
        clean = {k: _stringify(v) for k, v in params.items()}
        if clean:
            mlflow.log_params(clean)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        clean: dict[str, float] = {}
        for k, v in metrics.items():
            if v is None:
                continue
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                clean[k] = float(v)
        if clean:
            mlflow.log_metrics(clean, step=step)

    def log_dict(self, data: dict, artifact_file: str) -> None:
        mlflow.log_dict(data, artifact_file)

    def log_metric_card(
        self,
        rows: list[dict[str, Any]],
        artifact_file: str = "metrics_card.json",
    ) -> None:
        """Log a tabular metric view that renders in the MLflow UI artifact browser."""
        if not rows:
            return
        df = pd.DataFrame(rows)
        mlflow.log_table(data=df, artifact_file=artifact_file)

    def log_artifact(
        self,
        local_path: str | Path,
        artifact_path: str | None = None,
    ) -> None:
        path = Path(local_path)
        if not path.exists():
            logger.warning("Skipping missing artifact: {}", path)
            return
        mlflow.log_artifact(str(path), artifact_path=artifact_path)

    # -- dataset tracking + validation ---------------------------------------

    def log_dataset(
        self,
        df: pd.DataFrame,
        name: str,
        targets: str | None = None,
        context: str = "training",
    ) -> None:
        dataset = mlflow.data.from_pandas(df, source=name, name=name, targets=targets)
        mlflow.log_input(dataset, context=context)
        logger.info(
            "Logged dataset: name={}, rows={}, targets={}, context={}",
            name, len(df), targets, context,
        )

    def validate_dataset(
        self,
        df: pd.DataFrame,
        required_columns: list[str],
        min_rows: int = 1,
    ) -> dict[str, Any]:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")
        if len(df) < min_rows:
            raise ValueError(f"Dataset has {len(df)} rows, need >= {min_rows}")
        null_counts = {c: int(df[c].isna().sum()) for c in required_columns}
        report = {
            "schema_check": "ok",
            "row_count": int(len(df)),
            "required_columns": required_columns,
            "null_counts": null_counts,
        }
        mlflow.log_dict(report, "dataset_validation.json")
        logger.info("Dataset validated: rows={}, nulls={}", len(df), null_counts)
        return report

    # -- model logging + registry -------------------------------------------

    def log_and_register_model(
        self,
        model,
        artifact_path: str = "model",
        register: bool = True,
        alias: str | None = CHAMPION_ALIAS,
        pip_requirements: list[str] | None = None,
    ) -> str:
        """Log the trained sklearn pipeline and optionally promote to an alias.

        We do not pass an input signature because the pipeline's first step is a
        ``DictTextVectorizer`` that expects a ``list[dict]``, which MLflow's
        pandas-based signature inference cannot represent cleanly. The model
        still round-trips correctly via ``mlflow.sklearn.load_model``.
        """
        kwargs: dict[str, Any] = {"sk_model": model, "name": artifact_path}
        if register:
            kwargs["registered_model_name"] = self.config.registered_model_name
        if pip_requirements is not None:
            kwargs["pip_requirements"] = pip_requirements
        model_info = mlflow.sklearn.log_model(**kwargs)
        logger.info("Model logged: uri={}", model_info.model_uri)

        if register and alias:
            versions = self.client.search_model_versions(
                f"name='{self.config.registered_model_name}'"
            )
            if versions:
                latest = max(versions, key=lambda v: int(v.version))
                self.client.set_registered_model_alias(
                    self.config.registered_model_name, alias, latest.version
                )
                logger.info(
                    "Promoted {} v{} to alias '{}'",
                    self.config.registered_model_name, latest.version, alias,
                )
        return model_info.model_uri

    def load_model(self, alias: str = CHAMPION_ALIAS):
        uri = f"models:/{self.config.registered_model_name}@{alias}"
        logger.info("Loading model from registry: {}", uri)
        return mlflow.sklearn.load_model(uri)

    # -- tracing -------------------------------------------------------------

    @staticmethod
    def traced(name: str | None = None):
        """Decorator that wraps a callable in an MLflow span for tracing."""

        def wrapper(fn):
            return mlflow.trace(name=name or fn.__name__)(fn)

        return wrapper


# -- module-level helpers ----------------------------------------------------

def _stringify(v: Any) -> str:
    if v is None:
        return "None"
    if isinstance(v, (int, float, bool, str)):
        return str(v)
    try:
        return json.dumps(v, default=str)
    except TypeError:
        return str(v)


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        )
        return out.strip()
    except Exception:
        return None


def _records_to_dataframe(records) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in records])


def _build_metric_card(summary: dict) -> list[dict[str, Any]]:
    def g(path: list[str], default=None):
        node: Any = summary
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    rows: list[dict[str, Any]] = []
    rows.append({
        "benchmark": "Baseline item/vendor lookup",
        "accuracy": g(["baseline", "item_vendor_lookup_accuracy"]),
        "macro_f1": None,
    })
    for key, label in [
        ("random", "Random holdout"),
        ("group_item", "Grouped holdout (default)"),
        ("group_item_tuned", "Grouped holdout (tuned)"),
    ]:
        rows.append({
            "benchmark": label,
            "accuracy": g(["holdout", key, "accuracy"]),
            "macro_f1": g(["holdout", key, "macro_f1"]),
        })
    rows.append({
        "benchmark": "Balanced unseen (tuned)",
        "accuracy": g(["balanced_holdout", "accuracy"]),
        "macro_f1": g(["balanced_holdout", "macro_f1"]),
    })
    rows.append({
        "benchmark": "Calibrated fallback (balanced)",
        "accuracy": g(["calibrated_fallback_balanced", "accuracy"]),
        "macro_f1": g(["calibrated_fallback_balanced", "macro_f1"]),
    })
    return rows


def _flat_metrics_from_summary(summary: dict) -> dict[str, float]:
    metrics: dict[str, float] = {}

    def put(key: str, value):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[key] = float(value)

    put("baseline_lookup_accuracy", summary.get("baseline", {}).get("item_vendor_lookup_accuracy"))
    for k in ("random", "group_item", "group_item_tuned"):
        h = summary.get("holdout", {}).get(k, {})
        put(f"holdout_{k}_accuracy", h.get("accuracy"))
        put(f"holdout_{k}_macro_f1", h.get("macro_f1"))
        put(f"holdout_{k}_train_test_gap", h.get("accuracy_gap"))
    bh = summary.get("balanced_holdout", {})
    put("balanced_holdout_accuracy", bh.get("accuracy"))
    put("balanced_holdout_macro_f1", bh.get("macro_f1"))
    cfb = summary.get("calibrated_fallback_balanced", {})
    put("calibrated_fallback_balanced_accuracy", cfb.get("accuracy"))
    put("calibrated_fallback_balanced_macro_f1", cfb.get("macro_f1"))
    return metrics


# -- one-click retrain -------------------------------------------------------

def retrain(
    data_path: str | None = None,
    run_name: str | None = None,
    register: bool = True,
    config: TrackerConfig | None = None,
) -> dict:
    """Re-run the training pipeline inside a fresh MLflow run and register the model.

    Returns the evaluation summary from ``run_training_pipeline``.
    """
    from joblib import load

    from src.feature_engineering_pipeline import load_records
    from src.training_pipeline import DATA_PATH, run_training_pipeline

    tracker = MLflowTracker(config)
    path = data_path or DATA_PATH
    tags = {
        "trigger": "retrain",
        "git_commit": _git_commit() or "unknown",
        "data_path": path,
    }

    # Trace the training pipeline entry point so each retrain has a traceable span.
    traced_pipeline = mlflow.trace(name="run_training_pipeline")(run_training_pipeline)

    with tracker.start_run(run_name=run_name, tags=tags):
        records = load_records(path)
        df = _records_to_dataframe(records)
        tracker.validate_dataset(
            df,
            required_columns=["account_name", "vendor_id", "item_name", "item_total_amount"],
            min_rows=100,
        )
        tracker.log_dataset(df, name=path, targets="account_name", context="training")

        summary = traced_pipeline(path)

        tuning = summary.get("hyperparameter_tuning", {})
        balanced = summary.get("two_stage_tuning", {}).get("balanced_refinement", {}) or {}
        tracker.log_params({
            "best_C": balanced.get("C", tuning.get("best_C")),
            "best_class_weight": balanced.get("class_weight", tuning.get("best_class_weight")),
            "best_oversample_min_count": balanced.get(
                "oversample_min_count", tuning.get("best_oversample_min_count")
            ),
            "optimize_for": tuning.get("optimize_for"),
            "dataset_rows": len(records),
            "dataset_labels": df["account_name"].nunique(),
            "dataset_vendors": df["vendor_id"].nunique(),
        })

        tracker.log_metrics(_flat_metrics_from_summary(summary))
        tracker.log_metric_card(_build_metric_card(summary))

        tracker.log_dict(summary, "evaluation_summary.json")
        tracker.log_artifact("artifacts/evaluation_summary.json", artifact_path="reports")
        tracker.log_artifact(".docs/03_results.md", artifact_path="reports")
        tracker.log_artifact("artifacts/account_classifier.joblib", artifact_path="model_joblib")

        model_path = Path("artifacts/account_classifier.joblib")
        if model_path.exists():
            model = load(model_path)
            tracker.log_and_register_model(model, register=register)
        else:
            logger.warning("Trained model not found at {}; skipping registry step", model_path)

    return summary


# -- tracking server launcher ------------------------------------------------

DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 5000
DEFAULT_BACKEND_URI = "sqlite:///mlflow.db"
DEFAULT_ARTIFACTS_DIR = "./mlartifacts"


def _is_server_up(url: str, timeout: float = 1.0) -> bool:
    """Return True if an MLflow server is reachable at ``url``."""
    probe = url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(probe, timeout=timeout) as resp:
            return 200 <= resp.status < 500
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return False


def start_tracking_server(
    host: str = DEFAULT_SERVER_HOST,
    port: int = DEFAULT_SERVER_PORT,
    backend_store_uri: str = DEFAULT_BACKEND_URI,
    artifacts_destination: str = DEFAULT_ARTIFACTS_DIR,
) -> None:
    """Run ``mlflow server`` in the foreground (blocks until Ctrl+C)."""
    cmd = [
        sys.executable, "-m", "mlflow", "server",
        "--host", host,
        "--port", str(port),
        "--backend-store-uri", backend_store_uri,
        "--artifacts-destination", artifacts_destination,
    ]
    logger.info("Starting MLflow server: {}", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_server_running(
    host: str = DEFAULT_SERVER_HOST,
    port: int = DEFAULT_SERVER_PORT,
    backend_store_uri: str = DEFAULT_BACKEND_URI,
    artifacts_destination: str = DEFAULT_ARTIFACTS_DIR,
    readiness_timeout_s: float = 30.0,
) -> str:
    """Ensure an MLflow tracking server is live at ``host:port`` and return its URL.

    If one is already listening, reuse it. Otherwise spawn ``mlflow server`` as a
    detached background subprocess, wait for its ``/health`` endpoint, and return
    the URL. The subprocess outlives this Python process so the dashboard stays
    browsable after training finishes.
    """
    url = f"http://{host}:{port}"
    if _is_server_up(url):
        logger.info("MLflow server already running at {}", url)
        return url

    Path(artifacts_destination).mkdir(parents=True, exist_ok=True)
    log_path = Path("mlflow_server.log")
    log_file = log_path.open("ab")
    cmd = [
        sys.executable, "-m", "mlflow", "server",
        "--host", host,
        "--port", str(port),
        "--backend-store-uri", backend_store_uri,
        "--artifacts-destination", artifacts_destination,
    ]
    popen_kwargs: dict[str, Any] = {
        "stdout": log_file,
        "stderr": log_file,
        "stdin": subprocess.DEVNULL,
        "close_fds": True,
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
        )
    else:
        popen_kwargs["start_new_session"] = True

    logger.info("Starting MLflow server in background: {} (log: {})", url, log_path)
    subprocess.Popen(cmd, **popen_kwargs)

    deadline = time.monotonic() + readiness_timeout_s
    while time.monotonic() < deadline:
        if _is_server_up(url):
            logger.info("MLflow server ready at {}", url)
            return url
        time.sleep(0.5)
    raise RuntimeError(
        f"MLflow server did not become ready at {url} within {readiness_timeout_s}s; "
        f"see {log_path} for details"
    )


def run_tracked_pipeline(
    data_path: str | None = None,
    run_name: str | None = None,
    register: bool = True,
) -> dict:
    """High-level entrypoint for ``main.py``.

    Boots the dashboard if needed, points runs at it via ``MLFLOW_TRACKING_URI``,
    and executes ``retrain``. The dashboard stays up after this returns.
    """
    url = ensure_server_running()
    os.environ["MLFLOW_TRACKING_URI"] = url
    logger.info("MLflow dashboard: {}", url)
    print(f"MLflow dashboard: {url}")
    cfg = TrackerConfig(tracking_uri=url)
    return retrain(data_path=data_path, run_name=run_name, register=register, config=cfg)


# -- CLI ---------------------------------------------------------------------

def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MLflow tools for Peakflo")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_server = sub.add_parser("server", help="Start a local MLflow tracking server")
    p_server.add_argument("--host", default="127.0.0.1")
    p_server.add_argument("--port", type=int, default=5000)
    p_server.add_argument("--backend-store-uri", default="sqlite:///mlflow.db")
    p_server.add_argument("--artifacts-destination", default="./mlartifacts")

    p_retrain = sub.add_parser("retrain", help="One-click retrain under MLflow")
    p_retrain.add_argument("--data", default=None)
    p_retrain.add_argument("--run-name", default=None)
    p_retrain.add_argument("--no-register", action="store_true")

    p_load = sub.add_parser("load-champion", help="Load the champion model (smoke test)")
    p_load.add_argument("--alias", default=CHAMPION_ALIAS)

    args = parser.parse_args(argv)
    if args.cmd == "server":
        start_tracking_server(args.host, args.port, args.backend_store_uri, args.artifacts_destination)
    elif args.cmd == "retrain":
        retrain(args.data, args.run_name, register=not args.no_register)
    elif args.cmd == "load-champion":
        MLflowTracker().load_model(args.alias)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
