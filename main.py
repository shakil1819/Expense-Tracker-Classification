"""Peakflo entrypoint.

By default, ``uv run main.py``:

1. Ensures a local MLflow tracking server is running at http://127.0.0.1:5000
   (reused if already up, otherwise spawned as a detached background process).
2. Points all logging at that server via ``MLFLOW_TRACKING_URI``.
3. Runs the full training pipeline under a tracked MLflow run that logs params,
   metrics, metric card, dataset digest, artifacts, and registers the model.
4. Leaves the dashboard running so results are browsable immediately.

Use ``--no-track`` for a raw pipeline run without MLflow.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Peakflo expense classifier entrypoint")
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Skip MLflow; just run the training pipeline and write artifacts/ directly.",
    )
    parser.add_argument("--data", default=None, help="Override dataset path")
    parser.add_argument("--run-name", default=None, help="MLflow run name (tracked mode only)")
    args = parser.parse_args(argv)

    if args.no_track:
        if args.data is not None:
            print("warning: --data is ignored with --no-track; edit DATA_PATH in src/training_pipeline.py", file=sys.stderr)
        from src.training_pipeline import main as pipeline_main
        return pipeline_main([])

    from src.tracker import run_tracked_pipeline
    run_tracked_pipeline(data_path=args.data, run_name=args.run_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
