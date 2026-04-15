from __future__ import annotations

import argparse

from peakflo.data import load_records, summarize_records
from peakflo.model import (
    baseline_lookup_accuracy,
    evaluate_holdout,
    evaluate_repeated_splits,
    fit_full_model,
    save_model,
)
from peakflo.reporting import render_results_markdown, write_json, write_markdown


DATA_PATH = "accounts-bills.json"


def run_pipeline() -> dict[str, object]:
    records = load_records(DATA_PATH)
    summary = {
        "dataset": summarize_records(records),
        "baseline": {
            "item_vendor_lookup_accuracy": baseline_lookup_accuracy(records),
        },
        "repeated": {
            "random": evaluate_repeated_splits(records, strategy="random"),
            "group_item": evaluate_repeated_splits(records, strategy="group_item"),
        },
        "holdout": {
            "random": evaluate_holdout(records, strategy="random"),
            "group_item": evaluate_holdout(records, strategy="group_item"),
        },
    }

    model = fit_full_model(records)
    write_json("artifacts/evaluation_summary.json", summary)
    write_markdown(".docs/03_results.md", render_results_markdown(summary))
    save_model(model, "artifacts/account_classifier.joblib")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Peakflo expense classifier")
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=["run"],
        help="Run evaluation and train the final model.",
    )
    args = parser.parse_args(argv)
    if args.command == "run":
        summary = run_pipeline()
        print(
            "random_accuracy="
            f"{summary['holdout']['random']['accuracy']:.4f} "
            "group_accuracy="
            f"{summary['holdout']['group_item']['accuracy']:.4f}"
        )
    return 0
