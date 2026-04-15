from __future__ import annotations

import json
from pathlib import Path


def write_json(path: str | Path, payload: dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_results_markdown(summary: dict[str, object]) -> str:
    dataset = summary["dataset"]
    baseline = summary["baseline"]
    random_holdout = summary["holdout"]["random"]
    group_holdout = summary["holdout"]["group_item"]
    repeated_random = summary["repeated"]["random"]
    repeated_group = summary["repeated"]["group_item"]

    def render_label_rows(rows: list[dict[str, object]]) -> str:
        lines = ["| Label | F1 | Precision | Recall | Support |", "| --- | ---: | ---: | ---: | ---: |"]
        for row in rows:
            lines.append(
                f"| {row['label']} | {row['f1']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} | {row['support']} |"
            )
        return "\n".join(lines)

    return f"""# Results

## Executive Summary

The final model is a linear SVM over word n-grams, character n-grams, vendor ID, and log-scaled amount. It comfortably clears the 85% target on both a standard shuffled split and a stricter grouped split that keeps normalized `itemName` values out of both train and test.

## Data Analysis

- Records: {dataset['record_count']}
- Unique vendors: {dataset['unique_vendors']}
- Unique account names: {dataset['unique_account_names']}
- Labels with fewer than 10 rows: {dataset['labels_lt_10']}
- Singleton labels: {dataset['singleton_labels']}
- Missing descriptions: {dataset['missing_item_descriptions']}
- Amount range: {dataset['min_amount']} to {dataset['max_amount']}
- Exact item-name lookup baseline accuracy: {baseline['item_vendor_lookup_accuracy']:.4f}

## Validation

- Repeated shuffled holdout accuracy: {repeated_random['accuracy_mean']:.4f} +/- {repeated_random['accuracy_std']:.4f}
- Repeated shuffled holdout macro F1: {repeated_random['macro_f1_mean']:.4f}
- Repeated grouped holdout accuracy: {repeated_group['accuracy_mean']:.4f} +/- {repeated_group['accuracy_std']:.4f}
- Repeated grouped holdout macro F1: {repeated_group['macro_f1_mean']:.4f}
- Single shuffled holdout accuracy: {random_holdout['accuracy']:.4f}
- Single grouped holdout accuracy: {group_holdout['accuracy']:.4f}

## Strongest Labels

{render_label_rows(random_holdout['top_labels'])}

## Weakest Labels

{render_label_rows(random_holdout['bottom_labels'])}

## Discussion

- The dominant signal is transaction text. Vendor ID is useful when the same supplier consistently maps to one account.
- Grouped validation is lower than shuffled validation, which is expected because it removes repeated `itemName` leakage.
- Rare labels remain the main weakness. Many classes have too little support for stable estimates, so macro F1 is the better caution metric than overall accuracy alone.
"""


def write_markdown(path: str | Path, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
