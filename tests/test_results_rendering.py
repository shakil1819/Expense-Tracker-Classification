"""Regression tests for the markdown rendered to .docs/03_results.md.

Catches the exact class of bug where a template table header has N columns
but the separator row has N-1 cells, causing GitHub/VS Code to not render
the table.
"""

from __future__ import annotations

import re

import pytest

from src.training_pipeline import render_results_markdown


def _minimal_summary() -> dict:
    """Smallest summary dict that satisfies render_results_markdown's f-string."""
    tuning_row = {
        "C": 1.0,
        "class_weight": "None",
        "oversample_min_count": 0,
        "val_accuracy_mean": 0.88,
        "val_macro_f1_mean": 0.73,
        "val_accuracy_std": 0.02,
        "val_macro_f1_std": 0.02,
        "train_accuracy_mean": 0.99,
        "gap_mean": 0.11,
    }
    freq_bucket = {"total": 10, "correct": 8, "accuracy": 0.8}
    holdout = {
        "accuracy": 0.89,
        "macro_f1": 0.77,
        "accuracy_gap": 0.1,
        "error_analysis_by_frequency": {"<5": freq_bucket, "5-10": freq_bucket},
    }
    repeated = {"accuracy_mean": 0.87, "accuracy_std": 0.01, "macro_f1_mean": 0.75}
    balanced_model = {"accuracy_mean": 0.85, "accuracy_std": 0.01, "macro_f1_mean": 0.8}
    return {
        "dataset": {
            "record_count": 5000,
            "unique_account_names": 103,
            "labels_lt_5": 34,
            "labels_lt_3": 20,
            "singleton_labels": 10,
            "labels_with_one_unique_item": 25,
            "missing_item_descriptions": 0,
        },
        "baseline": {"item_vendor_lookup_accuracy": 0.75},
        "holdout": {
            "random": {"accuracy": 0.9, "macro_f1": 0.78, "accuracy_gap": 0.09},
            "group_item": holdout,
            "group_item_tuned": holdout,
        },
        "repeated": {
            "random": repeated,
            "group_item": repeated,
            "group_item_tuned": repeated,
        },
        "balanced_holdout": {
            "eligible_labels": 64,
            "samples_per_class": 3,
            "accuracy": 0.85,
            "macro_f1": 0.82,
        },
        "balanced_repeated": {
            "tuned_model": balanced_model,
            "class_weight_balanced_model": balanced_model,
        },
        "diagnostics": {
            "fit_assessment": {
                "underfitting": False,
                "overfitting": True,
                "C": 1.0,
                "class_weight": "None",
                "oversample_min_count": 0,
                "train_accuracy": 0.99,
                "grouped_val_accuracy": 0.88,
                "gap": 0.11,
                "summary": "Test summary.",
            },
            "learning_curve": [
                {"train_size": 500, "train_accuracy": 0.99, "validation_accuracy": 0.6},
                {"train_size": 2000, "train_accuracy": 0.99, "validation_accuracy": 0.85},
            ],
            "validation_curve": [
                {"C": 0.5, "train_accuracy": 0.98, "validation_accuracy": 0.87},
                {"C": 2.0, "train_accuracy": 0.99, "validation_accuracy": 0.88},
            ],
        },
        "hyperparameter_tuning": {
            "best_C": 2.0,
            "best_class_weight": None,
            "best_oversample_min_count": 0,
            "best_val_accuracy": 0.88,
            "best_val_macro_f1": 0.74,
            "best_gap": 0.11,
            "optimize_for": "macro_f1",
            "grid": [tuning_row, {**tuning_row, "C": 2.0}],
        },
    }


def _tables(markdown: str) -> list[list[str]]:
    """Return each table as a list of its lines (rows starting with '|')."""
    tables: list[list[str]] = []
    current: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("|"):
            current.append(line)
        else:
            if current:
                tables.append(current)
                current = []
    if current:
        tables.append(current)
    return tables


def _cell_count(row: str) -> int:
    # Strip leading/trailing pipes, then count cells.
    stripped = row.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return len(stripped.split("|"))


def test_every_rendered_table_has_matching_header_and_separator_widths():
    md = render_results_markdown(_minimal_summary())
    tables = _tables(md)
    assert tables, "render_results_markdown produced no tables"

    for i, table in enumerate(tables):
        assert len(table) >= 2, f"table {i} has no separator row"
        header_cells = _cell_count(table[0])
        separator_cells = _cell_count(table[1])
        assert re.fullmatch(r"\|[\s\-:|]+\|?", table[1].strip()), (
            f"table {i} row 1 is not a separator: {table[1]!r}"
        )
        assert header_cells == separator_cells, (
            f"table {i}: header has {header_cells} cells but separator has "
            f"{separator_cells}\nheader:    {table[0]}\nseparator: {table[1]}"
        )
        for row in table[2:]:
            assert _cell_count(row) == header_cells, (
                f"table {i}: data row has {_cell_count(row)} cells, expected {header_cells}\n{row}"
            )
