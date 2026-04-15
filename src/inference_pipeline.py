from __future__ import annotations

import math
from pathlib import Path

from joblib import dump, load
from sklearn.pipeline import Pipeline

from src.feature_engineering_pipeline import build_text, normalize_text


def save_trained_model(model: Pipeline, path: str | Path) -> None:
    dump(model, path)


def load_trained_model(path: str | Path) -> Pipeline:
    return load(path)


def prepare_inference_examples(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for row in rows:
        amount = float(row.get("itemTotalAmount") or 0.0)
        examples.append(
            {
                "text": build_text(row.get("itemName"), row.get("itemDescription")),
                "vendor_id": str(row.get("vendorId") or ""),
                "amount_log": math.log1p(abs(amount)),
                "normalized_item_name": normalize_text(row.get("itemName")),
            }
        )
    return examples


def predict_records(model: Pipeline, rows: list[dict[str, object]]) -> list[str]:
    examples = prepare_inference_examples(rows)
    return [str(label) for label in model.predict(examples)]
