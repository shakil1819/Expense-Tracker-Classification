from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from joblib import dump, load
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline

from src.feature_engineering_pipeline import ExpenseRecord, build_text, normalize_text


def save_trained_model(model: Pipeline, path: str | Path) -> None:
    logger.info("Saving trained model to {}", path)
    dump(model, path)


def load_trained_model(path: str | Path) -> Pipeline:
    logger.info("Loading trained model from {}", path)
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
    logger.info("Predicting {} records", len(rows))
    examples = prepare_inference_examples(rows)
    predictions = [str(label) for label in model.predict(examples)]
    logger.debug("Prediction complete: {} results", len(predictions))
    return predictions


def build_vendor_account_map(records: list[ExpenseRecord]) -> dict[str, str]:
    """Build a vendor -> majority accountName lookup from training records."""
    logger.debug("Building vendor-account map from {} records", len(records))
    vendor_votes: dict[str, Counter[str]] = {}
    for record in records:
        vendor_votes.setdefault(record.vendor_id, Counter())[record.account_name] += 1
    vendor_map = {
        vendor: max(counts, key=counts.get)
        for vendor, counts in vendor_votes.items()
    }
    logger.debug("Vendor-account map: {} vendors mapped", len(vendor_map))
    return vendor_map


class CalibratedPredictor:
    """Wraps a trained Pipeline with sigmoid calibration for probability estimates.

    Falls back to vendor-account majority lookup when model confidence is below threshold.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        vendor_map: dict[str, str],
        confidence_threshold: float = 0.5,
    ) -> None:
        self.vendor_map = vendor_map
        self.confidence_threshold = confidence_threshold
        self.calibrated_ = CalibratedClassifierCV(
            estimator=FrozenEstimator(pipeline), method="sigmoid"
        )

    def fit_calibration(self, X: list[dict[str, object]], y: list[str]) -> None:
        logger.info("Fitting sigmoid calibration on {} samples", len(y))
        self.calibrated_.fit(X, y)
        logger.debug("Calibration fit complete")
        return self

    def predict(self, examples: list[dict[str, object]]) -> list[str]:
        logger.debug("Predicting {} examples with confidence threshold={}", len(examples), self.confidence_threshold)
        proba = self.calibrated_.predict_proba(examples)
        classes = self.calibrated_.classes_
        predictions: list[str] = []
        fallback_count = 0
        for i, row in enumerate(examples):
            max_prob = float(proba[i].max())
            if max_prob >= self.confidence_threshold:
                predictions.append(str(classes[proba[i].argmax()]))
            else:
                vendor = str(row.get("vendor_id", ""))
                fallback = self.vendor_map.get(vendor)
                if fallback:
                    predictions.append(fallback)
                    fallback_count += 1
                else:
                    predictions.append(str(classes[proba[i].argmax()]))
        logger.info("Predictions: {} total, {} vendor fallbacks used", len(predictions), fallback_count)
        return predictions
