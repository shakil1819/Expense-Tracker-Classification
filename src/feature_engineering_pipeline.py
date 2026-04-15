from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder


SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ExpenseRecord:
    vendor_id: str
    item_name: str
    item_description: str
    account_name: str
    item_total_amount: float
    normalized_item_name: str
    text: str
    amount_log: float


class DictTextVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, key: str = "text", **kwargs: object) -> None:
        self.key = key
        self.kwargs = kwargs
        self.vectorizer = TfidfVectorizer(**kwargs)

    def fit(self, X: list[dict[str, object]], y: np.ndarray | None = None) -> "DictTextVectorizer":
        self.vectorizer.fit([str(row.get(self.key, "")) for row in X])
        return self

    def transform(self, X: list[dict[str, object]]) -> csr_matrix:
        return self.vectorizer.transform([str(row.get(self.key, "")) for row in X])


class DictOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, key: str = "vendor_id") -> None:
        self.key = key
        self.encoder = OneHotEncoder(handle_unknown="ignore")

    def fit(self, X: list[dict[str, object]], y: np.ndarray | None = None) -> "DictOneHotEncoder":
        values = np.array([[str(row.get(self.key, ""))] for row in X], dtype=object)
        self.encoder.fit(values)
        return self

    def transform(self, X: list[dict[str, object]]) -> csr_matrix:
        values = np.array([[str(row.get(self.key, ""))] for row in X], dtype=object)
        return self.encoder.transform(values)


class DictAmountScaler(BaseEstimator, TransformerMixin):
    def __init__(self, key: str = "amount_log") -> None:
        self.key = key
        self.scaler = MaxAbsScaler()

    def fit(self, X: list[dict[str, object]], y: np.ndarray | None = None) -> "DictAmountScaler":
        values = np.array([[float(row.get(self.key, 0.0))] for row in X], dtype=float)
        self.scaler.fit(values)
        return self

    def transform(self, X: list[dict[str, object]]) -> csr_matrix:
        values = np.array([[float(row.get(self.key, 0.0))] for row in X], dtype=float)
        return csr_matrix(self.scaler.transform(values))


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.lower().strip()
    return SPACE_RE.sub(" ", text)


def build_text(item_name: object, item_description: object) -> str:
    item = normalize_text(item_name)
    description = normalize_text(item_description)
    return f"{item} {description}".strip()


def load_records(path: str | Path) -> list[ExpenseRecord]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    records: list[ExpenseRecord] = []
    for row in rows:
        amount = float(row.get("itemTotalAmount") or 0.0)
        item_name = row.get("itemName") or ""
        item_description = row.get("itemDescription") or ""
        normalized_item_name = normalize_text(item_name)
        records.append(
            ExpenseRecord(
                vendor_id=str(row.get("vendorId") or ""),
                item_name=str(item_name),
                item_description=str(item_description),
                account_name=str(row["accountName"]),
                item_total_amount=amount,
                normalized_item_name=normalized_item_name,
                text=build_text(item_name, item_description),
                amount_log=math.log1p(abs(amount)),
            )
        )
    return records


def records_to_examples(
    records: list[ExpenseRecord],
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    examples = [
        {
            "text": record.text,
            "vendor_id": record.vendor_id,
            "amount_log": record.amount_log,
            "normalized_item_name": record.normalized_item_name,
        }
        for record in records
    ]
    labels = np.array([record.account_name for record in records], dtype=object)
    groups = np.array([record.normalized_item_name for record in records], dtype=object)
    return examples, labels, groups


def summarize_records(records: list[ExpenseRecord]) -> dict[str, object]:
    label_counts = Counter(record.account_name for record in records)
    vendor_counts = Counter(record.vendor_id for record in records)
    amounts = [record.item_total_amount for record in records]
    unique_items_by_label: dict[str, set[str]] = {}
    for record in records:
        unique_items_by_label.setdefault(record.account_name, set()).add(record.normalized_item_name)
    return {
        "record_count": len(records),
        "unique_vendors": len(vendor_counts),
        "unique_account_names": len(label_counts),
        "labels_lt_10": sum(count < 10 for count in label_counts.values()),
        "labels_lt_5": sum(count < 5 for count in label_counts.values()),
        "labels_lt_3": sum(count < 3 for count in label_counts.values()),
        "singleton_labels": sum(count == 1 for count in label_counts.values()),
        "labels_with_one_unique_item": sum(
            len(item_names) == 1 for item_names in unique_items_by_label.values()
        ),
        "labels_with_lt_3_unique_items": sum(
            len(item_names) < 3 for item_names in unique_items_by_label.values()
        ),
        "missing_item_descriptions": sum(not record.item_description.strip() for record in records),
        "min_amount": min(amounts),
        "max_amount": max(amounts),
        "top_accounts": label_counts.most_common(10),
    }


def build_feature_union() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word_tfidf",
                DictTextVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char_tfidf",
                DictTextVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            ("vendor", DictOneHotEncoder()),
            ("amount", DictAmountScaler()),
        ]
    )
