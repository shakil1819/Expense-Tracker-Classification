from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
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
    frame = pd.read_json(Path(path))
    frame = frame.assign(
        vendorId=frame["vendorId"].fillna("").astype(str),
        itemName=frame["itemName"].fillna("").astype(str),
        itemDescription=frame["itemDescription"].fillna("").astype(str),
        accountName=frame["accountName"].astype(str),
        itemTotalAmount=pd.to_numeric(frame["itemTotalAmount"], errors="coerce").fillna(0.0),
    )
    records: list[ExpenseRecord] = []
    for row in frame.itertuples(index=False):
        amount = float(row.itemTotalAmount)
        item_name = row.itemName
        item_description = row.itemDescription
        normalized_item_name = normalize_text(item_name)
        records.append(
            ExpenseRecord(
                vendor_id=str(row.vendorId),
                item_name=str(item_name),
                item_description=str(item_description),
                account_name=str(row.accountName),
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
    frame = pd.DataFrame.from_records(record.__dict__ for record in records)
    label_counts = frame["account_name"].value_counts()
    vendor_counts = frame["vendor_id"].value_counts()
    unique_items_by_label = frame.groupby("account_name")["normalized_item_name"].nunique()
    return {
        "record_count": int(len(frame)),
        "unique_vendors": int(vendor_counts.shape[0]),
        "unique_account_names": int(label_counts.shape[0]),
        "labels_lt_10": int((label_counts < 10).sum()),
        "labels_lt_5": int((label_counts < 5).sum()),
        "labels_lt_3": int((label_counts < 3).sum()),
        "singleton_labels": int((label_counts == 1).sum()),
        "labels_with_one_unique_item": int((unique_items_by_label == 1).sum()),
        "labels_with_lt_3_unique_items": int((unique_items_by_label < 3).sum()),
        "missing_item_descriptions": int(frame["item_description"].str.strip().eq("").sum()),
        "min_amount": float(frame["item_total_amount"].min()),
        "max_amount": float(frame["item_total_amount"].max()),
        "top_accounts": [
            [str(label), int(count)] for label, count in label_counts.head(10).items()
        ],
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
