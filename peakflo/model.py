from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from joblib import dump
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
from sklearn.svm import LinearSVC

from peakflo.data import ExpenseRecord, records_to_examples


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


@dataclass(frozen=True)
class SplitResult:
    accuracy: float
    weighted_f1: float
    macro_f1: float


def build_classifier() -> Pipeline:
    features = FeatureUnion(
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
    return Pipeline(
        [
            ("features", features),
            ("classifier", LinearSVC(dual="auto", C=1.0)),
        ]
    )


def _to_split_result(y_true: np.ndarray, y_pred: np.ndarray) -> SplitResult:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return SplitResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        weighted_f1=float(report["weighted avg"]["f1-score"]),
        macro_f1=float(report["macro avg"]["f1-score"]),
    )


def baseline_lookup_accuracy(records: list[ExpenseRecord], random_state: int = 42) -> float:
    examples, labels, _ = records_to_examples(records)
    splitter = ShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, test_idx = next(splitter.split(np.arange(len(examples))))
    majority_by_item: dict[str, str] = {}
    majority_by_vendor: dict[str, str] = {}
    train_labels, train_counts = np.unique(labels[train_idx], return_counts=True)
    majority_label = str(train_labels[np.argmax(train_counts)])

    item_votes: dict[str, list[str]] = {}
    vendor_votes: dict[str, list[str]] = {}
    for index in train_idx:
        item_votes.setdefault(str(examples[index]["normalized_item_name"]), []).append(str(labels[index]))
        vendor_votes.setdefault(str(examples[index]["vendor_id"]), []).append(str(labels[index]))
    for key, values in item_votes.items():
        majority_by_item[key] = max(set(values), key=values.count)
    for key, values in vendor_votes.items():
        majority_by_vendor[key] = max(set(values), key=values.count)

    predictions: list[str] = []
    for index in test_idx:
        item = str(examples[index]["normalized_item_name"])
        vendor = str(examples[index]["vendor_id"])
        predictions.append(
            majority_by_item.get(item)
            or majority_by_vendor.get(vendor)
            or majority_label
        )
    return float(accuracy_score(labels[test_idx], np.array(predictions, dtype=object)))


def evaluate_repeated_splits(
    records: list[ExpenseRecord],
    strategy: str,
    n_splits: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, object]:
    examples, labels, groups = records_to_examples(records)
    indices = np.arange(len(examples))
    if strategy == "random":
        splitter = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
        split_iter = splitter.split(indices)
    elif strategy == "group_item":
        splitter = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
        split_iter = splitter.split(indices, labels, groups=groups)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    results: list[SplitResult] = []
    for train_idx, test_idx in split_iter:
        model = build_classifier()
        train_x = [examples[index] for index in train_idx]
        test_x = [examples[index] for index in test_idx]
        train_y = labels[train_idx]
        test_y = labels[test_idx]
        model.fit(train_x, train_y)
        predictions = model.predict(test_x)
        results.append(_to_split_result(test_y, predictions))

    return {
        "strategy": strategy,
        "n_splits": n_splits,
        "test_size": test_size,
        "accuracy_mean": float(np.mean([result.accuracy for result in results])),
        "accuracy_std": float(np.std([result.accuracy for result in results])),
        "weighted_f1_mean": float(np.mean([result.weighted_f1 for result in results])),
        "macro_f1_mean": float(np.mean([result.macro_f1 for result in results])),
    }


def evaluate_holdout(
    records: list[ExpenseRecord],
    strategy: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, object]:
    examples, labels, groups = records_to_examples(records)
    indices = np.arange(len(examples))
    if strategy == "random":
        splitter = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(indices))
    elif strategy == "group_item":
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(indices, labels, groups=groups))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    train_x = [examples[index] for index in train_idx]
    test_x = [examples[index] for index in test_idx]
    train_y = labels[train_idx]
    test_y = labels[test_idx]

    model = build_classifier()
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    report = classification_report(test_y, predictions, output_dict=True, zero_division=0)

    label_rows = []
    for label, metrics in report.items():
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        label_rows.append(
            {
                "label": label,
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1-score"]),
                "support": int(metrics["support"]),
            }
        )
    label_rows.sort(key=lambda row: (-row["support"], -row["f1"], row["label"]))
    return {
        "strategy": strategy,
        "accuracy": float(accuracy_score(test_y, predictions)),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "labels_in_test": int(len(set(test_y))),
        "top_labels": sorted(
            [row for row in label_rows if row["support"] >= 5],
            key=lambda row: (-row["f1"], -row["support"], row["label"]),
        )[:10],
        "bottom_labels": sorted(
            [row for row in label_rows if row["support"] >= 5],
            key=lambda row: (row["f1"], -row["support"], row["label"]),
        )[:10],
    }


def fit_full_model(records: list[ExpenseRecord]) -> Pipeline:
    examples, labels, _ = records_to_examples(records)
    model = build_classifier()
    model.fit(examples, labels)
    return model


def save_model(model: Pipeline, path: str) -> None:
    dump(model, path)
