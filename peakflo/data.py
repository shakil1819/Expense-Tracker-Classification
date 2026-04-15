from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
    missing_descriptions = sum(not record.item_description.strip() for record in records)
    return {
        "record_count": len(records),
        "unique_vendors": len(vendor_counts),
        "unique_account_names": len(label_counts),
        "labels_lt_10": sum(count < 10 for count in label_counts.values()),
        "labels_lt_5": sum(count < 5 for count in label_counts.values()),
        "singleton_labels": sum(count == 1 for count in label_counts.values()),
        "missing_item_descriptions": missing_descriptions,
        "min_amount": min(amounts),
        "max_amount": max(amounts),
        "top_accounts": label_counts.most_common(10),
    }
