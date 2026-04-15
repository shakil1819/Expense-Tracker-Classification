from peakflo.data import load_records
from peakflo.model import baseline_lookup_accuracy, evaluate_holdout, evaluate_repeated_splits


def test_baseline_lookup_is_reasonable() -> None:
    records = load_records("accounts-bills.json")
    accuracy = baseline_lookup_accuracy(records)
    assert accuracy >= 0.70


def test_random_holdout_meets_target_accuracy() -> None:
    records = load_records("accounts-bills.json")
    result = evaluate_holdout(records, strategy="random")
    assert result["accuracy"] >= 0.85


def test_group_split_is_still_above_threshold() -> None:
    records = load_records("accounts-bills.json")
    result = evaluate_repeated_splits(records, strategy="group_item", n_splits=3)
    assert result["accuracy_mean"] >= 0.85
