from src.feature_engineering_pipeline import load_records
from src.training_pipeline import (
    baseline_lookup_accuracy,
    build_balanced_unseen_holdout,
    evaluate_balanced_holdout,
    evaluate_holdout,
    evaluate_repeated_splits,
)


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


def test_balanced_holdout_is_truly_balanced_and_unseen() -> None:
    records = load_records("accounts-bills.json")
    split = build_balanced_unseen_holdout(records)
    test_supports = {}
    for record in split.test_records:
        test_supports[record.account_name] = test_supports.get(record.account_name, 0) + 1
    assert min(test_supports.values()) == 3
    assert max(test_supports.values()) == 3
    train_pairs = {(record.account_name, record.normalized_item_name) for record in split.train_records}
    test_pairs = {(record.account_name, record.normalized_item_name) for record in split.test_records}
    assert train_pairs.isdisjoint(test_pairs)


def test_balanced_holdout_is_reasonable() -> None:
    records = load_records("accounts-bills.json")
    result = evaluate_balanced_holdout(records)
    assert result["eligible_labels"] >= 50
    assert result["accuracy"] >= 0.80
