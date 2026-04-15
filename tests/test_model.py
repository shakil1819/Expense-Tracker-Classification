from src.feature_engineering_pipeline import load_records
from src.training_pipeline import (
    baseline_lookup_accuracy,
    build_balanced_unseen_holdout,
    evaluate_balanced_holdout,
    evaluate_holdout,
    evaluate_repeated_splits,
    tune_hyperparameters_grouped_cv,
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


def test_tune_hyperparameters_grouped_cv_finds_valid_params() -> None:
    records = load_records("accounts-bills.json")
    # Use a small grid and 3 splits to keep this test fast.
    tuning = tune_hyperparameters_grouped_cv(
        records,
        C_values=[0.5, 1.0, 4.0],
        class_weights=[None, "balanced"],
        n_splits=3,
    )
    assert tuning["best_C"] in [0.5, 1.0, 4.0]
    assert tuning["best_class_weight"] in [None, "balanced"]
    assert tuning["best_val_accuracy"] >= 0.85
    # Gap should be positive (train > test) and bounded.
    assert 0.0 < tuning["best_gap"] < 0.20
    assert len(tuning["grid"]) == 6  # 3 C values × 2 class weights


def test_tuned_group_holdout_meets_target() -> None:
    records = load_records("accounts-bills.json")
    tuning = tune_hyperparameters_grouped_cv(
        records,
        C_values=[0.5, 1.0, 4.0],
        class_weights=[None, "balanced"],
        n_splits=3,
    )
    result = evaluate_holdout(
        records,
        strategy="group_item",
        C=tuning["best_C"],
        class_weight=tuning["best_class_weight"],
    )
    assert result["accuracy"] >= 0.85
