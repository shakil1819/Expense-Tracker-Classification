from src.feature_engineering_pipeline import DictAmountBinner, load_records, resample_minority_classes
from src.inference_pipeline import CalibratedPredictor, build_vendor_account_map
from src.training_pipeline import (
    analyze_errors_by_class_frequency,
    baseline_lookup_accuracy,
    build_balanced_unseen_holdout,
    build_classifier,
    compute_overfitting_diagnostics,
    evaluate_balanced_holdout,
    evaluate_calibrated_fallback_balanced,
    evaluate_holdout,
    evaluate_repeated_splits,
    records_to_examples,
    tune_for_balanced_unseen,
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
    assert result["accuracy"] >= 0.82


def test_balanced_holdout_reports_requested_model_params() -> None:
    records = load_records("accounts-bills.json")
    result = evaluate_balanced_holdout(records, C=4.0, class_weight=None, oversample_min_count=0)
    assert result["C"] == 4.0
    assert result["class_weight"] is None
    assert result["oversample_min_count"] == 0


def test_resample_minority_classes_reaches_min_count() -> None:
    records = load_records("accounts-bills.json")
    from collections import Counter
    original_counts = Counter(r.account_name for r in records)
    resampled = resample_minority_classes(records, min_count=10)
    resampled_counts = Counter(r.account_name for r in resampled)
    # Every class must reach at least min_count.
    assert all(resampled_counts[label] >= 10 for label in original_counts)
    # Total must be >= original.
    assert len(resampled) >= len(records)
    # Classes already at or above min_count must not shrink.
    for label, count in original_counts.items():
        if count >= 10:
            assert resampled_counts[label] == count


def test_analyze_errors_by_class_frequency_returns_buckets() -> None:
    records = load_records("accounts-bills.json")
    result = evaluate_holdout(records, strategy="group_item")
    analysis = result["error_analysis_by_frequency"]
    # At least two buckets must appear in the grouped holdout.
    assert len(analysis) >= 2
    for bucket, stats in analysis.items():
        assert stats["total"] > 0
        assert 0.0 <= stats["accuracy"] <= 1.0
        assert stats["correct"] <= stats["total"]
    # Rare classes should have lower accuracy than frequent classes.
    frequent_acc = analysis.get("frequent (100+)", {}).get("accuracy")
    rare_acc = analysis.get("very_rare (2-4)", {}) or analysis.get("singleton (n=1)", {})
    if frequent_acc and rare_acc.get("accuracy") is not None:
        assert frequent_acc > rare_acc["accuracy"]


def test_tune_hyperparameters_grouped_cv_finds_valid_params() -> None:
    records = load_records("accounts-bills.json")
    # Small grid and 3 splits to keep this test fast.
    tuning = tune_hyperparameters_grouped_cv(
        records,
        C_values=[0.5, 1.0, 4.0],
        class_weights=[None, "balanced"],
        oversample_min_count_values=[0, 5],
        n_splits=3,
    )
    assert tuning["best_C"] in [0.5, 1.0, 4.0]
    assert tuning["best_class_weight"] in [None, "balanced"]
    assert tuning["best_oversample_min_count"] in [0, 5]
    assert tuning["best_val_accuracy"] >= 0.85
    assert tuning["best_val_macro_f1"] >= 0.70
    assert 0.0 < tuning["best_gap"] < 0.20
    assert len(tuning["grid"]) == 12  # 3 C × 2 class_weight × 2 oversample


def test_tuned_group_holdout_meets_target() -> None:
    records = load_records("accounts-bills.json")
    tuning = tune_hyperparameters_grouped_cv(
        records,
        C_values=[0.5, 1.0, 4.0],
        class_weights=[None, "balanced"],
        oversample_min_count_values=[0, 5],
        n_splits=3,
    )
    result = evaluate_holdout(
        records,
        strategy="group_item",
        C=tuning["best_C"],
        class_weight=tuning["best_class_weight"],
        oversample_min_count=tuning["best_oversample_min_count"],
    )
    assert result["accuracy"] >= 0.85


def test_diagnostics_use_requested_model_params() -> None:
    records = load_records("accounts-bills.json")
    diagnostics = compute_overfitting_diagnostics(
        records,
        C=4.0,
        class_weight=None,
        oversample_min_count=0,
    )
    fit = diagnostics["fit_assessment"]
    assert fit["C"] == 4.0
    assert fit["class_weight"] is None
    assert fit["oversample_min_count"] == 0


def test_amount_binner_produces_correct_shape() -> None:
    records = load_records("accounts-bills.json")
    examples, _, _ = records_to_examples(records)
    binner = DictAmountBinner(n_bins=10)
    binner.fit(examples)
    result = binner.transform(examples)
    assert result.shape[0] == len(records)
    assert result.shape[1] <= 10


def test_vendor_account_map_covers_training_vendors() -> None:
    records = load_records("accounts-bills.json")
    vendor_map = build_vendor_account_map(records)
    vendor_ids = {r.vendor_id for r in records}
    assert len(vendor_map) == len(vendor_ids)
    assert all(isinstance(v, str) for v in vendor_map.values())


def test_calibrated_fallback_balanced_is_reasonable() -> None:
    records = load_records("accounts-bills.json")
    result = evaluate_calibrated_fallback_balanced(
        records, C=4.0, confidence_threshold=0.5
    )
    assert result["accuracy"] >= 0.80
    assert result["macro_f1"] >= 0.75
    assert result["strategy"] == "calibrated_fallback_balanced"


def test_tune_for_balanced_unseen_finds_valid_params() -> None:
    records = load_records("accounts-bills.json")
    result = tune_for_balanced_unseen(
        records,
        min_grouped_accuracy=0.85,
    )
    assert "grouped_cv" in result
    assert "balanced_refinement" in result
    ref = result["balanced_refinement"]
    assert ref["C"] > 0
    assert ref["class_weight"] in [None, "balanced"]
    assert ref["oversample_min_count"] >= 0
    assert ref["balanced_accuracy"] >= 0.80
    assert ref["balanced_macro_f1"] >= 0.75
