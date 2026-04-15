from peakflo.data import build_text, load_records, normalize_text


def test_normalize_text_collapses_whitespace_and_lowercases() -> None:
    assert normalize_text("  A  B\tC  ") == "a b c"


def test_build_text_handles_missing_description() -> None:
    assert build_text("Zoom Subscription", None) == "zoom subscription"


def test_load_records_parses_expected_fields() -> None:
    records = load_records("accounts-bills.json")
    assert len(records) == 4894
    first = records[0]
    assert first.vendor_id
    assert first.account_name
    assert first.text
