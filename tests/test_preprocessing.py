"""Loading, schema mapping, target parsing, data quality, cleaning and splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src.data_loader import (
    DataLoadError,
    apply_column_mapping,
    assess_target,
    load_sample_dataset,
    parse_target,
    read_csv_bytes,
    suggest_column_mapping,
)
from src.preprocessing import (
    CleaningOptions,
    build_preprocessor,
    build_quality_report,
    clean_dataset,
    coerce_types,
    imbalance_warning,
    split_data,
)


# --------------------------------------------------------------------------- #
# CSV loading
# --------------------------------------------------------------------------- #
def test_read_valid_csv() -> None:
    df = read_csv_bytes(b"weather,speed,accident\nRain,50,Yes\nClear,30,No\n", "x.csv")
    assert list(df.columns) == ["weather", "speed", "accident"]
    assert len(df) == 2


def test_read_semicolon_csv() -> None:
    df = read_csv_bytes(b"weather;speed;accident\nRain;50;Yes\nClear;30;No\n", "x.csv")
    assert df.shape == (2, 3)


@pytest.mark.parametrize(
    "content, filename, message",
    [
        (b"", "x.csv", "empty"),
        (b"   \n  ", "x.csv", "empty"),
        (b"a,b\n", "x.csv", "no data rows"),
        (b"only\n1\n2\n", "x.csv", "one column"),
        (b"a,b\n1,2\n", "data.xlsx", "Unsupported"),
        (b"a,b\n1,2\n3,4,5,6\n", "x.csv", "malformed"),
    ],
)
def test_invalid_csvs_raise_helpful_errors(content: bytes, filename: str, message: str) -> None:
    with pytest.raises(DataLoadError, match=message):
        read_csv_bytes(content, filename)


def test_sample_dataset_loads_and_is_labelled_synthetic() -> None:
    df = load_sample_dataset()
    assert len(df) >= 1000
    assert set(df["record_source"].dropna().unique()) == {"synthetic_demo"}


def test_missing_sample_dataset_raises(tmp_path) -> None:
    with pytest.raises(DataLoadError, match="missing"):
        load_sample_dataset(tmp_path / "nope.csv")


# --------------------------------------------------------------------------- #
# Mapping and target
# --------------------------------------------------------------------------- #
def test_column_mapping_recognises_aliases() -> None:
    mapping = suggest_column_mapping(["Weather_Condition", "Speed (km/h)", "Age", "Accident Occurred", "Notes"])
    assert mapping["weather"] == "Weather_Condition"
    assert mapping["speed"] == "Speed (km/h)"
    assert mapping["driver_age"] == "Age"
    assert mapping[config.TARGET_COLUMN] == "Accident Occurred"
    assert "Notes" not in mapping.values()


def test_apply_mapping_drops_unmapped_columns() -> None:
    df = pd.DataFrame({"w": ["Rain"], "casualties": [3], "acc": ["Yes"]})
    mapped, dropped = apply_column_mapping(df, {"weather": "w", config.TARGET_COLUMN: "acc"})
    assert list(mapped.columns) == ["weather", config.TARGET_COLUMN]
    assert dropped == ["casualties"]


def test_apply_mapping_rejects_reused_and_unknown_columns() -> None:
    df = pd.DataFrame({"w": ["Rain"], "acc": ["Yes"]})
    with pytest.raises(DataLoadError, match="only one field"):
        apply_column_mapping(df, {"weather": "w", "road_condition": "w"})
    with pytest.raises(DataLoadError, match="not found"):
        apply_column_mapping(df, {"weather": "missing_col"})
    with pytest.raises(DataLoadError, match="No columns"):
        apply_column_mapping(df, {"weather": None})


def test_parse_target_variants() -> None:
    parsed, invalid = parse_target(pd.Series(["Yes", "no", "1", 0, True, "maybe", None, " Y "]))
    assert parsed.tolist()[:5] == [1, 0, 1, 0, 1]
    assert parsed.isna().tolist()[5:] == [True, True, False]
    assert parsed.iloc[7] == 1
    assert invalid == 1


def test_assess_target_reports_missing_and_single_class() -> None:
    assert not assess_target(pd.DataFrame({"weather": ["Rain"]})).present
    single = pd.DataFrame({config.TARGET_COLUMN: pd.array([1] * 100, dtype="Int64")})
    result = assess_target(single)
    assert result.present and not result.usable
    assert "both" in result.reason


def test_assess_target_requires_enough_rows() -> None:
    small = pd.DataFrame({config.TARGET_COLUMN: pd.array([0, 1] * 10, dtype="Int64")})
    assert not assess_target(small).usable
    rare = pd.DataFrame({config.TARGET_COLUMN: pd.array([0] * 200 + [1] * 3, dtype="Int64")})
    assert "rarer class" in assess_target(rare).reason


# --------------------------------------------------------------------------- #
# Quality report and cleaning
# --------------------------------------------------------------------------- #
@pytest.fixture
def messy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "weather": [" rain ", "RAIN", "Rain", "Clear", None, "Clear"],
            "speed": [50, "fast", -5, 60, 70, 60],
            "driver_age": [30, 40, 130, 25, None, 25],
            "date": ["2023-01-01", "not a date", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-04"],
            "time": ["08:30", "2:15 PM", "25:00", "1730", None, "1730"],
            config.TARGET_COLUMN: ["Yes", "No", "No", "Yes", "maybe", "Yes"],
        }
    )


def test_category_spelling_is_normalised(messy: pd.DataFrame) -> None:
    coerced = coerce_types(messy).df
    assert set(coerced["weather"].dropna()) == {"Rain", "Clear"}
    times = coerced["time"].tolist()
    assert times[0] == "08:30" and times[1] == "14:15" and times[3] == "17:30"
    assert pd.isna(times[2])  # 25:00 is not a valid time
    assert pd.isna(coerced["date"].iloc[1])  # unparseable date becomes missing, not guessed


def test_quality_report_counts_problems(messy: pd.DataFrame) -> None:
    report = build_quality_report(messy)
    assert report.n_rows == 6
    assert report.n_duplicates == 1  # the last row repeats the fourth
    missing = dict(zip(report.missing["column"], report.missing["missing"]))
    assert missing["weather"] == 1
    assert missing["speed"] == 1  # "fast" is unparseable
    issues = {(i["column"], i["issue"].split(" ")[0]) for i in report.issues}
    assert ("speed", "Outside") in issues
    assert ("driver_age", "Outside") in issues
    assert (config.TARGET_COLUMN, "Unrecognised") in issues
    assert report.target.n_invalid == 1


def test_duplicates_detected_and_removed() -> None:
    df = pd.DataFrame({"weather": ["Rain", "Rain", "Clear"], "speed": [40, 40, 50],
                       config.TARGET_COLUMN: ["Yes", "Yes", "No"]})
    assert build_quality_report(df).n_duplicates == 1
    clean, log = clean_dataset(df)
    assert len(clean) == 2
    assert any("duplicate" in step for step in log.steps)
    kept, _ = clean_dataset(df, CleaningOptions(drop_duplicates=False))
    assert len(kept) == 3


def test_cleaning_never_imputes(messy: pd.DataFrame) -> None:
    clean, log = clean_dataset(messy)
    # One duplicate and the row with an unrecognised target are dropped; missing values stay missing.
    assert len(clean) == 4
    assert clean["speed"].isna().sum() == 2  # "fast" + out-of-range -5
    assert clean["driver_age"].isna().sum() == 1  # 130 set to missing; the None row was dropped with its target
    assert clean[config.TARGET_COLUMN].isna().sum() == 0
    assert log.rows_before == 6 and log.rows_after == 4


def test_out_of_range_can_be_kept(messy: pd.DataFrame) -> None:
    clean, _ = clean_dataset(messy, CleaningOptions(out_of_range_to_missing=False))
    assert (clean["speed"] == -5).any()


# --------------------------------------------------------------------------- #
# Splitting and preprocessing
# --------------------------------------------------------------------------- #
def test_stratified_split_preserves_class_share() -> None:
    y = pd.Series([0] * 80 + [1] * 20)
    X = pd.DataFrame({"x": range(100)})
    _, _, y_train, y_test, stratified = split_data(X, y, 0.25, 42)
    assert stratified
    assert y_test.mean() == pytest.approx(0.2, abs=0.01)
    assert y_train.mean() == pytest.approx(0.2, abs=0.01)


def test_split_is_reproducible_and_falls_back_without_stratification() -> None:
    X = pd.DataFrame({"x": range(50)})
    y = pd.Series([0] * 49 + [1])
    a = split_data(X, y, 0.2, 1)
    b = split_data(X, y, 0.2, 1)
    assert a[0].index.equals(b[0].index)
    assert a[4] is False
    with pytest.raises(ValueError):
        split_data(X, y, 0.9, 1)


def test_preprocessor_learns_only_from_training_data() -> None:
    train = pd.DataFrame({"speed": [10.0, 20.0, np.nan], "weather": ["Rain", None, "Clear"]})
    test = pd.DataFrame({"speed": [np.nan, 1000.0], "weather": ["Snow", "Rain"]})
    pre = build_preprocessor(["speed"], ["weather"], scale_numeric=False).fit(train)
    out = pre.transform(test)
    assert out[0][0] == pytest.approx(15.0)  # median of the training values only
    names = list(pre.get_feature_names_out())
    assert "cat__weather_Snow" not in names  # unseen test category ignored, not learned
    assert out.shape == (2, 1 + 3)


def test_imbalance_warning() -> None:
    assert imbalance_warning(pd.Series([0] * 95 + [1] * 5)) is not None
    assert imbalance_warning(pd.Series([0] * 70 + [1] * 30)) is None
