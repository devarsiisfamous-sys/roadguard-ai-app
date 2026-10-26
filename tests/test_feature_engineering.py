"""Derived features: parsing, indicators, categories, configuration and the sklearn transformer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    FeatureConfig,
    FeatureEngineer,
    add_engineered_features,
    categorize_speed,
    engineered_feature_names,
    model_feature_types,
    parse_dates,
    parse_holiday_text,
    parse_times,
    rain_indicator,
    speed_category_labels,
)


def test_parse_times_accepts_common_formats() -> None:
    values = pd.Series(["08:30", "2:30 PM", "12 am", "1430", "14:30:59", 7, "25:00", "abc", None, ""])
    minutes = parse_times(values).tolist()
    assert minutes[:6] == [510.0, 870.0, 0.0, 870.0, 870.0, 420.0]
    assert all(np.isnan(m) for m in minutes[6:])


def test_parse_dates_marks_invalid_as_missing() -> None:
    parsed = parse_dates(pd.Series(["2024-02-29", "2023-02-30", "garbage", None]))
    assert parsed.iloc[0] == pd.Timestamp("2024-02-29")
    assert parsed.iloc[1:].isna().all()


def test_engineered_features_values() -> None:
    df = pd.DataFrame({
        "date": ["2024-06-08", "2024-06-10", None],     # Saturday, Monday, missing
        "time": ["08:15", "13:00", "bad"],
        "weather": ["Light Rain", "Clear", None],
        "speed": [39.9, 70, np.nan],
    })
    out = add_engineered_features(df, FeatureConfig())
    assert out["is_weekend"].tolist()[:2] == [1.0, 0.0] and np.isnan(out["is_weekend"].iloc[2])
    assert out["day_of_week"].tolist()[:2] == ["Sat", "Mon"]
    assert out["is_peak_hour"].tolist()[:2] == [1.0, 0.0] and np.isnan(out["is_peak_hour"].iloc[2])
    assert out["time_of_day"].tolist()[:2] == ["Morning", "Afternoon"]
    assert out["is_rain"].tolist()[:2] == [1.0, 0.0] and np.isnan(out["is_rain"].iloc[2])
    labels = speed_category_labels(FeatureConfig().speed_bins)
    assert out["speed_category"].tolist()[:2] == [labels[0], labels[2]]
    assert pd.isna(out["speed_category"].iloc[2])
    assert "is_holiday" not in out.columns  # no holiday calendar supplied


def test_speed_category_boundaries_are_left_closed() -> None:
    labels = speed_category_labels((40.0, 70.0, 100.0))
    cats = categorize_speed(pd.Series([0, 40, 69.9, 100, 180]), (40.0, 70.0, 100.0)).tolist()
    assert cats == [labels[0], labels[1], labels[1], labels[3], labels[3]]


def test_rain_indicator_keywords() -> None:
    flags = rain_indicator(pd.Series(["Thunderstorm", "drizzle", "Fog", "Snow"])).tolist()
    assert flags == [1.0, 1.0, 0.0, 0.0]


def test_holiday_feature_only_with_calendar() -> None:
    df = pd.DataFrame({"date": ["2024-12-25", "2024-12-26"]})
    cfg = FeatureConfig(holidays=("2024-12-25",))
    out = add_engineered_features(df, cfg)
    assert out["is_holiday"].tolist() == [1.0, 0.0]
    assert "is_holiday" in engineered_feature_names(["date"], cfg)
    assert "is_holiday" not in engineered_feature_names(["date"], FeatureConfig())


def test_custom_peak_hours() -> None:
    df = pd.DataFrame({"time": ["06:00", "22:30"]})
    out = add_engineered_features(df, FeatureConfig(peak_hour_ranges=((22, 24),)))
    assert out["is_peak_hour"].tolist() == [0.0, 1.0]


def test_features_skipped_when_source_absent() -> None:
    out = add_engineered_features(pd.DataFrame({"driver_age": [30]}))
    assert list(out.columns) == ["driver_age"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"peak_hour_ranges": ((10, 7),)},
        {"peak_hour_ranges": ((0, 25),)},
        {"speed_bins": (70.0, 40.0)},
        {"speed_bins": ()},
        {"holidays": ("2024-13-01",)},
    ],
)
def test_feature_config_validation(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        FeatureConfig(**kwargs)


def test_parse_holiday_text() -> None:
    assert parse_holiday_text("2024-01-26, 2024-08-15\n2024-01-26") == ("2024-01-26", "2024-08-15")
    assert parse_holiday_text("") == ()
    with pytest.raises(ValueError):
        parse_holiday_text("26/01/2024")


def test_feature_config_roundtrip() -> None:
    cfg = FeatureConfig(peak_hour_ranges=((6, 9),), speed_bins=(30.0, 60.0), holidays=("2024-01-01",))
    assert FeatureConfig.from_dict(cfg.to_dict()) == cfg


def test_model_feature_types_replace_date_time() -> None:
    numeric, categorical = model_feature_types(["date", "time", "speed", "weather"], FeatureConfig())
    assert "date" not in numeric + categorical and "time" not in numeric + categorical
    assert {"speed", "hour", "is_peak_hour", "is_weekend", "month", "is_rain"} <= set(numeric)
    assert {"weather", "time_of_day", "day_of_week", "speed_category"} <= set(categorical)


def test_feature_engineer_transformer() -> None:
    X = pd.DataFrame({"date": ["2024-06-08"], "time": ["08:15"], "speed": ["55"], "weather": ["Rain"]})
    fe = FeatureEngineer(FeatureConfig().to_dict()).fit(X)
    out = fe.transform(X)
    assert list(out.columns) == list(fe.get_feature_names_out())
    assert out["speed"].dtype == float and out["speed"].iloc[0] == 55.0
    assert out["weather"].dtype == object
    with pytest.raises(ValueError, match="Missing input feature"):
        fe.transform(X.drop(columns=["weather"]))


def test_features_are_row_wise(clean_synthetic: pd.DataFrame) -> None:
    """Deriving features on a subset gives identical values: no cross-row (future) information."""
    full = add_engineered_features(clean_synthetic)
    subset = add_engineered_features(clean_synthetic.iloc[100:150])
    pd.testing.assert_frame_equal(full.iloc[100:150], subset)
