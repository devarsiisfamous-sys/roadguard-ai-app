"""Derived features from the presentation: peak hours, weekend, holiday, rain,
speed category and time of day.

All derivations are row-wise and use only information available at prediction
time (the observation's own date, time, weather and speed), so applying them
before a train/test split cannot leak future information. They are also
wrapped in ``FeatureEngineer`` so the saved model pipeline applies exactly the
same logic to new scenarios.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import config

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureConfig:
    """User-tunable assumptions behind the engineered features.

    Attributes:
        peak_hour_ranges: Half-open ``[start, end)`` hour ranges on a 24h clock.
        speed_bins: Increasing km/h edges separating the speed categories.
        holidays: ISO dates (YYYY-MM-DD) supplied by the user. The holiday
            indicator is only created when this is non-empty.
    """

    peak_hour_ranges: tuple[tuple[int, int], ...] = config.PEAK_HOUR_RANGES
    speed_bins: tuple[float, ...] = config.SPEED_BINS
    holidays: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for start, end in self.peak_hour_ranges:
            if not (0 <= start < end <= 24):
                raise ValueError(f"Invalid peak-hour range ({start}, {end}); need 0 <= start < end <= 24.")
        bins = list(self.speed_bins)
        if not bins or any(b <= 0 for b in bins) or bins != sorted(set(bins)):
            raise ValueError("Speed bins must be positive and strictly increasing.")
        for day in self.holidays:
            if pd.isna(pd.to_datetime(day, errors="coerce", format="%Y-%m-%d")):
                raise ValueError(f"Holiday '{day}' is not a valid YYYY-MM-DD date.")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["peak_hour_ranges"] = [list(r) for r in self.peak_hour_ranges]
        data["speed_bins"] = list(self.speed_bins)
        data["holidays"] = list(self.holidays)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "FeatureConfig":
        if not data:
            return cls()
        return cls(
            peak_hour_ranges=tuple(tuple(int(v) for v in r) for r in data.get("peak_hour_ranges", config.PEAK_HOUR_RANGES)),
            speed_bins=tuple(float(v) for v in data.get("speed_bins", config.SPEED_BINS)),
            holidays=tuple(str(v) for v in data.get("holidays", ())),
        )


def parse_holiday_text(text: str) -> tuple[str, ...]:
    """Parse comma/newline separated YYYY-MM-DD dates; raise ValueError on bad entries."""
    items = [t.strip() for t in re.split(r"[,\n;]+", text or "") if t.strip()]
    parsed: list[str] = []
    for item in items:
        stamp = pd.to_datetime(item, errors="coerce", format="%Y-%m-%d")
        if pd.isna(stamp):
            raise ValueError(f"'{item}' is not a valid YYYY-MM-DD date.")
        parsed.append(stamp.strftime("%Y-%m-%d"))
    return tuple(sorted(set(parsed)))


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def parse_dates(series: pd.Series) -> pd.Series:
    """Parse mixed-format dates; unparseable values become NaT (never guessed)."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.normalize()
    text = series.astype("object").where(series.notna(), None)
    parsed = pd.to_datetime(text, errors="coerce", format="mixed", dayfirst=False)
    return parsed.dt.normalize()


_TIME_PATTERNS = (
    re.compile(r"^(?P<h>\d{1,2}):(?P<m>\d{2})(?::(?P<s>\d{2}(?:\.\d+)?))?\s*(?P<ampm>[ap]\.?m\.?)?$", re.I),
    re.compile(r"^(?P<h>\d{1,2})\s*(?P<ampm>[ap]\.?m\.?)$", re.I),
    re.compile(r"^(?P<hm>\d{3,4})$"),
)


def _parse_time_value(value: object) -> float:
    if value is None or value is pd.NA:
        return np.nan
    if hasattr(value, "hour") and hasattr(value, "minute"):
        return float(value.hour * 60 + value.minute)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(value):
            return np.nan
        if float(value).is_integer() and 0 <= value <= 23:
            return float(value) * 60
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    if " " in text and ":" in text and re.match(r"^\d{4}-\d{2}-\d{2}", text):
        text = text.split(" ", 1)[1]
    for pattern in _TIME_PATTERNS:
        match = pattern.match(text)
        if not match:
            continue
        groups = match.groupdict()
        if groups.get("hm"):
            hour, minute = divmod(int(groups["hm"]), 100)
        else:
            hour = int(groups["h"])
            minute = int(groups.get("m") or 0)
        ampm = (groups.get("ampm") or "").lower().replace(".", "")
        if ampm:
            if not 1 <= hour <= 12:
                return np.nan
            hour = hour % 12 + (12 if ampm == "pm" else 0)
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return float(hour * 60 + minute)
        return np.nan
    if text.isdigit() and 0 <= int(text) <= 23:
        return float(int(text) * 60)
    return np.nan


def parse_times(series: pd.Series) -> pd.Series:
    """Parse times of day to minutes after midnight (NaN when unparseable)."""
    return series.astype("object").map(_parse_time_value).astype(float)


def format_minutes(minutes: pd.Series) -> pd.Series:
    """Format minutes after midnight as zero-padded HH:MM strings."""
    def fmt(v: float) -> object:
        if pd.isna(v):
            return np.nan
        return f"{int(v) // 60:02d}:{int(v) % 60:02d}"
    return minutes.map(fmt)


# --------------------------------------------------------------------------- #
# Individual features
# --------------------------------------------------------------------------- #
def speed_category_labels(bins: tuple[float, ...]) -> list[str]:
    """Readable labels for the speed categories implied by ``bins``."""
    names = ["Low", "Moderate", "High", "Very high", "Extreme", "Extreme+"]
    edges = [int(b) if float(b).is_integer() else b for b in bins]
    labels = [f"{names[0]} (<{edges[0]})"]
    for i in range(1, len(edges)):
        upper = edges[i]
        labels.append(f"{names[min(i, len(names) - 1)]} ({edges[i - 1]}–<{upper})")
    labels.append(f"{names[min(len(edges), len(names) - 1)]} (≥{edges[-1]})")
    return labels


def categorize_speed(speed: pd.Series, bins: tuple[float, ...]) -> pd.Series:
    """Bucket numeric speeds (km/h) into labelled categories; NaN stays NaN."""
    values = pd.to_numeric(speed, errors="coerce")
    labels = speed_category_labels(bins)
    edges = [-np.inf, *bins, np.inf]
    cats = pd.cut(values, bins=edges, labels=labels, right=False)
    return cats.astype("object").where(cats.notna(), np.nan)


def time_of_day(hour: pd.Series) -> pd.Series:
    """Map hours (0-23) to buckets defined in ``config.TIME_OF_DAY_BUCKETS``."""
    starts = [s for s, _ in config.TIME_OF_DAY_BUCKETS]
    names = [n for _, n in config.TIME_OF_DAY_BUCKETS]

    def bucket(h: float) -> object:
        if pd.isna(h):
            return np.nan
        idx = int(np.searchsorted(starts, h, side="right")) - 1
        return names[idx]
    return hour.map(bucket)


def is_peak_hour(hour: pd.Series, ranges: tuple[tuple[int, int], ...]) -> pd.Series:
    """1.0 if the hour falls inside any ``[start, end)`` range, NaN when unknown."""
    flag = pd.Series(0.0, index=hour.index)
    for start, end in ranges:
        flag[(hour >= start) & (hour < end)] = 1.0
    return flag.where(hour.notna(), np.nan)


def rain_indicator(weather: pd.Series) -> pd.Series:
    """1.0 when the weather text mentions rain-like conditions, NaN when missing."""
    text = weather.astype("object").where(weather.notna(), None)
    pattern = "|".join(config.RAIN_KEYWORDS)
    flag = text.map(lambda v: np.nan if v is None else float(bool(re.search(pattern, str(v), re.I))))
    return flag.astype(float)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
# name -> (kind, source column, description)
ENGINEERED_FEATURES: dict[str, tuple[str, str, str]] = {
    "hour": ("numeric", "time", "Hour of day (0-23)."),
    "is_peak_hour": ("numeric", "time", "1 if within configured peak-hour ranges."),
    "time_of_day": ("categorical", "time", "Night / Early morning / Morning / Afternoon / Evening."),
    "day_of_week": ("categorical", "date", "Mon ... Sun."),
    "is_weekend": ("numeric", "date", "1 for Saturday or Sunday."),
    "month": ("numeric", "date", "Calendar month (1-12)."),
    "is_holiday": ("numeric", "date", "1 if the date is in the user-supplied holiday list."),
    "is_rain": ("numeric", "weather", "1 if weather mentions rain, drizzle, shower, storm or thunder."),
    "speed_category": ("categorical", "speed", "Speed band from the configured thresholds."),
}


def engineered_feature_names(input_columns: list[str], cfg: FeatureConfig) -> list[str]:
    """Names of the engineered features that can be derived from ``input_columns``."""
    names = []
    for name, (_, source, _) in ENGINEERED_FEATURES.items():
        if source not in input_columns:
            continue
        if name == "is_holiday" and not cfg.holidays:
            continue
        names.append(name)
    return names


def add_engineered_features(df: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """Return a copy of ``df`` with every engineered feature its columns support.

    Derivations are skipped (not filled with defaults) when the source column
    is absent. Malformed dates/times yield NaN features that the model's
    imputer handles later.
    """
    cfg = cfg or FeatureConfig()
    out = df.copy()
    if "time" in out.columns:
        minutes = parse_times(out["time"])
        hour = np.floor(minutes / 60)
        out["hour"] = hour
        out["is_peak_hour"] = is_peak_hour(hour, cfg.peak_hour_ranges)
        out["time_of_day"] = time_of_day(hour)
    if "date" in out.columns:
        dates = parse_dates(out["date"])
        dow = dates.dt.day_of_week
        out["day_of_week"] = dow.map(lambda d: np.nan if pd.isna(d) else config.DAY_NAMES[int(d)])
        out["is_weekend"] = (dow >= 5).astype(float).where(dow.notna(), np.nan)
        out["month"] = dates.dt.month.astype(float)
        if cfg.holidays:
            holiday_set = set(pd.to_datetime(list(cfg.holidays)))
            out["is_holiday"] = dates.isin(holiday_set).astype(float).where(dates.notna(), np.nan)
    if "weather" in out.columns:
        out["is_rain"] = rain_indicator(out["weather"])
    if "speed" in out.columns:
        out["speed_category"] = categorize_speed(out["speed"], cfg.speed_bins)
    return out


def model_feature_types(input_columns: list[str], cfg: FeatureConfig) -> tuple[list[str], list[str]]:
    """Split the model-ready feature set into (numeric, categorical) column lists.

    Raw ``date`` and ``time`` are replaced by their derived features. Identifier
    columns (e.g. road segment) are treated as categorical when opted in.
    """
    numeric: list[str] = []
    categorical: list[str] = []
    for col in input_columns:
        kind = config.SCHEMA.get(col, {}).get("kind", "categorical")
        if kind in ("date", "time"):
            continue
        (numeric if kind == "numeric" else categorical).append(col)
    for name in engineered_feature_names(input_columns, cfg):
        (numeric if ENGINEERED_FEATURES[name][0] == "numeric" else categorical).append(name)
    return numeric, categorical


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer applying ``add_engineered_features``.

    It is the first step of every saved pipeline, so raw scenario inputs
    (date, time, weather, speed, ...) go through the same derivations at
    prediction time as at training time. It has no learned state beyond
    remembering the expected input columns.
    """

    def __init__(self, feature_config: dict[str, Any] | None = None, input_columns: list[str] | None = None):
        self.feature_config = feature_config
        self.input_columns = input_columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FeatureEngineer":
        cols = list(self.input_columns) if self.input_columns is not None else list(X.columns)
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ValueError(f"Missing input feature(s): {', '.join(missing)}")
        self.input_columns_ = cols
        self.config_ = FeatureConfig.from_dict(self.feature_config)
        numeric, categorical = model_feature_types(cols, self.config_)
        self.numeric_features_ = numeric
        self.categorical_features_ = categorical
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.input_columns_ if c not in X.columns]
        if missing:
            raise ValueError(f"Missing input feature(s): {', '.join(missing)}")
        frame = X[self.input_columns_].copy()
        for col in self.input_columns_:
            kind = config.SCHEMA.get(col, {}).get("kind")
            if kind == "numeric":
                frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(float)
        out = add_engineered_features(frame, self.config_)
        for col in self.categorical_features_:
            values = out[col].astype("object")
            out[col] = values.where(values.notna(), np.nan).map(
                lambda v: v if (isinstance(v, float) and np.isnan(v)) else str(v)
            ).astype(object)
        for col in self.numeric_features_:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
        return out[self.numeric_features_ + self.categorical_features_]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.numeric_features_ + self.categorical_features_, dtype=object)
