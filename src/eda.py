"""Exploratory analysis tables: counts, accident rates with confidence intervals,
temporal patterns, correlations and filtering.

A note on denominators: the *accident rate* of a group is accidents / records
in that group. It is only comparable across groups because every record is an
observation that could have had an accident. Raw accident *counts* mostly
reflect how many records a group has (exposure), so the dashboard labels counts
and rates separately and never presents counts as risk.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import config
from src.feature_engineering import FeatureConfig, add_engineered_features, speed_category_labels
from src.utils import wilson_interval

MISSING_LABEL = "(missing)"
AGE_BINS = (18, 25, 35, 45, 55, 65, 75)


def has_target(df: pd.DataFrame) -> bool:
    """True when the dataset has a parsed target with at least one labelled row."""
    return config.TARGET_COLUMN in df.columns and df[config.TARGET_COLUMN].notna().any()


def overview(df: pd.DataFrame) -> dict[str, Any]:
    """Headline counts for the dashboard KPI row."""
    n = len(df)
    cells = max(n * max(df.shape[1], 1), 1)
    summary: dict[str, Any] = {
        "records": n,
        "missing_share": float(df.isna().sum().sum() / cells) if n else 0.0,
        "has_target": has_target(df),
        "accidents": None,
        "non_accidents": None,
        "accident_share": None,
        "unlabelled": None,
    }
    if summary["has_target"]:
        target = df[config.TARGET_COLUMN]
        labelled = target.dropna().astype(int)
        summary.update(
            accidents=int((labelled == 1).sum()),
            non_accidents=int((labelled == 0).sum()),
            accident_share=float(labelled.mean()) if len(labelled) else None,
            unlabelled=int(target.isna().sum()),
        )
    return summary


def with_derived_columns(df: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """Add hour, weekday, weekend, month, speed band and age band columns for analysis."""
    out = add_engineered_features(df, cfg or FeatureConfig())
    if "date" in out.columns:
        dates = pd.to_datetime(out["date"], errors="coerce")
        out["year_month"] = dates.dt.to_period("M").astype(str).where(dates.notna(), np.nan)
    if "driver_age" in out.columns:
        out["age_band"] = age_bands(out["driver_age"])
    if "is_weekend" in out.columns:
        out["weekday_or_weekend"] = out["is_weekend"].map({1.0: "Weekend", 0.0: "Weekday"})
    return out


def age_bands(age: pd.Series) -> pd.Series:
    """Bucket driver ages into labelled bands (<25, 25-34, ...)."""
    edges = [-np.inf, *AGE_BINS[1:], np.inf]
    labels = ["<25"] + [f"{AGE_BINS[i]}–{AGE_BINS[i + 1] - 1}" for i in range(1, len(AGE_BINS) - 1)] + [f"{AGE_BINS[-1]}+"]
    cats = pd.cut(pd.to_numeric(age, errors="coerce"), bins=edges, labels=labels, right=False)
    return cats.astype("object").where(cats.notna(), np.nan)


def age_band_order() -> list[str]:
    return ["<25"] + [f"{AGE_BINS[i]}–{AGE_BINS[i + 1] - 1}" for i in range(1, len(AGE_BINS) - 1)] + [f"{AGE_BINS[-1]}+"]


def rate_by_group(df: pd.DataFrame, column: str, order: list[str] | None = None,
                  include_missing: bool = True) -> pd.DataFrame:
    """Records, accidents and accident rate (with 95% Wilson CI) per group.

    Columns: group, records, share_of_records, and when a target exists:
    accidents, accident_rate, ci_low, ci_high, share_of_accidents.
    """
    if column not in df.columns:
        raise KeyError(column)
    keys = df[column].astype("object")
    if include_missing:
        keys = keys.where(keys.notna(), MISSING_LABEL)
    frame = pd.DataFrame({"group": keys})
    target_present = has_target(df)
    if target_present:
        frame["y"] = df[config.TARGET_COLUMN].astype("Float64")
    frame = frame.dropna(subset=["group"])
    grouped = frame.groupby("group", sort=False)
    table = grouped.size().rename("records").to_frame()
    if target_present:
        labelled = frame.dropna(subset=["y"])
        stats = labelled.groupby("group", sort=False)["y"].agg(["sum", "count"])
        table["labelled"] = stats["count"].reindex(table.index).fillna(0).astype(int)
        table["accidents"] = stats["sum"].reindex(table.index).fillna(0).astype(int)
        with np.errstate(divide="ignore", invalid="ignore"):
            table["accident_rate"] = np.where(table["labelled"] > 0, table["accidents"] / table["labelled"], np.nan)
        low, high = wilson_interval(table["accidents"].to_numpy(), table["labelled"].to_numpy())
        table["ci_low"], table["ci_high"] = low, high
        total_acc = max(int(table["accidents"].sum()), 1)
        table["share_of_accidents"] = table["accidents"] / total_acc
    table["share_of_records"] = table["records"] / max(int(table["records"].sum()), 1)
    table = table.reset_index()
    table["group"] = table["group"].astype(str)
    if order is not None:
        rank = {str(k): i for i, k in enumerate(order)}
        table["_rank"] = table["group"].map(lambda g: rank.get(g, len(rank) + (1 if g == MISSING_LABEL else 0)))
        table = table.sort_values(["_rank", "group"]).drop(columns="_rank")
    elif target_present:
        table = table.sort_values("accident_rate", ascending=False, na_position="last")
    else:
        table = table.sort_values("records", ascending=False)
    return table.reset_index(drop=True)


def natural_order(column: str, values: pd.Series, cfg: FeatureConfig | None = None) -> list[str] | None:
    """Meaningful category order for ordinal-ish columns, else None (sort by rate)."""
    cfg = cfg or FeatureConfig()
    known = {
        "traffic_density": ["Low", "Medium", "High"],
        "light_condition": ["Daylight", "Dusk/Dawn", "Dark - Street Lights", "Dark - No Lights"],
        "day_of_week": list(config.DAY_NAMES),
        "time_of_day": ["Early morning", "Morning", "Afternoon", "Evening", "Night"],
        "speed_category": speed_category_labels(cfg.speed_bins),
        "age_band": age_band_order(),
        "weekday_or_weekend": ["Weekday", "Weekend"],
    }
    if column in known:
        present = set(values.dropna().astype(str))
        ordered = [v for v in known[column] if v in present]
        return ordered + sorted(present - set(ordered))
    if column in ("hour", "month"):
        return [str(v) for v in sorted(pd.to_numeric(values, errors="coerce").dropna().unique())]
    return None


def numeric_groups(df: pd.DataFrame, column: str) -> pd.Series:
    """Integer-valued numeric column (hour, month) as string group keys."""
    values = pd.to_numeric(df[column], errors="coerce")
    return values.map(lambda v: np.nan if pd.isna(v) else str(int(v)))


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame | None:
    """Pearson correlations among numeric inputs, binary indicators and the target.

    Returns None if fewer than two suitable columns exist.
    """
    candidates = ["speed", "driver_age", "hour", "is_peak_hour", "is_weekend", "is_rain", "month",
                  config.TARGET_COLUMN]
    cols = [c for c in candidates if c in df.columns]
    numeric = df[cols].apply(pd.to_numeric, errors="coerce").astype(float)
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    if numeric.shape[1] < 2:
        return None
    return numeric.corr()


def filter_options(df: pd.DataFrame) -> dict[str, Any]:
    """Values available for each filter control."""
    options: dict[str, Any] = {}
    for col in ("weather", "road_condition", "vehicle_type", "traffic_density", "light_condition"):
        if col in df.columns:
            options[col] = sorted(df[col].dropna().astype(str).unique().tolist())
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        if not dates.empty:
            options["date"] = (dates.min().date(), dates.max().date())
    for col in ("speed", "driver_age"):
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            if not values.empty:
                options[col] = (float(values.min()), float(values.max()))
    return options


def apply_filters(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    """Filter rows by category lists, a date range and numeric ranges.

    Category filters keep missing values only if ``include_missing`` is true.
    Range filters keep rows whose value is missing only in that case too.
    """
    mask = pd.Series(True, index=df.index)
    keep_missing = bool(filters.get("include_missing", True))
    for col, selected in filters.get("categories", {}).items():
        if col in df.columns and selected is not None:
            values = df[col].astype("object")
            mask &= values.astype(str).isin([str(s) for s in selected]) | (values.isna() & keep_missing)
    if "date" in filters and "date" in df.columns and filters["date"]:
        start, end = (pd.Timestamp(d) for d in filters["date"])
        dates = pd.to_datetime(df["date"], errors="coerce")
        mask &= dates.between(start, end) | (dates.isna() & keep_missing)
    for col, bounds in filters.get("ranges", {}).items():
        if col in df.columns and bounds:
            low, high = bounds
            values = pd.to_numeric(df[col], errors="coerce")
            mask &= values.between(low, high) | (values.isna() & keep_missing)
    return df[mask]
