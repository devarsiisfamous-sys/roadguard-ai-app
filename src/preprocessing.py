"""Data-quality analysis, cleaning and the model preprocessing pipeline.

Two stages, kept deliberately apart:

1. ``clean_dataset`` makes deterministic, row-wise fixes to the whole dataset
   (type coercion, spelling normalisation, out-of-range values set to missing,
   duplicate removal). It never imputes: missing values stay missing in the
   cleaned data, so nothing is invented.
2. ``build_preprocessor`` returns an unfitted ``ColumnTransformer`` (imputation,
   encoding, optional scaling). It sits inside each model pipeline, so its
   statistics are learned from training folds only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from src.data_loader import TargetAssessment, assess_target, parse_target
from src.feature_engineering import format_minutes, parse_dates, parse_times
from src.utils import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Type coercion
# --------------------------------------------------------------------------- #
def _normalize_categories(series: pd.Series) -> pd.Series:
    """Trim whitespace and unify case variants to their most frequent spelling.

    ``" rain "``, ``"RAIN"`` and ``"Rain"`` become whichever spelling is most
    common, so genuine categories are kept and only formatting noise goes.
    """
    text = series.astype("object").map(
        lambda v: np.nan if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v))
        else " ".join(str(v).split())
    )
    text = text.where(text != "", np.nan)
    present = text.dropna()
    if present.empty:
        return text
    def preferred(spellings: pd.Series) -> str:
        counts = spellings.value_counts()
        tied = counts[counts == counts.max()].index
        # Ties: prefer "Capitalised" over "ALL CAPS" or "lower case", then alphabetical.
        return sorted(tied, key=lambda v: (not v[:1].isupper(), v.isupper(), v))[0]

    canonical = present.groupby(present.str.lower()).agg(preferred).to_dict()
    return text.map(lambda v: canonical.get(v.lower(), v) if isinstance(v, str) else v)


@dataclass
class CoercionResult:
    df: pd.DataFrame
    notes: list[str]
    unparseable: dict[str, int]
    invalid_target: int


def coerce_types(df: pd.DataFrame) -> CoercionResult:
    """Convert canonical columns to their expected types without dropping rows."""
    out = df.copy()
    notes: list[str] = []
    unparseable: dict[str, int] = {}
    invalid_target = 0
    for col in out.columns:
        kind = config.SCHEMA.get(col, {}).get("kind")
        original = out[col]
        non_null_before = int(original.notna().sum())
        if kind == "numeric":
            out[col] = pd.to_numeric(original, errors="coerce").astype(float)
        elif kind == "date":
            parsed = parse_dates(original)
            out[col] = parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), np.nan).astype("object")
        elif kind == "time":
            out[col] = format_minutes(parse_times(original)).astype("object")
        elif kind in ("categorical", "identifier", "metadata"):
            out[col] = _normalize_categories(original)
        elif kind == "target":
            out[col], invalid_target = parse_target(original)
            if invalid_target:
                notes.append(f"{invalid_target} unrecognised accident values set to missing.")
            continue
        else:
            continue
        lost = non_null_before - int(out[col].notna().sum())
        if lost > 0:
            unparseable[col] = lost
            notes.append(f"{lost} unparseable value(s) in '{col}' set to missing.")
    return CoercionResult(out, notes, unparseable, invalid_target)


def find_out_of_range(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks of numeric values outside the plausible ranges in ``config.SCHEMA``."""
    masks: dict[str, pd.Series] = {}
    for col in df.columns:
        spec = config.SCHEMA.get(col, {})
        if spec.get("kind") != "numeric" or "valid_range" not in spec:
            continue
        low, high = spec["valid_range"]
        values = pd.to_numeric(df[col], errors="coerce")
        mask = values.notna() & ((values < low) | (values > high))
        if mask.any():
            masks[col] = mask
    return masks


# --------------------------------------------------------------------------- #
# Quality report
# --------------------------------------------------------------------------- #
@dataclass
class DataQualityReport:
    """Summary of a mapped dataset's shape and problems (computed before cleaning)."""

    n_rows: int
    n_columns: int
    missing: pd.DataFrame
    n_duplicates: int
    dtypes: pd.DataFrame
    issues: list[dict[str, object]]
    target: TargetAssessment
    total_missing_share: float

    @property
    def has_issues(self) -> bool:
        return bool(self.issues) or self.n_duplicates > 0 or self.total_missing_share > 0


def build_quality_report(df: pd.DataFrame) -> DataQualityReport:
    """Profile missing values, duplicates, types, invalid values and the target."""
    coerced = coerce_types(df)
    clean = coerced.df

    missing = pd.DataFrame(
        {
            "column": df.columns,
            "missing": [int(clean[c].isna().sum()) for c in df.columns],
        }
    )
    missing["missing_pct"] = (missing["missing"] / max(len(df), 1) * 100).round(2)

    dtypes = pd.DataFrame(
        {
            "column": df.columns,
            "schema_kind": [config.SCHEMA.get(c, {}).get("kind", "unmapped") for c in df.columns],
            "source_dtype": [str(df[c].dtype) for c in df.columns],
            "unique_values": [int(clean[c].nunique(dropna=True)) for c in df.columns],
            "example": [
                str(clean[c].dropna().iloc[0]) if clean[c].notna().any() else "" for c in df.columns
            ],
        }
    )

    issues: list[dict[str, object]] = []
    for col, count in coerced.unparseable.items():
        issues.append({"column": col, "issue": "Unparseable values (treated as missing)", "count": count})
    for col, mask in find_out_of_range(clean).items():
        low, high = config.SCHEMA[col]["valid_range"]
        issues.append(
            {"column": col, "issue": f"Outside plausible range [{low:g}, {high:g}]", "count": int(mask.sum())}
        )
    if coerced.invalid_target:
        issues.append(
            {"column": config.TARGET_COLUMN, "issue": "Unrecognised accident value", "count": coerced.invalid_target}
        )
    for col in df.columns:
        if config.SCHEMA.get(col, {}).get("kind") == "categorical":
            raw = df[col].dropna().astype(str)
            variants = int(raw.nunique() - raw.map(lambda v: " ".join(v.split()).lower()).nunique())
            if variants > 0:
                issues.append(
                    {"column": col, "issue": "Inconsistent spelling/case/whitespace variants", "count": variants}
                )

    total_cells = max(df.shape[0] * df.shape[1], 1)
    return DataQualityReport(
        n_rows=len(df),
        n_columns=df.shape[1],
        missing=missing,
        n_duplicates=int(df.duplicated().sum()),
        dtypes=dtypes,
        issues=issues,
        target=assess_target(clean, coerced.invalid_target),
        total_missing_share=float(clean.isna().sum().sum() / total_cells),
    )


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
@dataclass
class CleaningOptions:
    drop_duplicates: bool = True
    out_of_range_to_missing: bool = True
    drop_rows_missing_target: bool = True


@dataclass
class CleaningLog:
    rows_before: int
    rows_after: int
    steps: list[str] = field(default_factory=list)


def clean_dataset(df: pd.DataFrame, options: CleaningOptions | None = None) -> tuple[pd.DataFrame, CleaningLog]:
    """Apply deterministic cleaning to a schema-mapped DataFrame.

    Missing values are never filled in here. Imputation happens inside the
    model pipeline, fitted on training data only.
    """
    options = options or CleaningOptions()
    log = CleaningLog(rows_before=len(df), rows_after=len(df))
    coerced = coerce_types(df)
    out = coerced.df
    log.steps.append("Converted columns to schema types and normalised category spelling.")
    log.steps.extend(coerced.notes)

    if options.out_of_range_to_missing:
        for col, mask in find_out_of_range(out).items():
            out.loc[mask, col] = np.nan
            log.steps.append(f"Set {int(mask.sum())} out-of-range '{col}' value(s) to missing.")

    if options.drop_duplicates:
        n_dup = int(out.duplicated().sum())
        if n_dup:
            out = out.drop_duplicates().copy()
            log.steps.append(f"Removed {n_dup} exact duplicate row(s).")
        else:
            log.steps.append("No exact duplicate rows found.")

    if options.drop_rows_missing_target and config.TARGET_COLUMN in out.columns:
        n_missing = int(out[config.TARGET_COLUMN].isna().sum())
        if n_missing:
            out = out[out[config.TARGET_COLUMN].notna()].copy()
            log.steps.append(f"Dropped {n_missing} row(s) with a missing or unrecognised accident label.")

    out = out.reset_index(drop=True)
    log.rows_after = len(out)
    logger.info("Cleaned dataset: %d -> %d rows", log.rows_before, log.rows_after)
    return out, log


# --------------------------------------------------------------------------- #
# Model preprocessing
# --------------------------------------------------------------------------- #
def build_preprocessor(numeric: list[str], categorical: list[str], scale_numeric: bool) -> ColumnTransformer:
    """Unfitted ColumnTransformer for the given model-ready columns.

    Numeric: median imputation, then standard scaling only for models that
    need it (logistic regression, SVM). Categorical: a constant "Missing"
    category, then one-hot encoding that ignores unseen categories at
    prediction time.
    """
    numeric_steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline(numeric_steps), numeric))
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="Missing", keep_empty_features=True)),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise ValueError("No usable features were selected.")
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, bool]:
    """Reproducible train/test split, stratified on ``y`` when every class can be.

    Returns:
        X_train, X_test, y_train, y_test, and whether stratification was used.
    """
    if not 0.05 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.05 and 0.5.")
    counts = y.value_counts()
    n_test = int(np.ceil(len(y) * test_size))
    stratify = (
        y if len(counts) > 1 and counts.min() >= 2 and n_test >= len(counts) else None
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )
    return X_train, X_test, y_train, y_test, stratify is not None


def imbalance_warning(y: pd.Series) -> str | None:
    """Warning text when the rarer class is below ``config.SEVERE_IMBALANCE_SHARE``."""
    if y.empty:
        return None
    share = float(y.value_counts(normalize=True).min()) if y.nunique() > 1 else 0.0
    if share < config.SEVERE_IMBALANCE_SHARE:
        return (
            f"Severe class imbalance: the rarer class is only {share:.1%} of labelled rows. "
            "Accuracy will look high even for a useless model. Focus on recall, precision, F1 "
            "and the precision-recall curve, and consider class weighting."
        )
    return None
