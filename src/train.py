"""Model registry, leakage-safe training pipelines, tuning, calibration and explainability.

Every model is a single scikit-learn ``Pipeline``::

    FeatureEngineer -> ColumnTransformer (impute / encode / scale) -> classifier

That whole pipeline is what gets cross-validated, tuned, calibrated and
saved. Imputation statistics, category vocabularies and scalers are
therefore only ever fitted on training folds. The held-out test split is used
only for the final evaluation and permutation importance, never for fitting,
tuning, calibration or champion selection.
"""

from __future__ import annotations

import functools
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text

import config
from src import evaluate, model_store
from src.data_loader import assess_target
from src.feature_engineering import FeatureConfig, FeatureEngineer, engineered_feature_names, model_feature_types, parse_dates, parse_times, format_minutes
from src.preprocessing import build_preprocessor, imbalance_warning, split_data
from src.utils import get_logger

logger = get_logger(__name__)


class TrainingError(Exception):
    """Raised when a dataset or configuration cannot support training."""


# --------------------------------------------------------------------------- #
# Optional dependency: XGBoost
# --------------------------------------------------------------------------- #
def _import_xgboost() -> Any:
    import xgboost  # noqa: PLC0415 - optional dependency

    return xgboost


@functools.lru_cache(maxsize=1)
def xgboost_status() -> tuple[bool, str]:
    """(available, message). XGBoost can fail at import for missing native libs too."""
    try:
        xgb = _import_xgboost()
        return True, f"XGBoost {xgb.__version__} is available."
    except Exception as exc:  # ImportError, or XGBoostError when libomp is missing
        first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
        detail = f"{type(exc).__name__}: {first_line}" if first_line else type(exc).__name__
        hint = "Install it with `python -m pip install xgboost`."
        if "libomp" in str(exc) or "OpenMP" in str(exc):
            hint = (
                "The OpenMP runtime is missing. On macOS run `brew install libomp`; on Linux install libgomp."
            )
        return False, f"XGBoost is not available ({detail}). {hint} The other models still work."


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    needs_scaling: bool
    default_params: dict[str, Any]
    param_grid: dict[str, list[Any]]
    description: str
    probability_note: str
    always_calibrate: bool = False
    requires_xgboost: bool = False


MODEL_SPECS: dict[str, ModelSpec] = {
    "logistic_regression": ModelSpec(
        key="logistic_regression",
        name="Logistic Regression",
        needs_scaling=True,
        default_params={"C": 1.0, "max_iter": 2000},
        param_grid={"C": [0.1, 1.0, 10.0]},
        description="Linear model of the log-odds of an accident; transparent coefficients.",
        probability_note="Logistic regression outputs well-behaved probabilities when trained without class weighting.",
    ),
    "decision_tree": ModelSpec(
        key="decision_tree",
        name="Decision Tree",
        needs_scaling=False,
        default_params={"max_depth": 6, "min_samples_leaf": 20},
        param_grid={"max_depth": [4, 6, 10], "min_samples_leaf": [5, 20, 50]},
        description="A single tree of if/then splits; easy to read, prone to overfitting.",
        probability_note="Decision-tree probabilities are leaf frequencies: coarse steps that are often over-confident.",
    ),
    "random_forest": ModelSpec(
        key="random_forest",
        name="Random Forest",
        needs_scaling=False,
        default_params={"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 5},
        param_grid={"n_estimators": [300], "max_depth": [8, 12, None], "min_samples_leaf": [2, 5, 10]},
        description="Ensemble of decorrelated trees; robust to non-linear effects and interactions.",
        probability_note="Random-forest probabilities are vote shares; they are often squeezed toward the middle and benefit from calibration.",
    ),
    "xgboost": ModelSpec(
        key="xgboost",
        name="XGBoost",
        needs_scaling=False,
        default_params={"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.9},
        param_grid={"n_estimators": [200, 400], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
        description="Gradient-boosted trees; strong on tabular data.",
        probability_note="Boosted-tree probabilities can be over-confident; consider calibration.",
        requires_xgboost=True,
    ),
    "svm": ModelSpec(
        key="svm",
        name="Support Vector Machine",
        needs_scaling=True,
        default_params={"C": 1.0, "kernel": "rbf"},
        param_grid={"C": [0.5, 1.0, 4.0]},
        description="Maximum-margin classifier (RBF or linear kernel). Training slows down on large datasets.",
        probability_note="SVMs output margins, not probabilities. RoadGuard converts them with Platt (sigmoid) scaling fitted by cross-validation on the training data only.",
        always_calibrate=True,
    ),
}


def available_models() -> dict[str, tuple[bool, str]]:
    """Map each model key to (available, message)."""
    status = {}
    for key, spec in MODEL_SPECS.items():
        status[key] = xgboost_status() if spec.requires_xgboost else (True, "Available.")
    return status


def build_estimator(key: str, params: dict[str, Any], seed: int, balance: bool, y_train: pd.Series | None = None) -> Any:
    """Instantiate the classifier for ``key`` with user parameters merged over defaults."""
    spec = MODEL_SPECS[key]
    merged = {**spec.default_params, **(params or {})}
    class_weight = "balanced" if balance else None
    if key == "logistic_regression":
        return LogisticRegression(C=float(merged["C"]), max_iter=int(merged["max_iter"]),
                                  class_weight=class_weight, random_state=seed)
    if key == "decision_tree":
        return DecisionTreeClassifier(max_depth=merged["max_depth"], min_samples_leaf=int(merged["min_samples_leaf"]),
                                      class_weight=class_weight, random_state=seed)
    if key == "random_forest":
        return RandomForestClassifier(n_estimators=int(merged["n_estimators"]), max_depth=merged["max_depth"],
                                      min_samples_leaf=int(merged["min_samples_leaf"]), class_weight=class_weight,
                                      n_jobs=-1, random_state=seed)
    if key == "xgboost":
        available, message = xgboost_status()
        if not available:
            raise TrainingError(message)
        xgb = _import_xgboost()
        scale_pos_weight = 1.0
        if balance and y_train is not None and (y_train == 1).sum() > 0:
            scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
        return xgb.XGBClassifier(
            n_estimators=int(merged["n_estimators"]), max_depth=int(merged["max_depth"]),
            learning_rate=float(merged["learning_rate"]), subsample=float(merged["subsample"]),
            colsample_bytree=0.9, scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            tree_method="hist", n_jobs=-1, random_state=seed,
        )
    if key == "svm":
        return SVC(C=float(merged["C"]), kernel=str(merged["kernel"]), gamma="scale",
                   class_weight=class_weight, random_state=seed)
    raise KeyError(f"Unknown model '{key}'.")


def build_pipeline(key: str, input_features: list[str], feature_config: FeatureConfig, params: dict[str, Any],
                   seed: int, balance: bool, y_train: pd.Series | None = None) -> Pipeline:
    """Unfitted end-to-end pipeline for one model."""
    numeric, categorical = model_feature_types(input_features, feature_config)
    spec = MODEL_SPECS[key]
    return Pipeline(
        [
            ("features", FeatureEngineer(feature_config.to_dict(), list(input_features))),
            ("preprocess", build_preprocessor(numeric, categorical, scale_numeric=spec.needs_scaling)),
            ("model", build_estimator(key, params, seed, balance, y_train)),
        ]
    )


# --------------------------------------------------------------------------- #
# Settings and results
# --------------------------------------------------------------------------- #
@dataclass
class TrainingSettings:
    test_size: float = config.DEFAULT_TEST_SIZE
    seed: int = config.RANDOM_SEED
    balance_classes: bool = True
    calibration: str = "none"  # none | sigmoid | isotonic
    calibration_folds: int = 3
    tune: bool = False
    cv_folds: int = config.DEFAULT_CV_FOLDS
    threshold_strategy: str = "optimize_f1"  # optimize_f1 | fixed
    fixed_threshold: float = config.DECISION_THRESHOLD
    selection_metric: str = "f1"
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    params: dict[str, dict[str, Any]] = field(default_factory=dict)

    def validate(self) -> None:
        if self.calibration not in ("none", "sigmoid", "isotonic"):
            raise TrainingError(f"Unknown calibration method '{self.calibration}'.")
        if self.selection_metric not in evaluate.METRICS:
            raise TrainingError(f"Unknown selection metric '{self.selection_metric}'.")
        if not 2 <= self.cv_folds <= 10:
            raise TrainingError("Cross-validation folds must be between 2 and 10.")
        if not 2 <= self.calibration_folds <= 10:
            raise TrainingError("Calibration folds must be between 2 and 10.")
        if self.threshold_strategy not in ("optimize_f1", "fixed"):
            raise TrainingError(f"Unknown threshold strategy '{self.threshold_strategy}'.")
        if not 0.0 < self.fixed_threshold < 1.0:
            raise TrainingError("The fixed decision threshold must be between 0 and 1.")


@dataclass
class TrainingRun:
    run_id: str
    results: list[dict[str, Any]]
    errors: dict[str, str]
    champion_key: str | None
    champion_basis: str
    warnings: list[str]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def describe_feature_schema(X_train: pd.DataFrame, input_features: list[str], cfg: FeatureConfig) -> dict[str, Any]:
    """Describe the training inputs (types, categories, ranges) for prediction forms and validation."""
    features = []
    for col in input_features:
        spec = config.SCHEMA.get(col, {})
        kind = spec.get("kind", "categorical")
        series = X_train[col]
        info: dict[str, Any] = {"name": col, "kind": kind, "label": spec.get("label", col),
                                "missing_share": float(series.isna().mean())}
        if kind == "numeric":
            values = pd.to_numeric(series, errors="coerce").dropna()
            info.update({"unit": spec.get("unit", ""), "min": float(values.min()), "max": float(values.max()),
                         "median": float(values.median()), "p05": float(values.quantile(0.05)),
                         "p95": float(values.quantile(0.95)), "valid_range": list(spec.get("valid_range", (None, None)))})
        elif kind == "date":
            dates = parse_dates(series).dropna()
            info.update({"min": dates.min().strftime("%Y-%m-%d"), "max": dates.max().strftime("%Y-%m-%d"),
                         "median": dates.sort_values().iloc[len(dates) // 2].strftime("%Y-%m-%d")})
        elif kind == "time":
            minutes = parse_times(series).dropna()
            info.update({"median": format_minutes(pd.Series([float(np.median(minutes))])).iloc[0]})
        else:
            counts = series.dropna().astype(str).value_counts()
            info.update({"categories": counts.index[:60].tolist(), "mode": counts.index[0] if len(counts) else None,
                         "n_categories": int(len(counts))})
        features.append(info)
    numeric, categorical = model_feature_types(input_features, cfg)
    return {
        "input_names": list(input_features),
        "inputs": features,
        "engineered": engineered_feature_names(input_features, cfg),
        "model_numeric": numeric,
        "model_categorical": categorical,
        "feature_config": cfg.to_dict(),
    }


def _inner_pipeline(model: Any) -> Pipeline:
    """The fitted feature/preprocess/model pipeline, unwrapping calibration."""
    if isinstance(model, CalibratedClassifierCV):
        return model.calibrated_classifiers_[0].estimator
    return model


def _clean_feature_name(name: str) -> str:
    for prefix in ("num__", "cat__"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def model_specific_importance(model: Any, key: str) -> dict[str, Any] | None:
    """Impurity / gain importances for tree models, coefficients for linear ones."""
    inner = _inner_pipeline(model)
    clf = inner.named_steps["model"]
    names = [_clean_feature_name(n) for n in inner.named_steps["preprocess"].get_feature_names_out()]
    if hasattr(clf, "feature_importances_"):
        values = np.asarray(clf.feature_importances_, dtype=float)
        kind = "gain" if key == "xgboost" else "impurity"
    elif hasattr(clf, "coef_"):
        values = np.asarray(clf.coef_, dtype=float).ravel()
        kind = "coefficient"
    else:
        return None
    order = np.argsort(-np.abs(values))[:25]
    return {"kind": kind, "values": [{"feature": names[i], "value": float(values[i])} for i in order]}


def tree_rules(model: Any, max_depth: int = 3) -> str | None:
    """Text rendering of the top levels of a fitted decision tree."""
    inner = _inner_pipeline(model)
    clf = inner.named_steps["model"]
    if not isinstance(clf, DecisionTreeClassifier):
        return None
    names = [_clean_feature_name(n) for n in inner.named_steps["preprocess"].get_feature_names_out()]
    return export_text(clf, feature_names=names, max_depth=max_depth, decimals=1, show_weights=False)


def probability_note(spec: ModelSpec, settings: TrainingSettings, calibration_used: str) -> str:
    parts = [spec.probability_note]
    if settings.balance_classes:
        parts.append(
            "Class weighting was used. It raises recall but shifts raw probabilities upward relative "
            "to the true accident rate."
        )
    if calibration_used != "none":
        parts.append(
            f"Probabilities were calibrated ({calibration_used}) using {settings.calibration_folds}-fold "
            "cross-validation on the training split only. This maps scores closer to observed frequencies "
            "in the training data, which is not the same as real-world validation."
        )
    else:
        parts.append("Probabilities are not calibrated. Treat them as relative risk scores, not exact chances.")
    return " ".join(parts)


def _cv_splitter(y: pd.Series, folds: int, seed: int) -> StratifiedKFold:
    max_folds = int(y.value_counts().min())
    if max_folds < 2:
        raise TrainingError("Each class needs at least 2 training rows for cross-validation.")
    return StratifiedKFold(n_splits=min(folds, max_folds), shuffle=True, random_state=seed)


def _wrap_calibration(pipeline: Pipeline, method: str, folds: int, y_train: pd.Series, seed: int) -> Any:
    if method == "none":
        return pipeline
    return CalibratedClassifierCV(pipeline, method=method, cv=_cv_splitter(y_train, folds, seed + 1), ensemble=False)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def train_models(
    df: pd.DataFrame,
    input_features: list[str],
    model_keys: list[str],
    settings: TrainingSettings,
    dataset_info: dict[str, Any],
    models_dir: Path | None = None,
    progress: Callable[[float, str], None] | None = None,
    save: bool = True,
) -> TrainingRun:
    """Train, evaluate and (optionally) save each requested model.

    Args:
        df: Cleaned, schema-mapped data that includes the target column.
        input_features: Raw schema columns to use as model inputs.
        model_keys: Keys from ``MODEL_SPECS``.
        settings: Split, tuning, calibration and hyperparameter settings.
        dataset_info: ``source_label``, ``is_synthetic`` and ``fingerprint`` of the data.
        models_dir: Where artifacts are written.
        progress: Optional callback ``(fraction, message)`` for UI progress bars.
        save: Persist artifacts (disable for dry runs / tests).

    Raises:
        TrainingError: if the data cannot support training at all. Failures of
            individual models are reported in ``TrainingRun.errors`` and do not
            stop the other models.
    """
    settings.validate()
    report = progress or (lambda frac, msg: None)
    target = config.TARGET_COLUMN
    if not model_keys:
        raise TrainingError("Select at least one model to train.")
    unknown = [k for k in model_keys if k not in MODEL_SPECS]
    if unknown:
        raise TrainingError(f"Unknown model(s): {', '.join(unknown)}")
    if not input_features:
        raise TrainingError("Select at least one input feature.")
    missing = [c for c in input_features if c not in df.columns]
    if missing:
        raise TrainingError(f"Selected feature(s) not in the dataset: {', '.join(missing)}")
    if target in input_features:
        raise TrainingError("The accident target cannot be used as an input feature.")
    assessment = assess_target(df)
    if not assessment.usable:
        raise TrainingError(assessment.reason)

    data = df[df[target].notna()].reset_index(drop=True)
    X = data[input_features].copy()
    y = data[target].astype(int)
    X_train, X_test, y_train, y_test, stratified = split_data(X, y, settings.test_size, settings.seed)
    if y_train.nunique() < 2:
        raise TrainingError("The training split contains only one class; use a larger dataset or test size.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_warnings = evaluate.evaluation_warnings(y_train, y_test, bool(dataset_info.get("is_synthetic")))
    imbalance = imbalance_warning(y)
    if imbalance:
        run_warnings.append(imbalance)
    if "svm" in model_keys and len(X_train) > 20000:
        run_warnings.append("SVM training scales poorly beyond ~20,000 rows and may take several minutes.")
    schema = describe_feature_schema(X_train, input_features, settings.feature_config)
    xgb_version = None
    if xgboost_status()[0]:
        xgb_version = _import_xgboost().__version__

    results: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    total = len(model_keys)
    for i, key in enumerate(model_keys):
        spec = MODEL_SPECS[key]
        report(i / total, f"Training {spec.name} ({i + 1}/{total})…")
        if spec.requires_xgboost and not xgboost_status()[0]:
            errors[key] = xgboost_status()[1]
            continue
        try:
            results.append(
                _train_one(key, X_train, X_test, y_train, y_test, input_features, settings, schema,
                           dataset_info, data, run_id, stratified, run_warnings, xgb_version, models_dir, save)
            )
        except Exception as exc:  # report per-model failure, keep training the others
            logger.exception("Training %s failed", key)
            errors[key] = f"{type(exc).__name__}: {exc}"

    champion_key, basis = None, ""
    metric = settings.selection_metric
    scored = [(r["metrics"]["cv"].get(metric, {}).get("mean"), r["model"]["key"]) for r in results]
    scored = [(s, k) for s, k in scored if s is not None and np.isfinite(s)]
    if scored and len(scored) == len(results):
        champion_key = max(scored)[1]
        basis = f"highest mean cross-validated {evaluate.METRICS[metric]['label']} on the training split"
        if save:
            model_store.set_champion(champion_key, metric, basis, models_dir)
    elif results:
        basis = (
            "No champion was selected automatically because a cross-validated score was unavailable for "
            "some models. Choosing a model by test-set scores makes the test estimate optimistic."
        )
    report(1.0, "Training complete.")
    return TrainingRun(run_id, results, errors, champion_key, basis, run_warnings)


def _train_one(key: str, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
               input_features: list[str], settings: TrainingSettings, schema: dict[str, Any],
               dataset_info: dict[str, Any], data: pd.DataFrame, run_id: str, stratified: bool,
               run_warnings: list[str], xgb_version: str | None, models_dir: Path | None, save: bool) -> dict[str, Any]:
    spec = MODEL_SPECS[key]
    params = {**spec.default_params, **settings.params.get(key, {})}
    started = time.perf_counter()
    base = build_pipeline(key, input_features, settings.feature_config, params, settings.seed,
                          settings.balance_classes, y_train)
    splitter = _cv_splitter(y_train, settings.cv_folds, settings.seed)
    best_params: dict[str, Any] = {}
    notes: list[str] = []

    if settings.tune and spec.param_grid:
        grid = {f"model__{name}": values for name, values in spec.param_grid.items()}
        search = GridSearchCV(base, grid, scoring=evaluate.CV_SCORERS, refit=settings.selection_metric,
                              cv=splitter, error_score=np.nan)
        search.fit(X_train, y_train)
        best_params = {name.replace("model__", ""): value for name, value in search.best_params_.items()}
        params.update(best_params)
        base = clone(search.best_estimator_)
        notes.append(
            "Hyperparameters were chosen by grid search on these same training folds, so the "
            "cross-validation scores are slightly optimistic."
        )

    calibration_used = settings.calibration
    if spec.always_calibrate and calibration_used == "none":
        calibration_used = "sigmoid"
    model = _wrap_calibration(base, calibration_used, settings.calibration_folds, y_train, settings.seed)

    # Out-of-fold probabilities on the training split: used for CV metrics and,
    # optionally, to choose the decision threshold without touching the test set.
    oof = np.full(len(y_train), np.nan)
    folds: list[np.ndarray] = []
    for train_idx, val_idx in splitter.split(X_train, y_train):
        fold_model = clone(model).fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        oof[val_idx] = fold_model.predict_proba(X_train.iloc[val_idx])[:, 1]
        folds.append(val_idx)
    if settings.threshold_strategy == "optimize_f1":
        threshold = evaluate.best_f1_threshold(y_train.to_numpy(), oof)
        notes.append(
            f"Decision threshold {threshold:.3f} maximises F1 on out-of-fold training predictions; "
            "threshold-dependent CV scores are therefore slightly optimistic."
        )
    else:
        threshold = float(settings.fixed_threshold)
    fold_metrics = [
        evaluate.compute_metrics(y_train.iloc[idx], (oof[idx] >= threshold).astype(int), oof[idx]) for idx in folds
    ]
    cv_summary = evaluate.summarize_fold_metrics(fold_metrics)
    cv_note = " ".join(
        [f"{splitter.get_n_splits()}-fold stratified cross-validation on the training split only.", *notes]
    )

    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started

    y_score = model.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= threshold).astype(int)
    test_metrics = evaluate.compute_metrics(y_test, y_pred, y_score)

    both = y_test.nunique() == 2
    scoring = "roc_auc" if both else "accuracy"
    perm = permutation_importance(model, X_test, y_test, scoring=scoring, n_repeats=5,
                                  random_state=settings.seed)
    perm_rows = sorted(
        ({"feature": f, "importance_mean": float(m), "importance_std": float(s)}
         for f, m, s in zip(input_features, perm.importances_mean, perm.importances_std)),
        key=lambda r: -r["importance_mean"],
    )

    meta: dict[str, Any] = {
        "model": {"key": key, "name": spec.name, "description": spec.description},
        "training": {
            "run_id": run_id,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "test_size": settings.test_size, "seed": settings.seed,
            "n_train": int(len(X_train)), "n_test": int(len(X_test)),
            "train_positive_share": float(y_train.mean()), "test_positive_share": float(y_test.mean()),
            "stratified": stratified, "balance_classes": settings.balance_classes,
            "calibration": calibration_used, "calibration_folds": settings.calibration_folds,
            "tuned": bool(settings.tune and spec.param_grid), "best_params": best_params,
            "params": {k: v for k, v in params.items()}, "cv_folds": splitter.get_n_splits(),
            "selection_metric": settings.selection_metric, "decision_threshold": round(float(threshold), 4),
            "threshold_strategy": settings.threshold_strategy,
            "fit_seconds": round(fit_seconds, 2),
        },
        "dataset": {
            "source_label": dataset_info.get("source_label", "unknown"),
            "is_synthetic": bool(dataset_info.get("is_synthetic")),
            "fingerprint": dataset_info.get("fingerprint"),
            "n_rows": int(len(data)), "positive_share": float(data[config.TARGET_COLUMN].astype(int).mean()),
        },
        "target": {"column": config.TARGET_COLUMN, "positive_class": config.POSITIVE_CLASS,
                   "class_labels": config.CLASS_LABELS},
        "feature_schema": schema,
        "metrics": {"test": test_metrics, "cv": cv_summary, "cv_note": cv_note},
        "confusion_matrix": evaluate.confusion(y_test, y_pred),
        "classification_report": evaluate.report_dict(y_test, y_pred),
        "curves": evaluate.curves(y_test, y_score),
        "permutation_importance": perm_rows,
        "permutation_scoring": scoring,
        "model_importance": model_specific_importance(model, key),
        "tree_rules": tree_rules(model),
        "probability_note": probability_note(spec, settings, calibration_used),
        "warnings": list(run_warnings),
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__, "xgboost": xgb_version,
                        "platform": sys.platform},
    }
    if save:
        meta = model_store.save_model(key, model, meta, models_dir)
    meta["_pipeline"] = model
    return meta


__all__ = [
    "MODEL_SPECS", "ModelSpec", "TrainingError", "TrainingRun", "TrainingSettings",
    "available_models", "build_pipeline", "train_models", "xgboost_status",
]
