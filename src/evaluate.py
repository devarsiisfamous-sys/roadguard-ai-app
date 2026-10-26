"""Evaluation metrics, curves, comparison tables and text reports.

The accident class (label 1) is always the positive class. Every number here
comes from an actual evaluation run; nothing is hard-coded.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import config
from src.utils import format_metric, format_pct

METRICS: dict[str, dict[str, str]] = {
    "f1": {
        "label": "F1 score",
        "help": "Harmonic mean of precision and recall for the accident class. Balances missed accidents against false alarms.",
    },
    "recall": {
        "label": "Recall (sensitivity)",
        "help": "Share of actual accidents the model flags. Low recall means dangerous situations are missed (false negatives).",
    },
    "precision": {
        "label": "Precision",
        "help": "Share of flagged situations that really were accidents. Low precision means many false alarms (false positives).",
    },
    "roc_auc": {
        "label": "ROC-AUC",
        "help": "Probability that a random accident case is ranked above a random non-accident case. Threshold-independent.",
    },
    "average_precision": {
        "label": "Average precision (PR-AUC)",
        "help": "Area under the precision-recall curve. More informative than ROC-AUC when accidents are rare.",
    },
    "balanced_accuracy": {
        "label": "Balanced accuracy",
        "help": "Mean of recall on each class. Not inflated by class imbalance.",
    },
    "accuracy": {
        "label": "Accuracy",
        "help": "Share of all predictions that are correct. Misleading when accidents are rare.",
    },
}
SELECTION_METRICS = list(METRICS)
# sklearn scorer names for cross-validation
CV_SCORERS = {name: name for name in METRICS}

FP_FN_TRADEOFF = (
    "**False positives** (predicting an accident that does not happen) cost unnecessary alerts, "
    "patrols or diversions and erode trust in warnings. **False negatives** (missing an accident "
    "that does happen) mean a dangerous situation went unflagged. For safety screening, "
    "missing a dangerous situation is usually worse, which favours recall. Too many false alarms "
    "still make a system unusable, which is why F1 and precision-recall curves matter."
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None) -> dict[str, float | None]:
    """Classification metrics with the accident class (1) as positive.

    Threshold-free metrics (ROC-AUC, average precision) are ``None`` when the
    evaluation labels contain only one class, because they are undefined then.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    both_classes = len(np.unique(y_true)) == 2
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if both_classes else None,
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": None,
        "average_precision": None,
    }
    if y_score is not None and both_classes:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["average_precision"] = float(average_precision_score(y_true, y_score))
    return metrics


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    """2x2 confusion matrix [[TN, FP], [FN, TP]] with fixed label order (0, 1)."""
    return confusion_matrix(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int), labels=[0, 1]).tolist()


def report_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """sklearn classification report as a dict, with readable class names."""
    return classification_report(
        np.asarray(y_true).astype(int),
        np.asarray(y_pred).astype(int),
        labels=[0, 1],
        target_names=[config.CLASS_LABELS[0], config.CLASS_LABELS[1]],
        output_dict=True,
        zero_division=0,
    )


def _downsample(*arrays: np.ndarray, max_points: int = 250) -> list[list[float]]:
    n = len(arrays[0])
    if n <= max_points:
        idx = np.arange(n)
    else:
        idx = np.unique(np.linspace(0, n - 1, max_points).round().astype(int))
    return [np.asarray(a)[idx].astype(float).round(5).tolist() for a in arrays]


def curves(y_true: np.ndarray, y_score: np.ndarray | None) -> dict[str, Any]:
    """ROC and precision-recall curve points (downsampled), or empty when undefined."""
    y_true = np.asarray(y_true).astype(int)
    if y_score is None or len(np.unique(y_true)) < 2:
        return {}
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fpr_s, tpr_s = _downsample(fpr, tpr)
    rec_s, prec_s = _downsample(recall[::-1], precision[::-1])
    return {
        "roc": {"fpr": fpr_s, "tpr": tpr_s},
        "pr": {"recall": rec_s, "precision": prec_s, "baseline": float(y_true.mean())},
    }


def summarize_fold_metrics(fold_metrics: list[dict[str, float | None]]) -> dict[str, dict[str, float]]:
    """Mean and standard deviation per metric across cross-validation folds."""
    summary: dict[str, dict[str, float]] = {}
    for name in METRICS:
        values = np.asarray([m.get(name) for m in fold_metrics if m.get(name) is not None], dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            summary[name] = {"mean": float(values.mean()), "std": float(values.std())}
    return summary


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Probability cut-off that maximises F1 for the accident class.

    Call this only with out-of-fold *training* predictions so the test set
    stays untouched. Falls back to 0.5 when F1 is undefined.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_score)
    if mask.sum() == 0 or len(np.unique(y_true[mask])) < 2:
        return config.DECISION_THRESHOLD
    precision, recall, thresholds = precision_recall_curve(y_true[mask], y_score[mask])
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    best = int(np.nanargmax(f1[:-1])) if len(thresholds) else -1
    if best < 0:
        return config.DECISION_THRESHOLD
    return float(np.clip(thresholds[best], 0.01, 0.99))


def evaluation_warnings(y_train: pd.Series, y_test: pd.Series, is_synthetic: bool) -> list[str]:
    """Warnings about how much the measured metrics can be trusted."""
    warnings: list[str] = []
    if is_synthetic:
        warnings.append(
            "Measured on SYNTHETIC demo data. The data were simulated with a known risk formula, so "
            "these scores do not show real-world accident-prediction performance."
        )
    n_pos_test = int((y_test == 1).sum())
    if y_test.nunique() < 2:
        warnings.append("The test set contains only one class; ROC-AUC and PR curves cannot be computed.")
    elif n_pos_test < config.SMALL_TEST_POSITIVES:
        warnings.append(
            f"The test set has only {n_pos_test} accident cases; metrics will vary a lot between splits. "
            "Rely on the cross-validation spread as well."
        )
    total = len(y_train) + len(y_test)
    if total < 500:
        warnings.append(f"Small dataset ({total} labelled rows): estimates are unstable.")
    share = float(pd.concat([y_train, y_test]).mean())
    if min(share, 1 - share) < config.SEVERE_IMBALANCE_SHARE:
        warnings.append(
            f"Heavy class imbalance (accident share {share:.1%}). Accuracy is not informative; use recall, "
            "precision, F1 and average precision."
        )
    return warnings


def comparison_table(metadata_list: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per trained model with test and cross-validation metrics."""
    rows = []
    for meta in metadata_list:
        test = meta.get("metrics", {}).get("test", {})
        cv = meta.get("metrics", {}).get("cv", {})
        row: dict[str, Any] = {"model": meta["model"]["name"]}
        for name in METRICS:
            row[f"test_{name}"] = test.get(name)
        for name in METRICS:
            row[f"cv_{name}_mean"] = cv.get(name, {}).get("mean")
            row[f"cv_{name}_std"] = cv.get(name, {}).get("std")
        row["fit_seconds"] = meta.get("training", {}).get("fit_seconds")
        row["calibration"] = meta.get("training", {}).get("calibration")
        row["tuned"] = meta.get("training", {}).get("tuned")
        row["data_source"] = meta.get("dataset", {}).get("source_label")
        row["synthetic_data"] = meta.get("dataset", {}).get("is_synthetic")
        row["trained_at"] = meta.get("training", {}).get("trained_at")
        rows.append(row)
    return pd.DataFrame(rows)


def metrics_long_table(metadata_list: list[dict[str, Any]]) -> pd.DataFrame:
    """Long-format table: model, split (test / cv), metric, value, std."""
    rows = []
    for meta in metadata_list:
        name = meta["model"]["name"]
        for metric, value in meta.get("metrics", {}).get("test", {}).items():
            rows.append({"model": name, "evaluation": "held-out test", "metric": metric, "value": value, "std": None})
        for metric, stats in meta.get("metrics", {}).get("cv", {}).items():
            rows.append(
                {"model": name, "evaluation": "cross-validation (train)", "metric": metric,
                 "value": stats.get("mean"), "std": stats.get("std")}
            )
    return pd.DataFrame(rows)


def build_model_report(meta: dict[str, Any]) -> str:
    """Markdown summary of one model's evaluation, settings and limitations."""
    model = meta["model"]
    training = meta.get("training", {})
    dataset = meta.get("dataset", {})
    test = meta.get("metrics", {}).get("test", {})
    cv = meta.get("metrics", {}).get("cv", {})
    cm = meta.get("confusion_matrix")
    lines = [
        f"# RoadGuard AI model report: {model['name']}",
        "",
        f"- Trained at: {training.get('trained_at')}",
        f"- Run ID: {training.get('run_id')}",
        f"- Data source: {dataset.get('source_label')} "
        f"({'SYNTHETIC demo data' if dataset.get('is_synthetic') else 'user-provided data'})",
        f"- Labelled rows: {dataset.get('n_rows')} (accident share {format_pct(dataset.get('positive_share'))})",
        f"- Train / test rows: {training.get('n_train')} / {training.get('n_test')} "
        f"({'stratified' if training.get('stratified') else 'not stratified'}, seed {training.get('seed')})",
        f"- Input features: {', '.join(meta.get('feature_schema', {}).get('input_names', []))}",
        f"- Class weighting: {'balanced' if training.get('balance_classes') else 'none'}",
        f"- Probability calibration: {training.get('calibration')}",
        f"- Hyperparameter tuning: {'grid search CV on training data' if training.get('tuned') else 'none'}",
        f"- Parameters: {training.get('params')}",
        f"- Positive class: 1 = {config.CLASS_LABELS[1]}; decision threshold {training.get('decision_threshold')}",
        "",
        "## Held-out test metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for name, spec in METRICS.items():
        lines.append(f"| {spec['label']} | {format_metric(test.get(name))} |")
    if cv:
        lines += ["", f"## Cross-validation on training data ({training.get('cv_folds')} folds)", "",
                  "| Metric | Mean | Std |", "|---|---|---|"]
        for name, stats in cv.items():
            lines.append(f"| {METRICS[name]['label']} | {format_metric(stats['mean'])} | {format_metric(stats['std'])} |")
        if meta.get("metrics", {}).get("cv_note"):
            lines += ["", f"_{meta['metrics']['cv_note']}_"]
    if cm:
        (tn, fp), (fn, tp) = cm
        lines += ["", "## Confusion matrix (test set)", "",
                  "| | Predicted no accident | Predicted accident |", "|---|---|---|",
                  f"| Actual no accident | {tn} (TN) | {fp} (FP) |",
                  f"| Actual accident | {fn} (FN) | {tp} (TP) |"]
    perm = meta.get("permutation_importance", [])
    if perm:
        lines += ["", f"## Permutation importance (held-out test, scoring = {meta.get('permutation_scoring')})", ""]
        for item in perm[:10]:
            lines.append(f"- {item['feature']}: {item['importance_mean']:.4f} ± {item['importance_std']:.4f}")
        lines.append("")
        lines.append("Importance measures association with the model's predictions, not causation.")
    lines += ["", "## Probability interpretation", "", meta.get("probability_note", ""), ""]
    lines += ["## Warnings and limitations", ""]
    for warning in meta.get("warnings", []):
        lines.append(f"- {warning}")
    lines += [
        "- Feature importance and factor analyses describe statistical associations learned by the model; "
        "they are not causal evidence.",
        f"- {config.SAFETY_DISCLAIMER}",
        "",
    ]
    return "\n".join(lines)
