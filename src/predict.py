"""Scenario validation, prediction, risk bands and per-scenario factor analysis.

Predictions come from saved, verified pipelines. Probabilities are model
estimates whose meaning depends on calibration and the training data; the UI
always presents them with that context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Any

import numpy as np
import pandas as pd

import config
from src.feature_engineering import parse_dates, parse_times


class PredictionError(Exception):
    """Raised when a scenario or dataset cannot be scored by the selected model."""


# --------------------------------------------------------------------------- #
# Risk bands
# --------------------------------------------------------------------------- #
def validate_thresholds(thresholds: tuple[float, float, float]) -> tuple[float, float, float]:
    """Check the three band cut-offs are strictly increasing within (0, 1)."""
    low, mid, high = (float(t) for t in thresholds)
    if not 0.0 < low < mid < high < 1.0:
        raise ValueError("Risk thresholds must satisfy 0 < Low/Moderate < Moderate/High < High/Very High < 1.")
    return low, mid, high


def assign_risk_band(probability: float, thresholds: tuple[float, float, float] = config.DEFAULT_RISK_THRESHOLDS) -> str:
    """Map a probability to Low / Moderate / High / Very High using ``thresholds``."""
    low, mid, high = validate_thresholds(thresholds)
    if probability is None or not np.isfinite(probability):
        raise ValueError("Probability must be a finite number.")
    if probability < low:
        return config.RISK_BANDS[0]
    if probability < mid:
        return config.RISK_BANDS[1]
    if probability < high:
        return config.RISK_BANDS[2]
    return config.RISK_BANDS[3]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@dataclass
class ScenarioCheck:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def validate_scenario(scenario: dict[str, Any], meta: dict[str, Any]) -> ScenarioCheck:
    """Check that a scenario supplies every model input with a usable value.

    Missing inputs are errors: the app never substitutes an arbitrary value.
    Values outside the training range, and categories the model never saw,
    are warnings, because the model is extrapolating.
    """
    check = ScenarioCheck()
    schema = meta.get("feature_schema", {})
    for info in schema.get("inputs", []):
        name, kind, label = info["name"], info["kind"], info.get("label", info["name"])
        if name not in scenario or _is_missing(scenario[name]):
            check.errors.append(f"'{label}' is required by this model but was not provided.")
            continue
        value = scenario[name]
        if kind == "numeric":
            try:
                number = float(value)
            except (TypeError, ValueError):
                check.errors.append(f"'{label}' must be a number.")
                continue
            if not np.isfinite(number):
                check.errors.append(f"'{label}' must be a finite number.")
                continue
            low, high = info.get("valid_range", [None, None])
            if low is not None and high is not None and not low <= number <= high:
                check.errors.append(f"'{label}' = {number:g} is outside the plausible range [{low:g}, {high:g}].")
            elif not info["min"] <= number <= info["max"]:
                check.warnings.append(
                    f"'{label}' = {number:g} is outside the training range "
                    f"[{info['min']:g}, {info['max']:g}]; the model is extrapolating."
                )
        elif kind == "categorical" or kind == "identifier":
            categories = info.get("categories", [])
            if str(value) not in categories:
                check.warnings.append(
                    f"'{label}' = '{value}' was not seen during training; the model ignores it, so the "
                    "estimate relies on the other inputs."
                )
        elif kind == "date":
            if pd.isna(parse_dates(pd.Series([value])).iloc[0]):
                check.errors.append(f"'{label}' is not a valid date.")
        elif kind == "time":
            if pd.isna(parse_times(pd.Series([value])).iloc[0]):
                check.errors.append(f"'{label}' is not a valid time of day.")
    return check


def _to_input_value(value: Any, kind: str) -> Any:
    if kind == "date" and isinstance(value, (date, datetime)):
        return value.strftime("%Y-%m-%d")
    if kind == "time" and isinstance(value, time):
        return value.strftime("%H:%M")
    if kind == "numeric":
        return float(value)
    return value


def scenario_frame(scenario: dict[str, Any], meta: dict[str, Any]) -> pd.DataFrame:
    """One-row DataFrame with the model's input columns, in training order."""
    inputs = meta["feature_schema"]["inputs"]
    row = {info["name"]: _to_input_value(scenario.get(info["name"]), info["kind"]) for info in inputs}
    return pd.DataFrame([row], columns=[info["name"] for info in inputs])


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
@dataclass
class PredictionResult:
    probability: float
    predicted_class: int
    predicted_label: str
    risk_band: str
    threshold: float
    warnings: list[str]


def predict_proba_frame(model: Any, frame: pd.DataFrame) -> np.ndarray:
    """Accident-class probabilities for each row of ``frame``."""
    if not hasattr(model, "predict_proba"):
        raise PredictionError("This model does not provide probability estimates.")
    proba = model.predict_proba(frame)
    return np.asarray(proba)[:, 1]


def predict_scenario(model: Any, meta: dict[str, Any], scenario: dict[str, Any],
                     thresholds: tuple[float, float, float] = config.DEFAULT_RISK_THRESHOLDS) -> PredictionResult:
    """Validate and score a single scenario.

    Raises:
        PredictionError: if any required input is missing or invalid.
    """
    check = validate_scenario(scenario, meta)
    if not check.ok:
        raise PredictionError(" ".join(check.errors))
    frame = scenario_frame(scenario, meta)
    probability = float(predict_proba_frame(model, frame)[0])
    threshold = float(meta.get("training", {}).get("decision_threshold", config.DECISION_THRESHOLD))
    predicted = int(probability >= threshold)
    return PredictionResult(
        probability=probability,
        predicted_class=predicted,
        predicted_label=config.CLASS_LABELS[predicted],
        risk_band=assign_risk_band(probability, thresholds),
        threshold=threshold,
        warnings=check.warnings,
    )


def predict_dataframe(model: Any, meta: dict[str, Any], df: pd.DataFrame,
                      thresholds: tuple[float, float, float] = config.DEFAULT_RISK_THRESHOLDS) -> pd.DataFrame:
    """Score every row of ``df`` after checking it has all of the model's inputs."""
    required = meta["feature_schema"]["input_names"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise PredictionError(
            "The dataset is missing input(s) this model was trained on: " + ", ".join(missing)
        )
    if df.empty:
        raise PredictionError("The dataset has no rows to score.")
    probabilities = predict_proba_frame(model, df[required])
    threshold = float(meta.get("training", {}).get("decision_threshold", config.DECISION_THRESHOLD))
    out = df.copy()
    out["predicted_accident_probability"] = probabilities.round(4)
    out["predicted_class"] = np.where(probabilities >= threshold, config.CLASS_LABELS[1], config.CLASS_LABELS[0])
    out["risk_band"] = [assign_risk_band(p, thresholds) for p in probabilities]
    return out


# --------------------------------------------------------------------------- #
# Per-scenario factor analysis
# --------------------------------------------------------------------------- #
def reference_value(info: dict[str, Any]) -> Any:
    """A 'typical' training value: the median for numbers, dates and times, the most common category otherwise."""
    if info["kind"] in ("numeric", "date", "time"):
        return info.get("median")
    return info.get("mode")


def scenario_factors(model: Any, meta: dict[str, Any], scenario: dict[str, Any]) -> pd.DataFrame:
    """How much each input moves this scenario's estimate relative to a typical value.

    Each input is swapped, one at a time, for its typical training value (median
    or most common category) and the change in predicted probability is recorded.
    This is a model-dependent sensitivity analysis. It describes what the model
    learned, not what causes accidents.
    """
    base_frame = scenario_frame(scenario, meta)
    base = float(predict_proba_frame(model, base_frame)[0])
    variants = []
    rows = []
    for info in meta["feature_schema"]["inputs"]:
        ref = reference_value(info)
        current = base_frame.iloc[0][info["name"]]
        if ref is None or str(current) == str(ref):
            rows.append((info, current, ref, True))
            continue
        variant = base_frame.copy()
        variant.loc[variant.index[0], info["name"]] = ref if info["kind"] != "numeric" else float(ref)
        variants.append(variant)
        rows.append((info, current, ref, False))
    probs = iter(predict_proba_frame(model, pd.concat(variants, ignore_index=True))) if variants else iter(())
    records = []
    for info, current, ref, same in rows:
        delta = 0.0 if same else base - float(next(probs))
        records.append(
            {
                "feature": info.get("label", info["name"]),
                "your_value": current,
                "typical_value": ref,
                "effect_pp": round(delta * 100, 2),
                "direction": "raises" if delta > 0.0005 else ("lowers" if delta < -0.0005 else "no change"),
            }
        )
    out = pd.DataFrame(records)
    return out.reindex(out["effect_pp"].abs().sort_values(ascending=False).index).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Safety guidance
# --------------------------------------------------------------------------- #
def safety_guidance(scenario: dict[str, Any], risk_band: str) -> list[str]:
    """General, condition-based road-safety reminders (not model outputs)."""
    tips: list[str] = []
    weather = str(scenario.get("weather", "")).lower()
    road = str(scenario.get("road_condition", "")).lower()
    light = str(scenario.get("light_condition", "")).lower()
    vehicle = str(scenario.get("vehicle_type", "")).lower()
    traffic = str(scenario.get("traffic_density", "")).lower()
    speed = scenario.get("speed")
    if any(k in weather for k in ("rain", "drizzle", "storm", "shower")) or "wet" in road:
        tips.append("Wet surfaces lengthen braking distances: reduce speed and increase following distance.")
    if "fog" in weather:
        tips.append("In fog use low-beam headlights or fog lights and drive at a speed that lets you stop within visible distance.")
    if "snow" in weather or "icy" in road or "ice" in road:
        tips.append("On snow or ice, accelerate, brake and steer gently; avoid travel if conditions are severe.")
    if "damaged" in road or "pothole" in road:
        tips.append("Damaged road surfaces can destabilise vehicles; slow down and watch for hazards.")
    if "dark" in light:
        tips.append("In darkness reduce speed, use headlights correctly and watch for unlit pedestrians and cyclists.")
    if "motorcycle" in vehicle or "bicycle" in vehicle:
        tips.append("Two-wheeler riders should wear a helmet and high-visibility gear and avoid other vehicles' blind spots.")
    if "high" in traffic:
        tips.append("In dense traffic keep a safe gap, avoid abrupt lane changes and expect sudden braking.")
    try:
        if speed is not None and float(speed) >= 80:
            tips.append("Higher speeds increase both crash likelihood and severity: keep to posted limits.")
    except (TypeError, ValueError):
        pass
    if risk_band in ("High", "Very High") and not tips:
        tips.append("The model rates this scenario as elevated risk. Drive defensively and stay alert.")
    if not tips:
        tips.append("Stay alert, obey speed limits, and avoid distractions such as phone use.")
    return tips
