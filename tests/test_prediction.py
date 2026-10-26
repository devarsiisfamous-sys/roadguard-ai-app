"""Training, evaluation, persistence, prediction and optional-dependency handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src import evaluate, model_store, predict, train
from src.train import TrainingError, TrainingSettings, train_models
from tests.conftest import FEATURES

INFO = {"source_label": "pytest", "is_synthetic": True, "fingerprint": "pytest"}


def _scenario(**overrides) -> dict:
    base = {"date": "2024-06-08", "time": "22:30", "weather": "Rain", "speed": 95.0, "driver_age": 22.0,
            "vehicle_type": "Motorcycle", "road_condition": "Wet", "traffic_density": "High",
            "light_condition": "Dark - No Lights"}
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def test_training_produces_real_metrics_and_artifacts(trained) -> None:
    run, directory = trained
    assert not run.errors
    assert {r["model"]["key"] for r in run.results} == {"logistic_regression", "decision_tree"}
    for meta in run.results:
        for name, value in meta["metrics"]["test"].items():
            assert value is None or 0.0 <= value <= 1.0, name
        assert set(meta["metrics"]["cv"]) >= {"f1", "roc_auc", "recall"}
        assert 0.0 < meta["training"]["decision_threshold"] < 1.0
        assert meta["training"]["n_train"] + meta["training"]["n_test"] == meta["dataset"]["n_rows"]
        assert meta["dataset"]["is_synthetic"] is True
        assert any("SYNTHETIC" in w for w in meta["warnings"])
        assert (directory / meta["model"]["key"] / model_store.ARTIFACT_FILE).exists()
        assert meta["permutation_importance"]
    assert run.champion_key in {"logistic_regression", "decision_tree"}
    assert model_store.get_champion(directory)["key"] == run.champion_key


def test_champion_is_chosen_by_cv_not_test(trained) -> None:
    run, _ = trained
    metric = "f1"
    best_cv = max(run.results, key=lambda r: r["metrics"]["cv"][metric]["mean"])
    assert run.champion_key == best_cv["model"]["key"]


def test_saved_pipeline_reloads_with_identical_predictions(trained, clean_synthetic) -> None:
    run, directory = trained
    meta = run.results[0]
    loaded, loaded_meta = model_store.load_pipeline(meta["model"]["key"], directory)
    X = clean_synthetic[FEATURES].head(25)
    np.testing.assert_allclose(loaded.predict_proba(X), meta["_pipeline"].predict_proba(X))
    assert loaded_meta["artifact"]["sha256"] == meta["artifact"]["sha256"]


def test_tampered_artifact_is_refused(clean_synthetic, models_dir) -> None:
    run = train_models(clean_synthetic, FEATURES, ["logistic_regression"], TrainingSettings(cv_folds=3), INFO,
                       models_dir=models_dir)
    artifact = models_dir / "logistic_regression" / model_store.ARTIFACT_FILE
    artifact.write_bytes(artifact.read_bytes() + b"tampered")
    with pytest.raises(model_store.ModelStoreError, match="SHA-256"):
        model_store.load_pipeline("logistic_regression", models_dir)
    listed = model_store.list_models(models_dir)
    assert listed[0]["problems"]
    assert run.results


def test_missing_artifact_and_delete(clean_synthetic, models_dir) -> None:
    train_models(clean_synthetic, FEATURES, ["decision_tree"], TrainingSettings(cv_folds=3), INFO, models_dir=models_dir)
    assert model_store.get_champion(models_dir)["key"] == "decision_tree"
    model_store.delete_model("decision_tree", models_dir)
    assert model_store.list_models(models_dir) == []
    assert model_store.get_champion(models_dir) is None
    with pytest.raises(model_store.ModelStoreError):
        model_store.load_pipeline("decision_tree", models_dir)
    with pytest.raises(model_store.ModelStoreError, match="Invalid model key"):
        model_store.load_pipeline("../etc", models_dir)


def test_single_class_target_is_rejected(clean_synthetic) -> None:
    single = clean_synthetic.copy()
    single[config.TARGET_COLUMN] = pd.array([0] * len(single), dtype="Int64")
    with pytest.raises(TrainingError, match="both"):
        train_models(single, FEATURES, ["logistic_regression"], TrainingSettings(), INFO, save=False)


def test_invalid_training_requests(clean_synthetic) -> None:
    with pytest.raises(TrainingError, match="not in the dataset"):
        train_models(clean_synthetic, ["nonexistent"], ["logistic_regression"], TrainingSettings(), INFO, save=False)
    with pytest.raises(TrainingError, match="target"):
        train_models(clean_synthetic, [config.TARGET_COLUMN], ["logistic_regression"], TrainingSettings(), INFO, save=False)
    with pytest.raises(TrainingError, match="at least one model"):
        train_models(clean_synthetic, FEATURES, [], TrainingSettings(), INFO, save=False)
    with pytest.raises(TrainingError):
        train_models(clean_synthetic, FEATURES, ["logistic_regression"], TrainingSettings(calibration="magic"), INFO,
                     save=False)


def test_calibration_tuning_and_svm(clean_synthetic) -> None:
    settings = TrainingSettings(cv_folds=3, calibration="sigmoid", tune=True, threshold_strategy="fixed",
                                fixed_threshold=0.3)
    run = train_models(clean_synthetic, ["speed", "road_condition", "weather"], ["logistic_regression", "svm"],
                       settings, INFO, save=False)
    assert not run.errors
    by_key = {r["model"]["key"]: r for r in run.results}
    assert by_key["svm"]["training"]["calibration"] == "sigmoid"
    assert by_key["logistic_regression"]["training"]["tuned"] is True
    assert by_key["logistic_regression"]["training"]["best_params"]
    assert by_key["svm"]["training"]["decision_threshold"] == pytest.approx(0.3)


def test_missing_xgboost_degrades_gracefully(clean_synthetic, monkeypatch) -> None:
    def broken_import():
        raise ImportError("No module named 'xgboost'")

    monkeypatch.setattr(train, "_import_xgboost", broken_import)
    train.xgboost_status.cache_clear()
    try:
        ok, message = train.xgboost_status()
        assert not ok and "not available" in message
        assert train.available_models()["xgboost"][0] is False
        run = train_models(clean_synthetic, FEATURES, ["xgboost", "logistic_regression"],
                           TrainingSettings(cv_folds=3), INFO, save=False)
        assert "xgboost" in run.errors
        assert [r["model"]["key"] for r in run.results] == ["logistic_regression"]
    finally:
        train.xgboost_status.cache_clear()


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def test_metrics_handle_single_class_evaluation() -> None:
    metrics = evaluate.compute_metrics(np.array([0, 0, 0]), np.array([0, 1, 0]), np.array([0.1, 0.8, 0.2]))
    assert metrics["roc_auc"] is None and metrics["average_precision"] is None
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert evaluate.curves(np.array([0, 0]), np.array([0.2, 0.4])) == {}


def test_best_threshold_and_report(trained) -> None:
    y = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array([0.1, 0.3, 0.35, 0.8, 0.2, 0.9])
    assert 0.3 < evaluate.best_f1_threshold(y, scores) <= 0.35
    run, _ = trained
    report = evaluate.build_model_report(run.results[0])
    assert run.results[0]["model"]["name"] in report and "SYNTHETIC" in report and "not causal" in report
    table = evaluate.comparison_table(run.results)
    assert len(table) == 2 and "cv_f1_mean" in table.columns


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def test_predict_scenario(trained) -> None:
    run, directory = trained
    model, meta = model_store.load_pipeline(run.results[0]["model"]["key"], directory)
    risky = predict.predict_scenario(model, meta, _scenario())
    calm = predict.predict_scenario(model, meta, _scenario(weather="Clear", speed=30.0, road_condition="Dry",
                                                           light_condition="Daylight", time="11:00",
                                                           vehicle_type="Car", driver_age=40.0))
    assert 0.0 <= calm.probability <= 1.0 and 0.0 <= risky.probability <= 1.0
    assert risky.probability > calm.probability
    assert risky.risk_band in config.RISK_BANDS
    assert risky.predicted_label in config.CLASS_LABELS.values()


def test_prediction_rejects_missing_or_invalid_inputs(trained) -> None:
    run, directory = trained
    model, meta = model_store.load_pipeline(run.results[0]["model"]["key"], directory)
    scenario = _scenario()
    del scenario["weather"]
    with pytest.raises(predict.PredictionError, match="Weather"):
        predict.predict_scenario(model, meta, scenario)
    with pytest.raises(predict.PredictionError, match="plausible range"):
        predict.predict_scenario(model, meta, _scenario(speed=999.0))
    with pytest.raises(predict.PredictionError, match="valid date"):
        predict.predict_scenario(model, meta, _scenario(date="not a date"))


def test_prediction_warns_on_unseen_category(trained) -> None:
    run, directory = trained
    model, meta = model_store.load_pipeline(run.results[0]["model"]["key"], directory)
    result = predict.predict_scenario(model, meta, _scenario(vehicle_type="Hovercraft"))
    assert any("not seen during training" in w for w in result.warnings)


def test_predict_dataframe_checks_schema(trained, clean_synthetic) -> None:
    run, directory = trained
    model, meta = model_store.load_pipeline(run.results[0]["model"]["key"], directory)
    scored = predict.predict_dataframe(model, meta, clean_synthetic.head(30))
    assert {"predicted_accident_probability", "risk_band", "predicted_class"} <= set(scored.columns)
    with pytest.raises(predict.PredictionError, match="missing input"):
        predict.predict_dataframe(model, meta, clean_synthetic.drop(columns=["speed"]))


def test_scenario_factors(trained) -> None:
    run, directory = trained
    model, meta = model_store.load_pipeline(run.results[0]["model"]["key"], directory)
    factors = predict.scenario_factors(model, meta, _scenario())
    assert len(factors) == len(FEATURES)
    assert set(factors["direction"]) <= {"raises", "lowers", "no change"}
    typical = {i["name"]: predict.reference_value(i) for i in meta["feature_schema"]["inputs"]}
    assert (predict.scenario_factors(model, meta, typical)["effect_pp"] == 0).all()


@pytest.mark.parametrize("prob, band", [(0.0, "Low"), (0.249, "Low"), (0.25, "Moderate"), (0.5, "High"),
                                        (0.75, "Very High"), (1.0, "Very High")])
def test_risk_bands(prob: float, band: str) -> None:
    assert predict.assign_risk_band(prob) == band


def test_invalid_risk_thresholds() -> None:
    with pytest.raises(ValueError):
        predict.assign_risk_band(0.5, (0.5, 0.4, 0.8))
    with pytest.raises(ValueError):
        predict.assign_risk_band(float("nan"))


def test_safety_guidance_is_condition_based() -> None:
    tips = predict.safety_guidance(_scenario(weather="Fog"), "High")
    assert any("fog" in t.lower() for t in tips)


def test_static_evaluation_png(trained) -> None:
    from src.static_charts import evaluation_png

    run, _ = trained
    png = evaluation_png(run.results[0], dpi=60)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
