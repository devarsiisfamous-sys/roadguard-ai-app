"""Shared fixtures: small reproducible synthetic datasets and isolated model folders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import config
from src.data_loader import apply_column_mapping, suggest_column_mapping
from src.preprocessing import clean_dataset
from src.train import TrainingSettings, train_models

FEATURES = ["date", "time", "weather", "speed", "driver_age", "vehicle_type", "road_condition",
            "traffic_density", "light_condition"]
FAST_PARAMS = {"random_forest": {"n_estimators": 40}}


@pytest.fixture(scope="session")
def raw_synthetic() -> pd.DataFrame:
    from src.synthetic_data import generate_synthetic_dataset

    return generate_synthetic_dataset(n_rows=900, seed=7, inject_quality_issues=True)


@pytest.fixture(scope="session")
def clean_synthetic(raw_synthetic: pd.DataFrame) -> pd.DataFrame:
    mapped, _ = apply_column_mapping(raw_synthetic, suggest_column_mapping(list(raw_synthetic.columns)))
    clean, _ = clean_dataset(mapped)
    return clean


@pytest.fixture
def models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect every model read/write to a temporary folder."""
    directory = tmp_path / "models"
    directory.mkdir()
    monkeypatch.setattr(config, "MODELS_DIR", directory)
    return directory


@pytest.fixture(scope="session")
def trained(clean_synthetic: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory):
    """Logistic regression + decision tree trained once and saved to a temp folder."""
    directory = tmp_path_factory.mktemp("trained_models")
    settings = TrainingSettings(cv_folds=3, params=FAST_PARAMS)
    run = train_models(clean_synthetic, FEATURES, ["logistic_regression", "decision_tree"], settings,
                       {"source_label": "pytest synthetic", "is_synthetic": True, "fingerprint": "pytest"},
                       models_dir=directory)
    return run, directory
