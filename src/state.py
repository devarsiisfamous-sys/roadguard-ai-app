"""Streamlit application state shared by every page.

- Immutable inputs (CSV parsing) use ``st.cache_data``.
- Loaded model pipelines use ``st.cache_resource``, keyed by the artifact's
  SHA-256 so a retrained or deleted model can never be served stale.
- The active dataset, UI selections, scenarios and prediction history live in
  ``st.session_state``. Trained models are persisted on disk, not only in
  session state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

import config
from src import model_store
from src.data_loader import (
    DataLoadError,
    TargetAssessment,
    apply_column_mapping,
    assess_target,
    load_sample_dataset,
    read_csv_bytes,
    suggest_column_mapping,
)
from src.feature_engineering import FeatureConfig
from src.preprocessing import CleaningLog, CleaningOptions, DataQualityReport, build_quality_report, clean_dataset
from src.synthetic_data import SYNTHETIC_SOURCE_LABEL
from src.utils import dataframe_fingerprint

KEY_DATASET = "rg_dataset"
KEY_DATASET_ERROR = "rg_dataset_error"
KEY_FEATURE_CONFIG = "rg_feature_config"
KEY_RISK_THRESHOLDS = "rg_risk_thresholds"
KEY_HISTORY = "rg_prediction_history"
KEY_SCENARIOS = "rg_scenarios"
KEY_LAST_RUN = "rg_last_training_run"
KEY_SELECTED_MODEL = "rg_selected_model"


@dataclass
class ActiveDataset:
    """The dataset every page analyses, plus how it was produced."""

    df: pd.DataFrame
    mapped: pd.DataFrame
    source_label: str
    is_synthetic: bool
    mapping: dict[str, str | None]
    dropped_columns: list[str]
    quality: DataQualityReport
    cleaning_log: CleaningLog
    cleaning_options: CleaningOptions
    target: TargetAssessment
    fingerprint: str
    loaded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @property
    def info(self) -> dict[str, Any]:
        """Dataset description stored with trained models."""
        return {"source_label": self.source_label, "is_synthetic": self.is_synthetic, "fingerprint": self.fingerprint}

    @property
    def source_kind(self) -> str:
        return "SYNTHETIC demo data" if self.is_synthetic else "User-provided data"


@st.cache_data(show_spinner=False)
def _read_sample(path: str, mtime: float) -> pd.DataFrame:
    """Read the demo CSV; ``mtime`` is part of the cache key so a regenerated file is re-read."""
    return load_sample_dataset(Path(path))


@st.cache_data(show_spinner=False, max_entries=4)
def read_upload(raw: bytes, filename: str) -> pd.DataFrame:
    """Parse uploaded CSV bytes (cached by content; callers get a copy)."""
    return read_csv_bytes(raw, filename)


def build_active_dataset(raw: pd.DataFrame, mapping: dict[str, str | None], source_label: str,
                         is_synthetic: bool, options: CleaningOptions | None = None) -> ActiveDataset:
    """Map, profile and clean ``raw`` into an ``ActiveDataset``."""
    options = options or CleaningOptions()
    mapped, dropped = apply_column_mapping(raw, mapping)
    quality = build_quality_report(mapped)
    clean, log = clean_dataset(mapped, options)
    if "record_source" in clean.columns and clean["record_source"].notna().any():
        sources = set(clean["record_source"].dropna().astype(str).str.lower())
        is_synthetic = is_synthetic or SYNTHETIC_SOURCE_LABEL in sources
    return ActiveDataset(
        df=clean,
        mapped=mapped,
        source_label=source_label,
        is_synthetic=is_synthetic,
        mapping=mapping,
        dropped_columns=dropped,
        quality=quality,
        cleaning_log=log,
        cleaning_options=options,
        target=assess_target(clean, quality.target.n_invalid),
        fingerprint=dataframe_fingerprint(clean),
    )


def load_demo_dataset(options: CleaningOptions | None = None) -> ActiveDataset:
    """Build the active dataset from the bundled synthetic CSV."""
    path = config.SAMPLE_DATA_PATH
    mtime = path.stat().st_mtime if path.exists() else 0.0
    raw = _read_sample(str(path), mtime)
    return build_active_dataset(raw, suggest_column_mapping(list(raw.columns)),
                                "Built-in synthetic demo dataset", True, options)


def get_active_dataset() -> ActiveDataset | None:
    """Current dataset; loads the synthetic demo on first use. None if loading failed."""
    if KEY_DATASET not in st.session_state:
        try:
            st.session_state[KEY_DATASET] = load_demo_dataset()
            st.session_state.pop(KEY_DATASET_ERROR, None)
        except DataLoadError as exc:
            st.session_state[KEY_DATASET_ERROR] = str(exc)
            return None
    return st.session_state[KEY_DATASET]


def set_active_dataset(dataset: ActiveDataset) -> None:
    """Replace the active dataset and reset selections that depended on the old one."""
    st.session_state[KEY_DATASET] = dataset
    st.session_state.pop(KEY_DATASET_ERROR, None)
    for key in list(st.session_state.keys()):
        if str(key).startswith("flt_"):
            del st.session_state[key]


def dataset_error() -> str | None:
    return st.session_state.get(KEY_DATASET_ERROR)


# --------------------------------------------------------------------------- #
# Settings shared between pages
# --------------------------------------------------------------------------- #
def get_feature_config() -> FeatureConfig:
    return st.session_state.get(KEY_FEATURE_CONFIG, FeatureConfig())


def set_feature_config(cfg: FeatureConfig) -> None:
    st.session_state[KEY_FEATURE_CONFIG] = cfg


def get_risk_thresholds() -> tuple[float, float, float]:
    return st.session_state.get(KEY_RISK_THRESHOLDS, config.DEFAULT_RISK_THRESHOLDS)


def set_risk_thresholds(thresholds: tuple[float, float, float]) -> None:
    st.session_state[KEY_RISK_THRESHOLDS] = tuple(float(t) for t in thresholds)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False, max_entries=8)
def _load_pipeline_cached(key: str, models_dir: str, sha256: str) -> tuple[Any, dict[str, Any]]:
    return model_store.load_pipeline(key, Path(models_dir))


def load_model(key: str) -> tuple[Any, dict[str, Any]]:
    """Verified pipeline + metadata, cached per artifact hash."""
    meta = model_store.load_metadata(key, config.MODELS_DIR)
    sha = meta.get("artifact", {}).get("sha256", "")
    return _load_pipeline_cached(key, str(config.MODELS_DIR), sha)


def saved_models() -> list[dict[str, Any]]:
    """Metadata of all saved models with problems/warnings attached."""
    return model_store.list_models(config.MODELS_DIR)


def champion() -> dict[str, Any] | None:
    return model_store.get_champion(config.MODELS_DIR)


def default_model_key(models: list[dict[str, Any]]) -> str | None:
    """User's selection if still valid, else the champion, else the newest usable model."""
    usable = [m["model"]["key"] for m in models if not m.get("problems")]
    chosen = st.session_state.get(KEY_SELECTED_MODEL)
    if chosen in usable:
        return chosen
    champ = champion()
    if champ and champ["key"] in usable:
        return champ["key"]
    return usable[0] if usable else None


# --------------------------------------------------------------------------- #
# Session records
# --------------------------------------------------------------------------- #
def add_prediction(record: dict[str, Any]) -> None:
    st.session_state.setdefault(KEY_HISTORY, []).append(record)


def prediction_history() -> pd.DataFrame:
    return pd.DataFrame(st.session_state.get(KEY_HISTORY, []))


def clear_prediction_history() -> None:
    st.session_state[KEY_HISTORY] = []


def scenarios() -> list[dict[str, Any]]:
    return st.session_state.setdefault(KEY_SCENARIOS, [])


def clear_scenarios() -> None:
    st.session_state[KEY_SCENARIOS] = []
