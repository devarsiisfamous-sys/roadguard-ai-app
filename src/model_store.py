"""Saving, listing, loading and deleting trained model artifacts.

Layout::

    models/
      champion.json                 # which model the app uses by default
      <model_key>/pipeline.joblib   # full fitted pipeline (features + preprocessing + model)
      <model_key>/metadata.json     # schema, metrics, settings, SHA-256 of the joblib file

Joblib files can execute code when loaded, so the app only loads artifacts it
wrote itself. They must sit in the expected folder and match the SHA-256
recorded in their metadata. There is deliberately no feature for uploading
model files.
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import sklearn

import config
from src.utils import file_sha256, get_logger, read_json, write_json

logger = get_logger(__name__)

ARTIFACT_FILE = "pipeline.joblib"
METADATA_FILE = "metadata.json"
CHAMPION_FILE = "champion.json"
_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,40}$")


class ModelStoreError(Exception):
    """Raised for missing, tampered or incompatible model artifacts."""


def _root(models_dir: Path | None) -> Path:
    """Resolve the models directory at call time (so tests can redirect it)."""
    return Path(models_dir) if models_dir is not None else config.MODELS_DIR


def _model_dir(key: str, models_dir: Path | None) -> Path:
    models_dir = _root(models_dir)
    if not _KEY_PATTERN.match(key):
        raise ModelStoreError(f"Invalid model key '{key}'.")
    path = (_root(models_dir) / key).resolve()
    if _root(models_dir).resolve() not in path.parents:
        raise ModelStoreError("Model path escapes the models directory.")
    return path


def save_model(key: str, pipeline: Any, metadata: dict[str, Any], models_dir: Path | None = None) -> dict[str, Any]:
    """Persist a fitted pipeline and its metadata, recording the artifact hash."""
    directory = _model_dir(key, models_dir)
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / ARTIFACT_FILE
    tmp = artifact.with_suffix(".joblib.tmp")
    joblib.dump(pipeline, tmp, compress=3)
    tmp.replace(artifact)
    metadata = dict(metadata)
    metadata["artifact"] = {
        "file": ARTIFACT_FILE,
        "sha256": file_sha256(artifact),
        "format_version": config.ARTIFACT_FORMAT_VERSION,
        "size_bytes": artifact.stat().st_size,
    }
    write_json(directory / METADATA_FILE, metadata)
    logger.info("Saved model '%s' to %s", key, directory)
    return metadata


def load_metadata(key: str, models_dir: Path | None = None) -> dict[str, Any]:
    """Read one model's metadata, raising ``ModelStoreError`` if absent or unreadable."""
    path = _model_dir(key, models_dir) / METADATA_FILE
    if not path.exists():
        raise ModelStoreError(f"No saved metadata for model '{key}'.")
    try:
        return read_json(path)
    except (ValueError, OSError) as exc:
        raise ModelStoreError(f"Metadata for '{key}' is unreadable: {exc}") from exc


def artifact_problems(meta: dict[str, Any], models_dir: Path | None = None) -> list[str]:
    """Reasons a saved model cannot be used (empty list means it is loadable)."""
    problems: list[str] = []
    artifact_info = meta.get("artifact", {})
    if artifact_info.get("format_version") != config.ARTIFACT_FORMAT_VERSION:
        problems.append("Saved with an incompatible RoadGuard artifact format; please retrain.")
    try:
        path = _model_dir(meta["model"]["key"], models_dir) / ARTIFACT_FILE
    except (KeyError, ModelStoreError) as exc:
        return [f"Malformed metadata: {exc}"]
    if not path.exists():
        problems.append("The pipeline file is missing.")
    elif file_sha256(path) != artifact_info.get("sha256"):
        problems.append("The pipeline file does not match its recorded SHA-256 (modified or corrupted).")
    return problems


def version_warnings(meta: dict[str, Any]) -> list[str]:
    """Warnings when the model was trained under a different scikit-learn version."""
    trained = meta.get("environment", {}).get("sklearn")
    if trained and trained != sklearn.__version__:
        return [
            f"Trained with scikit-learn {trained}, running {sklearn.__version__}. "
            "Retrain if predictions look wrong."
        ]
    return []


def list_models(models_dir: Path | None = None) -> list[dict[str, Any]]:
    """Metadata of all saved models (newest first), each with ``problems`` and ``warnings``."""
    models_dir = _root(models_dir)
    if not models_dir.exists():
        return []
    found = []
    for directory in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        if not _KEY_PATTERN.match(directory.name) or not (directory / METADATA_FILE).exists():
            continue
        try:
            meta = load_metadata(directory.name, models_dir)
        except ModelStoreError as exc:
            logger.warning("Skipping model folder %s: %s", directory, exc)
            continue
        meta["problems"] = artifact_problems(meta, models_dir)
        meta["warnings_runtime"] = version_warnings(meta)
        found.append(meta)
    found.sort(key=lambda m: m.get("training", {}).get("trained_at", ""), reverse=True)
    return found


def load_pipeline(key: str, models_dir: Path | None = None) -> tuple[Any, dict[str, Any]]:
    """Load a verified pipeline and its metadata.

    Raises:
        ModelStoreError: if the artifact is missing, altered or incompatible.
    """
    meta = load_metadata(key, models_dir)
    problems = artifact_problems(meta, models_dir)
    if problems:
        raise ModelStoreError(" ".join(problems))
    path = _model_dir(key, models_dir) / ARTIFACT_FILE
    try:
        pipeline = joblib.load(path)
    except Exception as exc:  # unpickling errors vary by library version
        raise ModelStoreError(f"Could not load model '{key}': {type(exc).__name__}: {exc}") from exc
    return pipeline, meta


def delete_model(key: str, models_dir: Path | None = None) -> None:
    """Remove one model's folder (only the files this app writes) and unset it as champion."""
    directory = _model_dir(key, models_dir)
    if not directory.exists():
        return
    for name in (ARTIFACT_FILE, METADATA_FILE, ARTIFACT_FILE + ".tmp", METADATA_FILE + ".tmp"):
        target = directory / name
        if target.exists():
            target.unlink()
    try:
        directory.rmdir()
    except OSError:
        logger.warning("Left %s in place because it contains files RoadGuard did not create.", directory)
    champion = get_champion(models_dir)
    if champion and champion.get("key") == key:
        (_root(models_dir) / CHAMPION_FILE).unlink(missing_ok=True)
    logger.info("Deleted model '%s'", key)


def delete_all_models(models_dir: Path | None = None) -> int:
    """Delete every saved model; returns how many were removed."""
    keys = [m["model"]["key"] for m in list_models(models_dir)]
    for key in keys:
        delete_model(key, models_dir)
    (_root(models_dir) / CHAMPION_FILE).unlink(missing_ok=True)
    return len(keys)


def get_champion(models_dir: Path | None = None) -> dict[str, Any] | None:
    """Champion pointer ``{"key", "metric", "basis", "selected_at"}`` or None."""
    path = _root(models_dir) / CHAMPION_FILE
    if not path.exists():
        return None
    try:
        data = read_json(path)
    except (ValueError, OSError):
        return None
    key = data.get("key")
    if not key or not (_root(models_dir) / key / METADATA_FILE).exists():
        return None
    return data


def set_champion(key: str, metric: str, basis: str, models_dir: Path | None = None) -> None:
    """Mark ``key`` as the champion model used by default for predictions."""
    if not (_model_dir(key, models_dir) / METADATA_FILE).exists():
        raise ModelStoreError(f"Cannot select '{key}' as champion: it is not saved.")
    write_json(
        _root(models_dir) / CHAMPION_FILE,
        {"key": key, "metric": metric, "basis": basis, "selected_at": datetime.now(timezone.utc).isoformat()},
    )


def missing_features(meta: dict[str, Any], columns: list[str]) -> list[str]:
    """Input features the model needs that ``columns`` does not provide."""
    required = meta.get("feature_schema", {}).get("input_names", [])
    return [name for name in required if name not in columns]
