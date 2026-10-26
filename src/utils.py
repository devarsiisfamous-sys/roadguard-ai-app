"""Framework-independent helpers shared across RoadGuard AI modules."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_LOGGING_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring a console handler once per process."""
    global _LOGGING_CONFIGURED
    if not _LOGGING_CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        _LOGGING_CONFIGURED = True
    return logging.getLogger(name)


def normalize_name(name: object) -> str:
    """Lower-case a column name and collapse non-alphanumerics to underscores."""
    text = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower())
    return text.strip("_")


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """Short, order-sensitive content hash used to tell datasets apart."""
    hashed = pd.util.hash_pandas_object(df, index=False).to_numpy()
    digest = hashlib.sha256(hashed.tobytes())
    digest.update("|".join(map(str, df.columns)).encode())
    return digest.hexdigest()[:16]


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Encode a DataFrame as UTF-8 CSV bytes for download buttons."""
    return df.to_csv(index=False).encode("utf-8")


def to_jsonable(value: Any) -> Any:
    """Recursively convert numpy / pandas scalars and containers to JSON types."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically (temp file + replace) so a crash never leaves half a file."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object.")
    return data


def format_pct(value: float | None, digits: int = 1) -> str:
    """Format a 0-1 share as a percentage string, or an em dash when missing."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value * 100:.{digits}f}%"


def format_metric(value: float | None, digits: int = 3) -> str:
    """Format a metric value, or an em dash when it could not be computed."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}"


def wilson_interval(successes: np.ndarray, totals: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """95% Wilson score interval for binomial proportions (safe for small n).

    Returns arrays of (lower, upper); entries with ``totals == 0`` are NaN.
    """
    successes = np.asarray(successes, dtype=float)
    totals = np.asarray(totals, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = successes / totals
        denom = 1 + z**2 / totals
        centre = (p + z**2 / (2 * totals)) / denom
        half = z * np.sqrt(p * (1 - p) / totals + z**2 / (4 * totals**2)) / denom
    lower = np.where(totals > 0, np.clip(centre - half, 0, 1), np.nan)
    upper = np.where(totals > 0, np.clip(centre + half, 0, 1), np.nan)
    return lower, upper
