"""Loading CSV data, mapping columns to the canonical schema and parsing the target.

Uploaded files are only ever read into memory; nothing here writes to or
overwrites a user's source file.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

import config
from src.utils import get_logger, normalize_name

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = (".csv",)
MAX_UPLOAD_BYTES = 200 * 1024 * 1024


class DataLoadError(Exception):
    """Raised when a dataset cannot be read or does not fit the expected shape."""


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def _decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DataLoadError("The file could not be decoded as UTF-8 or Latin-1 text.")


def _sniff_delimiter(text: str) -> str:
    sample = "\n".join(text.splitlines()[:20])
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def read_csv_bytes(raw: bytes, filename: str | None = None) -> pd.DataFrame:
    """Parse CSV content into a DataFrame with actionable errors.

    Args:
        raw: File content.
        filename: Original file name, used to check the extension.

    Raises:
        DataLoadError: For unsupported types, empty files, malformed CSV or
            files without at least two columns and one data row.
    """
    if filename is not None and Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise DataLoadError(
            f"Unsupported file type '{Path(filename).suffix or 'none'}'. Please upload a .csv file."
        )
    if len(raw) == 0 or not raw.strip():
        raise DataLoadError("The file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise DataLoadError("The file is larger than 200 MB; please upload a smaller extract.")

    text = _decode(raw)
    delimiter = _sniff_delimiter(text)
    try:
        df = pd.read_csv(io.StringIO(text), sep=delimiter, skipinitialspace=True)
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError("The file contains no columns or data.") from exc
    except pd.errors.ParserError as exc:
        raise DataLoadError(f"The CSV appears to be malformed: {exc}".strip()) from exc

    df.columns = [str(c).strip() for c in df.columns]
    if df.shape[1] < 2:
        raise DataLoadError(
            "Only one column was found. Check that the file is comma-, semicolon- or tab-separated."
        )
    if df.empty:
        raise DataLoadError("The file has a header row but no data rows.")
    if any(c.startswith("Unnamed:") for c in df.columns) and df.shape[1] > 1:
        unnamed = [c for c in df.columns if c.startswith("Unnamed:")]
        if all(df[c].isna().all() for c in unnamed):
            df = df.drop(columns=unnamed)
    logger.info("Read CSV %s: %d rows x %d columns", filename or "<bytes>", *df.shape)
    return df


def read_csv_file(source: str | Path | BinaryIO, filename: str | None = None) -> pd.DataFrame:
    """Read a CSV from a path or a binary file-like object (e.g. a Streamlit upload)."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise DataLoadError(f"File not found: {path}")
        return read_csv_bytes(path.read_bytes(), filename or path.name)
    return read_csv_bytes(source.read(), filename or getattr(source, "name", None))


def load_sample_dataset(path: Path = config.SAMPLE_DATA_PATH) -> pd.DataFrame:
    """Load the bundled synthetic demonstration CSV."""
    if not Path(path).exists():
        raise DataLoadError(
            f"The demo dataset is missing at {path}. Regenerate it with: python -m src.synthetic_data"
        )
    return read_csv_file(path)


# --------------------------------------------------------------------------- #
# Column mapping
# --------------------------------------------------------------------------- #
def suggest_column_mapping(columns: list[str]) -> dict[str, str | None]:
    """Suggest which source column feeds each canonical field.

    Exact canonical-name matches win, then the alias lists in ``config``. Each
    source column is used at most once.
    """
    normalized = {normalize_name(c): c for c in columns}
    used: set[str] = set()
    mapping: dict[str, str | None] = {}
    for canonical in config.SCHEMA:
        choice = None
        for alias in (canonical, *config.COLUMN_ALIASES.get(canonical, ())):
            source = normalized.get(normalize_name(alias))
            if source is not None and source not in used:
                choice = source
                break
        mapping[canonical] = choice
        if choice is not None:
            used.add(choice)
    return mapping


def apply_column_mapping(
    df: pd.DataFrame, mapping: dict[str, str | None]
) -> tuple[pd.DataFrame, list[str]]:
    """Rename mapped columns to canonical names and keep only those columns.

    Unmapped source columns are dropped on purpose. Columns recorded after the
    event (casualty counts, severity, police notes) would leak the outcome
    into the model, so only explicitly mapped schema fields are kept.

    Returns:
        The mapped copy and the list of dropped source columns.
    """
    selected = {canon: src for canon, src in mapping.items() if src}
    unknown = [canon for canon in selected if canon not in config.SCHEMA]
    if unknown:
        raise DataLoadError(f"Unknown schema fields in mapping: {', '.join(unknown)}")
    sources = list(selected.values())
    duplicates = sorted({s for s in sources if sources.count(s) > 1})
    if duplicates:
        raise DataLoadError(
            f"Each source column can be mapped to only one field; reused: {', '.join(duplicates)}"
        )
    missing = [s for s in sources if s not in df.columns]
    if missing:
        raise DataLoadError(f"Mapped columns not found in the file: {', '.join(missing)}")
    if not selected:
        raise DataLoadError("No columns are mapped to the RoadGuard schema.")

    mapped = df[sources].copy()
    mapped.columns = list(selected.keys())
    dropped = [c for c in df.columns if c not in sources]
    return mapped, dropped


# --------------------------------------------------------------------------- #
# Target parsing
# --------------------------------------------------------------------------- #
def parse_target(series: pd.Series) -> tuple[pd.Series, int]:
    """Convert accident indicators (Yes/No, 1/0, True/False, ...) to nullable 0/1.

    Returns:
        (parsed ``Int64`` series, number of non-empty values that were not
        recognised and therefore became missing).
    """
    def convert(value: object) -> object:
        if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
            return pd.NA
        if isinstance(value, (bool, np.bool_)):
            return int(value)
        text = str(value).strip().lower()
        if text == "":
            return pd.NA
        if text in config.TARGET_TRUE_VALUES:
            return 1
        if text in config.TARGET_FALSE_VALUES:
            return 0
        return "INVALID"

    converted = series.map(convert)
    n_invalid = int((converted == "INVALID").sum())
    converted = converted.where(converted != "INVALID", pd.NA)
    return converted.astype("Int64"), n_invalid


@dataclass
class TargetAssessment:
    """Whether the target column supports supervised training, and why not."""

    present: bool
    usable: bool
    reason: str
    n_valid: int = 0
    n_missing: int = 0
    n_invalid: int = 0
    class_counts: dict[int, int] = field(default_factory=dict)

    @property
    def positive_share(self) -> float | None:
        if self.n_valid == 0:
            return None
        return self.class_counts.get(1, 0) / self.n_valid

    @property
    def minority_share(self) -> float | None:
        if self.n_valid == 0 or not self.class_counts:
            return None
        return min(self.class_counts.get(0, 0), self.class_counts.get(1, 0)) / self.n_valid


def assess_target(df: pd.DataFrame, n_invalid: int = 0) -> TargetAssessment:
    """Check that ``df`` has a parsed binary target with enough of both classes."""
    target = config.TARGET_COLUMN
    if target not in df.columns:
        return TargetAssessment(
            present=False,
            usable=False,
            reason="No accident (Yes/No) column is mapped, so supervised model training is disabled. "
            "Exploratory analysis of the other columns is still available.",
        )
    series = df[target]
    valid = series.dropna()
    counts = {int(k): int(v) for k, v in valid.value_counts().items()}
    counts = {0: counts.get(0, 0), 1: counts.get(1, 0)}
    assessment = TargetAssessment(
        present=True,
        usable=True,
        reason="",
        n_valid=int(valid.shape[0]),
        n_missing=int(series.isna().sum()),
        n_invalid=n_invalid,
        class_counts=counts,
    )
    if assessment.n_valid == 0:
        assessment.usable = False
        assessment.reason = "The accident column has no recognisable Yes/No (1/0) values."
    elif min(counts.values()) == 0:
        only = config.CLASS_LABELS[1 if counts[1] else 0]
        assessment.usable = False
        assessment.reason = (
            f"Every labelled record is '{only}'. A classifier needs both accident and non-accident "
            "observations (datasets of accident reports alone cannot estimate accident probability)."
        )
    elif assessment.n_valid < config.MIN_ROWS_FOR_TRAINING:
        assessment.usable = False
        assessment.reason = (
            f"Only {assessment.n_valid} labelled rows; at least {config.MIN_ROWS_FOR_TRAINING} are required."
        )
    elif min(counts.values()) < config.MIN_MINORITY_FOR_TRAINING:
        assessment.usable = False
        assessment.reason = (
            f"The rarer class has only {min(counts.values())} rows; at least "
            f"{config.MIN_MINORITY_FOR_TRAINING} are required for a train/test split and cross-validation."
        )
    return assessment
