"""Central configuration for RoadGuard AI.

Every path is derived from the location of this file so the project runs from
any working directory on macOS, Windows and Linux. Thresholds that shape the
analysis (peak hours, speed bands, risk bands) live here so they are documented
in one place and can be overridden from the UI where that is supported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
SAMPLE_DATA_PATH: Path = DATA_DIR / "sample_road_accidents.csv"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

APP_NAME = "RoadGuard AI"
APP_TAGLINE = "Road Accident Risk Prediction and Analytics System"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED: int = 42
SAMPLE_ROWS: int = 6000

# --------------------------------------------------------------------------- #
# Canonical schema
# --------------------------------------------------------------------------- #
TARGET_COLUMN = "accident"
POSITIVE_CLASS = 1
CLASS_LABELS: dict[int, str] = {0: "No accident", 1: "Accident"}

# kind: date | time | categorical | numeric | identifier | metadata | target
SCHEMA: dict[str, dict[str, Any]] = {
    "date": {
        "kind": "date",
        "label": "Date",
        "description": "Calendar date of the observation (any common date format).",
    },
    "time": {
        "kind": "time",
        "label": "Time",
        "description": "Time of day, e.g. 14:30, 2:30 PM or 1430.",
    },
    "weather": {
        "kind": "categorical",
        "label": "Weather",
        "description": "Weather at the time, e.g. Clear, Cloudy, Rain, Fog, Snow.",
    },
    "speed": {
        "kind": "numeric",
        "label": "Speed",
        "unit": "km/h",
        "valid_range": (0.0, 250.0),
        "description": "Vehicle speed in km/h.",
    },
    "driver_age": {
        "kind": "numeric",
        "label": "Driver age",
        "unit": "years",
        "valid_range": (14.0, 100.0),
        "description": "Age of the driver in years.",
    },
    "vehicle_type": {
        "kind": "categorical",
        "label": "Vehicle type",
        "description": "Car, Motorcycle, Truck, Bus, Van, Bicycle, ...",
    },
    "road_condition": {
        "kind": "categorical",
        "label": "Road condition",
        "description": "Dry, Wet, Icy, Damaged, ...",
    },
    "traffic_density": {
        "kind": "categorical",
        "label": "Traffic density",
        "description": "Low, Medium, High.",
    },
    "light_condition": {
        "kind": "categorical",
        "label": "Light condition",
        "description": "Daylight, Dusk/Dawn, Dark - Street Lights, Dark - No Lights.",
    },
    "road_segment": {
        "kind": "identifier",
        "label": "Road segment",
        "description": "Optional location / road-segment identifier (not used as a model feature by default).",
    },
    "record_source": {
        "kind": "metadata",
        "label": "Record source",
        "description": "Provenance label, e.g. 'synthetic_demo'. Never used as a model feature.",
    },
    TARGET_COLUMN: {
        "kind": "target",
        "label": "Accident (Yes/No)",
        "description": "Target: whether an accident occurred. Accepts Yes/No, 1/0, True/False.",
    },
}

FEATURE_KINDS = ("date", "time", "categorical", "numeric")
# Columns that may be used as model inputs (identifiers can be opted in).
CANDIDATE_FEATURES: tuple[str, ...] = tuple(
    name for name, spec in SCHEMA.items() if spec["kind"] in FEATURE_KINDS
)
DEFAULT_EXCLUDED_FEATURES: tuple[str, ...] = ("road_segment",)

# Alternative column names recognised when auto-mapping an uploaded CSV. Names
# are compared after lower-casing and replacing non-alphanumerics with "_".
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "date": ("date", "accident_date", "crash_date", "obs_date", "day", "datetime", "date_time", "timestamp"),
    "time": ("time", "accident_time", "crash_time", "time_of_accident", "hour_minute", "clock_time"),
    "weather": ("weather", "weather_condition", "weather_conditions", "weather_type", "climate"),
    "speed": ("speed", "speed_kmh", "speed_km_h", "vehicle_speed", "speed_limit", "avg_speed", "speed_kph"),
    "driver_age": ("driver_age", "age", "age_of_driver", "driverage", "driver_age_years", "age_driver"),
    "vehicle_type": ("vehicle_type", "vehicle", "type_of_vehicle", "vehicle_class", "vehicle_category"),
    "road_condition": (
        "road_condition", "road_conditions", "road_surface", "road_surface_conditions",
        "surface_condition", "road_state",
    ),
    "traffic_density": ("traffic_density", "traffic", "traffic_level", "traffic_volume", "congestion", "traffic_condition"),
    "light_condition": ("light_condition", "light_conditions", "lighting", "lighting_condition", "light", "visibility"),
    "road_segment": ("road_segment", "segment", "segment_id", "location", "location_id", "road_id", "road_name"),
    "record_source": ("record_source", "source", "data_source"),
    TARGET_COLUMN: (
        "accident", "accident_occurred", "is_accident", "accident_yes_no", "crash", "crash_occurred",
        "target", "label", "accident_flag",
    ),
}

TARGET_TRUE_VALUES = frozenset({"yes", "y", "1", "1.0", "true", "t", "accident", "accident occurred", "crash"})
TARGET_FALSE_VALUES = frozenset({"no", "n", "0", "0.0", "false", "f", "no accident", "none", "no crash"})

# --------------------------------------------------------------------------- #
# Feature engineering defaults (documented assumptions)
# --------------------------------------------------------------------------- #
# Peak hours as half-open [start, end) ranges on a 24h clock: 07:00-09:59 and
# 17:00-19:59. These are common urban commuting windows, not a standard.
PEAK_HOUR_RANGES: tuple[tuple[int, int], ...] = ((7, 10), (17, 20))
# Speed band edges in km/h -> Low (<40), Moderate (40-69), High (70-99), Very high (>=100).
SPEED_BINS: tuple[float, ...] = (40.0, 70.0, 100.0)
RAIN_KEYWORDS: tuple[str, ...] = ("rain", "drizzle", "shower", "storm", "thunder")
# Time-of-day buckets keyed by inclusive start hour.
TIME_OF_DAY_BUCKETS: tuple[tuple[int, str], ...] = (
    (0, "Night"),
    (5, "Early morning"),
    (8, "Morning"),
    (12, "Afternoon"),
    (17, "Evening"),
    (21, "Night"),
)
DAY_NAMES: tuple[str, ...] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

# --------------------------------------------------------------------------- #
# Modelling / evaluation
# --------------------------------------------------------------------------- #
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CV_FOLDS = 5
DECISION_THRESHOLD = 0.5
MIN_ROWS_FOR_TRAINING = 50
MIN_MINORITY_FOR_TRAINING = 10
SEVERE_IMBALANCE_SHARE = 0.10
SMALL_TEST_POSITIVES = 30
ARTIFACT_FORMAT_VERSION = 1

# Risk bands map a probability to a label. Defaults are illustrative cut-offs,
# NOT validated road-safety standards; users can change them in the UI.
RISK_BANDS: tuple[str, ...] = ("Low", "Moderate", "High", "Very High")
DEFAULT_RISK_THRESHOLDS: tuple[float, float, float] = (0.25, 0.50, 0.75)

SAFETY_DISCLAIMER = (
    "RoadGuard AI is a research and decision-support prototype, not a certified accident-warning "
    "system. Predictions are statistical estimates learned from historical or synthetic data, may be "
    "wrong or uncalibrated, and must not be the sole basis for driving, emergency-response or "
    "traffic-management decisions."
)
