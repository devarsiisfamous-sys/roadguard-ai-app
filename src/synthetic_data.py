"""Generator for the bundled, clearly-labelled SYNTHETIC demonstration dataset.

The records are simulated, not real accident observations. Accident outcomes
come from a hand-written logistic risk function over the generated conditions
(speed, weather, road surface, light, traffic, driver age, vehicle type, time).
That makes the demo useful for exercising the pipeline. It also means any model
trained on it learns the simulator's assumptions, not real-world risk.

A small, documented share of data-quality problems (missing values, duplicates,
out-of-range numbers, inconsistent spelling) is injected on purpose so that the
data-quality and cleaning features have something real to report.

Run ``python -m src.synthetic_data`` from the project root to regenerate
``data/sample_road_accidents.csv``.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

WEATHER = np.array(["Clear", "Cloudy", "Rain", "Fog", "Snow"])
ROAD = np.array(["Dry", "Wet", "Icy", "Damaged"])
TRAFFIC = np.array(["Low", "Medium", "High"])
VEHICLES = np.array(["Car", "Motorcycle", "Truck", "Bus", "Van", "Bicycle"])
SYNTHETIC_SOURCE_LABEL = "synthetic_demo"


def _weather_probs(month: int) -> np.ndarray:
    """Seasonal weather mix: snow/fog in winter, rain in the monsoon months."""
    if month in (12, 1, 2):
        return np.array([0.40, 0.25, 0.10, 0.12, 0.13])
    if month in (6, 7, 8, 9):
        return np.array([0.35, 0.25, 0.34, 0.06, 0.00])
    return np.array([0.55, 0.25, 0.14, 0.06, 0.00])


ROAD_GIVEN_WEATHER: dict[str, np.ndarray] = {
    "Clear": np.array([0.86, 0.04, 0.00, 0.10]),
    "Cloudy": np.array([0.74, 0.13, 0.00, 0.13]),
    "Rain": np.array([0.05, 0.80, 0.00, 0.15]),
    "Fog": np.array([0.55, 0.33, 0.02, 0.10]),
    "Snow": np.array([0.02, 0.30, 0.63, 0.05]),
}

# Relative traffic by hour of day (two commuting peaks).
HOUR_WEIGHTS = np.array(
    [1.2, 0.8, 0.6, 0.5, 0.6, 1.2, 2.6, 4.6, 5.4, 4.4, 3.6, 3.7,
     4.0, 3.8, 3.7, 4.0, 4.8, 5.6, 5.4, 4.4, 3.2, 2.5, 2.0, 1.6]
)


def _light_condition(hour: int, rng: np.random.Generator) -> str:
    if 7 <= hour <= 16:
        return "Daylight"
    if hour in (6, 17, 18):
        return "Dusk/Dawn" if rng.random() < 0.7 else "Daylight"
    return "Dark - Street Lights" if rng.random() < 0.65 else "Dark - No Lights"


def _traffic_density(hour: int, weekend: bool, rng: np.random.Generator) -> str:
    peak = hour in (7, 8, 9, 17, 18, 19) and not weekend
    if peak:
        probs = [0.10, 0.35, 0.55]
    elif 10 <= hour <= 20:
        probs = [0.25, 0.50, 0.25]
    else:
        probs = [0.70, 0.25, 0.05]
    return str(rng.choice(TRAFFIC, p=probs))


def generate_synthetic_dataset(
    n_rows: int = config.SAMPLE_ROWS,
    seed: int = config.RANDOM_SEED,
    inject_quality_issues: bool = True,
) -> pd.DataFrame:
    """Create a reproducible synthetic road-observation dataset.

    Each row is one simulated observation of a vehicle on a road segment, with
    a binary ``accident`` outcome ("Yes"/"No").

    Args:
        n_rows: Number of base observations (duplicates are added on top when
            ``inject_quality_issues`` is true).
        seed: Seed for ``numpy.random.default_rng``; the same seed gives the
            same file.
        inject_quality_issues: Add a small, documented share of missing,
            invalid, inconsistently-spelled and duplicated records.

    Returns:
        DataFrame with the canonical schema plus ``record_source``.
    """
    if n_rows < 10:
        raise ValueError("n_rows must be at least 10.")
    rng = np.random.default_rng(seed)

    start = pd.Timestamp("2022-01-01")
    day_offsets = rng.integers(0, 1096, n_rows)
    dates = start + pd.to_timedelta(day_offsets, unit="D")
    hours = rng.choice(24, size=n_rows, p=HOUR_WEIGHTS / HOUR_WEIGHTS.sum())
    minutes = rng.integers(0, 60, n_rows)
    segments = rng.integers(1, 41, n_rows)
    segment_effect = rng.normal(0.0, 0.3, 41)

    rows: list[dict[str, object]] = []
    logits = np.empty(n_rows)
    for i in range(n_rows):
        date = dates[i]
        hour = int(hours[i])
        weekend = date.day_of_week >= 5
        weather = str(rng.choice(WEATHER, p=_weather_probs(date.month)))
        road = str(rng.choice(ROAD, p=ROAD_GIVEN_WEATHER[weather]))
        light = _light_condition(hour, rng)
        traffic = _traffic_density(hour, weekend, rng)
        vehicle = str(rng.choice(VEHICLES, p=[0.45, 0.22, 0.10, 0.06, 0.10, 0.07]))

        age_scale = 7.0 if vehicle in ("Motorcycle", "Bicycle") else 10.0
        age = float(np.clip(18 + rng.gamma(2.2, age_scale), 18, 88))

        base_speed = {"Low": 72.0, "Medium": 55.0, "High": 32.0}[traffic]
        base_speed += {"Truck": -8.0, "Bus": -10.0, "Motorcycle": 6.0, "Van": -3.0}.get(vehicle, 0.0)
        base_speed += {"Rain": -5.0, "Fog": -10.0, "Snow": -15.0}.get(weather, 0.0)
        base_speed += 5.0 if age < 25 else 0.0
        speed = float(rng.normal(base_speed, 13.0))
        if vehicle == "Bicycle":
            speed = float(rng.normal(18.0, 5.0))
        speed = float(np.clip(speed, 5, 160))

        night = hour >= 22 or hour < 4
        z = -3.3
        z += 0.035 * (speed - 50.0) + (0.6 if speed > 100 else 0.0)
        z += {"Cloudy": 0.10, "Rain": 0.55, "Fog": 0.80, "Snow": 0.90}.get(weather, 0.0)
        z += {"Wet": 0.50, "Icy": 1.10, "Damaged": 0.60}.get(road, 0.0)
        z += {"Medium": 0.25, "High": 0.55}.get(traffic, 0.0)
        z += {"Dusk/Dawn": 0.30, "Dark - Street Lights": 0.45, "Dark - No Lights": 0.95}.get(light, 0.0)
        z += 0.55 if age < 25 else (0.45 if age > 70 else 0.0)
        z += {"Motorcycle": 0.70, "Bicycle": 0.50, "Truck": 0.25, "Bus": 0.10, "Van": 0.05}.get(vehicle, 0.0)
        z += 0.50 if (weekend and night) else 0.0
        z += 0.50 if (speed > 80 and road in ("Wet", "Icy")) else 0.0
        z += segment_effect[segments[i]]
        z += rng.normal(0.0, 0.35)
        logits[i] = z

        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "time": f"{hour:02d}:{int(minutes[i]):02d}",
                "weather": weather,
                "speed": round(speed),
                "driver_age": round(age),
                "vehicle_type": vehicle,
                "road_condition": road,
                "traffic_density": traffic,
                "light_condition": light,
                "road_segment": f"SEG-{int(segments[i]):03d}",
            }
        )

    df = pd.DataFrame(rows)
    probs = 1.0 / (1.0 + np.exp(-logits))
    df[config.TARGET_COLUMN] = np.where(rng.random(n_rows) < probs, "Yes", "No")
    df["record_source"] = SYNTHETIC_SOURCE_LABEL

    if inject_quality_issues:
        df = _inject_quality_issues(df, rng)
    return df.reset_index(drop=True)


def _inject_quality_issues(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add a small, documented share of realistic data-quality problems."""
    df = df.copy()
    df["speed"] = df["speed"].astype(float)
    df["driver_age"] = df["driver_age"].astype(float)
    n = len(df)

    def pick(share: float) -> np.ndarray:
        return rng.choice(n, size=max(1, int(n * share)), replace=False)

    df.loc[pick(0.015), "weather"] = np.nan
    df.loc[pick(0.020), "driver_age"] = np.nan
    df.loc[pick(0.010), "light_condition"] = np.nan
    df.loc[pick(0.010), "traffic_density"] = np.nan
    df.loc[pick(0.004), config.TARGET_COLUMN] = np.nan

    df.loc[rng.choice(n, 8, replace=False), "driver_age"] = rng.choice([5.0, 130.0, -1.0], 8)
    df.loc[rng.choice(n, 5, replace=False), "speed"] = rng.choice([-10.0, 400.0], 5)

    messy = pick(0.01)
    df.loc[messy, "weather"] = df.loc[messy, "weather"].map(
        lambda v: f" {str(v).lower()} " if isinstance(v, str) else v
    )

    duplicates = df.iloc[rng.choice(n, 40, replace=False)]
    df = pd.concat([df, duplicates], ignore_index=True)
    order = rng.permutation(len(df))
    return df.iloc[order].reset_index(drop=True)


def main() -> None:
    """CLI entry point: write the sample CSV to ``data/``."""
    parser = argparse.ArgumentParser(description="Generate the synthetic RoadGuard AI demo dataset.")
    parser.add_argument("--rows", type=int, default=config.SAMPLE_ROWS)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--output", type=str, default=str(config.SAMPLE_DATA_PATH))
    parser.add_argument("--clean", action="store_true", help="Do not inject data-quality issues.")
    args = parser.parse_args()

    df = generate_synthetic_dataset(args.rows, args.seed, inject_quality_issues=not args.clean)
    df.to_csv(args.output, index=False)
    positive = (df[config.TARGET_COLUMN] == "Yes").mean()
    print(f"Wrote {len(df):,} synthetic rows to {args.output} (accident share {positive:.1%}).")


if __name__ == "__main__":
    main()
