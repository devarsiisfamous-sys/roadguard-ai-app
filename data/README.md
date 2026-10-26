# Data folder

| Path | Contents |
|---|---|
| `sample_road_accidents.csv` | **Synthetic** demonstration data (6,040 rows, seed 42). Not real accident records. |
| `raw/` | Optional place to keep your own source CSVs. The app never writes here. |
| `processed/` | Cleaned copies saved from the Data Explorer (new timestamped files only; nothing is overwritten). |

## About the synthetic dataset

Every row is a simulated observation of one vehicle on a road segment, with columns
`date, time, weather, speed, driver_age, vehicle_type, road_condition, traffic_density,
light_condition, road_segment, accident, record_source` (`record_source = synthetic_demo`).

Accident outcomes are random draws from a hand-written logistic risk formula. The risk
rises with speed, rain/fog/snow, wet/icy/damaged roads, darkness, high traffic, very young
or old drivers, two-wheelers and weekend nights, plus noise. Models trained on this file can
only rediscover those assumptions; their scores say nothing about real-world performance.

Data-quality problems are injected **on purpose** so the quality checks have something to find:

- About 1–2% missing values in `weather`, `driver_age`, `light_condition`, `traffic_density`, and 0.4% in `accident`
- 8 implausible driver ages (5, 130, -1) and 5 implausible speeds (-10, 400 km/h)
- About 1% of weather values with inconsistent case/whitespace (e.g. `" rain "`)
- 40 exact duplicate rows

Regenerate it (same seed, same file) with:

```bash
python -m src.synthetic_data            # default: 6,000 rows + injected issues
python -m src.synthetic_data --clean    # without injected data-quality issues
```
