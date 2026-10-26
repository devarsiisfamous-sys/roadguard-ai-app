# RoadGuard AI: Road Accident Risk Prediction and Analytics System

RoadGuard AI is an interactive Streamlit application. It analyses road and traffic observations, trains
and compares machine-learning classifiers, and estimates accident risk for user-described road, traffic,
weather and driving conditions. It implements the presentation *Road Accident Prediction Model Using AI,
Machine Learning & Data Science* (Harsh Rana, Shubham Dalal, Aryan Gupta, Devarsi Roy).

> **Safety disclaimer.** RoadGuard AI is a research and decision-support prototype, **not** a certified
> accident-warning system. Predictions are statistical estimates learned from historical or synthetic data.
> They may be wrong or uncalibrated, and must not be the sole basis for driving, emergency-response or
> traffic-management decisions.

---

## Key features

| Page | What it does |
|---|---|
| **Overview** | Project summary, data-source status (synthetic vs user data), key statistics, model status, quick links. |
| **Accident Analytics** | Filterable dashboard: accident **rates** (with 95% confidence intervals) or counts by weather, road condition, traffic density, light, speed band, driver age, vehicle type, hour, weekday/weekend and month; outcome distributions; correlation matrix; downloadable summaries and filtered records. |
| **Risk Prediction** | Form generated from the trained model's feature schema. Shows estimated probability, a Low/Moderate/High/Very High band (adjustable cut-offs), the model's classification, a per-scenario factor analysis and condition-based safety reminders. Also includes scenario comparison, scoring the whole dataset, and a downloadable prediction history. |
| **Model Laboratory** | Train Logistic Regression, Decision Tree, Random Forest, XGBoost and SVM with leakage-safe pipelines; configure test size, seed, CV folds, class weighting, calibration, decision threshold, grid-search tuning, hyperparameters and feature-engineering assumptions; compare models; pick a champion; view confusion matrices, ROC/PR curves, CV results, permutation importance, coefficients and tree rules; download reports; manage saved models. |
| **Data Explorer** | Use the bundled synthetic dataset or upload a CSV; map columns to the schema; inspect missing values, duplicates, data types, invalid values and the target distribution; re-clean; download the cleaned data. |
| **About & Methodology** | Objectives, algorithms, data, evaluation methodology, interpretation guidance, limitations. |

## Technology stack

Python 3.11+ · Streamlit · pandas · NumPy · scikit-learn · XGBoost (optional) · Plotly (interactive charts) ·
Matplotlib + Seaborn (static report figures) · joblib (model persistence) · pytest.
No API keys, cloud services or databases are needed.

## Project structure

```
roadguard-ai/
├── app.py                      # entry point: theme, navigation, Overview dashboard
├── config.py                   # paths, schema, column aliases, thresholds, disclaimers
├── requirements.txt
├── pytest.ini
├── .streamlit/config.toml      # dark navy theme
├── data/
│   ├── sample_road_accidents.csv   # SYNTHETIC demo data (see data/README.md)
│   ├── raw/  processed/
├── models/                     # saved pipelines + metadata (created by training)
├── reports/                    # saved Markdown/PNG model reports
├── pages/
│   ├── 1_Accident_Analytics.py
│   ├── 2_Risk_Prediction.py
│   ├── 3_Model_Laboratory.py
│   ├── 4_Data_Explorer.py
│   └── 5_About.py
├── src/
│   ├── data_loader.py          # CSV reading, column mapping, target parsing
│   ├── preprocessing.py        # quality report, cleaning, ColumnTransformer, splitting
│   ├── feature_engineering.py  # peak hour, weekend, holiday, rain, speed band, time of day
│   ├── eda.py                  # rates with CIs, temporal tables, correlations, filters
│   ├── train.py                # model registry, pipelines, CV, tuning, calibration, importance
│   ├── evaluate.py             # metrics, curves, thresholds, comparison tables, reports
│   ├── predict.py              # scenario validation, prediction, risk bands, factor analysis
│   ├── model_store.py          # verified save/load/delete of artifacts, champion pointer
│   ├── synthetic_data.py       # reproducible synthetic dataset generator
│   ├── charts.py / static_charts.py / theme.py   # Plotly, Matplotlib/Seaborn, colour tokens
│   ├── state.py / ui.py        # Streamlit session state, caching and shared UI components
│   └── utils.py
└── tests/                      # pytest suite (unit + headless UI tests)
```

## Setup

Prerequisites: Python 3.11 or newer.

```bash
cd roadguard-ai
python -m venv .venv

# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

python -m pip install -r requirements.txt
```

**XGBoost on macOS** also needs the OpenMP runtime: `brew install libomp`. Without it the app shows
"XGBoost is not available" in the Model Laboratory and trains the other four models normally.

## Run

```bash
python -m streamlit run app.py
```

The app opens at http://localhost:8501 with the synthetic demo dataset already loaded.

## Typical workflow

1. **Data Explorer** (optional): check data quality, or upload your own CSV and map its columns.
2. **Model Laboratory → Train**: keep the defaults and click **Train selected models**. On the demo data,
   four models take about 30–40 seconds. Models are saved to `models/` and reloaded automatically next time.
3. **Compare & select**: compare cross-validated and test metrics; the champion is the best **mean
   cross-validated** score for the selected metric. You can override it.
4. **Risk Prediction**: enter conditions, click **Estimate accident risk**, add scenarios to the comparison,
   and download the prediction history.
5. **Accident Analytics**: explore descriptive patterns with filters and download summaries.

## Dataset schema and column mapping

| Field | Kind | Notes |
|---|---|---|
| `date` | date | Any common format; invalid dates become missing. |
| `time` | time | `14:30`, `2:30 PM`, `1430`, `14`; invalid times become missing. |
| `weather` | categorical | e.g. Clear, Cloudy, Rain, Fog, Snow |
| `speed` | numeric (km/h) | Plausible range 0–250 |
| `driver_age` | numeric (years) | Plausible range 14–100 |
| `vehicle_type` | categorical | e.g. Car, Motorcycle, Truck, Bus, Van, Bicycle |
| `road_condition` | categorical | e.g. Dry, Wet, Icy, Damaged |
| `traffic_density` | categorical | e.g. Low, Medium, High |
| `light_condition` | categorical | e.g. Daylight, Dusk/Dawn, Dark - Street Lights, Dark - No Lights |
| `road_segment` | identifier (optional) | Not a model input by default |
| `record_source` | metadata (optional) | Provenance; never a model input |
| `accident` | **target** | Yes/No, 1/0, True/False, Y/N (case-insensitive). 1 = accident = positive class. |

Uploaded columns are auto-matched by name and common aliases (e.g. `Weather_Condition`, `Speed (km/h)`,
`Age`, `Accident Occurred`). You can change every mapping in the UI. **Unmapped columns are dropped** so
that after-the-event fields such as casualties or severity cannot leak the outcome into a model.

Training needs a target with both classes, at least 50 labelled rows and at least 10 rows in the rarer
class. A file containing only accident reports cannot estimate accident probability. In that case the
app keeps exploratory analysis available and explains why training is disabled.

## Methodology in brief

- **Cleaning** (whole dataset, deterministic): type conversion, category spelling normalisation,
  implausible numbers set to missing, exact duplicates removed, rows without a label dropped.
  **No imputation**: nothing is invented.
- **Split first**: reproducible stratified train/test split. The test set is used only for final evaluation
  and permutation importance.
- **Pipeline per model**: `FeatureEngineer → ColumnTransformer (median / "Missing" imputation, one-hot
  encoding, scaling only for Logistic Regression and SVM) → classifier`. Everything is fitted inside
  training folds, so there is no leakage.
- **Imbalance**: optional balanced class weights (no resampling).
- **Tuning** (optional): small grid search with stratified CV on the training split.
- **Calibration** (optional; always sigmoid for SVM, since `SVC` has no native probabilities):
  `CalibratedClassifierCV` with CV on the training split.
- **Decision threshold**: fixed, or chosen to maximise F1 on out-of-fold *training* predictions.
- **Feature engineering**: peak hours (default 07–10 and 17–20), weekend, month, day of week, time of day,
  rain indicator, speed band (default edges 40/70/100 km/h), and a holiday flag only when you supply
  holiday dates. All are configurable in the Model Laboratory.
- **Persistence**: the full fitted pipeline is saved with joblib, plus `metadata.json` (feature schema,
  settings, metrics, curves, importance, environment and the SHA-256 of the artifact). Artifacts are loaded
  only from `models/` and only when the hash matches. There is no option to upload model files.

## Interpreting the metrics

- **Recall**: share of real accidents the model flags (missed accidents are false negatives).
- **Precision**: share of flagged cases that were accidents (false alarms are false positives).
- **F1**: balance of the two; the default champion metric.
- **ROC-AUC**: ranking quality, independent of the threshold. **Average precision**: area under the
  precision-recall curve, more informative when accidents are rare.
- **Accuracy** is misleading with rare accidents and is shown only for completeness.
- **Probabilities** from class-weighted or uncalibrated models are relative risk scores. Even calibrated
  probabilities are only as valid as the data they were calibrated on.
- **Feature importance and factor analysis** show associations the model learned, **not causes**.

On the synthetic demo data, expect modest scores (for example ROC-AUC around 0.7 and F1 around 0.4). The
simulated outcomes include deliberate randomness, so no model can be near-perfect. The app always shows
the metrics actually measured; nothing is hard-coded.

## Downloads

- Cleaned dataset CSV (Data Explorer) and an optional copy saved to `data/processed/`
- Filtered records and every chart's summary table (Accident Analytics)
- Model comparison CSV and long-format evaluation metrics CSV (Model Laboratory → Compare & select)
- Per-model Markdown report, metrics CSV and PNG evaluation figure (Model Laboratory → Evaluation),
  optionally saved to `reports/`
- Permutation importance CSV (Explainability)
- Prediction history, scenario comparison and scored-dataset CSVs (Risk Prediction)

## Deploying to Streamlit Community Cloud

1. Push this folder to a GitHub repository (the folder containing `app.py` should be the repository root).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, click **Create app → Deploy a public app from GitHub**.
3. Choose the repository, branch `main` and main file `app.py`, then **Deploy**.

`requirements.txt` installs the Python packages and `packages.txt` installs the OpenMP runtime (`libgomp1`) so
XGBoost works on the Linux servers. The cloud file system is **ephemeral**: models trained in the deployed app are
kept until the app restarts or goes to sleep, after which you need to train again (about 30–40 seconds).

## Testing

```bash
python -m pytest -q
```

The suite covers CSV loading and validation, column mapping, target parsing, missing values, duplicates,
cleaning, feature engineering, splitting, leakage-safe preprocessing, training, tuning, calibration,
single-class targets, invalid requests, missing optional dependencies (simulated XGBoost failure), model
persistence, tamper detection, deletion, prediction, risk bands and factor analysis. It also includes
headless UI tests (Streamlit `AppTest`) that render every page and run the full **train → predict** workflow
through the interface. Tests write models to temporary folders, never to `models/`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `XGBoost is not available … OpenMP runtime is missing` | macOS: `brew install libomp`, then restart the app. Other models work without it. |
| `The demo dataset is missing` | `python -m src.synthetic_data` |
| "Supervised training is disabled" | Map an accident (Yes/No) column that contains both outcomes, with at least 50 rows. |
| A saved model is shown as *Unusable* | The file changed or was saved in an older format; delete it in **Saved models** and retrain. |
| "Trained with scikit-learn X, running Y" | Retrain the models after upgrading packages. |
| `ModuleNotFoundError: src` / `config` | Run commands from the `roadguard-ai/` folder. |
| Upload rejected | Use a .csv with a header row (comma, semicolon or tab separated), under 200 MB. |

## Limitations

- The bundled data is synthetic; results do not transfer to real roads.
- Real accident data is often incomplete and biased toward reported incidents, and usually lacks
  exposure (traffic volume). Without proper non-accident observations, probabilities are not meaningful.
- The app predicts user-described scenarios; there are no real-time feeds.
- Future scope (from the presentation): IoT sensors, GPS tracking, deep learning, real-time prediction and
  smart traffic-system integration. The schema-mapping layer is the extension point for new data sources.
