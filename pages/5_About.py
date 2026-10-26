"""About & Methodology: objectives, algorithms, data, evaluation and limitations."""

from __future__ import annotations

import streamlit as st

import config
from src import train, ui
from src.feature_engineering import ENGINEERED_FEATURES, speed_category_labels

ui.setup_page("About & Methodology")
ui.page_header(
    "About & Methodology",
    "How RoadGuard AI works, what it can and cannot tell you, and how to use it responsibly.",
    "Documentation",
)
ui.disclaimer()

st.markdown("### Project objectives")
st.markdown(
    "Based on the presentation *Road Accident Prediction Model Using AI, Machine Learning & Data Science*:\n\n"
    "1. **Predict accident occurrence:** estimate accident likelihood for given road, traffic, weather and driving conditions.\n"
    "2. **Identify major risk factors:** show which inputs a model relies on and which patterns appear in the data.\n"
    "3. **Support traffic authorities:** compare scenarios to prioritise preventive measures.\n"
    "4. **Improve emergency response:** rank high-risk conditions for resource planning (decision support only).\n"
    "5. **Reduce fatalities:** this needs real data, validation and real-world action; the prototype is one step toward it."
)

st.markdown("### Data")
st.markdown(
    f"- **Schema:** {', '.join(f'`{c}`' for c in config.SCHEMA)}. Uploaded CSVs are mapped to these fields; "
    "unmapped columns are dropped to prevent target leakage from after-the-event fields.\n"
    f"- **Target:** `{config.TARGET_COLUMN}`, where 1 = accident (positive class) and 0 = no accident. A valid dataset needs "
    "observations both with and without accidents. A file of accident reports alone cannot estimate accident probability.\n"
    f"- **Demo data:** `{config.SAMPLE_DATA_PATH.name}` is **synthetic**. Outcomes were simulated from a hand-written "
    "logistic risk formula (speed, weather, road surface, light, traffic, driver age, vehicle type, weekend nights). "
    "About 1–2% missing values per affected column, 40 exact duplicates, a few implausible ages/speeds and "
    "inconsistent spellings were injected on purpose to exercise the quality checks. Models trained on it "
    "re-discover the simulator's assumptions; their scores say nothing about real-world performance.\n"
    "- **Cleaning** converts types, normalises category spelling, sets implausible numbers to missing and "
    "removes exact duplicates. It **never imputes**; imputation happens inside each model pipeline, fitted on "
    "training data only."
)

st.markdown("### Feature engineering")
peak = ", ".join(f"{s:02d}:00–{e:02d}:00" for s, e in config.PEAK_HOUR_RANGES)
st.markdown(
    "\n".join(f"- **{name}**: {desc}" for name, (_, _, desc) in ENGINEERED_FEATURES.items())
    + f"\n\nDefault assumptions (configurable in the Model Laboratory): peak hours {peak}; speed bands "
    f"{', '.join(speed_category_labels(config.SPEED_BINS))}; rain keywords {', '.join(config.RAIN_KEYWORDS)}. "
    "A holiday indicator is only created if you supply holiday dates. All features use only the observation's "
    "own fields, so no future information is used."
)

st.markdown("### Algorithms")
for spec in train.MODEL_SPECS.values():
    st.markdown(f"- **{spec.name}**: {spec.description} {spec.probability_note}")
ok, message = train.xgboost_status()
st.caption(("✅ " if ok else "ℹ️ ") + message)

st.markdown("### Training and evaluation methodology")
st.markdown(
    "1. **Split first:** a reproducible, stratified train/test split. The test set is used only for final "
    "evaluation and permutation importance.\n"
    "2. **One pipeline per model:** feature engineering → imputation → one-hot encoding → scaling (only for "
    "logistic regression and SVM) → classifier. All preprocessing is fitted inside training folds.\n"
    "3. **Imbalance:** optional balanced class weights (no resampling, so no synthetic rows enter training).\n"
    "4. **Tuning (optional):** grid search with stratified cross-validation on the training split only.\n"
    "5. **Calibration (optional, always for SVM):** `CalibratedClassifierCV` with cross-validation on the training split only.\n"
    "6. **Decision threshold:** fixed, or chosen to maximise F1 on out-of-fold *training* predictions.\n"
    "7. **Cross-validation metrics** are computed on the training split. The **champion** is the model with the best "
    "mean CV score for the chosen metric, never the best test score.\n"
    "8. **Test metrics:** accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, average precision, confusion "
    "matrix, ROC and precision-recall curves. ROC-AUC is reported as unavailable when the test set has one class.\n"
    "9. **Persistence:** the full fitted pipeline is saved with joblib, alongside metadata (schema, settings, metrics, "
    "environment and a SHA-256 hash checked before loading)."
)

st.markdown("### Interpreting the outputs")
st.markdown(
    "- **Accuracy alone is misleading** when accidents are rare: predicting 'no accident' for everything can score "
    "above 80%. Use recall (missed accidents), precision (false alarms), F1 and the precision-recall curve.\n"
    "- **Probabilities** from class-weighted or uncalibrated models are relative risk scores. Calibration "
    "improves them on the training distribution, but only validation on representative real data can justify reading "
    "them as real-world probabilities.\n"
    "- **Risk bands** (Low / Moderate / High / Very High) use adjustable cut-offs, default "
    f"{', '.join(f'{t:.0%}' for t in config.DEFAULT_RISK_THRESHOLDS)}. These are illustrative, not safety standards.\n"
    "- **Importance and factor analyses** describe associations the model learned. They are not causal effects."
)

st.markdown("### Limitations")
st.markdown(
    "- Synthetic demo data; results do not transfer to real roads.\n"
    "- Real accident data is often incomplete, biased toward reported incidents, and lacks exposure (traffic volume).\n"
    "- Weather and conditions change quickly; the model sees only a snapshot.\n"
    "- No real-time data feeds; predictions are for user-described scenarios.\n"
    "- Models can be wrong with high confidence, especially for conditions not seen during training."
)

st.markdown("### Future scope")
st.markdown(
    "IoT road sensors and GPS traces as live inputs; deep learning on richer spatio-temporal data; real-time "
    "prediction services; integration with smart traffic systems. The modular pipeline (`src/`) is designed so new "
    "data sources can be mapped onto the same schema."
)

st.markdown("### Team")
st.markdown("Presented by Harsh Rana, Shubham Dalal, Aryan Gupta and Devarsi Roy.")
