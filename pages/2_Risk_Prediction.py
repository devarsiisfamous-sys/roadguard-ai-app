"""Risk Prediction: score road scenarios with a saved, verified model."""

from __future__ import annotations

from datetime import date, datetime, time, timezone

import pandas as pd
import streamlit as st

import config
from src import charts, predict, state, theme, ui
from src.model_store import ModelStoreError
from src.utils import format_pct

ui.setup_page("Risk Prediction")
ui.page_header(
    "Risk Prediction",
    "Describe road, traffic, weather and driving conditions to get the model's estimated accident probability.",
    "Scenario analysis",
)

models = state.saved_models()
usable = [m for m in models if not m.get("problems")]
if not usable:
    st.info("No trained model is available yet. Train one in the Model Laboratory first.", icon=":material/info:")
    ui.nav_link("pages/3_Model_Laboratory.py", label="Open the Model Laboratory", icon=":material/model_training:")
    broken = [m for m in models if m.get("problems")]
    for meta in broken:
        st.warning(f"{meta['model']['name']}: {' '.join(meta['problems'])}")
    st.stop()

champ = state.champion()
keys = [m["model"]["key"] for m in usable]
by_key = {m["model"]["key"]: m for m in usable}
default_key = state.default_model_key(models)


def _model_label(key: str) -> str:
    meta = by_key[key]
    star = " ★ champion" if champ and champ["key"] == key else ""
    return f"{meta['model']['name']}{star} · trained {meta['training']['trained_at'][:16].replace('T', ' ')}"


top = st.columns([2, 1])
selected = top[0].selectbox("Model", keys, index=keys.index(default_key) if default_key in keys else 0,
                            format_func=_model_label, key="pred_model_select")
st.session_state[state.KEY_SELECTED_MODEL] = selected

try:
    model, meta = state.load_model(selected)
except ModelStoreError as exc:
    st.error(f"The model could not be loaded safely: {exc}")
    st.stop()

with top[1]:
    st.markdown(
        ui.badge("Trained on SYNTHETIC data", "synthetic") if meta["dataset"]["is_synthetic"]
        else ui.badge("Trained on user data", "user"),
        unsafe_allow_html=True,
    )
    st.caption(
        f"Decision threshold {meta['training']['decision_threshold']:.3f} · calibration: "
        f"{meta['training']['calibration']} · test F1 {meta['metrics']['test'].get('f1', 0):.3f}"
    )
for warning in by_key[selected].get("warnings_runtime", []):
    st.warning(warning)

with st.expander("Risk band settings", icon=":material/tune:"):
    st.caption(
        "Risk bands are display cut-offs on the estimated probability. The defaults are illustrative, "
        "**not validated road-safety standards**."
    )
    current = state.get_risk_thresholds()
    b1, b2, b3 = st.columns(3)
    low = b1.number_input("Low → Moderate at", 0.01, 0.98, float(current[0]), 0.01, key="band_low")
    mid = b2.number_input("Moderate → High at", 0.02, 0.98, float(current[1]), 0.01, key="band_mid")
    high = b3.number_input("High → Very High at", 0.03, 0.99, float(current[2]), 0.01, key="band_high")
    try:
        state.set_risk_thresholds(predict.validate_thresholds((low, mid, high)))
    except ValueError as exc:
        st.error(f"{exc} The previous thresholds are still in use.")
thresholds = state.get_risk_thresholds()

schema = meta["feature_schema"]
inputs = schema["inputs"]
tab_single, tab_compare, tab_batch, tab_history = st.tabs(
    ["Single scenario", "Compare scenarios", "Score dataset", "Prediction history"]
)


# --------------------------------------------------------------------------- #
# Scenario form (generated from the model's feature schema)
# --------------------------------------------------------------------------- #
def _input_widget(info: dict, container) -> object:
    name, kind, label = info["name"], info["kind"], info.get("label", info["name"])
    key = f"in_{selected}_{name}"
    if kind in ("categorical", "identifier"):
        categories = info.get("categories") or []
        if not categories:
            container.warning(f"No categories recorded for '{label}'.")
            return None
        index = categories.index(info["mode"]) if info.get("mode") in categories else 0
        return container.selectbox(label, categories, index=index, key=key,
                                   help=f"{info.get('n_categories', len(categories))} categories seen in training.")
    if kind == "numeric":
        low, high = info.get("valid_range") or [None, None]
        unit = info.get("unit", "")
        return container.number_input(
            f"{label} ({unit})" if unit else label, min_value=float(low) if low is not None else None,
            max_value=float(high) if high is not None else None, value=float(round(info["median"])), step=1.0, format="%.0f",
            key=key, help=f"Training range {info['min']:g}–{info['max']:g}; median {info['median']:g}.",
        )
    if kind == "date":
        default = datetime.strptime(info["median"], "%Y-%m-%d").date()
        return container.date_input(label, value=default, key=key,
                                    help=f"Training dates {info['min']} to {info['max']}. Used for weekday/weekend, month and holiday features.")
    if kind == "time":
        hh, mm = (int(x) for x in info["median"].split(":"))
        return container.time_input(label, value=time(hh, mm), step=900, key=key,
                                    help="Used for hour, peak-hour and time-of-day features.")
    container.warning(f"Unsupported input type for '{label}'.")
    return None


with tab_single:
    st.markdown("Every input below is required by the selected model. Defaults are typical training values.")
    with st.form(f"scenario_form_{selected}"):
        cols = st.columns(3)
        scenario: dict = {}
        for i, info in enumerate(inputs):
            scenario[info["name"]] = _input_widget(info, cols[i % 3])
        name_col, _ = st.columns([1, 2])
        scenario_name = name_col.text_input("Scenario name (optional)", key=f"scenario_name_{selected}",
                                            placeholder="e.g. Rainy night commute")
        submitted = st.form_submit_button("Estimate accident risk", type="primary", icon=":material/speed:")

    if submitted:
        try:
            result = predict.predict_scenario(model, meta, scenario, thresholds)
            factors = predict.scenario_factors(model, meta, scenario)
        except predict.PredictionError as exc:
            st.error(f"Cannot score this scenario: {exc}")
        except Exception as exc:  # model-specific failures must be visible, not swallowed
            st.error(f"Prediction failed: {type(exc).__name__}: {exc}")
        else:
            label = scenario_name.strip() or f"Scenario {len(state.prediction_history()) + 1}"
            st.session_state["rg_last_prediction"] = {
                "model_key": selected, "label": label, "scenario": scenario, "result": result, "factors": factors,
            }
            record = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "scenario": label,
                      "model": meta["model"]["name"], "model_trained_at": meta["training"]["trained_at"],
                      **{k: (v.isoformat() if isinstance(v, (date, time)) else v) for k, v in scenario.items()},
                      "probability": round(result.probability, 4), "risk_band": result.risk_band,
                      "predicted_class": result.predicted_label, "decision_threshold": round(result.threshold, 4),
                      "synthetic_training_data": meta["dataset"]["is_synthetic"]}
            state.add_prediction(record)

    last = st.session_state.get("rg_last_prediction")
    if last and last["model_key"] == selected:
        result = last["result"]
        # Re-derive the band in case the thresholds changed since the prediction.
        band = predict.assign_risk_band(result.probability, thresholds)
        st.divider()
        st.markdown(f"#### Result · {last['label']}")
        left, right = st.columns([1, 1.4], gap="large")
        with left:
            st.markdown(f'<div class="rg-muted">Estimated accident probability</div>'
                        f'<div class="rg-big">{result.probability:.1%}</div>', unsafe_allow_html=True)
            st.markdown(ui.risk_chip(band), unsafe_allow_html=True)
            st.markdown(
                f"Model classification: **{result.predicted_label}** "
                f"(probability {'≥' if result.predicted_class else '<'} decision threshold {result.threshold:.3f})"
            )
            ui.show_chart(charts.risk_strip(result.probability, thresholds), key="chart_risk_strip")
        with right:
            st.markdown("**How to read this estimate**")
            st.markdown(f"- {meta['probability_note']}")
            if meta["dataset"]["is_synthetic"]:
                st.markdown("- The model was trained on **synthetic** data, so this number reflects the simulator's "
                            "assumptions and is not a real-world probability.")
            st.markdown(f"- Risk bands use your cut-offs {thresholds[0]:.0%} / {thresholds[1]:.0%} / "
                        f"{thresholds[2]:.0%}; they are not official standards.")
            for warning in result.warnings:
                st.warning(warning, icon=":material/warning:")

        st.markdown("**Factors associated with this estimate**")
        st.caption(
            "Each input is swapped, one at a time, for a typical training value (median or most common category). "
            "The bars show how much the model's estimate changes. Red raises it and blue lowers it. "
            "This describes what the model learned from the data. It is **not** evidence that the factor causes accidents."
        )
        factors: pd.DataFrame = last["factors"]
        if (factors["direction"] == "no change").all():
            st.info("This scenario already uses typical values for every input, so there is nothing to compare against.")
        else:
            ui.show_chart(charts.factor_bars(factors), key="chart_factors")
        with st.expander("Factor table"):
            st.dataframe(factors.astype({"your_value": str, "typical_value": str}), hide_index=True)

        st.markdown("**General road-safety reminders**")
        st.caption("Generic guidance based on the conditions you entered. These are not model outputs.")
        for tip in predict.safety_guidance(last["scenario"], band):
            st.markdown(f"- {tip}")

        if st.button("Add to scenario comparison", icon=":material/add_chart:", key="add_compare"):
            entry = {"scenario": last["label"], "model": meta["model"]["name"],
                     **{k: (v.isoformat() if isinstance(v, (date, time)) else v) for k, v in last["scenario"].items()},
                     "probability": result.probability}
            existing = [s["scenario"] for s in state.scenarios()]
            if entry["scenario"] in existing:
                entry["scenario"] = f"{entry['scenario']} ({len(existing) + 1})"
            state.scenarios().append(entry)
            st.success(f"Added '{entry['scenario']}' to the comparison tab.")
        ui.disclaimer(compact=True)

with tab_compare:
    saved = state.scenarios()
    if not saved:
        st.info("Estimate a scenario, then click **Add to scenario comparison** to compare conditions side by side.")
    else:
        table = pd.DataFrame(saved)
        table["risk_band"] = [predict.assign_risk_band(p, thresholds) for p in table["probability"]]
        st.markdown("**Estimated accident probability by scenario**")
        ui.show_chart(charts.scenario_comparison(table), key="chart_compare")
        st.dataframe(table, hide_index=True,
                     column_config={"probability": st.column_config.NumberColumn("probability", format="percent")})
        c1, c2 = st.columns([1, 4])
        with c1:
            ui.download_csv(table, "roadguard_scenarios.csv", "Download CSV", key="dl_scenarios")
        if c2.button("Clear comparison", key="clear_compare"):
            state.clear_scenarios()
            st.rerun()
        st.caption("Scenarios scored by different models are not directly comparable.")

with tab_batch:
    dataset = state.get_active_dataset()
    st.markdown("Score every record of the active dataset with the selected model, for example to rank "
                "conditions by estimated risk.")
    if dataset is None:
        st.warning("No dataset is loaded.")
    else:
        missing = [c for c in schema["input_names"] if c not in dataset.df.columns]
        if missing:
            st.error("The active dataset is not compatible with this model. Missing input(s): " + ", ".join(missing))
        else:
            if meta["dataset"].get("fingerprint") == dataset.fingerprint:
                st.caption("ℹ️ This is the dataset the model was trained on. Its scores include training rows, so they are "
                           "**not** an evaluation of model accuracy. Use the Model Laboratory for that.")
            if st.button("Score active dataset", key="score_dataset", icon=":material/calculate:"):
                try:
                    scored = predict.predict_dataframe(model, meta, dataset.df, thresholds)
                except predict.PredictionError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["rg_scored"] = {"model": selected, "data": scored}
            scored_state = st.session_state.get("rg_scored")
            if scored_state and scored_state["model"] == selected:
                scored = scored_state["data"]
                counts = scored["risk_band"].value_counts().reindex(list(config.RISK_BANDS), fill_value=0)
                ui.kpi_row([(f"{theme.RISK_ICONS[b]} {b}", f"{int(counts[b]):,}", None) for b in config.RISK_BANDS])
                st.dataframe(scored.sort_values("predicted_accident_probability", ascending=False).head(500),
                             hide_index=True)
                export = scored.copy()
                if config.TARGET_COLUMN in export.columns:
                    export[config.TARGET_COLUMN] = export[config.TARGET_COLUMN].map({1: "Yes", 0: "No"})
                ui.download_csv(export, "roadguard_scored_dataset.csv", "Download scored dataset CSV", key="dl_scored")

with tab_history:
    history = state.prediction_history()
    if history.empty:
        st.info("No predictions yet in this session.")
    else:
        st.dataframe(history, hide_index=True)
        c1, c2 = st.columns([1, 4])
        with c1:
            ui.download_csv(history, "roadguard_prediction_history.csv", "Download history CSV", key="dl_history")
        if c2.button("Clear history", key="clear_history"):
            state.clear_prediction_history()
            st.rerun()
        st.caption("History is kept only for this browser session.")

st.caption(f"Model inputs: {', '.join(schema['input_names'])}. Engineered: {', '.join(schema['engineered']) or 'none'}. "
           f"Accident share in training data: {format_pct(meta['training']['train_positive_share'])}.")
