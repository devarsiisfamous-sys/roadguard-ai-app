"""Model Laboratory: train, evaluate, compare, explain and manage models."""

from __future__ import annotations


import pandas as pd
import streamlit as st

import config
from src import charts, evaluate, model_store, state, static_charts, train, ui
from src.feature_engineering import ENGINEERED_FEATURES, FeatureConfig, parse_holiday_text
from src.preprocessing import imbalance_warning
from src.utils import format_metric

ui.setup_page("Model Laboratory")
ui.page_header(
    "Model Laboratory",
    "Train the presentation's five algorithms with leakage-safe pipelines, then compare, explain and choose a champion.",
    "Machine learning",
)

dataset = state.get_active_dataset()
tab_train, tab_compare, tab_eval, tab_explain, tab_saved = st.tabs(
    ["Train", "Compare & select", "Evaluation", "Explainability", "Saved models"]
)


def _parse_peak_hours(text: str) -> tuple[tuple[int, int], ...]:
    ranges = []
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        start, end = (int(x) for x in part.split("-"))
        ranges.append((start, end))
    if not ranges:
        raise ValueError("Enter at least one peak-hour range, e.g. 7-10, 17-20.")
    return tuple(ranges)


def _parse_bins(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.replace(";", ",").split(",") if x.strip())


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
with tab_train:
    if dataset is None:
        st.error(f"No dataset is loaded: {state.dataset_error()}")
    elif not dataset.target.usable:
        st.warning(f"**Supervised training is disabled for this dataset.** {dataset.target.reason}",
                   icon=":material/block:")
        ui.nav_link("pages/4_Data_Explorer.py", label="Check the column mapping in the Data Explorer",
                     icon=":material/table:")
    else:
        ui.synthetic_notice(dataset)
        warning = imbalance_warning(dataset.df[config.TARGET_COLUMN].dropna().astype(int))
        if warning:
            st.warning(warning)
        availability = train.available_models()
        candidates = [c for c in config.SCHEMA if config.SCHEMA[c]["kind"] in (*config.FEATURE_KINDS, "identifier")
                      and c in dataset.df.columns]
        defaults = [c for c in candidates if c not in config.DEFAULT_EXCLUDED_FEATURES]
        cfg_now = state.get_feature_config()

        with st.form("train_form"):
            st.markdown("##### 1 · Inputs and algorithms")
            features = st.multiselect(
                "Input features", candidates, default=defaults,
                format_func=lambda c: config.SCHEMA[c]["label"],
                help="Raw schema columns given to the model. Date and time are expanded into engineered features. "
                     "The road-segment identifier is off by default because it does not generalise to new locations.",
            )
            model_cols = st.columns(len(train.MODEL_SPECS))
            chosen_models = []
            for col, (key, spec) in zip(model_cols, train.MODEL_SPECS.items()):
                ok, message = availability[key]
                if col.checkbox(spec.name, value=ok, disabled=not ok, key=f"train_pick_{key}",
                                help=spec.description if ok else message):
                    chosen_models.append(key)
            if not availability["xgboost"][0]:
                st.caption(f"ℹ️ {availability['xgboost'][1]}")

            st.markdown("##### 2 · Evaluation design")
            e1, e2, e3, e4 = st.columns(4)
            test_size = e1.slider("Test-set size", 0.10, 0.40, config.DEFAULT_TEST_SIZE, 0.05,
                                  help="Held out before any fitting; used only for final evaluation.")
            seed = e2.number_input("Random seed", 0, 1_000_000, config.RANDOM_SEED, 1,
                                   help="Controls the split, CV folds and model randomness.")
            cv_folds = e3.slider("Cross-validation folds", 3, 10, config.DEFAULT_CV_FOLDS,
                                 help="Stratified folds within the training split.")
            metric = e4.selectbox("Champion-selection metric", evaluate.SELECTION_METRICS,
                                  format_func=lambda m: evaluate.METRICS[m]["label"],
                                  help="The champion is the model with the best mean cross-validated score, never the test score.")

            st.markdown("##### 3 · Imbalance, threshold and probabilities")
            p1, p2, p3 = st.columns(3)
            balance = p1.toggle("Class weighting (balanced)", value=True,
                                help="Gives accident cases more weight during fitting. Raises recall but inflates raw probabilities.")
            tune = p1.toggle("Tune hyperparameters (grid search CV)", value=False,
                             help="Searches a small grid with cross-validation on the training split only. Slower.")
            calibration = p2.selectbox(
                "Probability calibration", ["none", "sigmoid", "isotonic"],
                help="Fits a calibration map with cross-validation on the training split only. "
                     "The SVM always uses sigmoid (Platt) scaling to produce probabilities.",
            )
            strategy = p3.radio(
                "Decision threshold", ["optimize_f1", "fixed"],
                format_func=lambda s: "Maximise F1 on training CV (recommended)" if s == "optimize_f1" else "Fixed value",
                help="Probability above which a case is classified as 'Accident'. Optimisation uses out-of-fold "
                     "training predictions, never the test set.",
            )
            fixed_threshold = p3.slider("Fixed threshold (used if 'Fixed value')", 0.05, 0.95, 0.50, 0.01)

            with st.expander("Hyperparameters"):
                params: dict[str, dict] = {}
                h = st.columns(5)
                params["logistic_regression"] = {
                    "C": h[0].select_slider("LogReg · C", [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0], value=1.0)}
                dt_depth = h[1].number_input("Tree · max depth (0 = none)", 0, 50, 6)
                params["decision_tree"] = {
                    "max_depth": None if dt_depth == 0 else int(dt_depth),
                    "min_samples_leaf": int(h[1].number_input("Tree · min samples leaf", 1, 500, 20))}
                rf_depth = h[2].number_input("Forest · max depth (0 = none)", 0, 50, 12)
                params["random_forest"] = {
                    "n_estimators": int(h[2].slider("Forest · trees", 50, 600, 300, 50)),
                    "max_depth": None if rf_depth == 0 else int(rf_depth),
                    "min_samples_leaf": int(h[2].number_input("Forest · min samples leaf", 1, 200, 5))}
                params["xgboost"] = {
                    "n_estimators": int(h[3].slider("XGBoost · trees", 50, 800, 300, 50)),
                    "max_depth": int(h[3].slider("XGBoost · max depth", 2, 10, 4)),
                    "learning_rate": float(h[3].select_slider("XGBoost · learning rate", [0.01, 0.03, 0.05, 0.1, 0.2, 0.3], value=0.05)),
                    "subsample": float(h[3].slider("XGBoost · subsample", 0.5, 1.0, 0.9, 0.05))}
                params["svm"] = {
                    "C": h[4].select_slider("SVM · C", [0.1, 0.5, 1.0, 2.0, 4.0, 10.0], value=1.0),
                    "kernel": h[4].selectbox("SVM · kernel", ["rbf", "linear"])}
                st.caption("When tuning is enabled, the tuned parameters replace these values for the searched settings.")

            with st.expander("Feature-engineering assumptions"):
                f1, f2 = st.columns(2)
                peak_text = f1.text_input(
                    "Peak-hour ranges [start-end) on a 24h clock",
                    ", ".join(f"{s}-{e}" for s, e in cfg_now.peak_hour_ranges),
                )
                bins_text = f1.text_input("Speed band edges (km/h)", ", ".join(f"{b:g}" for b in cfg_now.speed_bins))
                holiday_text = f2.text_area(
                    "Holiday dates (optional, YYYY-MM-DD; comma or newline separated)", "\n".join(cfg_now.holidays),
                    help="The holiday indicator is only created when you supply a holiday calendar. None is assumed.",
                )
                st.caption("Engineered features: " + "; ".join(
                    f"**{n}** ({d})" for n, (_, _, d) in ENGINEERED_FEATURES.items()))

            submitted = st.form_submit_button("Train selected models", type="primary", icon=":material/play_arrow:")

        if submitted:
            try:
                feature_cfg = FeatureConfig(
                    peak_hour_ranges=_parse_peak_hours(peak_text), speed_bins=_parse_bins(bins_text),
                    holidays=parse_holiday_text(holiday_text),
                )
            except ValueError as exc:
                st.error(f"Invalid feature-engineering settings: {exc}")
                feature_cfg = None
            if feature_cfg is not None:
                settings = train.TrainingSettings(
                    test_size=float(test_size), seed=int(seed), balance_classes=balance, calibration=calibration,
                    tune=tune, cv_folds=int(cv_folds), threshold_strategy=strategy,
                    fixed_threshold=float(fixed_threshold), selection_metric=metric, feature_config=feature_cfg,
                    params=params,
                )
                bar = st.progress(0.0, text="Preparing data…")
                try:
                    with st.spinner("Training. This can take a minute, especially with SVM or tuning."):
                        run = train.train_models(dataset.df, features, chosen_models, settings, dataset.info,
                                                 progress=lambda f, m: bar.progress(min(max(f, 0.0), 1.0), text=m))
                except train.TrainingError as exc:
                    st.error(f"Training could not start: {exc}")
                else:
                    state.set_feature_config(feature_cfg)
                    st.session_state[state.KEY_LAST_RUN] = {
                        "run_id": run.run_id, "trained": [r["model"]["key"] for r in run.results],
                        "errors": run.errors, "champion": run.champion_key, "basis": run.champion_basis,
                        "warnings": run.warnings, "metric": metric,
                    }
                    st.rerun()

        last = st.session_state.get(state.KEY_LAST_RUN)
        if last:
            st.divider()
            st.markdown(f"#### Last training run · `{last['run_id']}`")
            if last["trained"]:
                names = ", ".join(train.MODEL_SPECS[k].name for k in last["trained"])
                st.success(f"Trained and saved: {names}.", icon=":material/check_circle:")
            for key, error in last["errors"].items():
                st.error(f"**{train.MODEL_SPECS[key].name}** was not trained. {error}")
            if last["champion"]:
                st.info(f"Champion: **{train.MODEL_SPECS[last['champion']].name}**, selected by {last['basis']}. "
                        "Review it in **Compare & select**.", icon=":material/emoji_events:")
            elif last["basis"]:
                st.warning(last["basis"])
            for w in last["warnings"]:
                st.caption(f"⚠️ {w}")

# --------------------------------------------------------------------------- #
# Shared model lists
# --------------------------------------------------------------------------- #
models = state.saved_models()
usable = [m for m in models if not m.get("problems")]
champ = state.champion()
by_key = {m["model"]["key"]: m for m in usable}


def _pick_model(container, key: str) -> dict | None:
    if not usable:
        container.info("No trained models yet. Train some in the **Train** tab.")
        return None
    keys = list(by_key)
    default = state.default_model_key(models)
    choice = container.selectbox(
        "Model", keys, index=keys.index(default) if default in keys else 0, key=key,
        format_func=lambda k: by_key[k]["model"]["name"] + (" ★" if champ and champ["key"] == k else ""),
    )
    return by_key[choice]


# --------------------------------------------------------------------------- #
# Compare & select
# --------------------------------------------------------------------------- #
with tab_compare:
    if not usable:
        st.info("No trained models yet. Train some in the **Train** tab.")
    else:
        runs = sorted({m["training"]["run_id"] for m in usable}, reverse=True)
        show_all = st.toggle("Include models from earlier training runs", value=False,
                             help="Models from different runs may use different data, features or settings.")
        compared = usable if show_all else [m for m in usable if m["training"]["run_id"] == runs[0]]
        fingerprints = {m["dataset"].get("fingerprint") for m in compared}
        if len(fingerprints) > 1:
            st.warning("These models were trained on different datasets, so their scores are not directly comparable.")
        if any(m["dataset"]["is_synthetic"] for m in compared):
            st.caption("⚠️ Scores measured on SYNTHETIC data. They show how the pipeline works, not real-world performance.")

        table = evaluate.comparison_table(compared)
        metric = st.selectbox("Metric", evaluate.SELECTION_METRICS, format_func=lambda m: evaluate.METRICS[m]["label"],
                              key="compare_metric")
        st.caption(evaluate.METRICS[metric]["help"])
        st.markdown(f"**{evaluate.METRICS[metric]['label']}: cross-validated mean ± std (bars) and held-out test (◆)**")
        ui.show_chart(charts.model_comparison(table, metric, evaluate.METRICS[metric]["label"]), key="chart_model_cmp")

        display_cols = ["model"] + [f"test_{m}" for m in evaluate.METRICS] + [f"cv_{metric}_mean", f"cv_{metric}_std",
                                                                              "fit_seconds", "calibration", "tuned"]
        st.dataframe(
            table[display_cols], hide_index=True,
            column_config={c: st.column_config.NumberColumn(c.replace("test_", "test ").replace("_", " "), format="%.3f")
                           for c in display_cols if c.startswith(("test_", "cv_"))},
        )
        d1, d2, _ = st.columns([1, 1, 2])
        with d1:
            ui.download_csv(table, "roadguard_model_comparison.csv", "Comparison CSV", key="dl_cmp")
        with d2:
            ui.download_csv(evaluate.metrics_long_table(compared), "roadguard_evaluation_metrics.csv",
                            "Evaluation metrics CSV", key="dl_metrics")

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown("**ROC curves (held-out test)**")
            ui.show_chart(charts.roc_curves(compared), key="chart_roc_all")
        with c2:
            st.markdown("**Precision-recall curves (held-out test)**")
            ui.show_chart(charts.pr_curves(compared), key="chart_pr_all")

        with st.expander("False positives vs false negatives"):
            st.markdown(evaluate.FP_FN_TRADEOFF)

        st.markdown("#### Champion model")
        if champ and champ["key"] in by_key:
            st.markdown(f"Current champion: **{by_key[champ['key']].get('model')['name']}**, chosen by {champ.get('basis')}.")
        else:
            st.markdown("No champion is selected.")
        keys = [m["model"]["key"] for m in compared]
        cv_col = f"cv_{metric}_mean"
        best_cv = None
        if table[cv_col].notna().all():
            best_cv = compared[int(table[cv_col].astype(float).idxmax())]["model"]["key"]
            st.caption(f"Best mean CV {evaluate.METRICS[metric]['label']}: **{by_key[best_cv]['model']['name']}**.")
        sel1, sel2 = st.columns([2, 1])
        pick = sel1.selectbox("Use as champion", keys, index=keys.index(best_cv) if best_cv in keys else 0,
                              format_func=lambda k: by_key[k]["model"]["name"], key="champion_pick")
        if sel2.button("Set as champion", type="primary", key="set_champion", icon=":material/emoji_events:"):
            basis = (f"highest mean cross-validated {evaluate.METRICS[metric]['label']} (user confirmed)"
                     if pick == best_cv else f"manual selection by the user (comparison metric: {evaluate.METRICS[metric]['label']})")
            model_store.set_champion(pick, metric, basis, config.MODELS_DIR)
            st.session_state[state.KEY_SELECTED_MODEL] = pick
            st.rerun()
        st.caption("Choose champions by cross-validated scores. Picking by the test score turns the test set into a "
                   "selection set and makes its estimate optimistic.")

# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
with tab_eval:
    meta = _pick_model(st, "eval_model")
    if meta:
        tr = meta["training"]
        ui.badges(
            ui.badge("SYNTHETIC data" if meta["dataset"]["is_synthetic"] else "User data",
                     "synthetic" if meta["dataset"]["is_synthetic"] else "user"),
            ui.badge(f"Run {tr['run_id']}", "neutral"),
            ui.badge(f"Threshold {tr['decision_threshold']:.3f} ({tr.get('threshold_strategy', 'fixed')})", "neutral"),
            ui.badge(f"Calibration: {tr['calibration']}", "neutral"),
            ui.badge(f"{tr['n_train']:,} train / {tr['n_test']:,} test", "neutral"),
        )
        for w in meta.get("warnings", []) + meta.get("warnings_runtime", []):
            st.warning(w, icon=":material/warning:")

        test = meta["metrics"]["test"]
        cv = meta["metrics"]["cv"]
        st.markdown("**Held-out test metrics** (accident = positive class)")
        cols = st.columns(len(evaluate.METRICS))
        for col, (name, spec) in zip(cols, evaluate.METRICS.items()):
            cv_txt = f" CV: {cv[name]['mean']:.3f} ± {cv[name]['std']:.3f}." if name in cv else ""
            col.metric(spec["label"], format_metric(test.get(name)), help=spec["help"] + cv_txt)
        if test.get("roc_auc") is None:
            st.info("ROC-AUC and average precision could not be computed because the test set contains only one class.")

        c1, c2 = st.columns([1, 1.25], gap="medium")
        with c1:
            st.markdown("**Confusion matrix (test set)**")
            ui.show_chart(charts.confusion_heatmap(meta["confusion_matrix"]), key="chart_cm")
            (tn, fp), (fn, tp) = meta["confusion_matrix"]
            st.caption(f"Of {tp + fn:,} actual accidents, the model flagged {tp:,} and missed {fn:,} (false negatives). "
                       f"It raised {fp:,} false alarms among {tn + fp:,} non-accidents.")
        with c2:
            st.markdown("**Cross-validation on the training split**")
            if cv:
                cv_df = pd.DataFrame([{"metric": evaluate.METRICS[k]["label"], "mean": v["mean"], "std": v["std"],
                                       "test": test.get(k)} for k, v in cv.items()])
                st.dataframe(cv_df, hide_index=True, column_config={
                    c: st.column_config.NumberColumn(c, format="%.3f") for c in ("mean", "std", "test")})
                st.caption(meta["metrics"].get("cv_note", ""))
            else:
                st.info("No cross-validation results were recorded for this model.")
            st.markdown("**Classification report (test set)**")
            report = pd.DataFrame(meta["classification_report"]).T.reset_index().rename(columns={"index": "class"})
            st.dataframe(report, hide_index=True, column_config={
                c: st.column_config.NumberColumn(c, format="%.3f") for c in ("precision", "recall", "f1-score")})

        r1, r2 = st.columns(2, gap="medium")
        with r1:
            st.markdown("**ROC curve**")
            ui.show_chart(charts.roc_curves([meta]), key="chart_roc_one")
        with r2:
            st.markdown("**Precision-recall curve**")
            ui.show_chart(charts.pr_curves([meta]), key="chart_pr_one")

        st.markdown("**Probabilities**")
        st.markdown(meta.get("probability_note", ""))
        with st.expander("False positives vs false negatives"):
            st.markdown(evaluate.FP_FN_TRADEOFF)

        report_md = evaluate.build_model_report(meta)
        d1, d2, d3, d4 = st.columns([1, 1, 1, 1.4])
        d1.download_button("Model report (Markdown)", report_md.encode("utf-8"),
                           file_name=f"roadguard_report_{meta['model']['key']}.md", mime="text/markdown",
                           key="dl_report", icon=":material/description:")
        with d2:
            ui.download_csv(evaluate.metrics_long_table([meta]), f"roadguard_metrics_{meta['model']['key']}.csv",
                            "Metrics CSV", key="dl_metrics_one")
        if d3.button("Prepare PNG figure", key="make_png", icon=":material/image:",
                     help="Static confusion matrix, ROC and PR curves (Matplotlib/Seaborn) for documents."):
            st.session_state["rg_eval_png"] = (meta["artifact"]["sha256"], static_charts.evaluation_png(meta))
        png = st.session_state.get("rg_eval_png")
        if png and png[0] == meta["artifact"]["sha256"]:
            d3.download_button("Download PNG", png[1], file_name=f"roadguard_evaluation_{meta['model']['key']}.png",
                               mime="image/png", key="dl_png", icon=":material/download:")
        if d4.button("Save report + figure to reports/", key="save_report", icon=":material/save:"):
            config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            stem = f"report_{meta['model']['key']}_{tr['run_id']}"
            (config.REPORTS_DIR / f"{stem}.md").write_text(report_md, encoding="utf-8")
            (config.REPORTS_DIR / f"{stem}.png").write_bytes(static_charts.evaluation_png(meta))
            st.success(f"Saved reports/{stem}.md and reports/{stem}.png")

# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #
with tab_explain:
    st.info(
        "**Four different things, and only three are shown in this app.**\n\n"
        "1. *Descriptive patterns* (Accident Analytics): what the data shows, group by group.\n"
        "2. *Model associations* (this tab): which inputs the model relies on.\n"
        "3. *Scenario predictions* (Risk Prediction): the model's estimate for one set of conditions.\n"
        "4. *Causal conclusions*: **cannot** be established by any of these analyses. Importance is not causation.",
        icon=":material/psychology:",
    )
    meta = _pick_model(st, "explain_model")
    if meta:
        perm = meta.get("permutation_importance", [])
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown(f"**Permutation importance (held-out test, drop in {meta.get('permutation_scoring', 'score')})**")
            if perm:
                ui.show_chart(charts.importance_bars(perm, "importance_mean", "Mean score drop", "importance_std"),
                              key="chart_perm")
                st.caption("Each input is shuffled on the test set and the drop in score is measured (5 repeats, ± std). "
                           "This works for any model, so it is the fairest way to compare models. Correlated inputs share "
                           "importance, so each one can look less important than it is.")
                ui.download_csv(pd.DataFrame(perm), f"roadguard_permutation_{meta['model']['key']}.csv",
                                "Download CSV", key="dl_perm")
            else:
                st.info("No permutation importance was recorded.")
        with c2:
            imp = meta.get("model_importance")
            if imp is None:
                st.markdown("**Model-specific importance**")
                st.info(f"{meta['model']['name']} with this configuration has no built-in importance measure "
                        "(e.g. an RBF-kernel SVM). Use permutation importance.")
            elif imp["kind"] == "coefficient":
                st.markdown("**Model coefficients (log-odds scale)**")
                ui.show_chart(charts.importance_bars(imp["values"], "value", "Coefficient", signed=True), key="chart_coef")
                st.caption("Red raises the predicted log-odds of an accident and blue lowers them. Numeric inputs are "
                           "standardised and categories are one-hot encoded, so a category's coefficient is relative to the "
                           "other categories. Correlated inputs make individual coefficients unstable.")
            else:
                label = "gain" if imp["kind"] == "gain" else "impurity decrease"
                st.markdown(f"**Built-in feature importance ({label})**")
                ui.show_chart(charts.importance_bars(imp["values"], "value", "Importance"), key="chart_builtin")
                st.caption("Computed on training data at the level of encoded columns. It is biased toward numeric and "
                           "high-cardinality features. Prefer permutation importance for comparisons.")
        if meta.get("tree_rules"):
            with st.expander("Decision-tree rules (top 3 levels)", expanded=False):
                st.code(meta["tree_rules"], language="text")
                st.caption("Splits on one-hot columns read as 'category ≤ 0.5' = 'not this category'.")
        st.caption("SHAP explanations are not included, to keep dependencies light. Permutation importance is "
                   "used because it is model-agnostic and measured on held-out data.")

# --------------------------------------------------------------------------- #
# Saved models
# --------------------------------------------------------------------------- #
with tab_saved:
    st.markdown(
        "Trained pipelines (feature engineering, preprocessing and model) are saved in `models/` with their "
        "metadata. The app only loads files it wrote itself, after checking each file's SHA-256 against its "
        "metadata. There is no option to upload model files."
    )
    if not models:
        st.info("No saved models.")
    for meta in models:
        key = meta["model"]["key"]
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 2, 1.4])
            star = " ★ champion" if champ and champ["key"] == key else ""
            c1.markdown(f"**{meta['model']['name']}{star}**  \n"
                        f"Trained {meta['training']['trained_at']} · run `{meta['training']['run_id']}`  \n"
                        f"Data: {meta['dataset']['source_label']} "
                        f"({'synthetic' if meta['dataset']['is_synthetic'] else 'user'}) · "
                        f"{meta['artifact']['size_bytes'] / 1e6:.1f} MB")
            if meta.get("problems"):
                c2.markdown(ui.badge("Unusable", "warn"), unsafe_allow_html=True)
                c2.caption(" ".join(meta["problems"]))
            else:
                c2.markdown(ui.badge("Verified · ready", "ok"), unsafe_allow_html=True)
                for w in meta.get("warnings_runtime", []):
                    c2.caption(w)
            confirm = c3.checkbox("Confirm delete", key=f"confirm_del_{key}")
            if c3.button("Delete", key=f"del_{key}", disabled=not confirm, icon=":material/delete:"):
                model_store.delete_model(key, config.MODELS_DIR)
                st.session_state.pop("rg_last_prediction", None)
                st.rerun()
    if models:
        st.divider()
        confirm_all = st.checkbox("I understand this permanently deletes every saved model", key="confirm_del_all")
        if st.button("Delete all saved models", disabled=not confirm_all, key="del_all", icon=":material/delete_forever:"):
            removed = model_store.delete_all_models(config.MODELS_DIR)
            st.session_state.pop("rg_last_prediction", None)
            st.session_state.pop(state.KEY_LAST_RUN, None)
            st.success(f"Deleted {removed} model(s).")
            st.rerun()
