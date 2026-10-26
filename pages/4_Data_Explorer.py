"""Data Explorer: choose a data source, map columns, inspect quality, clean and download."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

import config
from src import state, ui
from src.data_loader import DataLoadError, suggest_column_mapping
from src.preprocessing import CleaningOptions
from src.utils import format_pct

ui.setup_page("Data Explorer")
ui.page_header(
    "Data Explorer",
    "Load the demo dataset or your own CSV, map its columns, check data quality and download the cleaned data.",
    "Data management",
)

NOT_MAPPED = "— not mapped —"
dataset = state.get_active_dataset()
if dataset is None:
    st.error(f"The demo dataset could not be loaded: {state.dataset_error()}. You can still upload a CSV below.")


def cleaning_controls(prefix: str, defaults: CleaningOptions) -> CleaningOptions:
    c1, c2, c3 = st.columns(3)
    return CleaningOptions(
        drop_duplicates=c1.checkbox("Remove exact duplicate rows", defaults.drop_duplicates, key=f"{prefix}_dup"),
        out_of_range_to_missing=c2.checkbox(
            "Set implausible numbers to missing", defaults.out_of_range_to_missing, key=f"{prefix}_oor",
            help="Speeds outside 0–250 km/h and driver ages outside 14–100 become missing. They are never replaced with guesses.",
        ),
        drop_rows_missing_target=c3.checkbox(
            "Drop rows without an accident label", defaults.drop_rows_missing_target, key=f"{prefix}_tgt"),
    )


# --------------------------------------------------------------------------- #
# Source selection
# --------------------------------------------------------------------------- #
with st.container(border=True):
    st.markdown("#### Data source")
    if dataset is not None:
        st.markdown(f"Active: **{dataset.source_label}** {ui.source_badge(dataset)} · loaded {dataset.loaded_at}",
                    unsafe_allow_html=True)
    source = st.radio("Choose a source", ["Built-in synthetic demo dataset", "Upload a CSV file"], horizontal=True,
                      key="source_choice")

    if source.startswith("Built-in"):
        st.caption(
            f"{config.SAMPLE_DATA_PATH.name}: simulated, clearly labelled synthetic observations (seed "
            f"{config.RANDOM_SEED}). They include a small, documented share of missing, duplicated and invalid "
            "values so that the quality checks have something to find."
        )
        options = cleaning_controls("demo", CleaningOptions())
        if st.button("Load demo dataset", type="primary", key="load_demo", icon=":material/dataset:"):
            try:
                state.set_active_dataset(state.load_demo_dataset(options))
                st.success("Demo dataset loaded.")
                st.rerun()
            except DataLoadError as exc:
                st.error(str(exc))
    else:
        upload = st.file_uploader("CSV file", type=["csv"], key="uploader",
                                  help="Comma, semicolon or tab separated, with a header row. Max 200 MB. "
                                       "The file is read in memory and never modified.")
        if upload is not None:
            try:
                raw = state.read_upload(upload.getvalue(), upload.name)
            except DataLoadError as exc:
                st.error(f"Could not read **{upload.name}**: {exc}")
                raw = None
            if raw is not None:
                st.caption(f"Read **{upload.name}**: {len(raw):,} rows × {raw.shape[1]} columns.")
                st.dataframe(raw.head(10), hide_index=True)
                st.markdown("##### Map columns to the RoadGuard schema")
                st.caption("Suggestions come from common column names. Unmapped columns are dropped. This keeps "
                           "after-the-event fields (casualties, severity, notes) from leaking into models.")
                suggested = suggest_column_mapping(list(raw.columns))
                choices = [NOT_MAPPED, *raw.columns]
                mapping: dict[str, str | None] = {}
                grid = st.columns(3)
                for i, (field, spec) in enumerate(config.SCHEMA.items()):
                    default = suggested.get(field)
                    picked = grid[i % 3].selectbox(
                        f"{spec['label']}" + (" · target" if spec["kind"] == "target" else ""),
                        choices, index=choices.index(default) if default in choices else 0,
                        key=f"map_{field}_{upload.name}", help=spec["description"],
                    )
                    mapping[field] = None if picked == NOT_MAPPED else picked
                upload_options = cleaning_controls("upload", CleaningOptions())
                if st.button("Use this dataset", type="primary", key="use_upload", icon=":material/check:"):
                    try:
                        new = state.build_active_dataset(raw, mapping, f"Upload: {upload.name}", False, upload_options)
                    except DataLoadError as exc:
                        st.error(f"Column mapping problem: {exc}")
                    except Exception as exc:  # surface unexpected parsing problems to the user
                        st.error(f"The dataset could not be prepared: {type(exc).__name__}: {exc}")
                    else:
                        if new.df.empty:
                            st.error("No rows remain after cleaning. Check the mapping and cleaning options.")
                        else:
                            state.set_active_dataset(new)
                            st.success(f"Now using **{upload.name}** ({len(new.df):,} rows after cleaning).")
                            if not new.target.usable:
                                st.warning(new.target.reason)
                            st.rerun()

if dataset is None:
    st.stop()

# --------------------------------------------------------------------------- #
# Active dataset details
# --------------------------------------------------------------------------- #
quality = dataset.quality
st.markdown("#### Active dataset")
ui.synthetic_notice(dataset)
ui.kpi_row([
    ("Raw rows", f"{quality.n_rows:,}", "Rows in the mapped source data."),
    ("Rows after cleaning", f"{len(dataset.df):,}", "After the cleaning options below."),
    ("Duplicate rows", f"{quality.n_duplicates:,}", "Exact duplicates in the raw data."),
    ("Missing cells", format_pct(quality.total_missing_share), "Empty or unparseable cells before cleaning."),
    ("Quality findings", f"{len(quality.issues)}", "Invalid values and inconsistent spellings."),
])
if not dataset.target.usable:
    st.warning(f"**Model training is unavailable:** {dataset.target.reason}", icon=":material/block:")

tab_quality, tab_target, tab_preview, tab_schema, tab_download = st.tabs(
    ["Quality report", "Target", "Preview", "Schema & mapping", "Clean & download"]
)

with tab_quality:
    c1, c2 = st.columns([1, 1], gap="medium")
    with c1:
        st.markdown("**Missing values by column (before cleaning)**")
        st.dataframe(quality.missing, hide_index=True, column_config={
            "missing_pct": st.column_config.ProgressColumn("missing %", min_value=0, max_value=100, format="%.2f%%")})
    with c2:
        st.markdown("**Invalid-value and consistency warnings**")
        if quality.issues:
            st.dataframe(pd.DataFrame(quality.issues), hide_index=True)
        else:
            st.success("No invalid values or inconsistent spellings detected.")
        st.markdown("**Cleaning log**")
        for step in dataset.cleaning_log.steps:
            st.markdown(f"- {step}")
        st.caption("Missing values are **not** filled in during cleaning. Models impute them inside their training "
                   "pipeline, using statistics from the training data only.")
    st.markdown("**Data types**")
    st.dataframe(quality.dtypes, hide_index=True)

with tab_target:
    target = dataset.target
    if not target.present:
        st.info(target.reason)
    else:
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Accident (1)", f"{target.class_counts.get(1, 0):,}")
        t2.metric("No accident (0)", f"{target.class_counts.get(0, 0):,}")
        t3.metric("Accident share", format_pct(target.positive_share))
        t4.metric("Missing / unrecognised", f"{quality.target.n_missing:,} / {quality.target.n_invalid:,}",
                  help="Counted before cleaning.")
        st.caption("Class mapping: 1 = accident occurred (positive class), 0 = no accident. Recognised labels: "
                   + ", ".join(sorted(config.TARGET_TRUE_VALUES)) + " → 1; "
                   + ", ".join(sorted(config.TARGET_FALSE_VALUES)) + " → 0 (case-insensitive).")
        if target.minority_share is not None and target.minority_share < config.SEVERE_IMBALANCE_SHARE:
            st.warning(f"Severe class imbalance: the rarer class is {target.minority_share:.1%} of labelled rows.")

with tab_preview:
    which = st.radio("Show", ["Cleaned", "Raw (mapped, before cleaning)"], horizontal=True, key="preview_which")
    frame = dataset.df if which != "Raw (mapped, before cleaning)" else dataset.mapped
    st.dataframe(frame.head(500), hide_index=True)
    st.caption(f"First {min(500, len(frame)):,} of {len(frame):,} rows.")
    numeric = frame.select_dtypes("number")
    if not numeric.empty:
        st.markdown("**Numeric summary**")
        st.dataframe(numeric.describe().T, column_config={c: st.column_config.NumberColumn(format="%.2f")
                                                          for c in ("mean", "std", "min", "25%", "50%", "75%", "max")})

with tab_schema:
    rows = []
    for field, spec in config.SCHEMA.items():
        rows.append({"field": field, "label": spec["label"], "kind": spec["kind"],
                     "mapped_from": dataset.mapping.get(field) or "—", "description": spec["description"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    if dataset.dropped_columns:
        st.caption("Columns dropped because they were not mapped: " + ", ".join(dataset.dropped_columns))
    st.markdown("**Recognised alternative column names**")
    st.dataframe(pd.DataFrame([{"field": f, "aliases": ", ".join(a)} for f, a in config.COLUMN_ALIASES.items()]),
                 hide_index=True)

with tab_download:
    st.markdown("**Re-apply cleaning with different options**")
    reclean = cleaning_controls("reclean", dataset.cleaning_options)
    if st.button("Re-apply cleaning", key="reclean_btn", icon=":material/cleaning_services:"):
        identity = {c: c for c in dataset.mapped.columns}
        new = state.build_active_dataset(dataset.mapped, identity, dataset.source_label, dataset.is_synthetic, reclean)
        new.mapping, new.dropped_columns = dataset.mapping, dataset.dropped_columns
        state.set_active_dataset(new)
        st.rerun()
    st.divider()
    export = dataset.df.copy()
    if config.TARGET_COLUMN in export.columns:
        export[config.TARGET_COLUMN] = export[config.TARGET_COLUMN].map({1: "Yes", 0: "No"})
    stem = "synthetic_demo" if dataset.is_synthetic else "uploaded"
    ui.download_csv(export, f"roadguard_cleaned_{stem}.csv", "Download cleaned dataset CSV", key="dl_clean",
                    help_text="Cleaned rows in the canonical schema. Missing values are left empty.")
    if st.button("Also save a copy to data/processed/", key="save_processed", icon=":material/save:"):
        config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = config.PROCESSED_DATA_DIR / f"cleaned_{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        export.to_csv(path, index=False)
        st.success(f"Saved a new file: {path.relative_to(config.PROJECT_ROOT)} (existing files are never overwritten).")
