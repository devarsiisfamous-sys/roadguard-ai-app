"""Accident Analytics: filterable exploratory dashboard of the active dataset."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from src import charts, eda, state, ui
from src.utils import format_pct

ui.setup_page("Accident Analytics")
ui.page_header(
    "Accident Analytics",
    "Descriptive patterns in the active dataset. Rates compare groups; counts mostly reflect how many records a group has.",
    "Exploratory data analysis",
)

dataset = state.get_active_dataset()
if dataset is None:
    st.error(f"No dataset is loaded: {state.dataset_error()}")
    ui.nav_link("pages/4_Data_Explorer.py", label="Load data in the Data Explorer", icon=":material/table:")
    st.stop()

ui.synthetic_notice(dataset)
cfg = state.get_feature_config()
base = eda.with_derived_columns(dataset.df, cfg)
options = eda.filter_options(dataset.df)
target_ok = eda.has_target(base)

# --------------------------------------------------------------------------- #
# Filters (one row above the charts)
# --------------------------------------------------------------------------- #
filters: dict = {"categories": {}, "ranges": {}}
with st.expander("Filters", expanded=False, icon=":material/filter_alt:"):
    row1 = st.columns(3)
    if "date" in options:
        start, end = options["date"]
        picked = row1[0].date_input("Date range", value=(start, end), min_value=start, max_value=end, key="flt_date")
        if isinstance(picked, (tuple, list)) and len(picked) == 2:
            filters["date"] = picked
    for i, col in enumerate(["speed", "driver_age"]):
        if col in options:
            low, high = options[col]
            unit = config.SCHEMA[col].get("unit", "")
            filters["ranges"][col] = row1[i + 1].slider(
                f"{config.SCHEMA[col]['label']} ({unit})", min_value=float(low), max_value=float(high),
                value=(float(low), float(high)), step=1.0, key=f"flt_{col}",
            )
    cat_cols = [c for c in ("weather", "road_condition", "vehicle_type", "traffic_density", "light_condition") if c in options]
    row2 = st.columns(max(len(cat_cols), 1))
    for col, slot in zip(cat_cols, row2):
        chosen = slot.multiselect(config.SCHEMA[col]["label"], options[col], default=[], placeholder="All",
                                  key=f"flt_{col}_cats")
        if chosen:
            filters["categories"][col] = chosen
    filters["include_missing"] = st.checkbox(
        "Keep records with missing values in filtered fields", value=True, key="flt_missing",
        help="If unticked, a record whose filtered field is missing is excluded.",
    )

filtered = eda.apply_filters(base, filters)
summary = eda.overview(filtered)
st.caption(f"Showing **{len(filtered):,}** of {len(base):,} records.")
if filtered.empty:
    st.warning("No records match the current filters. Widen the filters to see results.")
    st.stop()

ui.kpi_row([
    ("Records", f"{summary['records']:,}", "Rows after cleaning and filtering."),
    ("Accident-positive", f"{summary['accidents']:,}" if target_ok else "—", "Records with accident = Yes."),
    ("Non-accident", f"{summary['non_accidents']:,}" if target_ok else "—", "Records with accident = No."),
    ("Accident rate", format_pct(summary["accident_share"]) if target_ok else "—",
     "Accidents ÷ labelled records in the current selection."),
    ("Missing cells", format_pct(summary["missing_share"]), "Share of empty cells in the filtered, cleaned data."),
])

if not target_ok:
    st.info("This dataset has no usable accident (Yes/No) column, so only record counts are shown.", icon=":material/info:")
    measure_key = "records"
else:
    labels = {"Accident rate": "accident_rate", "Accident count": "accidents", "Record count": "records"}
    choice = st.radio("Measure", list(labels), horizontal=True, key="analytics_measure",
                      help="Rates are comparable across groups; counts depend on group size (exposure).")
    measure_key = labels.get(choice or "Accident rate", "accident_rate")
overall = summary["accident_share"]


def group_section(column: str, label: str, key: str, horizontal: bool = False) -> None:
    """Bar chart + data table + download for one grouping column."""
    if column not in filtered.columns:
        ui.require_columns(filtered, [column], f"The {label.lower()} chart")
        return
    order = eda.natural_order(column, filtered[column], cfg)
    table = eda.rate_by_group(filtered, column, order)
    title = f"{charts.MEASURES[measure_key]} by {label.lower()}"
    st.markdown(f"**{title}**")
    ui.show_chart(charts.group_bar(table, label, measure_key, overall, horizontal=horizontal), key=f"chart_{key}")
    with st.expander(f"Data · {label}"):
        st.dataframe(table, hide_index=True, column_config=_table_config())
        ui.download_csv(table, f"roadguard_summary_{column}.csv", "Download summary CSV", key=f"dl_{key}")


def _table_config() -> dict:
    pct_cols = ["accident_rate", "ci_low", "ci_high", "share_of_records", "share_of_accidents"]
    cfg_map = {c: st.column_config.NumberColumn(c.replace("_", " "), format="percent") for c in pct_cols}
    cfg_map["group"] = st.column_config.TextColumn("group")
    return cfg_map


tabs = st.tabs(["Road & environment", "Speed, driver & vehicle", "Time patterns", "Correlations", "Records"])

with tabs[0]:
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        group_section("weather", "Weather", "weather")
    with c2:
        group_section("road_condition", "Road condition", "road")
    c3, c4 = st.columns(2, gap="medium")
    with c3:
        group_section("traffic_density", "Traffic density", "traffic")
    with c4:
        group_section("light_condition", "Light condition", "light")

with tabs[1]:
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        group_section("speed_category", "Speed band (km/h)", "speedband")
    with c2:
        if target_ok and ui.require_columns(filtered, ["speed"], "The speed distribution"):
            st.markdown("**Speed distribution by outcome**")
            ui.show_chart(charts.outcome_distribution(filtered, "speed", "Speed (km/h)"), key="chart_speed_dist")
            st.caption("Each outcome is normalised to 100%, so the shapes can be compared despite different group sizes.")
    c3, c4 = st.columns(2, gap="medium")
    with c3:
        group_section("age_band", "Driver age band", "ageband")
    with c4:
        if target_ok and ui.require_columns(filtered, ["driver_age"], "The driver-age distribution"):
            st.markdown("**Driver age distribution by outcome**")
            ui.show_chart(charts.outcome_distribution(filtered, "driver_age", "Driver age (years)", nbins=25),
                          key="chart_age_dist")
    group_section("vehicle_type", "Vehicle type", "vehicle", horizontal=True)

with tabs[2]:
    if not ui.require_columns(filtered, ["time"], "Hourly patterns"):
        pass
    else:
        hourly = filtered.assign(hour_of_day=eda.numeric_groups(filtered, "hour"))
        order = [str(h) for h in range(24)]
        table = eda.rate_by_group(hourly, "hour_of_day", order, include_missing=False)
        st.markdown(f"**{charts.MEASURES[measure_key]} by hour of day**")
        if measure_key == "accident_rate":
            ui.show_chart(charts.rate_line(table, "Hour of day", overall), key="chart_hour")
        else:
            ui.show_chart(charts.group_bar(table, "Hour of day", measure_key), key="chart_hour")
        with st.expander("Data · hour of day"):
            st.dataframe(table, hide_index=True, column_config=_table_config())
            ui.download_csv(table, "roadguard_summary_hour.csv", "Download summary CSV", key="dl_hour")
        st.caption(
            f"Peak hours are configured as "
            + ", ".join(f"{s:02d}:00–{e:02d}:00" for s, e in cfg.peak_hour_ranges)
            + " (change them in the Model Laboratory's feature settings)."
        )
    if ui.require_columns(filtered, ["date"], "Daily and weekend patterns"):
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            group_section("day_of_week", "Day of week", "dow")
        with c2:
            group_section("weekday_or_weekend", "Weekday vs weekend", "weekend")
        if target_ok:
            monthly = eda.rate_by_group(filtered, "year_month", include_missing=False)
            monthly = monthly.sort_values("group").reset_index(drop=True)
            st.markdown("**Accident rate by month**")
            ui.show_chart(charts.rate_line(monthly, "Month", overall, height=300), key="chart_month")
            with st.expander("Data · month"):
                st.dataframe(monthly, hide_index=True, column_config=_table_config())
                ui.download_csv(monthly, "roadguard_summary_month.csv", "Download summary CSV", key="dl_month")

with tabs[3]:
    corr = eda.correlation_matrix(filtered)
    if corr is None:
        st.info("Not enough numeric columns for a correlation analysis.")
    else:
        st.markdown("**Pearson correlation between numeric variables, indicators and the accident outcome**")
        ui.show_chart(charts.correlation_heatmap(corr), key="chart_corr")
        st.caption(
            "Correlation measures only *linear* association between pairs of variables. With the 0/1 accident "
            "outcome it is the point-biserial correlation. Weak correlations can hide strong non-linear "
            "effects, and correlation is not causation."
        )
        corr_table = corr.round(4).reset_index().rename(columns={"index": "variable"})
        ui.download_csv(corr_table, "roadguard_correlations.csv", "Download correlation matrix CSV", key="dl_corr")

with tabs[4]:
    shown = [c for c in dataset.df.columns]
    st.dataframe(filtered[shown].head(1000), hide_index=True)
    st.caption(f"First {min(len(filtered), 1000):,} of {len(filtered):,} filtered records. The download contains all of them.")
    export = filtered[shown].copy()
    if config.TARGET_COLUMN in export.columns:
        export[config.TARGET_COLUMN] = export[config.TARGET_COLUMN].map({1: "Yes", 0: "No"})
    ui.download_csv(export, "roadguard_filtered_records.csv", "Download filtered records CSV", key="dl_filtered")

st.caption(
    "These are descriptive patterns in the data, not causal effects. Groups can differ for reasons the "
    "data does not capture, such as traffic volume or road type."
)
