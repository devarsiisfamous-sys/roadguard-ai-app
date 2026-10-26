"""RoadGuard AI: Streamlit entry point.

Run from the project root with::

    python -m streamlit run app.py

This file renders the Overview dashboard. The other pages live in ``pages/``
(Streamlit's standard multipage layout); ``ui.setup_page`` gives every page the
same theme, navigation and status sidebar.
"""

from __future__ import annotations

import streamlit as st

import config
from src import state, ui
from src.utils import format_metric, format_pct

ui.setup_page("Overview")


def overview_page() -> None:
    """Landing dashboard: project summary, data and model status, quick links."""
    dataset = state.get_active_dataset()
    models = state.saved_models()
    usable = [m for m in models if not m.get("problems")]
    champ_ptr = state.champion()
    champ = next((m for m in usable if champ_ptr and m["model"]["key"] == champ_ptr["key"]), None)

    st.markdown(
        f"""
        <div class="rg-hero">
          <div class="rg-eyebrow">Traffic intelligence · research prototype</div>
          <h1>{config.APP_NAME}</h1>
          <p>{config.APP_TAGLINE}. Explore historical road observations, train and compare machine-learning
          models, and estimate accident risk for road, traffic, weather and driving conditions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if dataset is None:
        st.error(f"The dataset could not be loaded: {state.dataset_error()}", icon=":material/error:")
        ui.nav_link("pages/4_Data_Explorer.py", label="Open the Data Explorer to upload a CSV", icon=":material/table:")
        return

    ui.badges(
        ui.source_badge(dataset),
        ui.badge(f"Model ready · {champ['model']['name']}", "ok") if champ else ui.badge("No champion model yet", "warn"),
        ui.badge("Supervised training available", "ok") if dataset.target.usable
        else ui.badge("Training unavailable for this dataset", "warn"),
    )
    ui.synthetic_notice(dataset)

    target = dataset.target
    ui.kpi_row([
        ("Observations (cleaned)", f"{len(dataset.df):,}", f"{dataset.quality.n_rows:,} rows before cleaning."),
        ("Accident-positive", f"{target.class_counts.get(1, 0):,}" if target.present else "—",
         "Labelled rows where an accident occurred."),
        ("Accident share", format_pct(target.positive_share) if target.present else "—",
         "Accidents ÷ labelled rows. Depends on how observations were sampled, so it is not a population risk."),
        ("Missing cells (raw)", format_pct(dataset.quality.total_missing_share),
         "Share of empty or unparseable cells before cleaning. Cleaning does not fill them in."),
        ("Saved models", f"{len(usable)}", "Trained pipelines stored in the models/ folder."),
    ])

    left, right = st.columns(2, gap="medium")
    with left:
        with st.container(border=True):
            st.markdown("#### Dataset")
            mapped = [c for c, s in dataset.mapping.items() if s]
            issues = len(dataset.quality.issues) + (1 if dataset.quality.n_duplicates else 0)
            st.markdown(
                f"- **Source:** {dataset.source_label} ({dataset.source_kind})\n"
                f"- **Rows:** {dataset.quality.n_rows:,} raw → {len(dataset.df):,} after cleaning\n"
                f"- **Schema fields mapped:** {len(mapped)} of {len(config.SCHEMA)}\n"
                f"- **Data-quality findings:** {issues} (duplicates: {dataset.quality.n_duplicates})\n"
                f"- **Target:** {'usable for training' if target.usable else target.reason}"
            )
            ui.nav_link("pages/4_Data_Explorer.py", label="Inspect data quality or upload a CSV", icon=":material/table:")
    with right:
        with st.container(border=True):
            st.markdown("#### Model")
            if champ:
                test = champ["metrics"]["test"]
                st.markdown(
                    f"- **Champion:** {champ['model']['name']} (chosen by {champ_ptr.get('basis', 'user selection')})\n"
                    f"- **Trained:** {champ['training']['trained_at']} on {champ['dataset']['source_label']}\n"
                    f"- **Held-out test:** F1 {format_metric(test.get('f1'))} · recall {format_metric(test.get('recall'))} · "
                    f"ROC-AUC {format_metric(test.get('roc_auc'))}\n"
                    f"- **Evaluated on:** {'SYNTHETIC data (not a real-world estimate)' if champ['dataset']['is_synthetic'] else 'user-provided data'}"
                )
                if champ["dataset"].get("fingerprint") != dataset.fingerprint:
                    st.caption("ℹ️ This model was trained on a different dataset from the one currently loaded.")
            elif usable:
                st.markdown(f"{len(usable)} model(s) are saved but no champion is selected. Choose one in the Model Laboratory.")
            else:
                st.markdown("No trained model yet. Train and compare models in the **Model Laboratory**, "
                            "about 30 seconds on the demo dataset.")
            ui.nav_link("pages/3_Model_Laboratory.py", label="Open the Model Laboratory", icon=":material/model_training:")

    st.markdown("#### Workflow")
    steps = ["Data collection", "Data cleaning", "Exploratory analysis", "Feature engineering",
             "Model training", "Evaluation", "Prediction"]
    cols = st.columns(len(steps))
    for i, (col, step) in enumerate(zip(cols, steps), start=1):
        col.markdown(f'<div class="rg-step"><b>{i:02d}</b>{step}</div>', unsafe_allow_html=True)

    st.markdown("#### Go to")
    links = st.columns(5)
    links[0].page_link("pages/1_Accident_Analytics.py", label="Accident analytics", icon=":material/insights:")
    links[1].page_link("pages/2_Risk_Prediction.py", label="Risk prediction", icon=":material/speed:")
    links[2].page_link("pages/3_Model_Laboratory.py", label="Model laboratory", icon=":material/model_training:")
    links[3].page_link("pages/4_Data_Explorer.py", label="Data explorer", icon=":material/table:")
    links[4].page_link("pages/5_About.py", label="About & methodology", icon=":material/info:")

    with st.expander("Why this matters"):
        st.markdown(
            "- The WHO *Global status report on road safety 2023* estimates about **1.19 million** road-traffic "
            "deaths per year worldwide.\n"
            "- Road-traffic injuries are the **leading cause of death for people aged 5–29** (WHO).\n"
            "- Predictive analytics can help prioritise preventive measures. Any operational use needs real, "
            "representative data and independent validation."
        )
    ui.disclaimer(compact=True)


overview_page()
