"""Shared Streamlit UI building blocks: theme CSS, headers, badges, sidebar status."""

from __future__ import annotations

import html
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit.errors import StreamlitAPIException

import config
from src import theme
from src.utils import df_to_csv_bytes, format_metric, format_pct

PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}

_CSS = f"""
<style>
:root {{
  --rg-bg: {theme.BG}; --rg-surface: {theme.SURFACE}; --rg-surface-2: {theme.SURFACE_2};
  --rg-border: {theme.BORDER}; --rg-text: {theme.TEXT}; --rg-muted: {theme.TEXT_MUTED};
  --rg-accent: {theme.ACCENT}; --rg-cyan: {theme.ACCENT_CYAN};
}}
.stApp {{ background: radial-gradient(1200px 500px at 85% -10%, rgba(59,130,246,0.10), transparent 60%), var(--rg-bg); }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px; }}
[data-testid="stSidebar"] {{ background: var(--rg-surface); border-right: 1px solid var(--rg-border); }}
[data-testid="stMetric"] {{
  background: var(--rg-surface); border: 1px solid var(--rg-border); border-radius: 12px;
  padding: 14px 16px; min-height: 104px;
}}
[data-testid="stMetricLabel"] p {{ color: var(--rg-muted) !important; font-size: 0.82rem; letter-spacing: .02em; }}
[data-testid="stMetricValue"] {{ color: var(--rg-text); font-weight: 650; }}
.rg-hero {{
  border: 1px solid var(--rg-border); border-radius: 16px; padding: 26px 28px; margin-bottom: 18px;
  background: linear-gradient(135deg, rgba(59,130,246,0.18), rgba(34,211,238,0.06) 55%, rgba(17,26,46,0.6));
}}
.rg-hero h1 {{ margin: 0 0 6px 0; font-size: 2.1rem; color: var(--rg-text); }}
.rg-hero p {{ margin: 0; color: var(--rg-muted); font-size: 1.02rem; max-width: 880px; }}
.rg-eyebrow {{ color: var(--rg-cyan); font-size: .78rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }}
.rg-header h2 {{ margin: 0; color: var(--rg-text); }}
.rg-header p {{ margin: 2px 0 14px 0; color: var(--rg-muted); }}
.rg-badge {{
  display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px;
  font-size: .78rem; font-weight: 600; border: 1px solid; margin: 2px 6px 2px 0; white-space: nowrap;
}}
.rg-badge.synthetic {{ color: #fab219; border-color: rgba(250,178,25,.55); background: rgba(250,178,25,.10); }}
.rg-badge.user {{ color: var(--rg-cyan); border-color: rgba(34,211,238,.5); background: rgba(34,211,238,.08); }}
.rg-badge.ok {{ color: #4ade80; border-color: rgba(12,163,12,.6); background: rgba(12,163,12,.10); }}
.rg-badge.warn {{ color: #ec835a; border-color: rgba(236,131,90,.6); background: rgba(236,131,90,.10); }}
.rg-badge.neutral {{ color: var(--rg-muted); border-color: var(--rg-border); background: var(--rg-surface-2); }}
.rg-risk {{
  display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px; border-radius: 10px;
  font-weight: 700; font-size: 1.05rem; color: #0B1220;
}}
.rg-card {{
  background: var(--rg-surface); border: 1px solid var(--rg-border); border-radius: 12px; padding: 16px 18px; height: 100%;
}}
.rg-card h4 {{ margin: 0 0 6px 0; color: var(--rg-text); font-size: 1rem; }}
.rg-card p, .rg-card li {{ color: var(--rg-muted); font-size: .9rem; margin: 0; }}
.rg-step {{ display:flex; gap:10px; align-items:center; background: var(--rg-surface); border:1px solid var(--rg-border);
  border-radius: 10px; padding: 10px 12px; color: var(--rg-text); font-size: .9rem; }}
.rg-step b {{ color: var(--rg-cyan); font-variant-numeric: tabular-nums; }}
.rg-big {{ font-size: 2.6rem; font-weight: 750; color: var(--rg-text); line-height: 1.1; font-variant-numeric: tabular-nums; }}
.rg-muted {{ color: var(--rg-muted); font-size: .88rem; }}
div[data-testid="stExpander"] details {{ background: var(--rg-surface); border-color: var(--rg-border); border-radius: 12px; }}
</style>
"""


def inject_css() -> None:
    """Apply the RoadGuard visual system (call once per run)."""
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str, eyebrow: str | None = None) -> None:
    """Consistent page title block."""
    eyebrow_html = f'<div class="rg-eyebrow">{html.escape(eyebrow)}</div>' if eyebrow else ""
    st.markdown(
        f'<div class="rg-header">{eyebrow_html}<h2>{html.escape(title)}</h2><p>{html.escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def badge(text: str, kind: str = "neutral") -> str:
    """HTML for a small pill badge (kinds: synthetic, user, ok, warn, neutral)."""
    return f'<span class="rg-badge {kind}">{html.escape(text)}</span>'


def badges(*items: str) -> None:
    st.markdown("".join(items), unsafe_allow_html=True)


def source_badge(dataset: Any) -> str:
    if dataset is None:
        return badge("No dataset", "warn")
    return badge("⚠ SYNTHETIC demo data", "synthetic") if dataset.is_synthetic else badge("User-provided data", "user")


def risk_chip(band: str) -> str:
    """Risk band label with icon + text + status colour (never colour alone)."""
    colour = theme.STATUS.get(band, theme.TEXT_MUTED)
    icon = theme.RISK_ICONS.get(band, "")
    return f'<span class="rg-risk" style="background:{colour}">{icon} {html.escape(band)} risk</span>'


def show_chart(fig: go.Figure, key: str | None = None) -> None:
    """Render a Plotly figure with the app's own theme (not Streamlit's override)."""
    st.plotly_chart(fig, theme=None, config=PLOTLY_CONFIG, key=key)


def download_csv(df: pd.DataFrame, filename: str, label: str, key: str, help_text: str | None = None) -> None:
    """CSV download button for a DataFrame (disabled when empty)."""
    st.download_button(label, data=df_to_csv_bytes(df), file_name=filename, mime="text/csv", key=key,
                       disabled=df.empty, help=help_text, icon=":material/download:")


def disclaimer(compact: bool = False) -> None:
    """Safety / intended-use notice."""
    if compact:
        st.caption(f"⚠️ {config.SAFETY_DISCLAIMER}")
    else:
        st.warning(config.SAFETY_DISCLAIMER, icon=":material/health_and_safety:")


def synthetic_notice(dataset: Any) -> None:
    if dataset is not None and dataset.is_synthetic:
        st.info(
            "You are viewing the **synthetic demonstration dataset**. Its records were simulated from a "
            "hand-written risk formula. Patterns and model scores show how the pipeline works, "
            "not real-world accident behaviour.",
            icon=":material/science:",
        )


def require_columns(df: pd.DataFrame, columns: list[str], what: str) -> bool:
    """Show an explanatory message and return False when ``columns`` are missing."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        labels = ", ".join(config.SCHEMA.get(c, {}).get("label", c) for c in missing)
        st.info(f"{what} needs: **{labels}**, which this dataset does not provide.", icon=":material/info:")
        return False
    return True


def render_sidebar_status(dataset: Any, models: list[dict[str, Any]], champion: dict[str, Any] | None) -> None:
    """Data-source and model status, shown on every page."""
    with st.sidebar:
        st.markdown("---")
        st.markdown("**Data source**")
        if dataset is None:
            st.markdown(badge("Not loaded", "warn"), unsafe_allow_html=True)
        else:
            st.markdown(source_badge(dataset), unsafe_allow_html=True)
            st.caption(f"{dataset.source_label}  \n{len(dataset.df):,} rows after cleaning")
        st.markdown("**Model status**")
        usable = [m for m in models if not m.get("problems")]
        if not usable:
            st.markdown(badge("No trained model", "warn"), unsafe_allow_html=True)
            st.caption("Train one in the Model Laboratory.")
        else:
            champ = next((m for m in usable if champion and m["model"]["key"] == champion["key"]), None)
            if champ:
                st.markdown(badge(f"Ready · {champ['model']['name']}", "ok"), unsafe_allow_html=True)
                f1 = champ["metrics"]["test"].get("f1")
                synthetic = " (synthetic data)" if champ["dataset"].get("is_synthetic") else ""
                st.caption(f"Champion · test F1 {format_metric(f1)}{synthetic}  \n{len(usable)} saved model(s)")
            else:
                st.markdown(badge(f"{len(usable)} model(s) saved", "ok"), unsafe_allow_html=True)
                st.caption("No champion selected yet.")
        st.caption("Research prototype: not a certified warning system.")


def kpi_row(items: list[tuple[str, str, str | None]]) -> None:
    """Row of st.metric cards: (label, value, help)."""
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        col.metric(label, value, help=help_text)


def pct(value: float | None) -> str:
    return format_pct(value)


def nav_link(page: str, label: str, icon: str | None = None) -> None:
    """``st.page_link`` that degrades to plain text when the page runs outside ``app.py`` navigation."""
    try:
        st.page_link(page, label=label, icon=icon)
    except StreamlitAPIException:
        st.caption(f"→ {label} (open via the sidebar)")



NAV_ITEMS: tuple[tuple[str, str, str], ...] = (
    ("app.py", "Overview", ":material/dashboard:"),
    ("pages/1_Accident_Analytics.py", "Accident Analytics", ":material/insights:"),
    ("pages/2_Risk_Prediction.py", "Risk Prediction", ":material/speed:"),
    ("pages/3_Model_Laboratory.py", "Model Laboratory", ":material/model_training:"),
    ("pages/4_Data_Explorer.py", "Data Explorer", ":material/table:"),
    ("pages/5_About.py", "About & Methodology", ":material/info:"),
)


def setup_page(title: str) -> None:
    """First call on every page: page config, theme, branded navigation and status sidebar.

    The app uses Streamlit's standard ``pages/`` multipage mode. The default
    sidebar menu is hidden (``.streamlit/config.toml``) and replaced by this one,
    so every page gets the same labels, icons and status panel.
    """
    from src import state  # local import: state depends on Streamlit session context

    st.set_page_config(page_title=f"{title} · {config.APP_NAME}", page_icon="🛣️", layout="wide",
                       initial_sidebar_state="expanded")
    inject_css()
    with st.sidebar:
        st.markdown(f"### 🛣️ {config.APP_NAME}")
        st.caption(config.APP_TAGLINE)
        for path, label, icon in NAV_ITEMS:
            nav_link(path, label, icon)
    render_sidebar_status(state.get_active_dataset(), state.saved_models(), state.champion())
