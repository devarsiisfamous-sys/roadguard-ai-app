"""Plotly figure builders with one consistent dark visual system.

Rules: one y-axis per chart; categorical colours follow the entity (outcome or
model), never its rank; status colours only for risk bands; every chart has
hover details, and pages pair charts with a data table.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import config
from src import theme
from src.utils import format_pct


def style(fig: go.Figure, title: str | None = None, height: int = 360, legend: bool = True,
          x_title: str | None = None, y_title: str | None = None) -> go.Figure:
    """Apply the RoadGuard chart theme in place and return the figure."""
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=theme.TEXT)) if title else None,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, 'Segoe UI', system-ui, sans-serif", size=12, color=theme.TEXT_MUTED),
        margin=dict(l=12, r=12, t=48 if title else 20, b=12),
        hoverlabel=dict(bgcolor=theme.SURFACE_2, bordercolor=theme.BORDER, font=dict(color=theme.TEXT, size=12)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=theme.TEXT)),
        bargap=0.28,
    )
    axis = dict(gridcolor=theme.GRID, zeroline=False, linecolor=theme.BORDER, tickfont=dict(color=theme.TEXT_MUTED),
                title_font=dict(color=theme.TEXT_MUTED), automargin=True, title_standoff=10)
    fig.update_xaxes(**axis, title_text=x_title)
    fig.update_yaxes(**axis, title_text=y_title)
    return fig


MEASURES = {
    "accident_rate": "Accident rate (% of records)",
    "accidents": "Accident count",
    "records": "Record count",
}


def group_bar(table: pd.DataFrame, group_label: str, measure: str = "accident_rate",
              overall_rate: float | None = None, horizontal: bool = False, height: int = 340) -> go.Figure:
    """One-series bar chart of a group table from ``eda.rate_by_group``.

    ``accident_rate`` shows 95% Wilson intervals and a dashed overall-rate
    reference. Counts are labelled as counts so they are not read as risk.
    """
    if measure not in table.columns:
        measure = "records"
    groups = table["group"].astype(str)
    values = table[measure].astype(float)
    hover = [f"<b>%{{{'y' if horizontal else 'x'}}}</b>", "Records: %{customdata[0]:,}"]
    custom_cols = ["records"]
    if "accidents" in table.columns:
        hover += ["Accidents: %{customdata[1]:,}", "Accident rate: %{customdata[2]:.1%}",
                  "95% CI: %{customdata[3]:.1%} – %{customdata[4]:.1%}"]
        custom_cols += ["accidents", "accident_rate", "ci_low", "ci_high"]
    custom = table[custom_cols].to_numpy()
    error = None
    if measure == "accident_rate":
        error = dict(type="data", symmetric=False, array=(table["ci_high"] - table["accident_rate"]).to_numpy(),
                     arrayminus=(table["accident_rate"] - table["ci_low"]).to_numpy(),
                     color=theme.TEXT_MUTED, thickness=1.2, width=4)
    bar = go.Bar(
        x=values if horizontal else groups,
        y=groups if horizontal else values,
        orientation="h" if horizontal else "v",
        marker=dict(color=theme.SERIES[0], cornerradius=4),
        error_x=error if horizontal else None,
        error_y=None if horizontal else error,
        customdata=custom,
        hovertemplate="<br>".join(hover) + "<extra></extra>",
        name=MEASURES[measure],
    )
    fig = go.Figure(bar)
    if measure == "accident_rate" and overall_rate is not None:
        line = dict(color=theme.TEXT_MUTED, dash="dash", width=1.5)
        label = f"Overall {format_pct(overall_rate)}"
        if horizontal:
            fig.add_vline(x=overall_rate, line=line, annotation_text=label, annotation_font_color=theme.TEXT_MUTED, annotation_bgcolor="rgba(11,18,32,0.8)")
        else:
            fig.add_hline(y=overall_rate, line=line, annotation_text=label, annotation_position="top left",
                          annotation_font_color=theme.TEXT_MUTED, annotation_bgcolor="rgba(11,18,32,0.8)")
    value_axis = MEASURES[measure]
    style(fig, height=height, legend=False,
          x_title=value_axis if horizontal else group_label, y_title=group_label if horizontal else value_axis)
    if measure == "accident_rate":
        (fig.update_xaxes if horizontal else fig.update_yaxes)(tickformat=".0%", rangemode="tozero")
    if horizontal:
        fig.update_yaxes(autorange="reversed")
    else:
        fig.update_xaxes(type="category")
    return fig


def rate_line(table: pd.DataFrame, x_label: str, overall_rate: float | None = None, height: int = 320) -> go.Figure:
    """Accident rate across an ordered axis (hour, month) with a 95% CI band."""
    x = table["group"].astype(str)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(x) + list(x[::-1]),
        y=list(table["ci_high"]) + list(table["ci_low"][::-1]),
        fill="toself", fillcolor="rgba(57,135,229,0.18)", line=dict(width=0), hoverinfo="skip",
        name="95% CI", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=table["accident_rate"], mode="lines+markers", name="Accident rate",
        line=dict(color=theme.SERIES[0], width=2), marker=dict(size=8, color=theme.SERIES[0],
                                                                 line=dict(color=theme.BG, width=2)),
        customdata=table[["records", "accidents", "ci_low", "ci_high"]].to_numpy(),
        hovertemplate=("<b>%{x}</b><br>Accident rate: %{y:.1%}<br>Records: %{customdata[0]:,}"
                       "<br>Accidents: %{customdata[1]:,}<br>95% CI: %{customdata[2]:.1%} – %{customdata[3]:.1%}"
                       "<extra></extra>"),
    ))
    if overall_rate is not None:
        fig.add_hline(y=overall_rate, line=dict(color=theme.TEXT_MUTED, dash="dash", width=1.5),
                      annotation_text=f"Overall {format_pct(overall_rate)}", annotation_position="top left",
                      annotation_font_color=theme.TEXT_MUTED, annotation_bgcolor="rgba(11,18,32,0.8)")
    style(fig, height=height, legend=False, x_title=x_label, y_title="Accident rate (% of records)")
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    fig.update_xaxes(type="category")
    if len(x) > 24:
        step = int(np.ceil(len(x) / 12))
        fig.update_xaxes(tickmode="array", tickvals=list(x[::step]), tickangle=0)
    return fig


def outcome_distribution(df: pd.DataFrame, column: str, label: str, nbins: int = 30, height: int = 320) -> go.Figure:
    """Distribution of a numeric column within each outcome, as % of that outcome's records."""
    fig = go.Figure()
    values = pd.to_numeric(df[column], errors="coerce")
    target = df[config.TARGET_COLUMN]
    finite = values.dropna()
    if finite.empty:
        return style(fig, height=height)
    edges = np.histogram_bin_edges(finite, bins=nbins)
    size = float(edges[1] - edges[0]) if len(edges) > 1 else 1.0
    for cls in (0, 1):
        name = config.CLASS_LABELS[cls]
        subset = values[(target == cls).fillna(False).to_numpy(dtype=bool)].dropna()
        fig.add_trace(go.Histogram(
            x=subset, name=f"{name} (n={len(subset):,})", histnorm="percent",
            xbins=dict(start=float(edges[0]), end=float(edges[-1]), size=size),
            marker=dict(color=theme.OUTCOME_COLORS[name], line=dict(color=theme.BG, width=1)),
            hovertemplate=f"{name}<br>{label}: %{{x}}<br>%{{y:.1f}}% of {name.lower()} records<extra></extra>",
        ))
    style(fig, height=height, x_title=label, y_title="% of records within outcome")
    fig.update_layout(barmode="group", bargap=0.12, bargroupgap=0.05)
    return fig


def correlation_heatmap(corr: pd.DataFrame, height: int = 420) -> go.Figure:
    """Diverging heatmap (blue negative, grey zero, red positive) with value labels."""
    labels = [c.replace("_", " ") for c in corr.columns]
    fig = go.Figure(go.Heatmap(
        z=corr.to_numpy(), x=labels, y=labels, zmin=-1, zmax=1, colorscale=theme.DIVERGING,
        text=np.round(corr.to_numpy(), 2), texttemplate="%{text}", textfont=dict(color=theme.TEXT, size=11),
        hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>", xgap=2, ygap=2,
        colorbar=dict(title=dict(text="r", font=dict(color=theme.TEXT_MUTED)), tickfont=dict(color=theme.TEXT_MUTED)),
    ))
    style(fig, height=height, legend=False)
    fig.update_yaxes(autorange="reversed")
    return fig


def confusion_heatmap(cm: list[list[int]], height: int = 320) -> go.Figure:
    """Confusion matrix with counts and row percentages."""
    matrix = np.asarray(cm, dtype=float)
    row_pct = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    names = [["True negative", "False positive"], ["False negative", "True positive"]]
    text = [[f"{names[i][j]}<br><b>{int(matrix[i, j]):,}</b><br>{row_pct[i, j]:.0%} of actual"
             for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=row_pct, x=["Predicted: No accident", "Predicted: Accident"],
        y=["Actual: No accident", "Actual: Accident"], colorscale=theme.SEQUENTIAL, zmin=0, zmax=1,
        text=text, texttemplate="%{text}", textfont=dict(color=theme.TEXT, size=12), showscale=False,
        hovertemplate="%{y}<br>%{x}<br>Share of actual class: %{z:.1%}<extra></extra>", xgap=3, ygap=3,
    ))
    style(fig, height=height, legend=False)
    fig.update_yaxes(autorange="reversed")
    return fig


def roc_curves(models: list[dict[str, Any]], height: int = 360) -> go.Figure:
    """ROC curves (test set) for several models, colour fixed per model."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                             line=dict(color=theme.TEXT_MUTED, dash="dot", width=1), hoverinfo="skip"))
    for meta in models:
        roc = meta.get("curves", {}).get("roc")
        if not roc:
            continue
        auc = meta["metrics"]["test"].get("roc_auc")
        fig.add_trace(go.Scatter(
            x=roc["fpr"], y=roc["tpr"], mode="lines",
            name=f"{meta['model']['name']} (AUC {auc:.3f})" if auc is not None else meta["model"]["name"],
            line=dict(color=theme.MODEL_COLORS.get(meta["model"]["key"], theme.SERIES[0]), width=2),
            hovertemplate=f"{meta['model']['name']}<br>FPR %{{x:.2f}}<br>TPR %{{y:.2f}}<extra></extra>",
        ))
    style(fig, height=height, x_title="False positive rate", y_title="True positive rate (recall)")
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1.02])
    fig.update_layout(legend=dict(orientation="v", yanchor="bottom", y=0.02, xanchor="right", x=0.98,
                                  bgcolor="rgba(17,26,46,0.85)"))
    return fig


def pr_curves(models: list[dict[str, Any]], height: int = 360) -> go.Figure:
    """Precision-recall curves (test set) with the no-skill baseline (accident prevalence)."""
    fig = go.Figure()
    baseline = None
    for meta in models:
        pr = meta.get("curves", {}).get("pr")
        if not pr:
            continue
        baseline = pr.get("baseline", baseline)
        ap = meta["metrics"]["test"].get("average_precision")
        fig.add_trace(go.Scatter(
            x=pr["recall"], y=pr["precision"], mode="lines",
            name=f"{meta['model']['name']} (AP {ap:.3f})" if ap is not None else meta["model"]["name"],
            line=dict(color=theme.MODEL_COLORS.get(meta["model"]["key"], theme.SERIES[0]), width=2),
            hovertemplate=f"{meta['model']['name']}<br>Recall %{{x:.2f}}<br>Precision %{{y:.2f}}<extra></extra>",
        ))
    if baseline is not None:
        fig.add_hline(y=baseline, line=dict(color=theme.TEXT_MUTED, dash="dot", width=1),
                      annotation_text=f"No-skill baseline {baseline:.1%}", annotation_position="bottom right",
                      annotation_font_color=theme.TEXT_MUTED, annotation_bgcolor="rgba(11,18,32,0.8)")
    style(fig, height=height, x_title="Recall", y_title="Precision")
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1.02])
    fig.update_layout(legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="right", x=0.98,
                                  bgcolor="rgba(17,26,46,0.85)"))
    return fig


def model_comparison(table: pd.DataFrame, metric: str, metric_label: str, height: int = 340) -> go.Figure:
    """Cross-validated mean (bar ± std) and held-out test score (marker) per model."""
    fig = go.Figure()
    cv_mean = table.get(f"cv_{metric}_mean")
    if cv_mean is not None and cv_mean.notna().any():
        fig.add_trace(go.Bar(
            y=table["model"], x=cv_mean, orientation="h", name="CV mean (training split)",
            marker=dict(color=theme.SERIES[0], cornerradius=4),
            error_x=dict(type="data", array=table[f"cv_{metric}_std"].fillna(0), color=theme.TEXT_MUTED,
                         thickness=1.2, width=4),
            hovertemplate="%{y}<br>CV mean: %{x:.3f}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        y=table["model"], x=table[f"test_{metric}"], mode="markers", name="Held-out test",
        marker=dict(color=theme.SERIES[1], size=12, symbol="diamond", line=dict(color=theme.BG, width=2)),
        hovertemplate="%{y}<br>Test: %{x:.3f}<extra></extra>",
    ))
    style(fig, height=height, x_title=metric_label)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(rangemode="tozero")
    return fig


def importance_bars(items: list[dict[str, Any]], value_key: str, label: str, error_key: str | None = None,
                    signed: bool = False, height: int | None = None, top: int = 15) -> go.Figure:
    """Horizontal importance bars. ``signed`` colours by sign (blue negative, red positive)."""
    df = pd.DataFrame(items[:top])
    if df.empty:
        return style(go.Figure(), height=height or 300)
    colors = ([theme.POSITIVE if v > 0 else theme.NEGATIVE for v in df[value_key]] if signed
              else theme.SERIES[0])
    error = None
    if error_key and error_key in df:
        error = dict(type="data", array=df[error_key], color=theme.TEXT_MUTED, thickness=1.2, width=4)
    fig = go.Figure(go.Bar(
        y=df["feature"].str.replace("_", " "), x=df[value_key], orientation="h",
        marker=dict(color=colors, cornerradius=4), error_x=error,
        hovertemplate="%{y}<br>" + label + ": %{x:.4f}<extra></extra>",
    ))
    style(fig, height=height or max(260, 28 * len(df) + 80), legend=False, x_title=label)
    fig.update_yaxes(autorange="reversed")
    if signed:
        fig.add_vline(x=0, line=dict(color=theme.TEXT_MUTED, width=1))
    return fig


def factor_bars(factors: pd.DataFrame, height: int | None = None) -> go.Figure:
    """Per-scenario sensitivity: percentage-point change vs a typical value."""
    df = factors[factors["direction"] != "no change"].copy()
    if df.empty:
        return style(go.Figure(), height=200)
    colors = [theme.POSITIVE if v > 0 else theme.NEGATIVE for v in df["effect_pp"]]
    labels = [f"{f}: {v} (typical {t})" for f, v, t in zip(df["feature"], df["your_value"], df["typical_value"])]
    fig = go.Figure(go.Bar(
        y=labels, x=df["effect_pp"], orientation="h", marker=dict(color=colors, cornerradius=4),
        text=[f"{v:+.1f} pp" for v in df["effect_pp"]], textposition="outside",
        textfont=dict(color=theme.TEXT),
        hovertemplate="%{y}<br>Effect vs typical value: %{x:+.2f} percentage points<extra></extra>",
        cliponaxis=False,
    ))
    span = float(df["effect_pp"].abs().max()) or 1.0
    style(fig, height=height or max(220, 34 * len(df) + 70), legend=False,
          x_title="Change in estimated probability vs typical value (percentage points)")
    fig.update_xaxes(range=[-span * 1.35, span * 1.35])
    fig.update_yaxes(autorange="reversed")
    fig.add_vline(x=0, line=dict(color=theme.TEXT_MUTED, width=1))
    return fig


def risk_strip(probability: float, thresholds: tuple[float, float, float], height: int = 140) -> go.Figure:
    """Horizontal band strip (Low → Very High) with a marker at the estimated probability."""
    edges = [0.0, *thresholds, 1.0]
    fig = go.Figure()
    for band, start, end in zip(config.RISK_BANDS, edges[:-1], edges[1:]):
        fig.add_trace(go.Bar(
            x=[end - start], y=["risk"], base=[start], orientation="h", name=band,
            marker=dict(color=theme.STATUS[band], line=dict(color=theme.BG, width=2)), opacity=0.85,
            text=[band], textposition="inside", insidetextanchor="middle", textfont=dict(color="#0B1220", size=12),
            hovertemplate=f"{band}: {start:.0%} – {end:.0%}<extra></extra>",
        ))
    fig.add_shape(type="line", x0=probability, x1=probability, y0=-0.55, y1=0.55,
                  line=dict(color=theme.TEXT, width=3))
    fig.add_annotation(x=probability, y=0.55, yanchor="bottom", text=f"<b>{probability:.1%}</b>", showarrow=False,
                       font=dict(color=theme.TEXT, size=13))
    style(fig, height=height, legend=False)
    fig.update_layout(barmode="overlay", margin=dict(l=8, r=8, t=30, b=8))
    fig.update_xaxes(range=[0, 1], tickformat=".0%", showgrid=False)
    fig.update_yaxes(visible=False)
    return fig


def scenario_comparison(df: pd.DataFrame, height: int | None = None) -> go.Figure:
    """Estimated probability per saved scenario, coloured and labelled by risk band."""
    fig = go.Figure(go.Bar(
        y=df["scenario"], x=df["probability"], orientation="h",
        marker=dict(color=[theme.STATUS.get(b, theme.SERIES[0]) for b in df["risk_band"]], cornerradius=4),
        text=[f"{p:.1%} · {b}" for p, b in zip(df["probability"], df["risk_band"])], textposition="outside",
        textfont=dict(color=theme.TEXT), cliponaxis=False,
        hovertemplate="%{y}<br>Estimated probability: %{x:.1%}<extra></extra>",
    ))
    style(fig, height=height or max(220, 40 * len(df) + 80), legend=False, x_title="Estimated accident probability")
    fig.update_xaxes(range=[0, 1.15], tickformat=".0%")
    fig.update_yaxes(autorange="reversed")
    return fig
