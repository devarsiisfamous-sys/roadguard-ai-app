"""Colour tokens shared by the Streamlit UI and the Plotly charts.

Chart series colours are the dark-mode steps of a validated categorical palette
(checked for colour-vision-deficiency separation against the navy surface).
Status colours are reserved for risk states and always appear with a text label.
"""

from __future__ import annotations

BG = "#0B1220"
SURFACE = "#111A2E"
SURFACE_2 = "#17223A"
BORDER = "#24324F"
TEXT = "#E6EDF7"
TEXT_MUTED = "#9AA8C1"
ACCENT = "#3B82F6"      # electric blue (UI accents, primary buttons)
ACCENT_CYAN = "#22D3EE"  # cyan highlight
GRID = "rgba(154, 168, 193, 0.14)"

# Categorical series slots, assigned in fixed order and never cycled.
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
OUTCOME_COLORS = {"No accident": SERIES[0], "Accident": SERIES[1]}
MODEL_COLORS = {
    "logistic_regression": SERIES[0],
    "decision_tree": SERIES[1],
    "random_forest": SERIES[2],
    "xgboost": SERIES[3],
    "svm": SERIES[4],
}

# Status palette for risk bands (always paired with a label).
STATUS = {"Low": "#0ca30c", "Moderate": "#fab219", "High": "#ec835a", "Very High": "#d03b3b"}
RISK_ICONS = {"Low": "🟢", "Moderate": "🟡", "High": "🟠", "Very High": "🔴"}

# Diverging (polarity) pair with a neutral midpoint, and a one-hue sequential ramp.
DIVERGING = [[0.0, "#3987e5"], [0.5, "#3a4256"], [1.0, "#e66767"]]
SEQUENTIAL = [[0.0, "#13284a"], [0.35, "#1c5cab"], [0.7, "#3987e5"], [1.0, "#b7d3f6"]]
NEGATIVE = "#3987e5"  # lowers estimate / negative association
POSITIVE = "#e66767"  # raises estimate / positive association
