"""Static (PNG) evaluation figures with Matplotlib + Seaborn, for reports and printing.

The interactive dashboard uses Plotly. These light-themed images go with the
Markdown model report so results can be pasted into documents or slides.
"""

from __future__ import annotations

import io
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend; no display required

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

BLUE = "#2a78d6"
GREY = "#52514e"


def evaluation_png(meta: dict[str, Any], dpi: int = 150) -> bytes:
    """Confusion matrix, ROC and precision-recall curves for one model as PNG bytes."""
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    name = meta["model"]["name"]
    data_kind = "synthetic demo data" if meta["dataset"].get("is_synthetic") else "user-provided data"
    fig.suptitle(f"{name}: held-out test evaluation ({data_kind})", fontsize=13, x=0.01, ha="left")

    cm = np.asarray(meta["confusion_matrix"], dtype=float)
    row_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    labels = np.array([[f"TN\n{int(cm[0, 0]):,}\n{row_pct[0, 0]:.0%}", f"FP\n{int(cm[0, 1]):,}\n{row_pct[0, 1]:.0%}"],
                       [f"FN\n{int(cm[1, 0]):,}\n{row_pct[1, 0]:.0%}", f"TP\n{int(cm[1, 1]):,}\n{row_pct[1, 1]:.0%}"]])
    sns.heatmap(row_pct, annot=labels, fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False, linewidths=2,
                linecolor="white", ax=axes[0],
                xticklabels=["Pred: no accident", "Pred: accident"], yticklabels=["Actual: no accident", "Actual: accident"])
    axes[0].set_title("Confusion matrix (row %)")

    curves = meta.get("curves", {})
    test = meta["metrics"]["test"]
    if curves.get("roc"):
        axes[1].plot(curves["roc"]["fpr"], curves["roc"]["tpr"], color=BLUE, lw=2,
                     label=f"AUC {test.get('roc_auc', float('nan')):.3f}")
        axes[1].plot([0, 1], [0, 1], ls=":", color=GREY, lw=1, label="Chance")
        axes[1].legend(loc="lower right")
    else:
        axes[1].text(0.5, 0.5, "ROC undefined\n(single-class test set)", ha="center", va="center")
    axes[1].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02))

    if curves.get("pr"):
        axes[2].plot(curves["pr"]["recall"], curves["pr"]["precision"], color=BLUE, lw=2,
                     label=f"AP {test.get('average_precision', float('nan')):.3f}")
        axes[2].axhline(curves["pr"]["baseline"], ls=":", color=GREY, lw=1,
                        label=f"No-skill {curves['pr']['baseline']:.1%}")
        axes[2].legend(loc="upper right")
    else:
        axes[2].text(0.5, 0.5, "PR curve undefined", ha="center", va="center")
    axes[2].set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))

    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()
