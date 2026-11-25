"""Charts for the README, built from the CSVs in outputs/."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGURES, OUTPUTS

SURFACE, INK, INK_2, INK_3 = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SHORT = {"simple_linear": "Simple\nLinear", "stepwise_ols": "Stepwise\nOLS", "ridge": "Ridge",
         "random_forest": "Random\nForest", "xgboost": "XGBoost", "arimax": "ARIMAX", "prophet": "Prophet"}
plt.rcParams["font.family"] = "DejaVu Sans"


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="y", colors=INK_3, labelsize=9)
    ax.set_title(title, fontsize=13, color=INK, fontweight="bold", loc="left", pad=14)


def model_comparison():
    t = pd.read_csv(OUTPUTS / "model_comparison.csv").sort_values("composite_mape")
    series = [("validation_mape", "Validation MAPE (2025)", BLUE),
              ("walk_forward_cv_mape", "Walk-forward CV MAPE", ORANGE),
              ("composite_mape", "Composite (average)", AQUA)]
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    x, w = np.arange(len(t)), 0.26
    for i, (col, label, color) in enumerate(series):
        bars = ax.bar(x + (i - 1) * w, t[col], w, label=label, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        for b, v in zip(bars, t[col]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.13, f"{v:.2f}%", ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in t["model"]], fontsize=10, color=INK)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("MAPE (%), lower is better", fontsize=10, color=INK_2)
    ax.set_ylim(0, t["walk_forward_cv_mape"].max() * 1.18)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK_2)
    style(ax, "Model comparison, sorted by composite score")
    plt.tight_layout()
    plt.savefig(FIGURES / "model_comparison.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def test_forecast():
    t = pd.read_csv(OUTPUTS / "test_predictions_2026.csv", parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.plot(t["date"], t["actual"], color=INK_2, linewidth=2, marker="o", markersize=4, label="Actual (EIA)", zorder=3)
    ax.plot(t["date"], t["prophet"], color=BLUE, linewidth=2, label="Prophet forecast", zorder=4)
    last = t.iloc[-1]
    ax.text(last["date"] + pd.Timedelta(days=3), last["actual"], "Actual", color=INK_2, fontsize=9, va="center")
    ax.text(last["date"] + pd.Timedelta(days=3), last["prophet"], "Prophet", color=INK_2, fontsize=9, va="center")
    ax.set_xlim(t["date"].min() - pd.Timedelta(days=4), t["date"].max() + pd.Timedelta(days=24))
    ax.set_ylabel("Thousand barrels per day", fontsize=10, color=INK_2)
    ax.tick_params(axis="x", colors=INK_3, labelsize=9)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=INK_2)
    style(ax, "2026 test period: weekly U.S. gasoline demand vs. forecast")
    plt.tight_layout()
    plt.savefig(FIGURES / "test_forecast_2026.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    model_comparison()
    test_forecast()
    print("Saved charts to outputs/figures/")
