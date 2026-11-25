"""Score every candidate model on the 2025 Validation year and on walk-forward CV inside Train.

Usage:
    python src/evaluate_models.py                  # all seven models (ARIMAX order search ~2 min)
    python src/evaluate_models.py prophet ridge    # a subset; results merge into existing outputs

Composite score = average of Validation MAPE and walk-forward CV MAPE. Validation alone rewards
a model for doing well on one specific year; the CV folds test the same specification across
several historical stretches. The composite is what the final model was selected on.

Writes:
    outputs/model_comparison.csv      one row per model, sorted by composite score
    outputs/validation_predictions.csv  2025 actuals and each model's predictions
    outputs/model_details.json        selected features, tuned parameters, ARIMAX order
"""
import json
import sys

import numpy as np
import pandas as pd

from config import MODEL_PANELS, OUTPUTS
from metrics import mape, rmse
from models import EVALUATORS

LABELS = {
    "simple_linear": "Model 1: Simple Linear Regression (baseline)",
    "stepwise_ols": "Model 2: Stepwise OLS",
    "ridge": "Model 3: Ridge Regression",
    "random_forest": "Model 4: Random Forest",
    "xgboost": "Model 5: XGBoost",
    "arimax": "Model 6: ARIMAX",
    "prophet": "Model 7: Prophet",
}


def main(models):
    comp_path = OUTPUTS / "model_comparison.csv"
    pred_path = OUTPUTS / "validation_predictions.csv"
    det_path = OUTPUTS / "model_details.json"
    comparison = pd.read_csv(comp_path).set_index("model").to_dict("index") if comp_path.exists() else {}
    preds = pd.read_csv(pred_path, parse_dates=["date"]) if pred_path.exists() else None
    details = json.loads(det_path.read_text()) if det_path.exists() else {}

    for name in models:
        print(f"Evaluating {name}...", flush=True)
        r = EVALUATORS[name]()
        val_mape, cv_mape = mape(r["actual"], r["val_pred"]), float(np.mean(r["fold_mapes"]))
        variant, window = MODEL_PANELS[name]
        comparison[name] = {
            "label": LABELS[name], "panel": variant, "train_window": window,
            "validation_mape": round(val_mape, 3), "validation_rmse": round(rmse(r["actual"], r["val_pred"]), 1),
            "walk_forward_cv_mape": round(cv_mape, 3), "composite_mape": round((val_mape + cv_mape) / 2, 3),
            "cv_fold_mapes": " / ".join(f"{x:.2f}" for x in r["fold_mapes"]),
        }
        details[name] = r["details"]
        frame = pd.DataFrame({"date": r["dates"], "actual": np.round(r["actual"], 3), name: r["val_pred"]})
        if preds is None:
            preds = frame
        else:
            preds = preds.drop(columns=[name], errors="ignore").merge(frame[["date", name]], on="date", how="outer")
        c = comparison[name]
        print(f"  validation {c['validation_mape']:.2f}%  walk-forward CV {c['walk_forward_cv_mape']:.2f}%  "
              f"composite {c['composite_mape']:.2f}%")

    table = pd.DataFrame.from_dict(comparison, orient="index").rename_axis("model").sort_values("composite_mape")
    table.to_csv(comp_path)
    order = ["date", "actual"] + [m for m in LABELS if m in preds.columns]
    preds[order].to_csv(pred_path, index=False)
    det_path.write_text(json.dumps(details, indent=2, default=str))
    print("\n" + table[["validation_mape", "walk_forward_cv_mape", "composite_mape"]].to_string())


if __name__ == "__main__":
    main(sys.argv[1:] or list(EVALUATORS))
