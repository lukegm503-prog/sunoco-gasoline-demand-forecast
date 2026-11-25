"""Produce the 2026 test-period forecast with the selected model (Prophet), plus the ARIMAX,
Stepwise OLS, and equal-weight ensemble forecasts for comparison.

Run evaluate_models.py first: this script reuses the stepwise-selected features and the
ARIMAX order chosen there (outputs/model_details.json).

Each model is refit on Train + Validate (1991/2001 through 2025) before forecasting 2026,
standard practice once the methodology has been validated.

The lagged demand moving averages need last week's actual demand. For the final test week,
the prior week's actual was not yet published, so its value is filled with the ensemble's own
forecast for that week (a one-step recursive forecast) before predicting the final week.
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import OUTPUTS, RAW, VALIDATE_END
from metrics import mape, rmse
from models import (ARIMAX_EXOG, arimax_data, fit_arimax, fit_ols, fit_prophet, load_panel, predict_ols,
                    predict_prophet, prophet_data, stepwise_data)

MA_WINDOWS = {"Target MA (4wk, lagged)": 4, "Target MA (8wk, lagged)": 8}


def through_validate(df, start=None):
    mask = df["date"] <= pd.Timestamp(VALIDATE_END)
    if start:
        mask &= df["date"] >= pd.Timestamp(start)
    return df[mask].reset_index(drop=True)


def test_rows(df):
    return df[df["split"] == "Test"].sort_values("date").reset_index(drop=True)


def main():
    details = json.loads((OUTPUTS / "model_details.json").read_text())
    stepwise_features = details["stepwise_ols"]["selected_features"]
    arimax_order = tuple(details["arimax"]["order"])

    p_df, s_df, a_df = prophet_data(), stepwise_data()[0], arimax_data()
    p_test, s_test, a_test = test_rows(p_df), test_rows(s_df), test_rows(a_df)
    n = len(p_test)
    print(f"Test window: {p_test['date'].min().date()} to {p_test['date'].max().date()} ({n} weeks, "
          f"{p_test['target'].notna().sum()} published when the forecast was made)")

    # ---- Fit on Train + Validate ----
    prophet = fit_prophet(through_validate(p_df))
    s_train = through_validate(s_df)
    stepwise = fit_ols(s_train, stepwise_features)
    a_train = through_validate(a_df)
    arimax = fit_arimax(a_train["target"], a_train[ARIMAX_EXOG], arimax_order)

    # ---- Round 1: every week except the last, whose lagged MAs need an unpublished actual ----
    p1 = predict_prophet(prophet, p_test.iloc[:-1])
    s1 = predict_ols(stepwise, s_test.iloc[:-1], stepwise_features)
    a1 = arimax.get_forecast(n - 1, exog=a_test.iloc[:-1][ARIMAX_EXOG]).predicted_mean.values
    prior_week_estimate = float(np.mean([p1[-1], s1[-1], a1[-1]]))

    # ---- Rebuild the final week's lagged MAs using the ensemble estimate for the prior week ----
    history = p_df.set_index("date")["target"].copy()
    history.loc[p_test["date"].iloc[-2]] = prior_week_estimate
    last_date = p_test["date"].iloc[-1]
    prior = history[history.index < last_date].sort_index()
    ma_fix = {col: float(prior.iloc[-k:].mean()) for col, k in MA_WINDOWS.items()}

    p_last = p_test.iloc[[-1]].copy()
    s_last = s_test.iloc[[-1]].copy()
    a_exog = a_test[ARIMAX_EXOG].copy()
    for col, val in ma_fix.items():
        p_last[col] = val
        s_last[col] = val
        a_exog.iloc[-1, a_exog.columns.get_loc(col)] = val

    prophet_pred = np.append(p1, predict_prophet(prophet, p_last))
    stepwise_pred = np.append(s1, predict_ols(stepwise, s_last, stepwise_features))
    arimax_pred = arimax.get_forecast(n, exog=a_exog).predicted_mean.values

    # Required baseline (time trend only), for an apples-to-apples comparison on the same weeks
    b_df, _ = load_panel("simple_linear")
    b_train, b_test = through_validate(b_df), test_rows(b_df)
    baseline_pred = LinearRegression().fit(b_train[["week_num"]], b_train["target"]).predict(b_test[["week_num"]])

    out = pd.DataFrame({
        "date": p_test["date"].dt.strftime("%Y-%m-%d"),
        "actual": p_test["target"].values,
        "prophet": prophet_pred,
        "arimax": arimax_pred,
        "stepwise_ols": stepwise_pred,
        "ensemble": (prophet_pred + arimax_pred + stepwise_pred) / 3,
        "simple_linear_baseline": baseline_pred,
    })

    # Score against EIA's later-published series, which covers all 36 test weeks. (It matches
    # the demand file used for modeling exactly on every overlapping week.)
    latest = pd.read_excel(RAW / "eia_gasoline_product_supplied_latest.xls", sheet_name="Data 1", header=2)
    latest.columns = ["date", "actual_final"]
    latest["date"] = pd.to_datetime(latest["date"]).dt.strftime("%Y-%m-%d")
    out = out.merge(latest, on="date", how="left")
    out["actual"] = out["actual_final"]
    out = out.drop(columns="actual_final")
    out.round(1).to_csv(OUTPUTS / "test_predictions_2026.csv", index=False)

    print(f"\nAccuracy on all {len(out)} test weeks:")
    for col in ["prophet", "ensemble", "stepwise_ols", "arimax", "simple_linear_baseline"]:
        print(f"  {col:23s} MAPE {mape(out['actual'], out[col]):.2f}%   RMSE {rmse(out['actual'], out[col]):.1f}")
    print("\nSaved outputs/test_predictions_2026.csv")


if __name__ == "__main__":
    main()
