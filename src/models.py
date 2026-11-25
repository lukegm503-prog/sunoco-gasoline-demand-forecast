"""The seven candidate models. Each evaluate_* function fits on Train, predicts Validate, and
re-scores the same specification on four walk-forward folds inside Train.

Models 1 and 2 were required by the case (simple linear baseline, stepwise OLS). Models 3-6
were chosen to span different modeling approaches. Model 7 (Prophet) was the refinement step,
built from what Models 1-6 showed about which predictors and interactions mattered.
"""
import itertools
import logging
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from config import MODEL_PANELS, PROCESSED
from features import (HOLIDAY_COLS, RIDGE_INTERACTIONS, add_extreme_weather, add_fred_extra,
                      add_holiday_weeks, add_prophet_interactions, add_ridge_interactions)
from metrics import mape, walk_forward_folds

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------------------------
def load_panel(model, kind="raw"):
    variant, window = MODEL_PANELS[model]
    df = pd.read_csv(PROCESSED / f"{variant}_{window}_{kind}.csv", parse_dates=["date"])
    candidates = (PROCESSED / f"{variant}_{window}_candidates.txt").read_text().splitlines()
    return df, candidates


def split(df):
    train = df[df["split"] == "Train"].reset_index(drop=True)
    val = df[df["split"] == "Validate"].reset_index(drop=True)
    return train, val


def _fold_mapes(train, fit_predict):
    out = []
    for tr_end, te_start, te_end in walk_forward_folds(train):
        tr, te = train.iloc[:tr_end], train.iloc[te_start:te_end]
        out.append(mape(te["target"].values, fit_predict(tr, te)))
    return out


def _result(val, val_pred, folds, **details):
    return dict(dates=val["date"], actual=val["target"].values, val_pred=np.asarray(val_pred),
                fold_mapes=folds, details=details)


# ---------------------------------------------------------------------------------------------
# Model 1: simple linear regression on a time trend (required baseline)
# ---------------------------------------------------------------------------------------------
def evaluate_simple_linear():
    df, _ = load_panel("simple_linear")
    train, val = split(df)

    def fit_predict(tr, te):
        m = LinearRegression().fit(tr[["week_num"]].values, tr["target"].values)
        return m.predict(te[["week_num"]].values)

    return _result(val, fit_predict(train, val), _fold_mapes(train, fit_predict), predictors=["week_num"])


# ---------------------------------------------------------------------------------------------
# Model 2: stepwise OLS (BIC forward-backward search, then drop predictors until VIF <= 10)
# ---------------------------------------------------------------------------------------------
def stepwise_bic(X, y):
    def bic(cols):
        if not cols:
            return sm.OLS(y, np.ones(len(y))).fit().bic
        return sm.OLS(y, sm.add_constant(X[cols])).fit().bic

    included, current = [], bic([])
    while True:
        changed = False
        scores = {c: bic(included + [c]) for c in X.columns if c not in included}
        best = min(scores, key=scores.get, default=None)
        if best is not None and scores[best] < current:
            included.append(best)
            current, changed = scores[best], True
        scores = {c: bic([x for x in included if x != c]) for c in included}
        worst = min(scores, key=scores.get, default=None)
        if worst is not None and scores[worst] < current:
            included.remove(worst)
            current, changed = scores[worst], True
        if not changed:
            return included


def vif_cap(X, cols, cap=10.0):
    cols = list(cols)
    while len(cols) > 1:
        Xc = sm.add_constant(X[cols])
        vifs = pd.Series([variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])], index=cols)
        if vifs.max() <= cap:
            break
        cols.remove(vifs.idxmax())
    return cols


def stepwise_data():
    df, cands = load_panel("stepwise_ols")
    return add_holiday_weeks(df), cands + HOLIDAY_COLS


def select_stepwise_features(train, cands):
    return vif_cap(train[cands], stepwise_bic(train[cands], train["target"]))


def fit_ols(train, features):
    return sm.OLS(train["target"], sm.add_constant(train[features])).fit()


def predict_ols(model, df, features):
    return model.predict(sm.add_constant(df[features], has_constant="add")).values


def evaluate_stepwise_ols():
    df, cands = stepwise_data()
    train, val = split(df)
    selected = select_stepwise_features(train, cands)

    def fit_predict(tr, te):
        return predict_ols(fit_ols(tr, selected), te, selected)

    return _result(val, fit_predict(train, val), _fold_mapes(train, fit_predict), selected_features=selected)


# ---------------------------------------------------------------------------------------------
# Model 3: ridge regression on standardized predictors (handles multicollinearity)
# ---------------------------------------------------------------------------------------------
def evaluate_ridge():
    df, cands = load_panel("ridge", kind="standardized")
    raw, _ = load_panel("ridge", kind="raw")
    df = df.merge(add_holiday_weeks(raw)[["date"] + HOLIDAY_COLS], on="date", how="left")
    df = add_ridge_interactions(df)
    cands = cands + HOLIDAY_COLS + RIDGE_INTERACTIONS
    variant, window = MODEL_PANELS["ridge"]
    stats = pd.read_csv(PROCESSED / f"{variant}_{window}_scaling_stats.csv", index_col=0)
    mu, sd = stats.loc["target", "mean"], stats.loc["target", "std"]

    def unscale(z):
        return np.asarray(z) * sd + mu

    train, val = split(df)
    n_splits = max(2, min(5, len(train) // 20))
    alphas = np.logspace(-3, 3, 50)
    ridge = RidgeCV(alphas=alphas, cv=TimeSeriesSplit(n_splits=n_splits)).fit(train[cands].values, train["target"].values)
    val_pred = unscale(ridge.predict(val[cands].values))

    folds = []
    for tr_end, te_start, te_end in walk_forward_folds(train):
        tr, te = train.iloc[:tr_end], train.iloc[te_start:te_end]
        m = RidgeCV(alphas=alphas, cv=min(3, n_splits)).fit(tr[cands].values, tr["target"].values)
        folds.append(mape(unscale(te["target"].values), unscale(m.predict(te[cands].values))))

    out = _result(val, val_pred, folds, alpha=float(ridge.alpha_), n_predictors=len(cands))
    out["actual"] = unscale(val["target"].values)
    return out


# ---------------------------------------------------------------------------------------------
# Models 4-5: tree ensembles (capture nonlinearity; NaN-aware, no scaling needed)
# ---------------------------------------------------------------------------------------------
def tree_data(model):
    df, cands = load_panel(model)
    return add_holiday_weeks(df), cands + HOLIDAY_COLS


def evaluate_random_forest():
    df, cands = tree_data("random_forest")
    train, val = split(df)
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_distributions={"n_estimators": [200, 400], "max_depth": [8, 12, 18],
                             "min_samples_leaf": [2, 3, 5], "max_features": ["sqrt", 0.5, 0.7, 1.0]},
        n_iter=10, cv=TimeSeriesSplit(n_splits=max(2, min(5, len(train) // 150))),
        scoring="neg_mean_absolute_percentage_error", random_state=42, n_jobs=-1,
    ).fit(train[cands], train["target"])
    params = search.best_params_

    def fit_predict(tr, te):
        m = RandomForestRegressor(random_state=42, n_jobs=-1, **params).fit(tr[cands], tr["target"])
        return m.predict(te[cands])

    val_pred = search.best_estimator_.predict(val[cands])
    return _result(val, val_pred, _fold_mapes(train, fit_predict), params=params)


XGB_PARAMS = dict(subsample=0.85, n_estimators=150, min_child_weight=5, max_depth=4, learning_rate=0.05,
                  colsample_bytree=0.7, random_state=42, n_jobs=-1, tree_method="hist")


def evaluate_xgboost():
    df, cands = tree_data("xgboost")
    train, val = split(df)

    def fit_predict(tr, te):
        return XGBRegressor(**XGB_PARAMS).fit(tr[cands], tr["target"]).predict(te[cands])

    return _result(val, fit_predict(train, val), _fold_mapes(train, fit_predict), params=XGB_PARAMS)


# ---------------------------------------------------------------------------------------------
# Model 6: ARIMAX (autoregressive errors plus exogenous predictors)
# ---------------------------------------------------------------------------------------------
ARIMAX_EXOG = ["Heating Degree Days (US)", "Cooling Degree Days (US)",
               "wti_spot_usd_per_bbl (real)", "refinery_utilization_pct",
               "Target MA (4wk, lagged)", "Target MA (8wk, lagged)",
               "New Years Week", "Labor Day Week",
               "Hurricane Landfall Week", "Winter Storm Week"]


def arimax_data():
    df, _ = load_panel("arimax")
    return add_extreme_weather(add_holiday_weeks(df))


def fit_arimax(y, exog, order):
    return SARIMAX(y, exog=exog, order=order, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)


def search_arimax_order(train):
    """Grid search (p, d, q) by 3-fold time-series CV MAPE inside Train. Takes ~2 minutes."""
    y, X = train["target"], train[ARIMAX_EXOG]
    best_order, best_cv = None, np.inf
    for order in itertools.product(range(4), [0, 1], range(4)):
        try:
            scores = []
            for tr, te in TimeSeriesSplit(n_splits=3).split(y):
                m = fit_arimax(y.iloc[tr], X.iloc[tr], order)
                scores.append(mape(y.iloc[te].values, m.get_forecast(len(te), exog=X.iloc[te]).predicted_mean.values))
            if np.mean(scores) < best_cv:
                best_order, best_cv = order, np.mean(scores)
        except Exception:
            continue
    return best_order


def evaluate_arimax():
    df = arimax_data()
    train, val = split(df)
    order = search_arimax_order(train)

    def fit_predict(tr, te):
        m = fit_arimax(tr["target"], tr[ARIMAX_EXOG], order)
        return m.get_forecast(len(te), exog=te[ARIMAX_EXOG]).predicted_mean.values

    return _result(val, fit_predict(train, val), _fold_mapes(train, fit_predict), order=list(order))


# ---------------------------------------------------------------------------------------------
# Model 7: Prophet (trend + yearly seasonality + named regressor effects)
# ---------------------------------------------------------------------------------------------
PROPHET_REGRESSORS = ["Heating Degree Days (US)", "Cooling Degree Days (US)",
                      "wti_spot_usd_per_bbl (real)", "refinery_utilization_pct",
                      "Unemployment Rate (%)", "initial_jobless_claims",
                      "Target MA (4wk, lagged)", "Target MA (8wk, lagged)",
                      "Hurricane Landfall Week", "Winter Storm Week",
                      "Price x Unemployment", "Price x Sentiment",
                      "Fed Funds Rate (%)", "Industrial Production Index"]
PROPHET_CHANGEPOINT_PRIOR = 0.015


def prophet_data():
    df, _ = load_panel("prophet")
    return add_fred_extra(add_prophet_interactions(add_extreme_weather(df)))


def fit_prophet(train):
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                changepoint_prior_scale=PROPHET_CHANGEPOINT_PRIOR)
    for r in PROPHET_REGRESSORS:
        m.add_regressor(r, mode="additive")
    m.fit(train.rename(columns={"date": "ds", "target": "y"})[["ds", "y"] + PROPHET_REGRESSORS])
    return m


def predict_prophet(model, df):
    return model.predict(df.rename(columns={"date": "ds"})[["ds"] + PROPHET_REGRESSORS])["yhat"].values


def evaluate_prophet():
    df = prophet_data()
    train, val = split(df)

    def fit_predict(tr, te):
        return predict_prophet(fit_prophet(tr), te)

    return _result(val, fit_predict(train, val), _fold_mapes(train, fit_predict), regressors=PROPHET_REGRESSORS)


EVALUATORS = {
    "simple_linear": evaluate_simple_linear,
    "stepwise_ols": evaluate_stepwise_ols,
    "ridge": evaluate_ridge,
    "random_forest": evaluate_random_forest,
    "xgboost": evaluate_xgboost,
    "arimax": evaluate_arimax,
    "prophet": evaluate_prophet,
}
