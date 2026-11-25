"""Build the weekly modeling panels from the raw source files in data/raw/.

Usage:
    python src/build_datasets.py            # builds every panel the models need
    python src/build_datasets.py tree full  # builds one (variant, window) panel

Every panel shares the same base: the weekly predictor panel, the EIA demand target, and a
set of series merged in from raw files. Each series is shifted forward by its real-world
publication lag before merging, so a value is only visible from the week it was released
(for example, January CPI is not known until mid-February).

Variants then differ in how they prepare features for different model families:
    linear_v4  backfill gaps, 4/12-week target moving averages, WTI/Brent price MAs,
               log-transform skewed predictors, standardize on Train statistics
    linear_v5  same as v4, plus a larger moving-average library and partial-history
               predictors promoted when they cover at least 90% of this window's Train rows
    tree       no backfilling, log transforms, or scaling; partial predictors stay as NaN
               so tree models can route on missingness natively
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import skew

from config import MODEL_PANELS, PROCESSED, RAW, TRAIN_END, VALIDATE_END, WINDOWS

FULL_RANGE_BASE = [
    "initial_jobless_claims", "rotary_rig_count", "refinery_utilization_pct",
    "vehicle_sales_saar_mil", "vehicle_miles_mil", "smoking_adults_pct", "us_population_thous",
]
DOLLAR = ["wti_spot_usd_per_bbl", "brent_spot_usd_per_bbl", "regular_gas_usd_per_gal",
          "gdp_saar_bil_usd", "pce_saar_bil_usd", "consumer_credit_mil_usd"]
PARTIAL = [
    "sp500_weekly_avg", "tsa_passengers_weekly_avg", "transit_unlinked_trips_thous",
    "ev_sales_vehicles", "regulated_afv_acquisitions", "household_debt_tril_usd",
    "eci_total_comp_index", "fed_broad_dollar_index", "home_price_expectation_pct",
    "inflation_expectation_1y_pct", "mean_annual_earnings_thous_usd",
]
POLICY_DATES = {
    "EV Credit Original Era (2008-10-03+)": "2008-10-03",
    "EV Credit IRA Era (2022-08-16+)": "2022-08-16",
    "EV Credit Expired (2025-09-30+)": "2025-09-30",
    "CAFE Phase 1 Finalized (2010-04-01+)": "2010-04-01",
    "CAFE Phase 2 Finalized (2012-08-01+)": "2012-08-01",
}
PROMOTE_THRESHOLD = 0.10

VARIANTS = {
    "linear_v4": dict(linear=True, promote_partials=False, target_ma=[4, 12], wide_predictor_ma=False),
    "linear_v5": dict(linear=True, promote_partials=True, target_ma=[2, 4, 8, 12, 26], wide_predictor_ma=True),
    "tree": dict(linear=False, promote_partials=False, target_ma=[2, 4, 8, 12, 26], wide_predictor_ma=True),
}


def _merge_lagged(src, d, lag_days):
    d = d.copy()
    d["date"] = d["date"] + pd.Timedelta(days=lag_days)
    return pd.merge_asof(src, d.sort_values("date"), on="date", direction="backward")


def build_panel(variant, window):
    cfg = VARIANTS[variant]
    train_start = pd.Timestamp(WINDOWS[window])

    # ---- Base panel + target ----
    src = pd.read_csv(RAW / "weekly_predictor_panel.csv", parse_dates=["date"])
    src = src.sort_values("date").reset_index(drop=True)
    tgt = pd.read_csv(RAW / "eia_weekly_gasoline_demand.csv", parse_dates=["date"])
    src = src.merge(tgt.rename(columns={"demand_kbbl_d": "target"}), on="date", how="left")

    def split_label(d):
        if d < train_start:
            return "Excluded"
        if d <= pd.Timestamp(TRAIN_END):
            return "Train"
        if d <= pd.Timestamp(VALIDATE_END):
            return "Validate"
        return "Test"
    src["split"] = src["date"].apply(split_label)

    full_range = FULL_RANGE_BASE + DOLLAR
    if cfg["linear"]:
        if cfg["promote_partials"]:
            window_train = src[src["split"] == "Train"]
            full_range += [c for c in PARTIAL if window_train[c].isna().mean() < PROMOTE_THRESHOLD]
        for c in full_range:
            src[c] = src[c].bfill()
        cands = full_range
    else:
        cands = full_range + PARTIAL

    # ---- Refinery sub-series (EIA weekly, released ~5 days after the reporting Friday) ----
    ref = pd.read_excel(RAW / "eia_refinery_utilization.xls", sheet_name="Data 1", header=2)
    ref = ref.rename(columns={ref.columns[0]: "date"})
    ref["date"] = pd.to_datetime(ref["date"])
    ref_map = {
        "Gross Inputs": "Refinery Gross Inputs (Kbbl/d)",
        "Operable Crude Oil Distillation Capacity": "Refinery Operable Capacity (Kbbl/d)",
        "Operating Crude Oil Distillation Capacity": "Refinery Operating Capacity (Kbbl/d)",
        "Idle Crude Oil Distillation Capacity": "Refinery Idle Capacity (Kbbl/d)",
    }
    ref_cols = []
    for raw_col, out_col in ref_map.items():
        match = [c for c in ref.columns if raw_col.lower() in str(c).lower()]
        if match:
            ref_cols.append((match[0], out_col))
    ref_small = ref[["date"] + [m for m, _ in ref_cols]].dropna(subset=["date"])
    src = _merge_lagged(src, ref_small, 7).rename(columns={m: o for m, o in ref_cols})
    cands += [o for _, o in ref_cols]

    # ---- Monthly labor-market and sentiment series (FRED) ----
    for fname, col, out_name, lag in [
        ("fred_unrate.csv", "UNRATE", "Unemployment Rate (%)", 40),
        ("fred_umcsent.csv", "UMCSENT", "Consumer Sentiment (UMich)", 35),
        ("fred_payems.csv", "PAYEMS", "Nonfarm Payrolls (thous)", 40),
    ]:
        d = pd.read_csv(RAW / fname)
        d = pd.DataFrame({"date": pd.to_datetime(d["observation_date"]), out_name: d[col]})
        src = _merge_lagged(src, d, lag)
        # Fills reporting gaps (e.g. the Oct 2025 BLS shutdown delay) and the leading edge.
        src[out_name] = src[out_name].ffill().bfill()
        cands.append(out_name)

    # ---- Gasoline stocks (EIA weekly) ----
    gs = pd.read_csv(RAW / "eia_gasoline_stocks.csv", skiprows=4)
    gs.columns = ["date", "Gasoline Stocks (Kbbl)"]
    gs["date"] = pd.to_datetime(gs["date"], format="%m/%d/%Y")
    src = _merge_lagged(src, gs, 7)
    cands.append("Gasoline Stocks (Kbbl)")

    # ---- Crude oil field production (EIA monthly, ~2-month lag) ----
    cop = pd.read_excel(RAW / "eia_crude_field_production.xls", sheet_name="Data 1", header=2)
    cop = cop.rename(columns={cop.columns[0]: "date", cop.columns[1]: "Crude Oil Field Production (Kbbl/d)"})
    cop["date"] = pd.to_datetime(cop["date"])
    src = _merge_lagged(src, cop[["date", "Crude Oil Field Production (Kbbl/d)"]], 65)
    cands.append("Crude Oil Field Production (Kbbl/d)")

    # ---- Heating and cooling degree days (NOAA monthly) ----
    for fname, desc, out_name in [
        ("noaa_heating_degree_days.csv", "Heating Degree-Days, United States", "Heating Degree Days (US)"),
        ("noaa_cooling_degree_days.csv", "Cooling Degree-Days, United States", "Cooling Degree Days (US)"),
    ]:
        d = pd.read_csv(RAW / fname)
        us = d[d["Description"] == desc].copy()
        us["month"] = us["YYYYMM"].astype(str).str[-2:]
        us = us[us["month"] != "13"]  # drop annual totals
        us["date"] = pd.to_datetime(us["YYYYMM"].astype(str).str[:4] + "-" + us["month"] + "-01")
        src = _merge_lagged(src, us[["date", "Value"]].rename(columns={"Value": out_name}), 20)
        cands.append(out_name)

    # ---- Deflate dollar series to real terms with CPI ----
    cpi = pd.read_excel(RAW / "fred_cpiaucsl.xlsx", sheet_name="Monthly")
    cpi = cpi.rename(columns={"observation_date": "date"})
    cpi["date"] = pd.to_datetime(cpi["date"])
    src = _merge_lagged(src, cpi, 45)
    src["CPIAUCSL"] = src["CPIAUCSL"].ffill().bfill()
    real = {}
    for c in DOLLAR:
        real[c] = f"{c} (real)"
        src[real[c]] = src[c] / (src["CPIAUCSL"] / 100)
        cands.remove(c)
        cands.append(real[c])

    # ---- Crude price changes ----
    for base_col, label in [(real["wti_spot_usd_per_bbl"], "WTI"), (real["brent_spot_usd_per_bbl"], "Brent")]:
        chg = src[base_col].diff()
        if cfg["linear"]:
            # Asymmetric: demand may respond differently to price rises and falls.
            src[f"{label} Price Chg (rise, $ real)"] = chg.clip(lower=0).fillna(0)
            src[f"{label} Price Chg (fall, $ real)"] = (-chg.clip(upper=0)).fillna(0)
            cands += [f"{label} Price Chg (rise, $ real)", f"{label} Price Chg (fall, $ real)"]
        else:
            src[f"{label} Price Chg ($ real)"] = chg  # trees find the sign split themselves
            cands.append(f"{label} Price Chg ($ real)")

    # ---- EV tax credit and CAFE fuel-economy policy eras ----
    for col, thresh in POLICY_DATES.items():
        src[col] = (src["date"] >= pd.Timestamp(thresh)).astype(int)
    train_rows = src[src["split"] == "Train"]
    cands += [c for c in POLICY_DATES if train_rows[c].nunique() > 1]  # drop constant-in-Train

    # ---- Time trend (counts from this window's start) ----
    src = src.sort_values("date").reset_index(drop=True)
    in_window = src["date"] >= train_start
    src["week_num"] = np.nan
    src.loc[in_window, "week_num"] = np.arange(1, in_window.sum() + 1)
    cands.append("week_num")

    # ---- Moving averages ----
    # Target MAs are strictly lagged (shift(1)) so week t never sees its own demand value.
    for k in cfg["target_ma"]:
        col = f"Target MA ({k}wk, lagged)"
        src[col] = src["target"].shift(1).rolling(k).mean()
        cands.append(col)
    ma_sources = [(real["wti_spot_usd_per_bbl"], "WTI Real Price"), (real["brent_spot_usd_per_bbl"], "Brent Real Price")]
    if cfg["wide_predictor_ma"]:
        ma_sources += [("initial_jobless_claims", "Jobless Claims"),
                       (real["regular_gas_usd_per_gal"], "Retail Gas Real Price"),
                       ("refinery_utilization_pct", "Refinery Utilization")]
    for base_col, label in ma_sources:
        for k in [4, 12]:
            col = f"{label} MA ({k}wk)"
            src[col] = src[base_col].rolling(k).mean()
            cands.append(col)

    out_name = f"{variant}_{window}"
    if not cfg["linear"]:
        out = src[src["split"] != "Excluded"].reset_index(drop=True)
        cols = list(dict.fromkeys(["date"] + cands + ["target", "split"]))
        out[cols].to_csv(PROCESSED / f"{out_name}_raw.csv", index=False)
        (PROCESSED / f"{out_name}_candidates.txt").write_text("\n".join(cands))
        return out_name

    for c in [c for c in cands if " MA (" in c]:
        src[c] = src[c].bfill()  # leading edge of the full-history window only

    # ---- Log-transform right-skewed predictors (decided on this window's Train) ----
    train_mask = src["split"] == "Train"
    final_cands = []
    for c in cands:
        vals = src.loc[train_mask, c].dropna()
        if len(vals) > 10 and (vals > 0).all() and skew(vals) > 1:
            src[f"{c} (log)"] = np.log(src[c])
            final_cands.append(f"{c} (log)")
        else:
            final_cands.append(c)

    out = src[src["split"] != "Excluded"].reset_index(drop=True)
    raw_cols = ["date"] + [c for c in out.columns if c not in ("date", "target", "split")] + ["target", "split"]
    out[raw_cols].to_csv(PROCESSED / f"{out_name}_raw.csv", index=False)

    # ---- Standardize on Train statistics only ----
    train_mask = out["split"] == "Train"
    stats, std = {}, out.copy()
    for c in final_cands + ["target"]:
        mu, sd = out.loc[train_mask, c].mean(), out.loc[train_mask, c].std()
        stats[c] = {"mean": mu, "std": sd}
        std[c] = (out[c] - mu) / sd if sd and not np.isnan(sd) else 0.0
    std_cols = list(dict.fromkeys(["date"] + [c for c in std.columns if c not in ("date", "split")] + ["split"]))
    std[std_cols].to_csv(PROCESSED / f"{out_name}_standardized.csv", index=False)
    pd.DataFrame(stats).T.to_csv(PROCESSED / f"{out_name}_scaling_stats.csv")
    (PROCESSED / f"{out_name}_candidates.txt").write_text("\n".join(final_cands))
    return out_name


if __name__ == "__main__":
    targets = [tuple(sys.argv[1:3])] if len(sys.argv) > 2 else sorted(set(MODEL_PANELS.values()))
    for variant, window in targets:
        name = build_panel(variant, window)
        rows = pd.read_csv(PROCESSED / f"{name}_raw.csv", usecols=["split"])["split"].value_counts()
        print(f"built {name:20s} Train={rows.get('Train', 0)}  Validate={rows.get('Validate', 0)}  Test={rows.get('Test', 0)}")
