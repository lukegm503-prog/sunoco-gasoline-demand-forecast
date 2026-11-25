"""Engineered features added on top of the weekly panels.

Every panel row is dated on a Friday (the EIA reporting week). Event-style features are set on
the panel week whose trailing 7-day window (Saturday through Friday) contains the event date.
"""
import holidays
import pandas as pd

from config import RAW

# ---------------------------------------------------------------------------------------------
# Holiday travel weeks
# ---------------------------------------------------------------------------------------------
_US = holidays.US(years=range(1990, 2027))
_HOLIDAYS = {
    "New Years Week": {d for d, n in _US.items() if "New Year" in n},
    "Labor Day Week": {d for d, n in _US.items() if "Labor Day" in n},
    "Memorial Day Week": {d for d, n in _US.items() if "Memorial Day" in n},
    "July 4th Week": {d for d, n in _US.items() if "Independence Day" in n and "observed" not in n},
    "Thanksgiving Week": {d for d, n in _US.items() if "Thanksgiving" in n},
}
HOLIDAY_COLS = list(_HOLIDAYS)


def _week_contains(dates, event_dates):
    event_dates = set(pd.Timestamp(d).date() for d in event_dates)
    return [int(any((d - pd.Timedelta(days=o)).date() in event_dates for o in range(7))) for d in dates]


def add_holiday_weeks(df):
    df = df.copy()
    dates = pd.to_datetime(df["date"])
    for col, days in _HOLIDAYS.items():
        df[col] = _week_contains(dates, days)
    return df


# ---------------------------------------------------------------------------------------------
# Gulf Coast extreme weather events (refining-corridor disruptions)
# ---------------------------------------------------------------------------------------------
# Hurricanes: NOAA NHC HURDAT2 landfalls at hurricane strength (>= 64 kt) inside a TX/LA
# refining-corridor box (27.5-30.7N, 89.0-98.0W), deduplicated to one date per storm.
GULF_HURRICANE_LANDFALLS = [
    "1992-08-26", "1997-07-18", "1998-09-28", "2002-10-03", "2003-07-15",
    "2005-07-06", "2005-08-29", "2005-09-24", "2007-09-13", "2008-09-01",
    "2008-09-13", "2012-08-29", "2017-08-26", "2017-10-08", "2019-07-13",
    "2020-08-27", "2020-10-09", "2020-10-28", "2021-08-29", "2021-09-14",
    "2024-07-08", "2024-09-11",
]
# Winter storms: HURDAT2 does not cover these. Sourced from NOAA NCEI billion-dollar disaster
# records for Texas, kept only where EIA or industry reporting confirmed refinery or grid
# shutdowns (a Jan 2014 candidate was excluded for lack of evidence).
WINTER_STORM_EVENTS = [
    "2011-02-03",  # Groundhog Day blizzard / ERCOT rolling blackouts
    "2021-02-19",  # Winter Storm Uri (EIA: refinery inputs fell 18% that week)
    "2022-12-26",  # Winter Storm Elliott
    "2024-01-18",  # Gulf Coast cold snap (~15% of USGC refining capacity offline)
]


def _trailing_week_flag(dates, event_dates):
    events = pd.to_datetime(event_dates)
    return [int(any(d - pd.Timedelta(days=6) <= e <= d for e in events)) for d in dates]


def add_extreme_weather(df):
    df = df.copy()
    dates = pd.to_datetime(df["date"])
    df["Hurricane Landfall Week"] = _trailing_week_flag(dates, GULF_HURRICANE_LANDFALLS)
    df["Winter Storm Week"] = _trailing_week_flag(dates, WINTER_STORM_EVENTS)
    return df


# ---------------------------------------------------------------------------------------------
# Additional FRED predictors (financing cost and economic activity channels)
# ---------------------------------------------------------------------------------------------
# (file, column, output name, publication lag in days). Each value only becomes visible on or
# after the date it would actually have been published, to avoid look-ahead leakage.
FRED_EXTRA = [
    ("fred_nasdaqcom.csv", "NASDAQCOM", "NASDAQ Composite", 1),
    ("fred_fedfunds.csv", "FEDFUNDS", "Fed Funds Rate (%)", 10),
    ("fred_dgs10.csv", "DGS10", "10Y Treasury Yield (%)", 1),
    ("fred_houst.csv", "HOUST", "Housing Starts (thous)", 45),
    ("fred_indpro.csv", "INDPRO", "Industrial Production Index", 45),
]


def add_fred_extra(df):
    df = df.sort_values("date").reset_index(drop=True)
    for fname, col, out_name, lag_days in FRED_EXTRA:
        d = pd.read_csv(RAW / fname)
        d["date"] = pd.to_datetime(d["observation_date"]) + pd.Timedelta(days=lag_days)
        d[out_name] = pd.to_numeric(d[col], errors="coerce")
        d = d[["date", out_name]].sort_values("date").dropna()
        df = pd.merge_asof(df, d, on="date", direction="backward")
        df[out_name] = df[out_name].ffill().bfill()  # leading edge only
    return df


# ---------------------------------------------------------------------------------------------
# Interaction terms (each tied to a stated hypothesis before testing)
# ---------------------------------------------------------------------------------------------
#   Price x Unemployment: price sensitivity should be sharper when the labor market is weak.
#   Price x Sentiment:    pessimistic consumers should cut driving faster when prices rise.
#   Price x CDD:          price elasticity may differ in the summer driving season.
RIDGE_INTERACTIONS = ["Price x Unemployment", "Price x CDD", "Price x Sentiment"]
PROPHET_INTERACTIONS = ["Price x Unemployment", "Price x Sentiment"]


def add_ridge_interactions(df):
    """Retail gas price interactions, computed on the standardized panel (z-score x z-score)."""
    df = df.copy()
    price = df["regular_gas_usd_per_gal (real)"]
    df["Price x Unemployment"] = price * df["Unemployment Rate (%)"]
    df["Price x CDD"] = price * df["Cooling Degree Days (US)"]
    df["Price x Sentiment"] = price * df["Consumer Sentiment (UMich)"]
    return df


def add_prophet_interactions(df):
    """Crude price interactions, computed on the raw-scale panel."""
    df = df.copy()
    price = df["wti_spot_usd_per_bbl (real)"]
    df["Price x Unemployment"] = price * df["Unemployment Rate (%)"]
    df["Price x Sentiment"] = price * df["Consumer Sentiment (UMich)"]
    return df
