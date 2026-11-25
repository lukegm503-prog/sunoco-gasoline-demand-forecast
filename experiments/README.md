# What was tested along the way

The `src/` pipeline contains only the final specification of each model. Getting there took
roughly 140 exploratory scripts, which are not included here. This is a summary of what they
tested and what was kept.

## Training window start dates
Each model family was evaluated with training windows starting in 1991 (full history), 1996,
1999, 2000, 2001, 2002, 2003, 2005, 2006, 2010, 2012, and 2015. A longer history is not
automatically better if the relationship between predictors and demand has shifted.

| Model | Window used | Note |
|---|---|---|
| Stepwise OLS, Ridge, Prophet | 2001 | Best balance of validation and walk-forward accuracy |
| ARIMAX, Simple linear | 1991 | Full history |
| Random Forest, XGBoost | 2012 | Chosen on validation accuracy; see "What I'd revisit" in the main README |

## Feature panel versions
Five versions of the weekly panel were built (v1 to v5). v4 and v5 are kept in
`build_datasets.py` because the final models use them. v5 added a larger moving-average
library and promoted partial-history predictors (S&P 500, TSA passengers, household debt, and
others) whenever they covered at least 90% of a window's training rows.

## Interaction terms
| Interaction | Hypothesis | Result |
|---|---|---|
| Price x Unemployment | Price sensitivity is sharper when jobs are scarce | Kept |
| Price x Consumer Sentiment | Pessimistic consumers cut driving faster when prices rise | Kept |
| Unemployment x Cooling Degree Days | Weak labor market suppresses summer driving specifically | Dropped |
| Vehicle Miles x EV era | The same miles use less gasoline after 2022 | Dropped (the EV flag already captured it) |
| Refinery Utilization x Heating Degree Days | Winter distillate demand competes for refinery capacity | Dropped |

## Other candidates tested and dropped
- **Generalized additive model (GAM) and PLS regression:** built and scored but left out of
  the final seven, since the case allowed seven model slots. Neither beat Prophet. On the
  composite, GAM (full history) scored 3.07%, slightly ahead of ARIMAX and Stepwise OLS, and PLS
  scored 3.35%.
- **Holiday weeks beyond New Year's and Labor Day** (Memorial Day, July 4th, Thanksgiving):
  offered to Stepwise OLS, Ridge, and the tree models as candidates. Stepwise selection kept
  only New Year's and Labor Day, so ARIMAX uses only those two.
- **Refinery shock indicators** from week-over-week utilization drops: failed because the
  utilization series is monthly data repeated weekly, so it cannot see a one-week event. This
  is why extreme weather events were sourced from NOAA records instead.
- **Additional FRED series** (NASDAQ, 10-year Treasury yield, housing starts): tested in
  Prophet; only the Fed Funds rate and industrial production were kept.
- **Ensembles:** equal-weight and CV-optimized weights of the top three models. Simple
  averaging beat optimized weights, since with four folds the optimizer overfits its own
  weights. The equal-weight ensemble is reported alongside Prophet in the 2026 results.
