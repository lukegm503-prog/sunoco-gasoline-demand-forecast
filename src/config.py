"""Project paths, the official Train/Validate/Test split, and which panel each model uses."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"

# Official split from the case: Train 1991-2024, Validate 2025, Test 2026 (unseen, graded).
TRAIN_END = "2024-12-27"
VALIDATE_END = "2025-12-26"

# Training windows tested. Each model family was allowed to pick its own start date, since a
# longer history is not automatically better if the demand relationship has shifted.
WINDOWS = {
    "full": "1991-02-08",
    "y2001": "2001-01-05",
    "y2012": "2012-01-06",
}

# Feature panel variants (see build_datasets.py):
#   linear_v4  smaller moving-average library; used by Stepwise OLS
#   linear_v5  expanded moving averages + partial-history predictors promoted per window
#   tree       no backfilling, scaling, or log transforms (trees handle NaN and skew natively)
# Final (variant, window) per model, as reported in the write-up.
MODEL_PANELS = {
    "simple_linear": ("linear_v5", "full"),
    "stepwise_ols": ("linear_v4", "y2001"),
    "ridge": ("linear_v5", "y2001"),
    "random_forest": ("tree", "y2012"),
    "xgboost": ("tree", "y2012"),
    "arimax": ("linear_v5", "full"),
    "prophet": ("linear_v5", "y2001"),
}

for _d in (PROCESSED, OUTPUTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)
