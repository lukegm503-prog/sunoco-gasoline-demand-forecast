"""Error metrics and the walk-forward cross-validation folds used to score every model."""
import numpy as np


def mape(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return float(np.mean(np.abs((y - pred) / y)) * 100)


def rmse(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def r2(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def adj_r2(y, pred, k):
    n = len(y)
    denom = n - k - 1
    if denom <= 0:
        return float("nan")
    return float(1 - (1 - r2(y, pred)) * (n - 1) / denom)


def walk_forward_folds(train_df, start_frac=0.6, n_folds=4):
    """Expanding-window folds inside Train: (train_end, test_start, test_end) row indices.

    The first 60% of Train is always used for fitting; the remaining 40% is split into four
    consecutive test blocks, each scored with a model fit on everything before it. This tests
    stability across several historical stretches, not just the one Validation year.
    """
    n = len(train_df)
    edges = np.linspace(int(n * start_frac), n, n_folds + 1).astype(int)
    return [(edges[i], edges[i], edges[i + 1]) for i in range(n_folds)]
