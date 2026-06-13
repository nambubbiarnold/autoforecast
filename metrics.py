"""Forecasting error metrics."""
import numpy as np
from typing import Dict


def smape(a, p):
    a, p = np.array(a, float), np.array(p, float)
    d = (np.abs(a) + np.abs(p)) / 2
    mask = d > 0
    return float(np.mean(np.abs(a[mask] - p[mask]) / d[mask]) * 100) if mask.any() else 0.0

def mae(a, p):
    return float(np.mean(np.abs(np.array(a) - np.array(p))))

def rmse(a, p):
    return float(np.sqrt(np.mean((np.array(a) - np.array(p)) ** 2)))

def mape(a, p):
    a, p = np.array(a, float), np.array(p, float)
    mask = a != 0
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100) if mask.any() else 0.0

METRICS = {"SMAPE": smape, "MAE": mae, "RMSE": rmse, "MAPE": mape}

def score(actual, predicted, metric="SMAPE") -> float:
    return METRICS.get(metric.upper(), smape)(actual, predicted)

def all_metrics(actual, predicted) -> Dict[str, float]:
    return {name: fn(actual, predicted) for name, fn in METRICS.items()}
