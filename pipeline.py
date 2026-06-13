"""Runs all cleaning × model combinations for one SKU."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import warnings, numpy as np, pandas as pd
from typing import Dict, Any, List, Optional, Callable

from data.cleaner import apply_cleaning
from data.splitter import temporal_split
from evaluation.metrics import score as compute_score, all_metrics
from optimization.searcher import search

warnings.filterwarnings("ignore")


def _infer_freq(dates: pd.Series) -> str:
    if len(dates) < 2:
        return "MS"
    diff = (dates.sort_values().diff().dropna().median()).days
    if diff >= 25:   return "MS"
    if diff >= 6:    return "W"
    return "D"


def _load_model_classes(model_names: List[str]) -> Dict[str, Any]:
    classes = {}
    loaders = {
        "ARIMA":         ("models.arima_model",   "ARIMAForecaster"),
        "Exp. Smoothing":("models.es_model",       "ESForecaster"),
        "XGBoost":       ("models.xgboost_model",  "XGBoostForecaster"),
        "Prophet":       ("models.prophet_model",  "ProphetForecaster"),
        "Theta":         ("models.theta_model",    "ThetaForecaster"),
    }
    for name in model_names:
        if name in loaders:
            mod_path, cls_name = loaders[name]
            try:
                import importlib
                mod = importlib.import_module(mod_path)
                classes[name] = getattr(mod, cls_name)
            except Exception as e:
                pass
    return classes


def run_sku_pipeline(
    sku_id: str,
    sku_df: pd.DataFrame,
    cleaning_methods: List[str],
    model_names: List[str],
    metric: str = "SMAPE",
    split_method: str = "last_n",
    n_test: int = 6,
    optim_method: str = "random",
    n_trials: int = 10,
    horizon: int = 12,
    seed: int = 42,
    status_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Run full pipeline for one SKU. Returns dict with results, best pipeline, forecasts.
    status_fn(msg: str) called to report progress.
    """
    def log(msg):
        if status_fn:
            status_fn(msg)

    freq = _infer_freq(sku_df["ds"])
    model_classes = _load_model_classes(model_names)

    train_df, test_df, train_idx, test_idx = temporal_split(
        sku_df, method=split_method, n_test=n_test, seed=seed
    )

    metric_fn = lambda a, p: compute_score(a, p, metric)
    results = []

    for cleaning in cleaning_methods:
        cleaned_train = apply_cleaning(train_df, cleaning)
        cleaned_test  = apply_cleaning(test_df, cleaning)

        for model_name, ModelClass in model_classes.items():
            log(f"  {sku_id} › {cleaning} + {model_name}…")
            model_seed = (seed + hash(f"{sku_id}_{cleaning}_{model_name}")) % (2**31)
            try:
                dummy = ModelClass(seed=model_seed)
                grid = dummy.param_grid()

                best_params, best_score, _ = search(
                    ModelClass, grid, cleaned_train, cleaned_test,
                    metric_fn, method=optim_method, n_trials=n_trials, seed=model_seed
                )

                # Refit best model on full cleaned series
                best_model = ModelClass(seed=model_seed, **best_params)
                best_model.fit(apply_cleaning(sku_df, cleaning))

                # Validation metrics
                val_model = ModelClass(seed=model_seed, **best_params)
                val_model.fit(cleaned_train)
                val_preds = val_model.predict(len(test_df), freq=freq)
                val_metrics = all_metrics(
                    test_df["y"].values,
                    val_preds["yhat"].values[:len(test_df)]
                )

                # Final forecast
                last_date = sku_df["ds"].iloc[-1]
                try:
                    if hasattr(best_model, "predict") and "last_date" in best_model.predict.__code__.co_varnames:
                        fc_df = best_model.predict(horizon, last_date=last_date, freq=freq)
                    else:
                        fc_df = best_model.predict(horizon, freq=freq)
                except Exception:
                    fc_df = best_model.predict(horizon)
                    future = pd.date_range(
                        start=last_date + pd.tseries.frequencies.to_offset(freq),
                        periods=horizon, freq=freq
                    )
                    fc_df["ds"] = future

                results.append({
                    "sku_id":      sku_id,
                    "cleaning":    cleaning,
                    "model":       model_name,
                    "params":      best_params,
                    "score":       best_score,
                    "metrics":     val_metrics,
                    "model_obj":   best_model,
                    "forecast_df": fc_df,
                    "train_df":    train_df,
                    "test_df":     test_df,
                    "error":       None,
                })
                log(f"  ✓ {sku_id} › {cleaning} + {model_name} → {metric}={best_score:.3f}")

            except Exception as e:
                results.append({
                    "sku_id": sku_id, "cleaning": cleaning, "model": model_name,
                    "params": {}, "score": float("inf"), "metrics": {},
                    "model_obj": None, "forecast_df": None,
                    "train_df": train_df, "test_df": test_df,
                    "error": str(e),
                })
                log(f"  ✗ {sku_id} › {cleaning} + {model_name} failed: {e}")

    ranked = sorted([r for r in results if r["error"] is None], key=lambda x: x["score"])
    return {
        "sku_id":   sku_id,
        "results":  results,
        "ranked":   ranked,
        "best":     ranked[0] if ranked else None,
        "freq":     freq,
        "split":    {"train_idx": train_idx, "test_idx": test_idx, "n_test": n_test},
    }
