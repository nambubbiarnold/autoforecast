"""Exponential Smoothing (Holt-Winters) via statsmodels."""
import warnings, pandas as pd, numpy as np
from typing import Dict

warnings.filterwarnings("ignore")


class ESForecaster:
    name = "Exp. Smoothing"

    def __init__(self, seed=42, trend="add", damped_trend=False,
                 seasonal="add", seasonal_periods=12, use_boxcox=False):
        self.seed = seed
        self.params = dict(trend=trend, damped_trend=damped_trend,
                           seasonal=seasonal, seasonal_periods=seasonal_periods,
                           use_boxcox=use_boxcox)
        self.model = None
        self._train = None
        self._last_date = None
        self._freq = "MS"

    def fit(self, train_df: pd.DataFrame):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        y = train_df["y"].values.astype(float)
        self._train = y
        self._last_date = train_df["ds"].iloc[-1]
        sp = self.params["seasonal_periods"]
        seasonal = self.params["seasonal"] if len(y) >= 2 * sp else None
        # Force additive if any non-positive values (multiplicative requires strictly positive)
        has_nonpos = (y <= 0).any()
        trend    = "add" if has_nonpos and self.params["trend"] == "mul" else self.params["trend"]
        seasonal = "add" if has_nonpos and seasonal == "mul" else seasonal
        use_bc   = False if has_nonpos else self.params["use_boxcox"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.model = ExponentialSmoothing(
                    y, trend=trend, damped_trend=self.params["damped_trend"],
                    seasonal=seasonal, seasonal_periods=sp if seasonal else None,
                    use_boxcox=use_bc,
                ).fit(optimized=True, use_brute=True)
            except Exception:
                # Minimal fallback
                self.model = ExponentialSmoothing(y, trend="add", seasonal=None).fit()
        return self

    def predict(self, horizon: int, freq: str = "MS") -> pd.DataFrame:
        self._freq = freq
        fc = self.model.forecast(horizon)
        ci_width = np.std(self.model.resid) * 1.96
        dates = pd.date_range(
            start=self._last_date + pd.tseries.frequencies.to_offset(freq),
            periods=horizon, freq=freq
        )
        fc_arr = np.array(fc)
        yhat = np.maximum(0, fc_arr)
        return pd.DataFrame({
            "ds": dates, "yhat": yhat,
            "yhat_lower": np.maximum(0, yhat - ci_width),
            "yhat_upper": yhat + ci_width,
        })

    def param_grid(self) -> Dict[str, list]:
        return {
            "trend":            ["add", "mul"],
            "damped_trend":     [False, True],
            "seasonal":         ["add", "mul"],
            "use_boxcox":       [False, True],
        }
