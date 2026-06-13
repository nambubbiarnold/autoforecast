"""Theta forecasting method."""
import numpy as np, pandas as pd
from typing import Dict


class ThetaForecaster:
    name = "Theta"

    def __init__(self, seed=42, theta=2.0):
        self.seed = seed
        self.params = dict(theta=theta)
        self._train = None
        self._last_date = None
        self._l = None
        self._b = None

    def fit(self, train_df: pd.DataFrame):
        y = train_df["y"].values.astype(float)
        self._train = y
        self._last_date = train_df["ds"].iloc[-1]
        n = len(y)
        t = np.arange(1, n + 1)
        # OLS trend
        A = np.vstack([t, np.ones(n)]).T
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        self._b = coef[0]
        self._intercept = coef[1]
        detrended = y - (self._b * t + self._intercept)
        # SES on detrended
        alpha = 0.5
        l = detrended[0]
        for v in detrended[1:]:
            l = alpha * v + (1 - alpha) * l
        self._l = l
        self._n = n
        return self

    def predict(self, horizon: int, freq: str = "MS") -> pd.DataFrame:
        n = self._n
        preds = []
        for h in range(1, horizon + 1):
            trend = self._b * (n + h) + self._intercept
            preds.append(max(0, trend + self._l))
        yhat = np.array(preds)
        ci_w = np.std(self._train) * 0.15
        dates = pd.date_range(
            start=self._last_date + pd.tseries.frequencies.to_offset(freq),
            periods=horizon, freq=freq
        )
        return pd.DataFrame({
            "ds": dates, "yhat": yhat,
            "yhat_lower": np.maximum(0, yhat - 1.96 * ci_w),
            "yhat_upper": yhat + 1.96 * ci_w,
        })

    def param_grid(self) -> Dict[str, list]:
        return {"theta": [1.5, 2.0, 3.0]}
