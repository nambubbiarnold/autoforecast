"""ARIMA/SARIMA forecasting model."""
import warnings, numpy as np, pandas as pd
from typing import Dict, Any

warnings.filterwarnings("ignore")


class ARIMAForecaster:
    name = "ARIMA"

    def __init__(self, seed=42, p=1, d=1, q=1, P=0, D=0, Q=0, s=0):
        self.seed = seed
        self.params = dict(p=p, d=d, q=q, P=P, D=D, Q=Q, s=s)
        self.model = None
        self._last_dates = None
        self.aic = self.bic = None

    def fit(self, train_df: pd.DataFrame):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        self._last_dates = train_df["ds"].values
        y = train_df["y"].values
        p = self.params
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = SARIMAX(
                y,
                order=(p["p"], p["d"], p["q"]),
                seasonal_order=(p["P"], p["D"], p["Q"], p["s"]),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        self.aic = self.model.aic
        self.bic = self.model.bic
        return self

    def predict(self, horizon: int, freq: str = "MS") -> pd.DataFrame:
        fc = self.model.get_forecast(steps=horizon)
        mean = fc.predicted_mean
        ci = fc.conf_int(alpha=0.05)
        lower = ci.iloc[:, 0].values if hasattr(ci, "iloc") else ci[:, 0]
        upper = ci.iloc[:, 1].values if hasattr(ci, "iloc") else ci[:, 1]
        last = pd.Timestamp(self._last_dates[-1])
        dates = pd.date_range(start=last + pd.tseries.frequencies.to_offset(freq), periods=horizon, freq=freq)
        return pd.DataFrame({"ds": dates, "yhat": mean, "yhat_lower": lower, "yhat_upper": upper})

    def param_grid(self) -> Dict[str, list]:
        return {"p": [0,1,2], "d": [0,1], "q": [0,1,2], "P": [0,1], "D": [0,1], "Q": [0,1], "s": [0,12]}
