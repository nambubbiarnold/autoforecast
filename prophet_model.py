"""Facebook Prophet forecasting model."""
import warnings, pandas as pd, numpy as np
from typing import Dict

warnings.filterwarnings("ignore")


class ProphetForecaster:
    name = "Prophet"

    def __init__(self, seed=42, changepoint_prior_scale=0.05,
                 seasonality_prior_scale=10.0, seasonality_mode="additive"):
        self.seed = seed
        self.params = dict(changepoint_prior_scale=changepoint_prior_scale,
                           seasonality_prior_scale=seasonality_prior_scale,
                           seasonality_mode=seasonality_mode)
        self.model = None
        self._train_df = None

    def fit(self, train_df: pd.DataFrame):
        from prophet import Prophet
        self._train_df = train_df[["ds", "y"]].copy()
        self.model = Prophet(**self.params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self._train_df)
        return self

    def predict(self, horizon: int, freq: str = "MS") -> pd.DataFrame:
        future = self.model.make_future_dataframe(periods=horizon, freq=freq)
        fc = self.model.predict(future)
        result = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon).reset_index(drop=True)
        result["yhat"] = result["yhat"].clip(lower=0)
        return result

    def component_fig(self):
        try:
            future = self.model.make_future_dataframe(periods=0)
            fc = self.model.predict(future)
            return self.model.plot_components(fc)
        except Exception:
            return None

    def param_grid(self) -> Dict[str, list]:
        return {
            "changepoint_prior_scale":  [0.001, 0.01, 0.05, 0.1, 0.5],
            "seasonality_prior_scale":  [0.01, 1.0, 10.0],
            "seasonality_mode":         ["additive", "multiplicative"],
        }
