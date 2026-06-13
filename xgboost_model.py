"""XGBoost time-series model with lag features."""
import numpy as np, pandas as pd
from typing import Dict


def _lag_features(series: np.ndarray, lags: int = 12):
    df = pd.DataFrame({"y": series})
    for l in range(1, lags + 1):
        df[f"lag_{l}"] = df["y"].shift(l)
    df["roll_mean_6"]  = df["y"].shift(1).rolling(6).mean()
    df["roll_std_6"]   = df["y"].shift(1).rolling(6).std()
    df["roll_mean_12"] = df["y"].shift(1).rolling(12).mean()
    return df.dropna()


class XGBoostForecaster:
    name = "XGBoost"

    def __init__(self, seed=42, n_estimators=200, max_depth=4,
                 learning_rate=0.05, subsample=0.8, reg_alpha=0.0, reg_lambda=1.0):
        self.seed = seed
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth,
                           learning_rate=learning_rate, subsample=subsample,
                           reg_alpha=reg_alpha, reg_lambda=reg_lambda)
        self.lags = 12
        self._train = None
        self._feat_cols = None
        self.model = None
        self.feature_importance_ = None

    def fit(self, train_df: pd.DataFrame):
        import xgboost as xgb
        self._train = train_df["y"].values.copy()
        self._last_date = train_df["ds"].iloc[-1]
        feat_df = _lag_features(self._train, self.lags)
        self._feat_cols = [c for c in feat_df.columns if c != "y"]
        self.model = xgb.XGBRegressor(random_state=self.seed, verbosity=0, **self.params)
        self.model.fit(feat_df[self._feat_cols].values, feat_df["y"].values)
        self.feature_importance_ = pd.DataFrame({
            "feature": self._feat_cols,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)
        return self

    def predict(self, horizon: int, last_date: pd.Timestamp = None, freq: str = "MS") -> pd.DataFrame:
        if last_date is None:
            last_date = self._last_date
        buf = list(self._train)
        preds = []
        for _ in range(horizon):
            fd = _lag_features(np.array(buf), self.lags)
            if len(fd) == 0:
                preds.append(np.nan); buf.append(np.nan); continue
            X = fd[self._feat_cols].values[-1:]
            p = float(self.model.predict(X)[0])
            preds.append(max(0, p)); buf.append(max(0, p))
        yhat = np.array(preds)
        resid_std = np.std(self._train) * 0.1
        dates = pd.date_range(start=last_date + pd.tseries.frequencies.to_offset(freq), periods=horizon, freq=freq)
        return pd.DataFrame({"ds": dates, "yhat": yhat,
                             "yhat_lower": yhat - 1.96 * resid_std,
                             "yhat_upper": yhat + 1.96 * resid_std})

    def param_grid(self) -> Dict[str, list]:
        return {
            "n_estimators":  [100, 200, 500],
            "max_depth":     [3, 4, 6],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample":     [0.7, 0.8, 1.0],
            "reg_alpha":     [0.0, 0.1],
            "reg_lambda":    [1.0, 5.0],
        }
