"""
AutoForecast Pro — Single-file Streamlit app
All modules inlined. Works with flat GitHub repo structure.
Run: streamlit run app.py
"""
import sys, os, json, io, warnings, itertools, random
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AutoForecast Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"]   { background: #0f0d1c; }
[data-testid="stSidebar"]            { background: #161428 !important; border-right:1px solid #2a2750; }
[data-testid="stHeader"]             { background: transparent; }
body, .stMarkdown, p, label          { color: #e0daf5 !important; font-family:'Segoe UI',system-ui,sans-serif; }
h1,h2,h3                             { color: #f0ecff !important; }
.af-metric { background:#1c1a32; border:1px solid #2a2750; border-radius:10px; padding:14px 18px; }
.af-metric .label { font-size:11px; color:#7c6fad; text-transform:uppercase; letter-spacing:.8px; }
.af-metric .value { font-size:26px; font-weight:700; color:#f0ecff; line-height:1.1; margin-top:4px; }
.af-metric .sub   { font-size:11px; color:#7c6fad; margin-top:3px; }
.sb-section { font-size:10px; font-weight:600; color:#7c6fad; text-transform:uppercase;
              letter-spacing:1px; margin:16px 0 6px; border-bottom:1px solid #2a2750; padding-bottom:4px; }
.status-log { background:#0a0914; border:1px solid #2a2750; border-radius:8px;
              padding:12px; font-family:'Courier New',monospace; font-size:12px;
              color:#7c6fad; height:200px; overflow-y:auto; }
.log-ok  { color:#10b981; }
.log-err { color:#ef4444; }
.log-run { color:#a78bfa; }
.stButton > button { background:#6d28d9 !important; color:white !important;
                     border:none !important; border-radius:8px !important; font-weight:600 !important; }
.stButton > button:hover { background:#7c3aed !important; }
.stTabs [data-baseweb="tab-list"] { background:#1c1a32; border-radius:8px; padding:4px; gap:4px; }
.stTabs [data-baseweb="tab"]      { background:transparent; color:#7c6fad; border-radius:6px; }
.stTabs [aria-selected="true"]    { background:#6d28d9 !important; color:white !important; }
.stProgress > div > div           { background:#6d28d9 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def get_sheet_names(uploaded_file) -> List[str]:
    try:
        xl = pd.ExcelFile(uploaded_file)
        return xl.sheet_names
    except Exception:
        return []

def load_uploaded_file(uploaded_file, sheet_name=None) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        kwargs = {"sheet_name": sheet_name} if sheet_name else {}
        return pd.read_excel(uploaded_file, **kwargs)
    elif name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    raise ValueError(f"Unsupported file: {uploaded_file.name}. Use .xlsx, .xls, or .csv")

def auto_detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    date_kw = ["date","ds","time","timestamp","period","month","week","day"]
    val_kw  = ["y","value","sales","demand","qty","quantity","units","revenue","amount","volume"]
    sku_kw  = ["sku","product","item","store","region","category","sku_id","product_id","item_id","group"]
    def find(kws, exclude=None):
        exclude = exclude or []
        for col in df.columns:
            if col not in exclude and any(k in col.lower() for k in kws):
                return col
        return None
    date_col = find(date_kw)
    val_col  = find(val_kw,  exclude=[date_col] if date_col else [])
    sku_col  = find(sku_kw,  exclude=[c for c in [date_col, val_col] if c])
    if date_col is None:
        for col in df.columns:
            try: pd.to_datetime(df[col].head(5)); date_col = col; break
            except: pass
    if val_col is None:
        for c in df.select_dtypes(include=[np.number]).columns:
            if c != date_col: val_col = c; break
    return date_col, val_col, sku_col

def validate_and_prepare(df, date_col, val_col, sku_col=None) -> pd.DataFrame:
    missing = [c for c in [date_col, val_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")
    cols = [date_col, val_col] + ([sku_col] if sku_col and sku_col in df.columns else [])
    result = df[cols].copy().rename(columns={date_col:"ds", val_col:"y"})
    if sku_col and sku_col in df.columns:
        result = result.rename(columns={sku_col:"sku_id"})
    result["ds"] = pd.to_datetime(result["ds"], errors="coerce")
    result["y"]  = pd.to_numeric(result["y"],   errors="coerce")
    result = result.dropna(subset=["ds","y"]).sort_values("ds").reset_index(drop=True)
    if len(result) < 10:
        raise ValueError(f"Only {len(result)} valid rows — need at least 10.")
    return result

def split_by_sku(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if "sku_id" not in df.columns:
        return {"All": df[["ds","y"]].reset_index(drop=True)}
    return {str(sku): grp[["ds","y"]].reset_index(drop=True) for sku, grp in df.groupby("sku_id")}

# ══════════════════════════════════════════════════════════════════════════════
# CLEANING
# ══════════════════════════════════════════════════════════════════════════════
def hampel_filter(arr, window=7, sigma=3.0):
    k, out = 1.4826, arr.copy().astype(float)
    for i in range(len(arr)):
        win = arr[max(0,i-window):min(len(arr),i+window+1)]
        med = np.median(win)
        mad = k * np.median(np.abs(win - med))
        if mad > 0 and np.abs(arr[i]-med) > sigma*mad:
            out[i] = med
    return out

def dwt_denoise(arr, wavelet="db4", level=3, mode="soft"):
    try:
        import pywt
        coeffs = pywt.wavedec(arr, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        thr   = sigma * np.sqrt(2 * np.log(len(arr)))
        rec   = pywt.waverec([coeffs[0]] + [pywt.threshold(c,thr,mode=mode) for c in coeffs[1:]], wavelet)
        return rec[:len(arr)]
    except Exception:
        return arr

def ssa_denoise(arr, window=None, n_components=3):
    n = len(arr)
    L = min(window or min(max(2,n//4),20), n-1)
    K = n - L + 1
    X = np.array([arr[i:i+L] for i in range(K)]).T
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    rec, cnt = np.zeros(n), np.zeros(n)
    for i in range(min(n_components, len(s))):
        Xi = s[i] * np.outer(U[:,i], Vt[i,:])
        for r in range(L):
            for c in range(K):
                rec[r+c] += Xi[r,c]; cnt[r+c] += 1
    return np.where(cnt>0, rec/cnt, arr)

def apply_cleaning(df, method, params=None):
    params = params or {}
    df = df.copy()
    y = df["y"].values.astype(float)
    if   method == "hampel": y = hampel_filter(y, **{k:params.get(k,v) for k,v in [("window",7),("sigma",3.0)]})
    elif method == "dwt":    y = dwt_denoise(y)
    elif method == "ssa":    y = ssa_denoise(y)
    df["y"] = y
    return df

CLEANING_LABELS = {"none":"No cleaning","hampel":"Hampel filter","dwt":"DWT denoising","ssa":"SSA denoising"}

# ══════════════════════════════════════════════════════════════════════════════
# SPLITTING
# ══════════════════════════════════════════════════════════════════════════════
def temporal_split(df, n_test=6, seed=42):
    np.random.seed(seed)
    n = len(df)
    split = max(1, n - n_test)
    ti, vi = list(range(split)), list(range(split, n))
    return df.iloc[ti].reset_index(drop=True), df.iloc[vi].reset_index(drop=True), ti, vi

# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════
def smape(a,p):
    a,p = np.array(a,float), np.array(p,float)
    d = (np.abs(a)+np.abs(p))/2; mask = d>0
    return float(np.mean(np.abs(a[mask]-p[mask])/d[mask])*100) if mask.any() else 0.0
def mae(a,p):   return float(np.mean(np.abs(np.array(a)-np.array(p))))
def rmse(a,p):  return float(np.sqrt(np.mean((np.array(a)-np.array(p))**2)))
def mape(a,p):
    a,p = np.array(a,float),np.array(p,float); mask = a!=0
    return float(np.mean(np.abs((a[mask]-p[mask])/a[mask]))*100) if mask.any() else 0.0
METRICS = {"SMAPE":smape,"MAE":mae,"RMSE":rmse,"MAPE":mape}
def score(actual,predicted,metric="SMAPE"): return METRICS.get(metric.upper(),smape)(actual,predicted)
def all_metrics(actual,predicted): return {n:fn(actual,predicted) for n,fn in METRICS.items()}

# ══════════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════════
def _infer_freq(dates):
    if len(dates)<2: return "MS"
    diff = (pd.Series(dates).sort_values().diff().dropna().median()).days
    return "MS" if diff>=25 else ("W" if diff>=6 else "D")

class ARIMAForecaster:
    name = "ARIMA"
    def __init__(self,seed=42,p=1,d=1,q=1,P=0,D=0,Q=0,s=0):
        self.seed=seed; self.params=dict(p=p,d=d,q=q,P=P,D=D,Q=Q,s=s)
        self.model=None; self._last_dates=None; self.aic=self.bic=None
    def fit(self,df):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        self._last_dates=df["ds"].values; p=self.params
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model=SARIMAX(df["y"].values,order=(p["p"],p["d"],p["q"]),
                seasonal_order=(p["P"],p["D"],p["Q"],p["s"]),
                enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)
        self.aic=self.model.aic; self.bic=self.model.bic; return self
    def predict(self,horizon,freq="MS"):
        fc=self.model.get_forecast(steps=horizon); mean=fc.predicted_mean
        ci=fc.conf_int(alpha=0.05)
        lo=ci.iloc[:,0].values if hasattr(ci,"iloc") else ci[:,0]
        hi=ci.iloc[:,1].values if hasattr(ci,"iloc") else ci[:,1]
        dates=pd.date_range(start=pd.Timestamp(self._last_dates[-1])+pd.tseries.frequencies.to_offset(freq),periods=horizon,freq=freq)
        return pd.DataFrame({"ds":dates,"yhat":mean,"yhat_lower":lo,"yhat_upper":hi})
    def param_grid(self):
        return {"p":[1,2],"d":[0,1],"q":[0,1],"P":[0],"D":[0],"Q":[0],"s":[0]}

class ESForecaster:
    name = "Exp. Smoothing"
    def __init__(self,seed=42,trend="add",damped_trend=False,seasonal="add",seasonal_periods=12,use_boxcox=False):
        self.seed=seed; self.params=dict(trend=trend,damped_trend=damped_trend,
            seasonal=seasonal,seasonal_periods=seasonal_periods,use_boxcox=use_boxcox)
        self.model=None; self._last_date=None
    def fit(self,df):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        y=df["y"].values.astype(float); self._last_date=df["ds"].iloc[-1]
        sp=self.params["seasonal_periods"]
        seasonal=self.params["seasonal"] if len(y)>=2*sp else None
        has_nonpos=(y<=0).any()
        trend="add" if has_nonpos and self.params["trend"]=="mul" else self.params["trend"]
        seasonal="add" if has_nonpos and seasonal=="mul" else seasonal
        use_bc=False if has_nonpos else self.params["use_boxcox"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.model=ExponentialSmoothing(y,trend=trend,damped_trend=self.params["damped_trend"],
                    seasonal=seasonal,seasonal_periods=sp if seasonal else None,
                    use_boxcox=use_bc).fit(optimized=True,use_brute=True)
            except Exception:
                self.model=ExponentialSmoothing(y,trend="add",seasonal=None).fit()
        return self
    def predict(self,horizon,freq="MS"):
        fc=np.array(self.model.forecast(horizon)); ci=np.std(self.model.resid)*1.96
        yhat=np.maximum(0,fc)
        dates=pd.date_range(start=self._last_date+pd.tseries.frequencies.to_offset(freq),periods=horizon,freq=freq)
        return pd.DataFrame({"ds":dates,"yhat":yhat,"yhat_lower":np.maximum(0,yhat-ci),"yhat_upper":yhat+ci})
    def param_grid(self):
        return {"trend":["add"],"damped_trend":[False],"seasonal":["add"],"use_boxcox":[False]}

def _lag_features(series, lags=12):
    df=pd.DataFrame({"y":series})
    for l in range(1,lags+1): df[f"lag_{l}"]=df["y"].shift(l)
    df["roll_mean_6"]=df["y"].shift(1).rolling(6).mean()
    df["roll_std_6"]=df["y"].shift(1).rolling(6).std()
    df["roll_mean_12"]=df["y"].shift(1).rolling(12).mean()
    return df.dropna()

class XGBoostForecaster:
    name = "XGBoost"
    def __init__(self,seed=42,n_estimators=200,max_depth=4,learning_rate=0.05,subsample=0.8,reg_alpha=0.0,reg_lambda=1.0):
        self.seed=seed; self.params=dict(n_estimators=n_estimators,max_depth=max_depth,
            learning_rate=learning_rate,subsample=subsample,reg_alpha=reg_alpha,reg_lambda=reg_lambda)
        self.lags=12; self._train=None; self._last_date=None; self._feat_cols=None
        self.model=None; self.feature_importance_=None
    def fit(self,df):
        import xgboost as xgb
        self._train=df["y"].values.copy(); self._last_date=df["ds"].iloc[-1]
        fd=_lag_features(self._train,self.lags); self._feat_cols=[c for c in fd.columns if c!="y"]
        self.model=xgb.XGBRegressor(random_state=self.seed,verbosity=0,**self.params)
        self.model.fit(fd[self._feat_cols].values,fd["y"].values)
        self.feature_importance_=pd.DataFrame({"feature":self._feat_cols,"importance":self.model.feature_importances_}).sort_values("importance",ascending=False)
        return self
    def predict(self,horizon,freq="MS",last_date=None):
        last_date=last_date or self._last_date; buf=list(self._train); preds=[]
        for _ in range(horizon):
            fd=_lag_features(np.array(buf),self.lags)
            if len(fd)==0: preds.append(np.nan); buf.append(np.nan); continue
            p=float(self.model.predict(fd[self._feat_cols].values[-1:])[0])
            preds.append(max(0,p)); buf.append(max(0,p))
        yhat=np.array(preds); std=np.std(self._train)*0.1
        dates=pd.date_range(start=last_date+pd.tseries.frequencies.to_offset(freq),periods=horizon,freq=freq)
        return pd.DataFrame({"ds":dates,"yhat":yhat,"yhat_lower":yhat-1.96*std,"yhat_upper":yhat+1.96*std})
    def param_grid(self):
        return {"n_estimators":[100,200],"max_depth":[3,4],"learning_rate":[0.05,0.1],
                "subsample":[0.8],"reg_alpha":[0.0],"reg_lambda":[1.0]}

class ThetaForecaster:
    name = "Theta"
    def __init__(self,seed=42,theta=2.0):
        self.seed=seed; self.params=dict(theta=theta)
        self._train=None; self._last_date=None; self._l=None; self._b=None; self._intercept=None; self._n=0
    def fit(self,df):
        y=df["y"].values.astype(float); self._train=y; self._last_date=df["ds"].iloc[-1]; self._n=len(y)
        t=np.arange(1,self._n+1); A=np.vstack([t,np.ones(self._n)]).T
        coef,_,_,_=np.linalg.lstsq(A,y,rcond=None); self._b=coef[0]; self._intercept=coef[1]
        detrended=y-(self._b*t+self._intercept); alpha=0.5; l=detrended[0]
        for v in detrended[1:]: l=alpha*v+(1-alpha)*l
        self._l=l; return self
    def predict(self,horizon,freq="MS"):
        preds=[max(0,self._b*(self._n+h)+self._intercept+self._l) for h in range(1,horizon+1)]
        yhat=np.array(preds); ci_w=np.std(self._train)*0.15
        dates=pd.date_range(start=self._last_date+pd.tseries.frequencies.to_offset(freq),periods=horizon,freq=freq)
        return pd.DataFrame({"ds":dates,"yhat":yhat,"yhat_lower":np.maximum(0,yhat-1.96*ci_w),"yhat_upper":yhat+1.96*ci_w})
    def param_grid(self): return {"theta":[1.5,2.0,3.0]}

class ProphetForecaster:
    name = "Prophet"
    def __init__(self,seed=42,changepoint_prior_scale=0.05,seasonality_prior_scale=10.0,seasonality_mode="additive"):
        self.seed=seed; self.params=dict(changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,seasonality_mode=seasonality_mode)
        self.model=None; self._train_df=None
    def fit(self,df):
        from prophet import Prophet
        self._train_df=df[["ds","y"]].copy()
        self.model=Prophet(**self.params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); self.model.fit(self._train_df)
        return self
    def predict(self,horizon,freq="MS"):
        future=self.model.make_future_dataframe(periods=horizon,freq=freq)
        fc=self.model.predict(future)
        result=fc[["ds","yhat","yhat_lower","yhat_upper"]].tail(horizon).reset_index(drop=True)
        result["yhat"]=result["yhat"].clip(lower=0); return result
    def param_grid(self):
        return {"changepoint_prior_scale":[0.001,0.01,0.05,0.1,0.5],
                "seasonality_prior_scale":[0.01,1.0,10.0],"seasonality_mode":["additive","multiplicative"]}

# ══════════════════════════════════════════════════════════════════════════════
# DATA QUALITY — AUTO TRIM LEADING ZEROS
# ══════════════════════════════════════════════════════════════════════════════
def detect_launch_date(df: pd.DataFrame, max_leading_zero_run: int = 3) -> dict:
    """
    Detect the effective launch date by finding the end of the leading-zero period.
    Returns dict with: trimmed_df, launch_date, original_rows, trimmed_rows,
                       leading_zeros_removed, status
    Rules:
      - Find the last position where there is a run of >= max_leading_zero_run consecutive zeros
        starting from the beginning of the series.
      - Everything before and including that run is pre-launch and gets removed.
      - If fewer than 12 usable rows remain after trimming, flag as INSUFFICIENT.
    """
    y = df["y"].values
    n = len(y)
    cut = 0  # index from which real data starts

    # Walk through finding runs of consecutive zeros from start
    i = 0
    while i < n:
        if y[i] == 0:
            # Find end of this zero run
            j = i
            while j < n and y[j] == 0:
                j += 1
            run_len = j - i
            # Only trim if this run starts within the first half of the series
            # AND it's a long run (>= threshold) — keeps genuine sparse months
            if i == cut and run_len >= max_leading_zero_run:
                cut = j  # advance cut past this zero run
            i = j
        else:
            i += 1

    trimmed_df = df.iloc[cut:].reset_index(drop=True)
    usable = len(trimmed_df)
    removed = cut
    launch_date = trimmed_df["ds"].iloc[0] if usable > 0 else df["ds"].iloc[0]

    if usable < 12:
        status = "INSUFFICIENT"
    elif removed > 0:
        status = "TRIMMED"
    else:
        status = "OK"

    return {
        "trimmed_df":        trimmed_df,
        "launch_date":       launch_date,
        "original_rows":     n,
        "usable_rows":       usable,
        "leading_zeros_removed": removed,
        "status":            status,
    }

def build_data_quality_report(skus: dict, min_months: int = 12) -> pd.DataFrame:
    """Build a data quality report for all SKUs before running forecasts."""
    rows = []
    for sku_id, df in skus.items():
        info = detect_launch_date(df)
        zero_pct = int((df["y"] == 0).mean() * 100)
        non_zero = int((df["y"] > 0).sum())
        rows.append({
            "SKU":               sku_id,
            "Total months":      info["original_rows"],
            "Usable months":     info["usable_rows"],
            "Leading zeros cut": info["leading_zeros_removed"],
            "Launch date":       info["launch_date"].strftime("%Y-%m") if hasattr(info["launch_date"], "strftime") else str(info["launch_date"]),
            "Zero months (all)": f"{zero_pct}%",
            "Non-zero months":   non_zero,
            "Status":            info["status"],
        })
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════════════════
# CROSTON'S METHOD — for intermittent / sparse demand
# ══════════════════════════════════════════════════════════════════════════════
class CrostonForecaster:
    """
    Croston's method for intermittent demand.
    Separately smooths demand size and inter-demand interval,
    producing a stable non-zero forecast rate.
    """
    name = "Croston"

    def __init__(self, seed=42, alpha=0.1, beta=0.1):
        self.seed   = seed
        self.params = dict(alpha=alpha, beta=beta)
        self._rate  = None
        self._last_date = None
        self._train = None

    def fit(self, df: pd.DataFrame):
        y = df["y"].values.astype(float)
        self._train     = y
        self._last_date = df["ds"].iloc[-1]
        alpha = self.params["alpha"]
        beta  = self.params["beta"]

        # Initialise on first non-zero observation
        non_zero_idx = np.where(y > 0)[0]
        if len(non_zero_idx) == 0:
            self._rate = 0.0
            return self

        q = float(y[non_zero_idx[0]])   # demand size estimate
        p = 1.0                          # inter-demand interval estimate
        last_demand_t = non_zero_idx[0]

        for t in range(non_zero_idx[0] + 1, len(y)):
            if y[t] > 0:
                interval = t - last_demand_t
                q = alpha * y[t]  + (1 - alpha) * q
                p = beta  * interval + (1 - beta)  * p
                last_demand_t = t

        self._rate = q / p if p > 0 else 0.0
        return self

    def predict(self, horizon: int, freq: str = "MS") -> pd.DataFrame:
        rate  = max(0.0, self._rate)
        yhat  = np.round(np.full(horizon, rate)).astype(int)
        ci_w  = int(np.std(self._train) * 0.5) if self._train is not None else 0
        dates = pd.date_range(
            start=self._last_date + pd.tseries.frequencies.to_offset(freq),
            periods=horizon, freq=freq
        )
        return pd.DataFrame({
            "ds":         dates,
            "yhat":       yhat,
            "yhat_lower": np.maximum(0, yhat - ci_w),
            "yhat_upper": yhat + ci_w,
        })

    def param_grid(self) -> dict:
        return {"alpha": [0.05, 0.1, 0.2, 0.3], "beta": [0.05, 0.1, 0.2]}

MODEL_REGISTRY = {
    "ARIMA":          ARIMAForecaster,
    "Exp. Smoothing": ESForecaster,
    "XGBoost":        XGBoostForecaster,
    "Theta":          ThetaForecaster,
    "Prophet":        ProphetForecaster,
    "Croston":        CrostonForecaster,
}

# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════
def hyper_search(ModelClass, grid, train_df, test_df, metric_fn, method="random", n_trials=10, seed=42):
    def _try(params):
        try:
            m=ModelClass(seed=seed,**params); m.fit(train_df)
            p=m.predict(len(test_df)); return metric_fn(test_df["y"].values,p["yhat"].values[:len(test_df)])
        except Exception: return float("inf")

    if method=="optuna":
        try:
            import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
            def obj(trial):
                return _try({k:trial.suggest_categorical(k,v) for k,v in grid.items()})
            study=optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=seed))
            study.optimize(obj,n_trials=n_trials,catch=(Exception,))
            return study.best_params if study.best_trial else {}, (study.best_value if study.best_trial else float("inf"))
        except ImportError:
            method="random"

    random.seed(seed); np.random.seed(seed)
    combos=list(itertools.product(*grid.values()))
    if method=="random" and len(combos)>n_trials:
        combos=random.sample(combos,n_trials)
    best_p,best_s={},float("inf")
    for combo in combos:
        params=dict(zip(grid.keys(),combo)); s=_try(params)
        if s<best_s: best_s=s; best_p=params.copy()
    return best_p, best_s

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def run_sku_pipeline(sku_id, sku_df, cleaning_methods, model_names, metric="SMAPE",
                     n_test=6, optim_method="random", n_trials=8, horizon=12, seed=42,
                     auto_trim=True, min_months=12, status_fn=None):
    def log(msg):
        if status_fn: status_fn(msg)

    # ── Auto-trim leading zeros ──────────────────────────────────────────────
    dq = detect_launch_date(sku_df)
    if auto_trim and dq["leading_zeros_removed"] > 0:
        log(f"  {sku_id}: trimmed {dq['leading_zeros_removed']} pre-launch zero months → using from {dq['launch_date'].strftime('%Y-%m')}")
        sku_df = dq["trimmed_df"]

    # ── Insufficient data check ──────────────────────────────────────────────
    if dq["usable_rows"] < min_months:
        log(f"  {sku_id}: only {dq['usable_rows']} usable months after trimming — skipping (need {min_months}+)")
        return {"sku_id":sku_id,"results":[],"ranked":[],"best":None,
                "freq":"MS","split":{},"dq":dq,"status":"INSUFFICIENT"}

    freq = _infer_freq(sku_df["ds"])
    # Adjust n_test if series is short after trimming
    n_test = min(n_test, max(3, len(sku_df) // 5))
    train_df, test_df, train_idx, test_idx = temporal_split(sku_df, n_test=n_test, seed=seed)
    metric_fn = lambda a,p: score(a,p,metric)
    results = []

    for cleaning in cleaning_methods:
        c_train = apply_cleaning(train_df, cleaning)
        c_test  = apply_cleaning(test_df,  cleaning)
        for model_name in model_names:
            ModelClass = MODEL_REGISTRY.get(model_name)
            if ModelClass is None: continue
            log(f"  {sku_id} › {CLEANING_LABELS.get(cleaning,cleaning)} + {model_name}…")
            model_seed = (seed + abs(hash(f"{sku_id}_{cleaning}_{model_name}"))) % (2**31)
            try:
                dummy = ModelClass(seed=model_seed)
                best_params, best_score = hyper_search(
                    ModelClass, dummy.param_grid(), c_train, c_test,
                    metric_fn, method=optim_method, n_trials=n_trials, seed=model_seed)

                # Refit on full series
                full_clean = apply_cleaning(sku_df, cleaning)
                best_model = ModelClass(seed=model_seed, **best_params)
                best_model.fit(full_clean)

                # Validation metrics on train/test
                val_m = ModelClass(seed=model_seed, **best_params)
                val_m.fit(c_train)
                val_p = val_m.predict(len(test_df), freq=freq)
                val_metrics = all_metrics(test_df["y"].values, val_p["yhat"].values[:len(test_df)])

                # Train-fit metrics (in-sample) for overfitting detection
                try:
                    train_p = val_m.predict(len(c_train), freq=freq)
                    train_metrics = all_metrics(c_train["y"].values, train_p["yhat"].values[:len(c_train)])
                except Exception:
                    train_metrics = {}

                # Overfitting flag: test error > 2x train error on chosen metric = overfit
                train_err = train_metrics.get(metric, 0)
                test_err  = val_metrics.get(metric, 0)
                overfit_ratio = (test_err / train_err) if train_err > 1 else 1.0
                overfit_flag  = overfit_ratio > 2.5

                # Final forecast — integers, floor 0
                try:
                    fc_df = best_model.predict(horizon, freq=freq)
                except Exception:
                    fc_df = best_model.predict(horizon)
                    last = sku_df["ds"].iloc[-1]
                    fc_df["ds"] = pd.date_range(start=last+pd.tseries.frequencies.to_offset(freq),periods=horizon,freq=freq)

                # Enforce non-negative whole numbers on all forecast columns
                for col in ["yhat","yhat_lower","yhat_upper"]:
                    if col in fc_df.columns:
                        fc_df[col] = np.maximum(0, fc_df[col]).round(0).astype(int)

                results.append({"sku_id":sku_id,"cleaning":cleaning,"model":model_name,
                    "params":best_params,"score":best_score,"metrics":val_metrics,
                    "train_metrics":train_metrics,"overfit_ratio":overfit_ratio,"overfit_flag":overfit_flag,
                    "model_obj":best_model,"forecast_df":fc_df,
                    "train_df":train_df,"test_df":test_df,"error":None})
                log(f"  ✓ {sku_id} › {model_name} → {metric}={best_score:.3f}")
            except Exception as e:
                results.append({"sku_id":sku_id,"cleaning":cleaning,"model":model_name,
                    "params":{},"score":float("inf"),"metrics":{},"model_obj":None,
                    "forecast_df":None,"train_df":train_df,"test_df":test_df,"error":str(e)})
                log(f"  ✗ {sku_id} › {model_name} failed: {e}")

    ranked = sorted([r for r in results if r["error"] is None], key=lambda x: x["score"])
    return {"sku_id":sku_id,"results":results,"ranked":ranked,
            "best":ranked[0] if ranked else None,"freq":freq,
            "split":{"train_idx":train_idx,"test_idx":test_idx,"n_test":n_test},
            "dq":dq,"status":"OK"}

# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════
THEME = dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e0daf5",family="Segoe UI"),
    xaxis=dict(gridcolor="#2a2750",showgrid=True,zeroline=False),
    yaxis=dict(gridcolor="#2a2750",showgrid=True,zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)",bordercolor="#2a2750"))
BRAND="#6d28d9"; ACCENT="#f59e0b"; PALETTE=["#6d28d9","#f59e0b","#10b981","#ef4444","#3b82f6"]

def forecast_chart(train_df, test_df, fc_df, model_name, cleaning, sku_id, sc, mn):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_df["ds"],y=train_df["y"],name="Train",
        line=dict(color="#5b5296",width=1.5),mode="lines"))
    if test_df is not None and len(test_df):
        fig.add_trace(go.Scatter(x=test_df["ds"],y=test_df["y"],name="Actuals (test)",
            line=dict(color=ACCENT,width=2),mode="lines+markers",marker=dict(size=4)))
    if fc_df is not None and len(fc_df):
        if "yhat_upper" in fc_df:
            fig.add_trace(go.Scatter(
                x=pd.concat([fc_df["ds"],fc_df["ds"][::-1]]),
                y=pd.concat([fc_df["yhat_upper"],fc_df["yhat_lower"][::-1]]),
                fill="toself",fillcolor="rgba(109,40,217,0.15)",
                line=dict(color="rgba(0,0,0,0)"),name="95% CI"))
        fig.add_trace(go.Scatter(x=fc_df["ds"],y=fc_df["yhat"],name=f"{model_name} forecast",
            line=dict(color=BRAND,width=2.5),mode="lines"))
    fig.update_layout(title=f"{sku_id} — {cleaning} + {model_name} ({mn}={sc:.3f})",
        hovermode="x unified",height=420,**THEME)
    return fig

def comparison_bar(ranked, sku_id, metric_name):
    if not ranked: return go.Figure()
    labels=[f"{CLEANING_LABELS.get(r['cleaning'],r['cleaning'])} + {r['model']}" for r in ranked]
    scores=[r["score"] for r in ranked]
    fig=go.Figure(go.Bar(x=labels,y=scores,
        marker_color=[BRAND if i==0 else "#2a2750" for i in range(len(scores))],
        marker_line_color=[BRAND if i==0 else "#4a4275" for i in range(len(scores))],
        marker_line_width=1,text=[f"{s:.3f}" for s in scores],textposition="outside"))
    fig.update_layout(title=f"{sku_id} — Pipeline Comparison ({metric_name})",
        yaxis_title=metric_name,height=380,**THEME)
    return fig

def heatmap_chart(results_by_sku, metric_name):
    rows=[]
    for sku_id,data in results_by_sku.items():
        for r in data.get("results",[]):
            if r["error"] is None:
                rows.append({"SKU":sku_id,"Pipeline":f"{r['cleaning']}+{r['model']}","Score":round(r["score"],3)})
    if not rows: return None
    pivot=pd.DataFrame(rows).pivot_table(index="SKU",columns="Pipeline",values="Score",aggfunc="min")
    fig=px.imshow(pivot,color_continuous_scale=["#10b981","#f59e0b","#ef4444"],
        aspect="auto",title=f"Validation {metric_name} Heatmap (lower = better)")
    fig.update_layout(height=max(300,80*len(pivot)),**THEME)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for k,v in {"raw_df":None,"skus":{},"results":{},"run_complete":False,"partial_results":{}}.items():
    if k not in st.session_state: st.session_state[k]=v

@st.cache_data(show_spinner=False)
def _lib_ok(lib):
    try: __import__(lib); return True
    except: return False

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    cfg={}
    with st.sidebar:
        st.markdown('<div style="font-size:20px;font-weight:800;color:#f0ecff;padding-bottom:14px;'
            'border-bottom:1px solid #2a2750">📈 AutoForecast Pro</div>',unsafe_allow_html=True)

        st.markdown('<div class="sb-section">📂 Upload Data File</div>',unsafe_allow_html=True)
        uploaded=st.file_uploader("Upload Excel or CSV",type=["xlsx","xls","csv"],
            help="Excel (.xlsx/.xls) or CSV. Must have a date column and a numeric value column.",
            label_visibility="collapsed")

        if uploaded is None:
            st.info("⬆️ Upload an Excel (.xlsx/.xls) or CSV file to begin.")
            return {"uploaded":None}

        cfg["uploaded"]=uploaded
        sheet_name=None
        if uploaded.name.lower().endswith((".xlsx",".xls")):
            sheets=get_sheet_names(uploaded); uploaded.seek(0)
            if len(sheets)>1:
                sheet_name=st.selectbox("Sheet",sheets)
        cfg["sheet_name"]=sheet_name

        try:
            df_raw=load_uploaded_file(uploaded,sheet_name=sheet_name); uploaded.seek(0)
            st.session_state["raw_df"]=df_raw
            st.success(f"✓ {len(df_raw):,} rows · {df_raw.shape[1]} columns")
        except Exception as e:
            st.error(f"❌ {e}"); return {"uploaded":None}

        st.markdown('<div class="sb-section">🗂 Column Mapping</div>',unsafe_allow_html=True)
        cols=df_raw.columns.tolist()
        auto_date,auto_val,auto_sku=auto_detect_columns(df_raw)
        date_col=st.selectbox("Date column (ds)",cols,
            index=cols.index(auto_date) if auto_date in cols else 0)
        val_col=st.selectbox("Value column (y)",cols,
            index=cols.index(auto_val) if auto_val in cols else min(1,len(cols)-1))
        sku_opts=["— none (single series) —"]+[c for c in cols if c not in (date_col,val_col)]
        sku_sel=st.selectbox("SKU / Group column (optional)",sku_opts)
        sku_col=sku_sel if sku_sel!="— none (single series) —" else None
        cfg.update({"date_col":date_col,"val_col":val_col,"sku_col":sku_col})

        try:
            prepared=validate_and_prepare(df_raw,date_col,val_col,sku_col)
            skus=split_by_sku(prepared)
            st.session_state["skus"]=skus
            n_sku=len(skus); total=sum(len(v) for v in skus.values())
            st.caption(f"{n_sku} SKU{'s' if n_sku>1 else ''} · {total:,} total rows")
        except Exception as e:
            st.error(f"❌ {e}"); return {"uploaded":None}

        st.markdown('<div class="sb-section">🔮 Forecast Settings</div>',unsafe_allow_html=True)
        c1,c2=st.columns(2)
        cfg["horizon"]=c1.number_input("Horizon",1,365,12)
        cfg["n_test"] =c2.number_input("Test size",3,200,6)
        cfg["metric"] =st.selectbox("Metric",["SMAPE","MAE","RMSE","MAPE"])
        cfg["seed"]   =st.number_input("Seed",0,9999,42)

        st.markdown('<div class="sb-section">🧹 Cleaning Methods</div>',unsafe_allow_html=True)
        cleaning_sel=[]
        if st.checkbox("No cleaning (raw)",True):       cleaning_sel.append("none")
        if st.checkbox("Hampel filter",True):           cleaning_sel.append("hampel")
        if st.checkbox("DWT denoising (wavelet)",False):cleaning_sel.append("dwt")
        if st.checkbox("SSA denoising",False):          cleaning_sel.append("ssa")
        cfg["cleaning_methods"]=cleaning_sel or ["none"]

        st.markdown('<div class="sb-section">🤖 Models</div>',unsafe_allow_html=True)
        avail={"ARIMA":_lib_ok("statsmodels"),"Exp. Smoothing":_lib_ok("statsmodels"),
               "XGBoost":_lib_ok("xgboost"),"Theta":True,"Prophet":_lib_ok("prophet"),
               "Croston":True}
        model_sel=[]
        for m,ok in avail.items():
            lbl=m if ok else f"{m} *(not installed)*"
            default=ok and m in ("ARIMA","Exp. Smoothing","XGBoost","Theta","Croston")
            if st.checkbox(lbl,value=default,disabled=not ok): model_sel.append(m)
        cfg["models"]=model_sel or ["Theta"]

        st.markdown('<div class="sb-section">🔍 Data Quality</div>',unsafe_allow_html=True)
        cfg["auto_trim"]=st.toggle("Auto-trim leading zeros (recommended)",value=True,
            help="Automatically removes pre-launch zero months from each SKU before training. Prevents new products from skewing forecasts.")
        cfg["min_months"]=st.number_input("Min usable months required",3,36,12,
            help="SKUs with fewer than this many non-zero months after trimming are flagged as Insufficient and skipped.")

        st.markdown('<div class="sb-section">🎯 Optimization</div>',unsafe_allow_html=True)
        cfg["optim_method"]=st.selectbox("Method",["random","optuna","grid"],
            format_func=lambda x:{"random":"Random search","optuna":"Optuna (Bayesian)","grid":"Grid search"}[x])
        cfg["n_trials"]=st.number_input("Trials per pipeline",1,100,3)

        st.divider()
        cfg["can_run"]=bool(st.session_state.get("skus")) and bool(model_sel)
    return cfg

# ══════════════════════════════════════════════════════════════════════════════
# RUN ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def run_all(cfg):
    """Run pipeline with incremental saves — each SKU saved immediately so progress survives resets."""
    skus=st.session_state["skus"]
    # Resume from partial results if they exist
    existing=st.session_state.get("partial_results",{})
    done_skus=set(existing.keys())
    log_lines=[]; results=dict(existing)
    status_box=st.empty(); prog=st.progress(0.0,"Starting…"); log_area=st.empty()
    def log(msg,kind="run"):
        log_lines.append(f'<div class="log-{kind}">{msg}</div>')
        log_area.markdown(f'<div class="status-log">{"".join(log_lines[-25:])}</div>',unsafe_allow_html=True)
    n_skus=len(skus)
    for i,(sku_id,sku_df) in enumerate(skus.items()):
        prog.progress(i/n_skus,text=f"SKU {i+1}/{n_skus}")
        if sku_id in done_skus:
            log(f"⏭ {sku_id}: already done (skipping)","ok"); continue
        status_box.markdown(
            f'<div style="background:#1c1a32;border:1px solid #2a2750;border-radius:10px;padding:12px 18px">'
            f'<span style="color:#a78bfa">Processing:</span> <strong style="color:#f0ecff">{sku_id}</strong>'
            f' <span style="color:#7c6fad">({i+1}/{n_skus})</span></div>',unsafe_allow_html=True)
        try:
            r=run_sku_pipeline(sku_id=sku_id,sku_df=sku_df,
                cleaning_methods=cfg["cleaning_methods"],model_names=cfg["models"],
                metric=cfg["metric"],n_test=cfg["n_test"],
                optim_method=cfg["optim_method"],n_trials=cfg["n_trials"],
                horizon=cfg["horizon"],seed=cfg["seed"],
                auto_trim=cfg.get("auto_trim",True),
                min_months=cfg.get("min_months",12),
                status_fn=lambda m:log(m,"run"))
            results[sku_id]=r
            # Save immediately so a crash doesn't lose this SKU
            st.session_state["partial_results"]=dict(results)
            best=r.get("best")
            if best: log(f"✓ {sku_id}: BEST={best['model']} {cfg['metric']}={best['score']:.2f}","ok")
            else:    log(f"⚠ {sku_id}: No successful pipelines","err")
        except Exception as e:
            log(f"✗ {sku_id}: {e}","err")
        prog.progress((i+1)/n_skus,text=f"SKU {i+1}/{n_skus} complete")
    prog.progress(1.0,text="✅ All complete!"); status_box.empty()
    # Clear partial cache on full completion
    st.session_state["partial_results"]={}
    return results

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS UI
# ══════════════════════════════════════════════════════════════════════════════
def render_results(results, metric_name):
    if not results: st.warning("No results."); return

    tab_sum,tab_fc,tab_det,tab_dq,tab_rep=st.tabs(["📊 Summary","📈 Forecast","🔬 Model Details","🔎 Data Quality","🔁 Reproducibility"])

    # Build summary rows
    rows=[]
    for sku_id,data in results.items():
        for i,r in enumerate(data.get("ranked",[])):
            m=r.get("metrics",{})
            overfit = r.get("overfit_flag", False)
            ratio   = r.get("overfit_ratio", 1.0)
            rows.append({"Rank":i+1,"SKU":sku_id,
                "Cleaning":CLEANING_LABELS.get(r["cleaning"],r["cleaning"]),
                "Model":r["model"],
                metric_name:round(r["score"],4),
                "MAPE %":round(m.get("MAPE",0),2),
                "MAE":round(m.get("MAE",0),2),
                "RMSE":round(m.get("RMSE",0),2),
                "Overfit?":"⚠️ Yes" if overfit else "✅ OK",
                "Test/Train ratio":round(ratio,2),
                "Best?":"⭐" if i==0 else ""})
    summary_df=pd.DataFrame(rows) if rows else pd.DataFrame()

    # ── Summary tab ──────────────────────────────────────────────────────
    with tab_sum:
        if not summary_df.empty:
            best_df=summary_df[summary_df["Rank"]==1].drop(columns=["Rank"])
            n_skus=len(results)
            ok=sum(1 for d in results.values() if d.get("best"))
            scores=[d["best"]["score"] for d in results.values() if d.get("best")]
            avg=np.mean(scores) if scores else 0
            total_p=sum(len(d.get("results",[])) for d in results.values())
            c1,c2,c3,c4=st.columns(4)
            for col,label,val,sub in [(c1,"SKUs processed",str(n_skus),"total"),
                (c2,"Successful",str(ok),"with results"),
                (c3,f"Avg {metric_name}",f"{avg:.2f}","best per SKU"),
                (c4,"Pipelines tested",str(total_p),"across all SKUs")]:
                col.markdown(f'<div class="af-metric"><div class="label">{label}</div>'
                    f'<div class="value">{val}</div><div class="sub">{sub}</div></div>',unsafe_allow_html=True)
            st.markdown("---")
            # Overfitting alert
            overfit_skus=[row["SKU"] for _,row in best_df.iterrows() if row.get("Overfit?","")=="⚠️ Yes"]
            if overfit_skus:
                st.warning(f"⚠️ **Overfitting detected** on {len(overfit_skus)} SKU(s): {', '.join(overfit_skus)}. "
                    f"The model fits training data well but performs poorly on the held-out test set. "
                    f"Consider: fewer trials, simpler models, or more training data.")
            else:
                st.success("✅ No overfitting detected across all SKUs.")
            st.markdown("**Best pipeline per SKU**")
            st.dataframe(best_df,use_container_width=True,hide_index=True)
            hm=heatmap_chart(results,metric_name)
            if hm: st.plotly_chart(hm,use_container_width=True)
            st.markdown("**⬇️ Downloads**")
            dl1,dl2,dl3=st.columns(3)
            with dl1:
                buf=io.StringIO(); summary_df.to_csv(buf,index=False)
                st.download_button("📋 Pipeline results (CSV)",buf.getvalue(),"pipeline_results.csv","text/csv")
            with dl2:
                all_fc=[]
                for sku_id,data in results.items():
                    best=data.get("best")
                    if best and best.get("forecast_df") is not None:
                        fc=best["forecast_df"].copy()
                        fc["sku_id"]=sku_id; fc["model"]=best["model"]; fc["cleaning"]=best["cleaning"]
                        all_fc.append(fc)
                if all_fc:
                    all_fc_df=pd.concat(all_fc,ignore_index=True)
                    for col in ["yhat","yhat_lower","yhat_upper"]:
                        if col in all_fc_df.columns:
                            all_fc_df[col]=np.maximum(0,all_fc_df[col]).round(0).astype(int)
                    buf2=io.StringIO(); all_fc_df.to_csv(buf2,index=False)
                    st.download_button("📈 All forecasts (CSV)",buf2.getvalue(),"all_forecasts.csv","text/csv")
            with dl3:
                repro=build_repro(results)
                st.download_button("🔁 Reproducibility (JSON)",json.dumps(repro,indent=2,default=str),
                    "reproducibility.json","application/json")

    # ── Forecast tab ─────────────────────────────────────────────────────
    with tab_fc:
        sel=st.selectbox("Select SKU",list(results.keys()),key="fc_sel")
        if sel:
            data=results[sel]; best=data.get("best"); ranked=data.get("ranked",[])
            if best:
                fig=forecast_chart(best["train_df"],best["test_df"],best.get("forecast_df"),
                    best["model"],CLEANING_LABELS.get(best["cleaning"],best["cleaning"]),
                    sel,best["score"],metric_name)
                st.plotly_chart(fig,use_container_width=True)
                if len(ranked)>1:
                    with st.expander("🔀 Overlay all pipeline forecasts"):
                        fig2=go.Figure()
                        fig2.add_trace(go.Scatter(x=best["train_df"]["ds"],y=best["train_df"]["y"],
                            name="Historical",line=dict(color="#5b5296",width=1)))
                        for i,r in enumerate(ranked[:5]):
                            if r.get("forecast_df") is not None:
                                fig2.add_trace(go.Scatter(x=r["forecast_df"]["ds"],y=r["forecast_df"]["yhat"],
                                    name=f"{'★ ' if i==0 else ''}{r['cleaning']}+{r['model']}",
                                    line=dict(color=PALETTE[i%len(PALETTE)],
                                        width=2.5 if i==0 else 1.5,dash="solid" if i==0 else "dot")))
                        fig2.update_layout(hovermode="x unified",height=400,**THEME)
                        st.plotly_chart(fig2,use_container_width=True)
                if best.get("forecast_df") is not None:
                    fc=best["forecast_df"].copy()
                    # Enforce whole numbers ≥ 0 (units)
                    for col in ["yhat","yhat_lower","yhat_upper"]:
                        if col in fc.columns:
                            fc[col]=np.maximum(0,fc[col]).round(0).astype(int)
                    fc=fc.rename(columns={"yhat":"Forecast (units)","yhat_lower":"Lower (95%)","yhat_upper":"Upper (95%)"})
                    fc["ds"]=fc["ds"].dt.strftime("%Y-%m-%d") if hasattr(fc["ds"],"dt") else fc["ds"].astype(str)
                    fc=fc.rename(columns={"ds":"Date"})
                    st.markdown("**Forecast values — whole units, floor 0**")
                    st.dataframe(fc,use_container_width=True,hide_index=True)
                    buf=io.StringIO(); fc.to_csv(buf,index=False)
                    st.download_button(f"⬇️ Download {sel} forecast (CSV)",buf.getvalue(),f"forecast_{sel}.csv","text/csv")
            else:
                st.warning(f"No successful forecast for {sel}")

    # ── Model details tab ─────────────────────────────────────────────────
    with tab_det:
        sel2=st.selectbox("Select SKU",list(results.keys()),key="det_sel")
        if sel2:
            data=results[sel2]; ranked=data.get("ranked",[]); best=data.get("best")
            if best:
                # ── Overfitting panel ──────────────────────────────────────
                st.markdown("#### 🔬 Overfitting Diagnostics — Best Model")
                tm = best.get("train_metrics",{}); vm = best.get("metrics",{})
                ratio = best.get("overfit_ratio",1.0); flag = best.get("overfit_flag",False)
                oc1,oc2,oc3,oc4 = st.columns(4)
                oc1.metric("Train SMAPE",  f"{tm.get('SMAPE',0):.2f}%")
                oc2.metric("Test SMAPE",   f"{vm.get('SMAPE',0):.2f}%",
                    delta=f"+{vm.get('SMAPE',0)-tm.get('SMAPE',0):.2f}% vs train",
                    delta_color="inverse")
                oc3.metric("Test/Train ratio", f"{ratio:.2f}x",
                    help="<1.5 = good · 1.5–2.5 = caution · >2.5 = overfitting")
                oc4.metric("Overfit status", "⚠️ Overfit" if flag else "✅ OK")

                if flag:
                    st.warning("⚠️ **Overfitting detected.** The model memorises training data but fails on new data. "
                        "Try: reduce Trials per pipeline · switch to Theta or Exp. Smoothing · increase test size.")
                else:
                    st.success("✅ Model generalises well — test error is within acceptable range of train error.")

                # ── Full metrics table ─────────────────────────────────────
                st.markdown("#### 📊 Full Validation Metrics — Best Model")
                mc1,mc2,mc3,mc4,mc5 = st.columns(5)
                for col,label,key in [(mc1,"SMAPE %","SMAPE"),(mc2,"MAPE %","MAPE"),
                                      (mc3,"MAE","MAE"),(mc4,"RMSE","RMSE")]:
                    col.metric(label, f"{vm.get(key,0):.2f}")
                mc5.metric("Forecast type","Whole units ≥ 0")

                st.markdown("---")
                # ── All pipeline results ────────────────────────────────────
                st.markdown("#### All pipeline results")
                pipe_rows=[]
                for r in data.get("results",[]):
                    m2=r.get("metrics",{}); tm2=r.get("train_metrics",{})
                    pipe_rows.append({
                        "Cleaning":CLEANING_LABELS.get(r["cleaning"],r["cleaning"]),
                        "Model":r["model"],
                        metric_name:round(r["score"],4) if r["score"]<1e8 else "Failed",
                        "MAPE %":round(m2.get("MAPE",0),2),
                        "MAE":round(m2.get("MAE",0),2),
                        "RMSE":round(m2.get("RMSE",0),2),
                        "Train SMAPE":round(tm2.get("SMAPE",0),2),
                        "Overfit?":"⚠️ Yes" if r.get("overfit_flag") else "✅ OK",
                        "Error":r.get("error","") or ""})
                if pipe_rows: st.dataframe(pd.DataFrame(pipe_rows),use_container_width=True,hide_index=True)
                st.plotly_chart(comparison_bar(ranked,sel2,metric_name),use_container_width=True)

                # ── Residuals plot ──────────────────────────────────────────
                with st.expander("📉 Residuals analysis (best model)"):
                    try:
                        val_m2 = MODEL_REGISTRY[best["model"]](seed=42,**best.get("params",{}))
                        val_m2.fit(apply_cleaning(best["train_df"],best["cleaning"]))
                        val_p2 = val_m2.predict(len(best["test_df"]),freq=data.get("freq","MS"))
                        actuals = best["test_df"]["y"].values
                        preds   = val_p2["yhat"].values[:len(actuals)]
                        residuals = actuals - preds
                        fig_r = go.Figure()
                        fig_r.add_trace(go.Bar(x=best["test_df"]["ds"],y=residuals,
                            marker_color=[BRAND if r>=0 else "#ef4444" for r in residuals],
                            name="Residual (actual − forecast)"))
                        fig_r.add_hline(y=0,line_dash="dash",line_color="#7c6fad")
                        fig_r.update_layout(title="Residuals — Random scatter = good · Patterns = model missing something",
                            height=300,**THEME)
                        st.plotly_chart(fig_r,use_container_width=True)
                        bias = float(np.mean(residuals))
                        st.caption(f"Mean bias: {bias:+.2f} units  "
                            f"({'model tends to under-forecast' if bias>0 else 'model tends to over-forecast'})")
                    except Exception as e:
                        st.caption(f"Residuals not available: {e}")

                # ── Model-specific extras ───────────────────────────────────
                model_obj=best.get("model_obj")
                if model_obj and best["model"]=="XGBoost" and hasattr(model_obj,"feature_importance_") and model_obj.feature_importance_ is not None:
                    with st.expander("📊 XGBoost Feature Importance"):
                        fi=model_obj.feature_importance_.head(15)
                        fig=px.bar(fi,x="importance",y="feature",orientation="h",
                            color="importance",color_continuous_scale=["#2a2750",BRAND])
                        fig.update_layout(title="Feature Importance",height=400,coloraxis_showscale=False,**THEME)
                        st.plotly_chart(fig,use_container_width=True)
                if model_obj and best["model"]=="ARIMA" and hasattr(model_obj,"aic") and model_obj.aic:
                    st.info(f"ARIMA — AIC: {model_obj.aic:.2f}  |  BIC: {model_obj.bic:.2f}  |  "
                        f"Order: p={best['params'].get('p',1)} d={best['params'].get('d',1)} q={best['params'].get('q',1)}")
                with st.expander("⚙️ Best hyperparameters"):
                    st.json(best.get("params",{}))

    # ── Data Quality tab ──────────────────────────────────────────────────
    with tab_dq:
        st.markdown("#### 🔎 Data Quality Report")
        st.markdown(
            "This table shows how each SKU's data was assessed before forecasting. "
            "Leading pre-launch zeros are automatically removed so models only learn from real demand. "
            "SKUs flagged **INSUFFICIENT** were skipped — they don't have enough usable history.")

        dq_rows = []
        for sku_id, data in results.items():
            dq = data.get("dq", {})
            status = data.get("status", "OK")
            dq_rows.append({
                "SKU":               sku_id,
                "Status":            status,
                "Total months":      dq.get("original_rows", "—"),
                "Leading zeros cut": dq.get("leading_zeros_removed", 0),
                "Usable months":     dq.get("usable_rows", "—"),
                "Effective launch":  dq.get("launch_date", pd.NaT),
                "Pipelines run":     len(data.get("results", [])),
                "Best model":        data["best"]["model"] if data.get("best") else "N/A",
            })
        if dq_rows:
            dq_df = pd.DataFrame(dq_rows)
            # Colour-code status
            def style_status(val):
                if val == "INSUFFICIENT": return "background-color:#7f1d1d;color:white"
                if val == "TRIMMED":      return "background-color:#78350f;color:white"
                return "background-color:#14532d;color:white"
            st.dataframe(dq_df, use_container_width=True, hide_index=True)

        # Summary counts
        statuses = [data.get("status","OK") for data in results.values()]
        c1, c2, c3 = st.columns(3)
        c1.metric("✅ Full data", statuses.count("OK"))
        c2.metric("✂️ Trimmed (launch zeros removed)", statuses.count("TRIMMED"))
        c3.metric("⛔ Insufficient data (skipped)", statuses.count("INSUFFICIENT"))

        if any(s == "INSUFFICIENT" for s in statuses):
            insuf = [sid for sid, d in results.items() if d.get("status") == "INSUFFICIENT"]
            st.warning(
                f"**{len(insuf)} SKU(s) skipped due to insufficient data after trimming:** "
                f"{', '.join(insuf)}. "
                f"Lower the 'Min usable months' setting in the sidebar, or "
                f"provide more historical data for these products.")

        if any(s == "TRIMMED" for s in statuses):
            trimmed = [(sid, results[sid]["dq"].get("leading_zeros_removed",0),
                        results[sid]["dq"].get("launch_date",""))
                       for sid in results if results[sid].get("status") == "TRIMMED"]
            with st.expander("📋 Trimmed SKUs detail"):
                for sid, cut, ld in sorted(trimmed, key=lambda x: -x[1]):
                    launch_str = ld.strftime("%b %Y") if hasattr(ld,"strftime") else str(ld)
                    st.caption(f"**{sid}**: {cut} pre-launch zero months removed · effective launch {launch_str}")

        # Croston recommendation
        sparse_skus = []
        for sku_id, data in results.items():
            dq = data.get("dq", {})
            usable = dq.get("usable_rows", 39)
            df_tmp = data.get("best", {}).get("train_df")
            if df_tmp is not None:
                zero_pct = (df_tmp["y"] == 0).mean()
                if zero_pct > 0.25:
                    sparse_skus.append(sku_id)
        if sparse_skus:
            st.info(
                f"💡 **Croston's method recommended** for: {', '.join(sparse_skus)}. "
                f"These SKUs have >25% zero-sales months even after trimming — "
                f"Croston handles intermittent demand better than ARIMA or Prophet. "
                f"Make sure Croston is checked in the Models section and re-run.")

    # ── Reproducibility tab ───────────────────────────────────────────────
    with tab_rep:
        repro=build_repro(results)
        st.json(repro)
        st.download_button("⬇️ Download reproducibility.json",
            json.dumps(repro,indent=2,default=str),"reproducibility.json","application/json")

def build_repro(results):
    return {"generated_at":datetime.now().isoformat(),"skus":{
        sku_id:{"best_pipeline":{"cleaning":d["best"]["cleaning"],"model":d["best"]["model"],
            "params":d["best"]["params"],"score":d["best"]["score"]} if d.get("best") else None,
            "split":d.get("split",{}),
            "all_results":[{"cleaning":r["cleaning"],"model":r["model"],
                "params":r["params"],"score":r["score"]} for r in d.get("results",[])]}
        for sku_id,d in results.items()}}

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    cfg=render_sidebar()
    st.markdown('<div style="background:linear-gradient(135deg,#2d1b69 0%,#1c1a32 70%,#0f0d1c 100%);'
        'border-radius:14px;padding:28px 32px;margin-bottom:24px;border:1px solid #2a2750">'
        '<div style="font-size:28px;font-weight:800;color:#f0ecff">📈 AutoForecast Pro</div>'
        '<div style="color:#9d94c8;font-size:14px;margin-top:6px">'
        'Multi-SKU · 5 models · 4 cleaning methods · automated pipeline selection</div></div>',
        unsafe_allow_html=True)

    if cfg.get("uploaded") is None:
        st.markdown('<div style="text-align:center;padding:60px 40px;color:#7c6fad">'
            '<div style="font-size:56px;margin-bottom:16px">📊</div>'
            '<h2 style="color:#b8b0e0 !important">No data loaded</h2>'
            '<p style="font-size:14px;line-height:1.8">Upload an Excel (.xlsx/.xls) or CSV file in the sidebar<br>'
            'The app will auto-detect your date, value, and SKU columns<br><br>'
            '<strong style="color:#b8b0e0">Expected columns:</strong><br>'
            '<code style="background:#1c1a32;padding:4px 10px;border-radius:4px;color:#a78bfa">'
            'ds (date) &nbsp;|&nbsp; y (value) &nbsp;|&nbsp; sku_id (optional)</code></p></div>',
            unsafe_allow_html=True)
        return

    skus=st.session_state.get("skus",{})
    if not skus:
        st.warning("❌ No valid data. Check column selections in the sidebar."); return

    n_pipes=len(cfg.get("cleaning_methods",[]))*len(cfg.get("models",[]))
    # ── Pre-run data quality preview ────────────────────────────────────────
    if skus:
        dq_report = build_data_quality_report(skus)
        insuf = dq_report[dq_report["Status"]=="INSUFFICIENT"]
        trimmed = dq_report[dq_report["Status"]=="TRIMMED"]
        if not insuf.empty:
            st.warning(f"⚠️ {len(insuf)} SKU(s) may have insufficient data after trimming: "
                f"{', '.join(insuf['SKU'].tolist())}. They will be skipped unless you lower "
                f"'Min usable months' in the sidebar.")
        if not trimmed.empty:
            st.info(f"✂️ {len(trimmed)} SKU(s) will have pre-launch zero months auto-removed: "
                f"{', '.join(trimmed['SKU'].tolist())}.")

    col_run,col_info,col_clear=st.columns([2,2,1])
    with col_run:
        run_clicked=st.button("🚀 Run Full Comparison",use_container_width=True,
            disabled=not cfg.get("can_run",False))
    with col_clear:
        if st.button("🔄 Reset",use_container_width=True):
            st.session_state["results"]={}
            st.session_state["partial_results"]={}
            st.session_state["run_complete"]=False
            st.rerun()
    with col_info:
        st.markdown(f'<div style="padding:10px;color:#9d94c8;font-size:13px">'
            f'{len(skus)} SKU(s) · {len(cfg.get("cleaning_methods",[]))} cleaning × '
            f'{len(cfg.get("models",[]))} models = '
            f'<strong style="color:#a78bfa">{n_pipes} pipelines/SKU</strong></div>',
            unsafe_allow_html=True)

    partial=st.session_state.get("partial_results",{})
    if partial and not st.session_state.get("run_complete"):
        st.info(f"⏸ {len(partial)}/{len(skus)} SKUs completed from last run. Click Run to continue from where it stopped.")

    if run_clicked:
        with st.expander("▶ Progress log",expanded=True):
            results=run_all(cfg)
        st.session_state["results"]=results
        st.session_state["run_complete"]=True
        st.success(f"✅ Done! {len(results)} SKU(s) processed.")
        st.rerun()

    if st.session_state.get("run_complete") and st.session_state.get("results"):
        render_results(st.session_state["results"],cfg.get("metric","SMAPE"))

if __name__=="__main__":
    main()
