"""Data cleaning: DWT, SSA, Hampel filter."""
import numpy as np
import pandas as pd
from typing import Optional


def hampel_filter(arr: np.ndarray, window: int = 7, sigma: float = 3.0) -> np.ndarray:
    k = 1.4826
    out = arr.copy().astype(float)
    for i in range(len(arr)):
        lo, hi = max(0, i - window), min(len(arr), i + window + 1)
        win = arr[lo:hi]
        med = np.median(win)
        mad = k * np.median(np.abs(win - med))
        if mad > 0 and np.abs(arr[i] - med) > sigma * mad:
            out[i] = med
    return out


def dwt_denoise(arr: np.ndarray, wavelet: str = "db4", level: int = 3, mode: str = "soft") -> np.ndarray:
    try:
        import pywt
    except ImportError:
        return arr
    coeffs = pywt.wavedec(arr, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(arr)))
    denoised = [coeffs[0]] + [pywt.threshold(c, threshold, mode=mode) for c in coeffs[1:]]
    rec = pywt.waverec(denoised, wavelet)
    return rec[:len(arr)]


def ssa_denoise(arr: np.ndarray, window: Optional[int] = None, n_components: int = 3) -> np.ndarray:
    n = len(arr)
    L = window or min(max(2, n // 4), 20)
    L = min(L, n - 1)
    K = n - L + 1
    X = np.array([arr[i:i + L] for i in range(K)]).T
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    nc = min(n_components, len(s))
    rec = np.zeros(n)
    cnt = np.zeros(n)
    for i in range(nc):
        Xi = s[i] * np.outer(U[:, i], Vt[i, :])
        for r in range(L):
            for c in range(K):
                rec[r + c] += Xi[r, c]
                cnt[r + c] += 1
    return np.where(cnt > 0, rec / cnt, arr)


def apply_cleaning(df: pd.DataFrame, method: str, params: dict = None) -> pd.DataFrame:
    params = params or {}
    df = df.copy()
    y = df["y"].values.astype(float)
    if method == "hampel":
        y = hampel_filter(y, window=params.get("window", 7), sigma=params.get("sigma", 3.0))
    elif method == "dwt":
        y = dwt_denoise(y, wavelet=params.get("wavelet", "db4"), level=params.get("level", 3))
    elif method == "ssa":
        y = ssa_denoise(y, window=params.get("window", None), n_components=params.get("n_components", 3))
    df["y"] = y
    return df


CLEANING_LABELS = {
    "none":   "No cleaning",
    "hampel": "Hampel filter",
    "dwt":    "DWT denoising",
    "ssa":    "SSA denoising",
}
