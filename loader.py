"""Data loading with full Excel support."""
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List
import io


def get_sheet_names(uploaded_file) -> List[str]:
    """Return sheet names from an Excel file."""
    try:
        xl = pd.ExcelFile(uploaded_file)
        return xl.sheet_names
    except Exception:
        return []


def load_uploaded_file(uploaded_file, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Load CSV or Excel from a Streamlit UploadedFile object."""
    name = uploaded_file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        kwargs = {"sheet_name": sheet_name} if sheet_name else {}
        df = pd.read_excel(uploaded_file, **kwargs)
    elif name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}. Please upload .xlsx, .xls, or .csv")
    return df


def auto_detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Auto-detect date, value, and SKU columns."""
    date_kw  = ["date", "ds", "time", "timestamp", "period", "month", "week", "day"]
    val_kw   = ["y", "value", "sales", "demand", "qty", "quantity", "units", "revenue", "amount", "volume"]
    sku_kw   = ["sku", "product", "item", "store", "region", "category", "sku_id", "product_id", "item_id", "group"]

    def find(kws, exclude=None):
        exclude = exclude or []
        for col in df.columns:
            cl = col.lower()
            if col not in exclude and any(k in cl for k in kws):
                return col
        return None

    date_col = find(date_kw)
    val_col  = find(val_kw, exclude=[date_col] if date_col else [])
    sku_col  = find(sku_kw, exclude=[c for c in [date_col, val_col] if c])

    # Fallback: use first parseable date col, first numeric col
    if date_col is None:
        for col in df.columns:
            try:
                pd.to_datetime(df[col].head(10))
                date_col = col
                break
            except Exception:
                pass

    if val_col is None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for c in num_cols:
            if c != date_col:
                val_col = c
                break

    return date_col, val_col, sku_col


def validate_and_prepare(
    df: pd.DataFrame,
    date_col: str,
    val_col: str,
    sku_col: Optional[str] = None,
) -> pd.DataFrame:
    """Validate columns, parse types, sort."""
    missing = [c for c in [date_col, val_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in file: {missing}")

    result = df[[date_col, val_col] + ([sku_col] if sku_col and sku_col in df.columns else [])].copy()
    result = result.rename(columns={date_col: "ds", val_col: "y"})
    if sku_col and sku_col in df.columns:
        result = result.rename(columns={sku_col: "sku_id"})

    result["ds"] = pd.to_datetime(result["ds"], errors="coerce")
    result["y"]  = pd.to_numeric(result["y"], errors="coerce")
    result = result.dropna(subset=["ds", "y"])
    result = result.sort_values("ds").reset_index(drop=True)

    if len(result) < 10:
        raise ValueError(f"Only {len(result)} valid rows after cleaning. Need at least 10.")

    return result


def split_by_sku(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split multi-SKU dataframe."""
    if "sku_id" not in df.columns:
        return {"All": df[["ds", "y"]].reset_index(drop=True)}
    skus = {}
    for sku, grp in df.groupby("sku_id"):
        skus[str(sku)] = grp[["ds", "y"]].reset_index(drop=True)
    return skus
