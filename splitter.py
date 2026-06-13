"""Reproducible temporal train/test splitting."""
import numpy as np
import pandas as pd
from typing import Tuple, List


def temporal_split(
    df: pd.DataFrame,
    method: str = "last_n",
    n_test: int = 12,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], List[int]]:
    np.random.seed(seed)
    n = len(df)
    if method == "last_n":
        split = max(1, n - n_test)
    else:
        split = max(1, int(n * (1 - test_fraction)))
    train_idx = list(range(split))
    test_idx  = list(range(split, n))
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
        train_idx,
        test_idx,
    )
