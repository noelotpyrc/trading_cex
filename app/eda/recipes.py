from __future__ import annotations

import numpy as np
import pandas as pd


def forward_return(df: pd.DataFrame, h: int, close_col: str = "close") -> pd.Series:
    if close_col not in df.columns:
        raise ValueError(f"Column '{close_col}' not in DataFrame")
    close = pd.to_numeric(df[close_col], errors="coerce")
    fr = close.shift(-int(h)) / close - 1.0
    return fr


def direction_label(df: pd.DataFrame, h: int, thr: float = 0.0, close_col: str = "close") -> pd.Series:
    fr = forward_return(df, h=h, close_col=close_col)
    lab = (fr > float(thr)).astype("Int64").astype("float").astype("Int64")
    return lab


def triple_barrier(
    df: pd.DataFrame,
    h: int,
    tp: float,
    sl: float,
    *,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
) -> pd.Series:
    """Simple triple-barrier labeling over the next h bars.

    Returns:
        pd.Series of int labels in {1, -1, 0}
    Notes:
        - This implementation is optimized for clarity over speed and is fine for EDA windows.
        - Uses close as the entry reference; checks future highs/lows for barrier hits.
    """
    for c in (close_col, high_col, low_col):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not in DataFrame")

    close = pd.to_numeric(df[close_col], errors="coerce").to_numpy()
    high = pd.to_numeric(df[high_col], errors="coerce").to_numpy()
    low = pd.to_numeric(df[low_col], errors="coerce").to_numpy()

    n = len(df)
    out = np.zeros(n, dtype=np.int8)
    horizon = int(max(1, h))

    # For each index i, look ahead up to i+h (exclusive of i, inclusive of i+h)
    for i in range(n):
        j_end = min(n - 1, i + horizon)
        # No full future coverage -> label will be NaN later (we'll drop tail outside)
        if j_end <= i:
            continue
        up = close[i] * (1.0 + float(tp))
        dn = close[i] * (1.0 - float(sl))
        label = 0
        # Find first hit time if any
        for j in range(i + 1, j_end + 1):
            if high[j] >= up:
                label = 1
                break
            if low[j] <= dn:
                label = -1
                break
        out[i] = label

    s = pd.Series(out, index=df.index)
    # Tail rows without full horizon should not be used; caller drops them explicitly
    return s

