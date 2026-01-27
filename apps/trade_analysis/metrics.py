from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal, Tuple, Dict, Any

try:
    from scipy.stats import wasserstein_distance as _wdist
except Exception:
    _wdist = None


ScaleMethod = Literal["iqr", "q90q10", "mad"]


def _wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    """
    1D Wasserstein-1 distance (Earth Mover's Distance) for equal weights.
    Falls back to a simple numpy implementation if SciPy isn't available.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan

    if _wdist is not None:
        return float(_wdist(a, b))

    # Numpy fallback (equal weights): interpolate empirical quantile functions
    a = np.sort(a)
    b = np.sort(b)
    na, nb = a.size, b.size
    # grid of cumulative probabilities
    p = np.linspace(0.0, 1.0, max(na, nb), endpoint=True)
    qa = np.quantile(a, p, method="linear")
    qb = np.quantile(b, p, method="linear")
    # integral of absolute difference ~ mean on uniform grid
    return float(np.mean(np.abs(qa - qb)))


def tqss(
    x: pd.Series,
    y: pd.Series,
    q: float = 0.25,
    y_hi: float = 0.9,
    y_lo: float = 0.1,
    eps: float = 1e-12,
):
    """
    Tail Quantile Shift Score (TQSS):
      (Q_hi(Y|X top-q) - Q_hi(Y|X bottom-q)) + (Q_lo(Y|X top-q) - Q_lo(Y|X bottom-q))

    Emphasizes two tails of Y by conditioning on two tails of X.
    """
    if not (0 < q < 0.5):
        raise ValueError("q must be in (0, 0.5)")
    if not (0 < y_lo < y_hi < 1):
        raise ValueError("Require 0 < y_lo < y_hi < 1")

    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if df.empty:
        return np.nan, {"n": 0, "n_low": 0, "n_high": 0}

    x_vals = df["x"].to_numpy()
    y_vals = df["y"].to_numpy()

    lo_thr = np.quantile(x_vals, q)
    hi_thr = np.quantile(x_vals, 1 - q)

    # Keep tails disjoint even when many values equal the threshold
    low_mask = x_vals <= lo_thr
    high_mask = x_vals >= hi_thr
    if lo_thr >= hi_thr - eps:
        # extreme tie case; use ranks to split deterministically
        ranks = pd.Series(x_vals).rank(method="first").to_numpy()
        n = ranks.size
        k = int(np.floor(q * n))
        low_mask = ranks <= k
        high_mask = ranks > n - k

    y_low = y_vals[low_mask]
    y_high = y_vals[high_mask]

    if y_low.size < 10 or y_high.size < 10:
        return np.nan, {"n": len(df), "n_low": y_low.size, "n_high": y_high.size}

    qh_high = np.quantile(y_high, y_hi)
    qh_low = np.quantile(y_low, y_hi)
    ql_high = np.quantile(y_high, y_lo)
    ql_low = np.quantile(y_low, y_lo)

    score = (qh_high - qh_low) + (ql_high - ql_low)
    info = {
        "n": len(df),
        "n_low": y_low.size,
        "n_high": y_high.size,
        "x_lo_thr": float(lo_thr),
        "x_hi_thr": float(hi_thr),
        "Qy_hi_high": float(qh_high),
        "Qy_hi_low": float(qh_low),
        "Qy_lo_high": float(ql_high),
        "Qy_lo_low": float(ql_low),
    }
    return float(score), info


def signed_wasserstein(
    x: pd.Series,
    y: pd.Series,
    q: float = 0.25,
    sign_center: str = "median",  # or "mean"
    eps: float = 1e-12,
):
    """
    Signed Wasserstein distance between Y distributions in X top-q vs bottom-q tails:
      sW = sign(center(Y|top) - center(Y|bottom)) * W1(Y|top, Y|bottom)

    W1 is Wasserstein-1 distance (Earth Mover's Distance).
    """
    if not (0 < q < 0.5):
        raise ValueError("q must be in (0, 0.5)")
    if sign_center not in ("median", "mean"):
        raise ValueError("sign_center must be 'median' or 'mean'")

    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if df.empty:
        return np.nan, {"n": 0, "n_low": 0, "n_high": 0}

    x_vals = df["x"].to_numpy()
    y_vals = df["y"].to_numpy()

    lo_thr = np.quantile(x_vals, q)
    hi_thr = np.quantile(x_vals, 1 - q)

    low_mask = x_vals <= lo_thr
    high_mask = x_vals >= hi_thr
    if lo_thr >= hi_thr - eps:
        ranks = pd.Series(x_vals).rank(method="first").to_numpy()
        n = ranks.size
        k = int(np.floor(q * n))
        low_mask = ranks <= k
        high_mask = ranks > n - k

    y_low = y_vals[low_mask]
    y_high = y_vals[high_mask]

    if y_low.size < 10 or y_high.size < 10:
        return np.nan, {"n": len(df), "n_low": y_low.size, "n_high": y_high.size}

    w1 = _wasserstein_1d(y_high, y_low)

    if sign_center == "median":
        center_diff = np.median(y_high) - np.median(y_low)
    else:
        center_diff = np.mean(y_high) - np.mean(y_low)

    sgn = 0.0 if np.isclose(center_diff, 0.0) else float(np.sign(center_diff))
    sw = sgn * w1

    info = {
        "n": len(df),
        "n_low": y_low.size,
        "n_high": y_high.size,
        "x_lo_thr": float(lo_thr),
        "x_hi_thr": float(hi_thr),
        "w1": float(w1),
        "center_diff": float(center_diff),
        "sign": float(sgn),
    }
    return float(sw), info


def robust_scale(y: pd.Series, method: ScaleMethod = "iqr", eps: float = 1e-12) -> float:
    """
    Robust scale for normalizing metrics measured in 'units of y'.

    - iqr    : Q75 - Q25
    - q90q10 : Q90 - Q10   (more tail-focused)
    - mad    : median(|y - median(y)|)

    Returns np.nan if scale is degenerate.
    """
    yv = pd.Series(y).dropna().to_numpy(dtype=float)
    if yv.size == 0:
        return np.nan

    if method == "iqr":
        s = float(np.quantile(yv, 0.75) - np.quantile(yv, 0.25))
    elif method == "q90q10":
        s = float(np.quantile(yv, 0.90) - np.quantile(yv, 0.10))
    elif method == "mad":
        med = float(np.median(yv))
        s = float(np.median(np.abs(yv - med)))
    else:
        raise ValueError(f"Unknown method: {method}")

    return s if np.isfinite(s) and s > eps else np.nan


def normalize_by_y_scale(score: float, y: pd.Series, method: ScaleMethod = "iqr", eps: float = 1e-12) -> float:
    """Return score / robust_scale(y)."""
    s = robust_scale(y, method=method, eps=eps)
    return float(score) / s if np.isfinite(s) else np.nan


# ---- Wrappers around your existing raw metrics ----
# Assumes you already have:
#   tqss(x, y, q=..., y_hi=..., y_lo=...) -> (score, info)
#   signed_wasserstein(x, y, q=..., sign_center=...) -> (score, info)

def tqss_norm(
    x: pd.Series,
    y: pd.Series,
    q: float = 0.25,
    y_hi: float = 0.9,
    y_lo: float = 0.1,
    scale: ScaleMethod = "iqr",
    eps: float = 1e-12,
) -> Tuple[float, Dict[str, Any]]:
    """Normalized TQSS = TQSS / robust_scale(y)."""
    score, info = tqss(x, y, q=q, y_hi=y_hi, y_lo=y_lo, eps=eps)  # your raw fn
    score_norm = normalize_by_y_scale(score, y, method=scale, eps=eps)
    info = dict(info or {})
    info.update({"tqss_raw": score, "tqss_norm": score_norm, "y_scale": robust_scale(y, scale, eps), "scale_kind": scale})
    return score_norm, info


def signed_wasserstein_norm(
    x: pd.Series,
    y: pd.Series,
    q: float = 0.25,
    scale: ScaleMethod = "iqr",
    sign_center: Literal["median", "mean"] = "median",
    eps: float = 1e-12,
) -> Tuple[float, Dict[str, Any]]:
    """Normalized signed Wasserstein = sW / robust_scale(y)."""
    score, info = signed_wasserstein(x, y, q=q, sign_center=sign_center, eps=eps)  # your raw fn
    score_norm = normalize_by_y_scale(score, y, method=scale, eps=eps)
    info = dict(info or {})
    info.update({"sw_raw": score, "sw_norm": score_norm, "y_scale": robust_scale(y, scale, eps), "scale_kind": scale})
    return score_norm, info


def wasserstein_magnitude_norm(
    x: pd.Series,
    y: pd.Series,
    q: float = 0.25,
    scale: ScaleMethod = "iqr",
    eps: float = 1e-12,
) -> Tuple[float, Dict[str, Any]]:
    """
    Normalized Wasserstein magnitude (W1) = |W1| / robust_scale(y).

    Uses your signed_wasserstein() info["w1"] if present.
    """
    _, info = signed_wasserstein(x, y, q=q, sign_center="median", eps=eps)  # your raw fn
    w1 = float(info.get("w1", np.nan))
    w1_norm = normalize_by_y_scale(w1, y, method=scale, eps=eps)
    out = dict(info or {})
    out.update({"w1_raw": w1, "w1_norm": w1_norm, "y_scale": robust_scale(y, scale, eps), "scale_kind": scale})
    return w1_norm, out

def tail_masks(x: pd.Series, q: float = 0.2, eps: float = 1e-12) -> Tuple[pd.Series, pd.Series, float, float]:
    """
    Boolean masks for bottom-q and top-q tails of x (disjoint, robust to ties).
    Returns: (low_mask, high_mask, lo_thr, hi_thr) aligned to x.index.
    """
    xv = x.to_numpy(dtype=float)
    lo_thr = float(np.nanquantile(xv, q))
    hi_thr = float(np.nanquantile(xv, 1 - q))

    low_mask = x <= lo_thr
    high_mask = x >= hi_thr

    # If thresholds overlap (many ties), split deterministically by rank
    if lo_thr >= hi_thr - eps:
        ranks = x.rank(method="first")
        n = ranks.notna().sum()
        k = int(np.floor(q * n))
        low_mask = ranks <= k
        high_mask = ranks > n - k

    return low_mask, high_mask, lo_thr, hi_thr


def median_shift_norm(
    x: pd.Series,
    y: pd.Series,
    q: float = 0.2,
    scale: ScaleMethod = "q90q10",
    eps: float = 1e-12,
) -> Tuple[float, Dict[str, Any]]:
    """
    Median shift normalized by a robust scale of y:

      med_norm = (median(Y | X top-q) - median(Y | X bottom-q)) / scale(Y)

    This is a simple, explainable "center shift" metric to complement TQSS and Wasserstein.
    """
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if df.empty:
        return np.nan, {"n": 0, "n_low": 0, "n_high": 0}

    low_mask, high_mask, lo_thr, hi_thr = tail_masks(df["x"], q=q, eps=eps)

    y_low = df.loc[low_mask, "y"].to_numpy(dtype=float)
    y_high = df.loc[high_mask, "y"].to_numpy(dtype=float)

    out = {
        "n": int(len(df)),
        "n_low": int(y_low.size),
        "n_high": int(y_high.size),
        "x_lo_thr": float(lo_thr),
        "x_hi_thr": float(hi_thr),
        "scale_kind": scale,
    }

    if y_low.size < 10 or y_high.size < 10:
        out.update({"med_low": np.nan, "med_high": np.nan, "med_diff": np.nan, "y_scale": np.nan})
        return np.nan, out

    med_low = float(np.median(y_low))
    med_high = float(np.median(y_high))
    med_diff = med_high - med_low

    y_scale = robust_scale(df["y"], method=scale, eps=eps)
    med_norm = med_diff / y_scale if np.isfinite(y_scale) else np.nan

    out.update({
        "med_low": med_low,
        "med_high": med_high,
        "med_diff": float(med_diff),
        "y_scale": float(y_scale) if np.isfinite(y_scale) else np.nan,
        "med_norm": float(med_norm) if np.isfinite(med_norm) else np.nan,
    })
    return float(med_norm) if np.isfinite(med_norm) else np.nan, out