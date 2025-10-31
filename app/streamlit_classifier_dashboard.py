"""Classifier Dashboard: evaluate 24h event probabilities over recent data.

Features:
- OHLCV candlestick with signal markers (y_pred >= threshold).
- Rolling AUC (default 90 days) computed on all predictions.
- Rolling win-rate (default 90 days) computed on triggered trades.
- Rolling trading returns (default 90 days), showing average and total (simple sum).

Data sources (defaults to your local paths; override via env or UI):
- OHLCV DuckDB: /Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_ohlcv.duckdb
- Classifier predictions DuckDB: /Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_prediction_classifier.duckdb
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure repo root on sys.path so `from app...` works when run via `streamlit run`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Local imports (reuse data loaders & metrics)
from app.dashboard_data import (
    DataLoadError,
    load_ohlcv,
    load_predictions,
    compute_signal_metrics,
)


# ===== Paths and defaults =====

FALLBACK_OHLCV_DB = Path(
    "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_ohlcv.duckdb"
)
FALLBACK_CLASSIFIER_DB = Path(
    "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_prediction_classifier.duckdb"
)


def _resolve_path(env_var: str, fallback: Path) -> str:
    val = os.environ.get(env_var)
    return val if val else str(fallback)


DEFAULT_OHLCV_DB = _resolve_path("DASHBOARD_OHLCV_DUCKDB", FALLBACK_OHLCV_DB)
DEFAULT_CLASSIFIER_DB = _resolve_path(
    "DASHBOARD_PREDICTIONS_CLASSIFIER_DUCKDB", FALLBACK_CLASSIFIER_DB
)


# ===== Cached loaders =====


@st.cache_data(ttl=None, show_spinner=False)
def cached_load_ohlcv(path_str: str) -> pd.DataFrame:
    if not path_str:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    return load_ohlcv(Path(path_str))


@st.cache_data(ttl=None, show_spinner=False)
def cached_load_predictions(path_str: str) -> pd.DataFrame:
    if not path_str:
        return pd.DataFrame(columns=["timestamp", "y_pred", "model_path", "feature_key", "created_at"])
    # Load full table; we'll filter by model/date in-app (table is small enough)
    return load_predictions(Path(path_str))


# ===== Utilities =====


def _timestamp_bounds(start_date: date, end_date: date) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = pd.Timestamp(start_date).floor("D")
    end_ts = pd.Timestamp(end_date + timedelta(days=1)).floor("D")
    return start_ts, end_ts


def _filter_by_range(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    if df.empty or "timestamp" not in df.columns:
        return df
    mask = (df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)
    return df.loc[mask].copy()


def _quantile_default(series: pd.Series, q: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.5
    q = float(np.clip(q, 0.0, 1.0))
    return float(s.quantile(q))


def _compute_ytrue_all(preds: pd.DataFrame, ohlcv: pd.DataFrame, *, horizon_h: int) -> pd.DataFrame:
    """Attach forward_return/log_return for ALL rows via compute_signal_metrics trick.

    We call with direction_threshold=-inf, include_long=True to bypass filtering and
    obtain forward_return/log_return for every prediction row.
    """
    if preds.empty:
        return preds.assign(forward_return=pd.NA, forward_log_return=pd.NA, has_full_horizon=False)
    augmented = compute_signal_metrics(
        preds,
        ohlcv,
        horizon_hours=horizon_h,
        direction_threshold=float("-inf"),
        include_long=True,
        include_short=False,
    )
    # y_true: forward_return > 0 (only when full horizon is available)
    y = augmented.copy()
    y["y_true"] = np.where(y["forward_return"].notna(), (y["forward_return"] > 0).astype(float), np.nan)
    return y


def _compute_tp_before_sl_labels(
    preds: pd.DataFrame,
    ohlcv: pd.DataFrame,
    *,
    horizon_h: int = 24,
    up_frac: float = 0.04,
    dn_frac: float = 0.02,
) -> pd.DataFrame:
    """Compute binary y_true indicating TP(+up_frac) occurs before SL(-dn_frac) within horizon.

    - Returns DataFrame with columns: timestamp, y_true_tp_sl, has_full_horizon
    - y_true_tp_sl ∈ {0,1} where 1=TP before SL, 0=otherwise (including SL first or neither)
    - NaN when full horizon not available after the prediction timestamp.
    """
    if preds.empty or ohlcv.empty or "timestamp" not in preds.columns or "timestamp" not in ohlcv.columns:
        return preds[["timestamp"]].assign(y_true_tp_sl=np.nan, has_full_horizon=False) if "timestamp" in preds.columns else pd.DataFrame(columns=["timestamp", "y_true_tp_sl", "has_full_horizon"])

    o = ohlcv.sort_values("timestamp").reset_index(drop=True)
    p = preds.sort_values("timestamp").reset_index(drop=True)

    ts_to_idx = {pd.Timestamp(t): i for i, t in enumerate(o["timestamp"]) }
    close = pd.to_numeric(o["close"], errors="coerce").to_numpy()
    high = pd.to_numeric(o.get("high", np.nan), errors="coerce").to_numpy()
    low = pd.to_numeric(o.get("low", np.nan), errors="coerce").to_numpy()

    n = len(o)
    y_true_vals: List[float] = []
    full_flags: List[bool] = []

    for t in p["timestamp"]:
        idx = ts_to_idx.get(pd.Timestamp(t))
        if idx is None or idx >= n:
            y_true_vals.append(np.nan)
            full_flags.append(False)
            continue
        j_end = min(n - 1, idx + int(horizon_h))
        if j_end <= idx:
            y_true_vals.append(np.nan)
            full_flags.append(False)
            continue

        entry = close[idx]
        if not np.isfinite(entry):
            y_true_vals.append(np.nan)
            full_flags.append(False)
            continue

        up = entry * (1.0 + float(up_frac))
        dn = entry * (1.0 - float(dn_frac))

        label = 0.0
        hit = False
        for j in range(idx + 1, j_end + 1):
            hj = high[j]
            lj = low[j]
            if np.isfinite(hj) and hj >= up:
                label = 1.0
                hit = True
                break
            if np.isfinite(lj) and lj <= dn:
                label = 0.0
                hit = True
                break
        if hit:
            y_true_vals.append(label)
        else:
            # Neither barrier hit within horizon → treat as negative
            y_true_vals.append(0.0)
        # Full horizon available if we had at least one future bar and idx+horizon exists
        full_flags.append((idx + int(horizon_h)) < n)

    out = p[["timestamp"]].copy()
    out["y_true_tp_sl"] = y_true_vals
    out["has_full_horizon"] = full_flags
    return out


def _compute_strategy_return_tp_sl(
    timestamps: pd.Series,
    ohlcv: pd.DataFrame,
    *,
    horizon_h: int = 24,
    up_frac: float = 0.04,
    dn_frac: float = 0.02,
) -> pd.Series:
    """For long-only entries at given timestamps, compute strategy return using
    TP(+up_frac) before SL(-dn_frac); otherwise natural close at 24h. If the
    horizon is incomplete and neither barrier is hit, return NaN.

    Returns a Series aligned to input timestamps.
    """
    if timestamps is None or timestamps.empty or ohlcv.empty:
        return pd.Series([], dtype=float)

    o = ohlcv.sort_values("timestamp").reset_index(drop=True)
    ts_to_idx = {pd.Timestamp(t): i for i, t in enumerate(o["timestamp"]) }
    close = pd.to_numeric(o["close"], errors="coerce").to_numpy()
    high = pd.to_numeric(o.get("high", np.nan), errors="coerce").to_numpy()
    low = pd.to_numeric(o.get("low", np.nan), errors="coerce").to_numpy()

    n = len(o)
    out_vals: List[float] = []

    for t in timestamps:
        idx = ts_to_idx.get(pd.Timestamp(t))
        if idx is None or idx >= n:
            out_vals.append(np.nan)
            continue
        entry = close[idx] if np.isfinite(close[idx]) else np.nan
        if not np.isfinite(entry):
            out_vals.append(np.nan)
            continue
        j_end = min(n - 1, idx + int(horizon_h))
        if j_end <= idx:
            out_vals.append(np.nan)
            continue
        up = entry * (1.0 + float(up_frac))
        dn = entry * (1.0 - float(dn_frac))
        realized = False
        ret = np.nan
        for j in range(idx + 1, j_end + 1):
            hj = high[j]
            lj = low[j]
            if np.isfinite(hj) and hj >= up:
                ret = float(up / entry - 1.0)  # ~+up_frac
                realized = True
                break
            if np.isfinite(lj) and lj <= dn:
                ret = float(dn / entry - 1.0)  # ~-dn_frac
                realized = True
                break
        if not realized:
            # Natural close if full horizon available
            if (idx + int(horizon_h)) < n and np.isfinite(close[j_end]):
                ret = float(close[j_end] / entry - 1.0)
            else:
                ret = np.nan
        out_vals.append(ret)

    return pd.Series(out_vals, index=timestamps.index, dtype=float)

def _compute_rolling_auc(df: pd.DataFrame, *, window_days: int, min_obs: int = 100) -> pd.DataFrame:
    """Time-based rolling AUC using all predictions (y_true vs y_pred).

    Returns a DataFrame with columns: timestamp, auc
    """
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore
    except Exception:
        st.error("scikit-learn is required for AUC. Install with: pip install scikit-learn")
        return pd.DataFrame(columns=["timestamp", "auc"])

    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame(columns=["timestamp", "auc"])

    data = df.dropna(subset=["y_true", "y_pred"]).sort_values("timestamp").reset_index(drop=True)
    if data.empty:
        return pd.DataFrame(columns=["timestamp", "auc"])

    ts = data["timestamp"].to_numpy()
    y_true = data["y_true"].astype(float).to_numpy()
    y_score = pd.to_numeric(data["y_pred"], errors="coerce").to_numpy()

    out_ts: List[pd.Timestamp] = []
    out_auc: List[float] = []
    w = pd.Timedelta(days=int(max(1, window_days)))

    # Two-pointer sliding window for time-based selection
    j = 0
    for i in range(len(ts)):
        t_hi = pd.Timestamp(ts[i])
        t_lo = t_hi - w
        while j < i and pd.Timestamp(ts[j]) < t_lo:
            j += 1
        y_win = y_true[j : i + 1]
        s_win = y_score[j : i + 1]
        # Need both classes and min observations
        valid = np.isfinite(y_win) & np.isfinite(s_win)
        yv = y_win[valid]
        sv = s_win[valid]
        auc_val = np.nan
        if yv.size >= max(10, min_obs) and len(np.unique(yv)) >= 2:
            try:
                auc_val = float(roc_auc_score(yv, sv))
            except Exception:
                auc_val = np.nan
        out_ts.append(t_hi)
        out_auc.append(auc_val)

    return pd.DataFrame({"timestamp": out_ts, "auc": out_auc})


def _compute_rolling_trade_stats(trades: pd.DataFrame, *, window_days: int) -> pd.DataFrame:
    """Compute rolling win-rate, avg return, total return over a time window for triggered trades.

    returns DataFrame with: timestamp, win_rate, avg_return, total_return, n_trades
    """
    if trades.empty:
        return pd.DataFrame(columns=["timestamp", "win_rate", "avg_return", "total_return", "n_trades"])

    df = trades.copy()
    df = df[df["has_full_horizon"] == True]  # noqa: E712
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "win_rate", "avg_return", "total_return", "n_trades"])

    df["win"] = (pd.to_numeric(df["forward_return"], errors="coerce") > 0).astype(float)
    df = df.dropna(subset=["forward_return"])  # ensure realized only
    df = df.sort_values("timestamp").reset_index(drop=True)

    window = f"{int(max(1, window_days))}D"

    # Rolling metrics (time-based windows via 'on=timestamp')
    win_rate = df.rolling(window=window, on="timestamp")["win"].mean()
    avg_return = df.rolling(window=window, on="timestamp")["forward_return"].mean()
    total_return = df.rolling(window=window, on="timestamp")["forward_return"].sum()
    n_trades = df.rolling(window=window, on="timestamp")["win"].count()

    out = df[["timestamp"]].copy()
    out["win_rate"] = win_rate.values
    out["avg_return"] = avg_return.values
    out["total_return"] = total_return.values
    out["n_trades"] = n_trades.values
    return out


def _build_market_figure(ohlcv_df: pd.DataFrame, trades_df: pd.DataFrame) -> go.Figure:
    """Candles + markers for triggered trades.

    Marker color encodes correctness (green=win, red=loss, grey=pending),
    opacity encodes drawdown magnitude (lighter for larger drawdown).
    """
    fig = go.Figure()

    if not ohlcv_df.empty:
        fig.add_trace(
            go.Candlestick(
                x=ohlcv_df["timestamp"],
                open=ohlcv_df["open"],
                high=ohlcv_df["high"],
                low=ohlcv_df["low"],
                close=ohlcv_df["close"],
                name="OHLCV (1h)",
                increasing_line_color="#2ca02c",
                decreasing_line_color="#d62728",
                showlegend=False,
            )
        )

    if not trades_df.empty:
        close_series = trades_df["close"] if "close" in trades_df.columns else pd.Series(np.nan, index=trades_df.index)
        close_values = pd.to_numeric(close_series, errors="coerce")
        dd_series = trades_df.get("max_drawdown", pd.Series(dtype=float))
        dd = pd.to_numeric(dd_series, errors="coerce").abs().fillna(0.0)
        opacity = (1.0 - np.clip(dd / 0.10, 0.0, 0.7)).clip(0.3, 1.0)

        status = trades_df.get("is_correct")
        colors = status.map({True: "#2ca02c", False: "#d62728"}).fillna("#7f7f7f").tolist()

        hover_text = []
        for _, row in trades_df.iterrows():
            pred = row.get("y_pred")
            fr = row.get("forward_return")
            ddv = row.get("max_drawdown")
            pred_str = "n/a" if pd.isna(pred) else f"{float(pred):.4f}"
            fr_str = "n/a" if pd.isna(fr) else f"{float(fr)*100:.2f}%"
            dd_str = "n/a" if pd.isna(ddv) else f"{float(ddv)*100:.2f}%"
            status_str = (
                "Correct" if row.get("is_correct") is True else ("Incorrect" if row.get("is_correct") is False else "Pending")
            )
            model_path = row.get("model_path")
            lines = [f"y_pred: {pred_str}", f"24h return: {fr_str}", f"Max drawdown: {dd_str}", f"Status: {status_str}"]
            if isinstance(model_path, str) and model_path:
                lines.append(model_path)
            hover_text.append("<br>".join(lines))

        fig.add_trace(
            go.Scatter(
                x=trades_df["timestamp"],
                y=close_values,
                mode="markers",
                name="Signals",
                marker=dict(size=9, color=colors, opacity=opacity, line=dict(color="#1f2937", width=1)),
                hovertext=hover_text,
                hovertemplate="%{x}<br>%{hovertext}<extra></extra>",
            )
        )

    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Timestamp", rangeslider=dict(visible=False)),
        yaxis=dict(title="Price"),
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def _build_line_figure(df: pd.DataFrame, y_col: str, *, title: str, y_range: Optional[Tuple[float, float]] = None) -> go.Figure:
    fig = go.Figure()
    if not df.empty and y_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[y_col],
                mode="lines",
                name=y_col,
            )
        )
    layout = dict(
        height=320,
        margin=dict(l=0, r=0, t=30, b=30),
        xaxis=dict(title="Timestamp"),
        yaxis=dict(title=y_col),
        template="plotly_white",
        title=title,
    )
    if y_range is not None:
        layout["yaxis"]["range"] = list(y_range)
    fig.update_layout(**layout)
    return fig


# Rolling stats for all bars (not only trades)
def _compute_rolling_all_bars_stats(all_rows: pd.DataFrame, *, window_days: int) -> pd.DataFrame:
    """Compute rolling win-rate, avg return, total return over all bars with full horizon."""
    if all_rows.empty:
        return pd.DataFrame(columns=["timestamp", "win_rate", "avg_return", "total_return", "n_rows"])
    df = all_rows.copy()
    df = df[df["has_full_horizon"] == True]  # noqa: E712
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "win_rate", "avg_return", "total_return", "n_rows"])
    df["win"] = (pd.to_numeric(df["forward_return"], errors="coerce") > 0).astype(float)
    df = df.dropna(subset=["forward_return"]).sort_values("timestamp").reset_index(drop=True)
    window = f"{int(max(1, window_days))}D"
    win_rate = df.rolling(window=window, on="timestamp")["win"].mean()
    avg_return = df.rolling(window=window, on="timestamp")["forward_return"].mean()
    total_return = df.rolling(window=window, on="timestamp")["forward_return"].sum()
    n_rows = df.rolling(window=window, on="timestamp")["win"].count()
    out = df[["timestamp"]].copy()
    out["win_rate"] = win_rate.values
    out["avg_return"] = avg_return.values
    out["total_return"] = total_return.values
    out["n_rows"] = n_rows.values
    return out


def _build_overlay_line_figure(
    traces: List[Tuple[pd.DataFrame, str, str]], *, title: str, y_range: Optional[Tuple[float, float]] = None
) -> go.Figure:
    """Build a multi-line figure from (df, y_col, name) tuples."""
    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for idx, (df, y_col, name) in enumerate(traces):
        if df is None or df.empty or y_col not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[y_col],
                mode="lines",
                name=name,
                line=dict(color=colors[idx % len(colors)]),
            )
        )
    layout = dict(
        height=320,
        margin=dict(l=0, r=0, t=30, b=30),
        xaxis=dict(title="Timestamp"),
        yaxis=dict(title="value"),
        template="plotly_white",
        title=title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if y_range is not None:
        layout["yaxis"]["range"] = list(y_range)
    fig.update_layout(**layout)
    return fig


# Build cumulative P&L figure (sum of per-trade returns, $1 notional per trade)
def _build_cum_pnl_df(ts: pd.Series, rets: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"timestamp": pd.to_datetime(ts), "ret": pd.to_numeric(rets, errors="coerce")})
    df = df.dropna(subset=["timestamp", "ret"]).sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "cum_return"])
    df["cum_return"] = df["ret"].cumsum()
    return df[["timestamp", "cum_return"]]


# Rolling stats for strategy returns on trades
def _compute_rolling_strategy_stats(
    trades: pd.DataFrame,
    ohlcv: pd.DataFrame,
    *,
    window_days: int,
    horizon_h: int = 24,
    up_frac: float = 0.04,
    dn_frac: float = 0.02,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["timestamp", "avg_return_strategy", "total_return_strategy", "n_trades_strategy"])
    # Compute per-trade strategy return (TP/SL else natural close)
    strat_ret = _compute_strategy_return_tp_sl(trades["timestamp"], ohlcv, horizon_h=horizon_h, up_frac=up_frac, dn_frac=dn_frac)
    df = trades[["timestamp"]].copy()
    df["strategy_return"] = pd.to_numeric(strat_ret, errors="coerce")
    # Keep realized strategy outcomes
    df = df.dropna(subset=["strategy_return"]).sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "avg_return_strategy", "total_return_strategy", "n_trades_strategy"])
    window = f"{int(max(1, window_days))}D"
    avg_ret = df.rolling(window=window, on="timestamp")["strategy_return"].mean()
    total_ret = df.rolling(window=window, on="timestamp")["strategy_return"].sum()
    n_tr = df.rolling(window=window, on="timestamp")["strategy_return"].count()
    out = df[["timestamp"]].copy()
    out["avg_return_strategy"] = avg_ret.values
    out["total_return_strategy"] = total_ret.values
    out["n_trades_strategy"] = n_tr.values
    return out


# ===== Model artifacts: metrics and feature importance =====


def _normalise_importance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    col_map = {c.lower(): c for c in df.columns}
    feature_col = col_map.get("feature") or next((c for c in df.columns if "feature" in c.lower()), None)
    importance_col = col_map.get("importance") or next((c for c in df.columns if "importance" in c.lower() or "gain" in c.lower()), None)
    if not feature_col or not importance_col:
        return pd.DataFrame(columns=["feature", "importance"])
    cleaned = df[[feature_col, importance_col]].rename(columns={feature_col: "feature", importance_col: "importance"})
    cleaned = cleaned.dropna(subset=["feature"]).copy()
    cleaned["importance"] = pd.to_numeric(cleaned["importance"], errors="coerce")
    cleaned = cleaned.dropna(subset=["importance"]).sort_values("importance", ascending=False)
    return cleaned


@st.cache_data(ttl=None, show_spinner=False)
def cached_load_feature_importance(model_path: str) -> pd.DataFrame:
    if not model_path:
        return pd.DataFrame(columns=["feature", "importance", "__source_path__"])
    mp = Path(model_path)
    candidates: List[Path] = []
    if mp.is_file():
        candidates.append(mp.parent / "feature_importance.csv")
    if mp.is_dir():
        candidates.append(mp / "feature_importance.csv")
    if mp.parent != mp:
        candidates.append(mp.parent / "feature_importance.csv")
    if mp.parent.parent != mp.parent:
        candidates.append(mp.parent.parent / "feature_importance.csv")

    seen: set[Path] = set()
    for cand in candidates:
        if cand in seen or not cand.exists():
            continue
        seen.add(cand)
        try:
            df = pd.read_csv(cand)
        except Exception:
            continue
        if df.empty:
            continue
        df = _normalise_importance_columns(df)
        if df.empty:
            continue
        df["__source_path__"] = str(cand)
        return df
    return pd.DataFrame(columns=["feature", "importance", "__source_path__"])


def _flatten_dict(d: dict, parent: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{parent}.{k}" if parent else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        else:
            out[key] = v
    return out


@st.cache_data(ttl=None, show_spinner=False)
def cached_load_model_metrics(model_path: str) -> pd.DataFrame:
    if not model_path:
        return pd.DataFrame(columns=["metric", "value", "__source_path__"])
    mp = Path(model_path)
    search_dirs: List[Path] = []
    if mp.is_file():
        search_dirs.extend([mp.parent, mp.parent.parent])
    elif mp.is_dir():
        search_dirs.extend([mp, mp.parent])
    tried: set[Path] = set()
    # Preferred JSON, then CSV
    for d in search_dirs:
        if not d or d in tried or not d.exists():
            continue
        tried.add(d)
        json_path = d / "metrics.json"
        if json_path.exists():
            try:
                import json as _json  # local import to avoid top-level dependency
                with open(json_path, "r") as f:
                    raw = _json.load(f)
                if isinstance(raw, dict):
                    flat = _flatten_dict(raw)
                    df = pd.DataFrame({"metric": list(flat.keys()), "value": list(flat.values())})
                elif isinstance(raw, list):
                    # If list of dicts, try to find 'metric'/'value' or collapse numerics
                    tmp = pd.DataFrame(raw)
                    if {"metric", "value"}.issubset(tmp.columns):
                        df = tmp[["metric", "value"]].copy()
                    else:
                        num = tmp.select_dtypes(include=[np.number]).mean().to_frame(name="value").reset_index().rename(columns={"index": "metric"})
                        df = num
                else:
                    df = pd.DataFrame(columns=["metric", "value"])
                df["__source_path__"] = str(json_path)
                return df
            except Exception:
                pass
        csv_path = d / "metrics.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if {"metric", "value"}.issubset(df.columns):
                    out = df[["metric", "value"]].copy()
                elif {"name", "value"}.issubset(df.columns):
                    out = df[["name", "value"]].rename(columns={"name": "metric"})
                else:
                    # summarize numeric columns as a single-row, then melt
                    nums = df.select_dtypes(include=[np.number])
                    if not nums.empty:
                        ser = nums.mean().rename_axis("metric").reset_index(name="value")
                        out = ser
                    else:
                        out = pd.DataFrame(columns=["metric", "value"])
                out["__source_path__"] = str(csv_path)
                return out
            except Exception:
                pass
    return pd.DataFrame(columns=["metric", "value", "__source_path__"])


def _build_feature_importance_fig(df: pd.DataFrame, *, top_n: int = 20) -> go.Figure:
    fig = go.Figure()
    if not df.empty:
        top = df.head(top_n)
        fig.add_bar(x=top["feature"], y=top["importance"], name="importance")
        fig.update_layout(xaxis_tickangle=-45)
    fig.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=30, b=80),
        template="plotly_white",
        title=f"Top-{top_n} Feature Importance",
        yaxis_title="importance",
    )
    return fig


def _build_all_models_predictions_fig(df: pd.DataFrame, *, models: List[str]) -> go.Figure:
    fig = go.Figure()
    if not df.empty:
        for run_id in models:
            sub = df[df["run_id"] == run_id]
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub["timestamp"],
                    y=sub["y_pred"],
                    mode="lines",
                    name=str(run_id),
                    hovertemplate="%{x}<br>y_pred=%{y:.4f}<extra></extra>",
                )
            )
    fig.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=30, b=30),
        template="plotly_white",
        title="All Models — y_pred over time",
        yaxis_title="y_pred",
        xaxis_title="timestamp",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ===== Decile calibration utilities =====

def _compute_decile_edges(scores: pd.Series, *, n_bins: int = 10) -> Optional[np.ndarray]:
    """Legacy qcut-based edges (may drop bins on ties). Kept for reference."""
    s = pd.to_numeric(scores, errors="coerce").dropna()
    if s.empty:
        return None
    try:
        _, edges = pd.qcut(s, q=min(int(n_bins), max(1, s.nunique())), retbins=True, duplicates="drop")
    except Exception:
        return None
    return np.asarray(edges)


def _decile_stats(df: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    """qcut-based decile stats (may return < n bins on ties)."""
    if df.empty or edges is None or len(edges) < 2:
        return pd.DataFrame(columns=["decile", "count", "mean_true", "mean_pred", "low", "high"])
    data = df.copy()
    data["y_pred"] = pd.to_numeric(data["y_pred"], errors="coerce")
    data["y_true_tp_sl"] = pd.to_numeric(data["y_true_tp_sl"], errors="coerce")
    data = data[(data["has_full_horizon"] == True) & data["y_true_tp_sl"].notna()]  # noqa: E712
    if data.empty:
        return pd.DataFrame(columns=["decile", "count", "mean_true", "mean_pred", "low", "high"])
    k = len(edges) - 1
    labels = list(range(1, k + 1))
    cats = pd.cut(data["y_pred"], bins=edges, include_lowest=True, right=True, labels=labels)
    g = pd.DataFrame({"decile": cats, "y_true": data["y_true_tp_sl"], "y_pred": data["y_pred"]})
    grouped = g.groupby("decile", dropna=False).agg(
        count=("y_true", "count"),
        mean_true=("y_true", "mean"),
        mean_pred=("y_pred", "mean"),
        low=("y_pred", "min"),
        high=("y_pred", "max"),
    )
    full_index = pd.Index(labels, name="decile")
    out = grouped.reindex(full_index).reset_index()
    return out


def _rank_decile_stats(df: pd.DataFrame, *, n_bins: int = 10) -> pd.DataFrame:
    """Rank-based, size-balanced deciles with deterministic tie-breaking.

    Expects columns: 'y_pred', 'y_true_tp_sl', 'has_full_horizon'.
    Only counts rows with full horizon and non-null y_true.
    Returns columns: decile(1..n), count, mean_true, mean_pred, low, high.
    """
    if df.empty or n_bins < 1:
        return pd.DataFrame(columns=["decile", "count", "mean_true", "mean_pred", "low", "high"])
    data = df.copy()
    data["y_pred"] = pd.to_numeric(data["y_pred"], errors="coerce")
    data["y_true_tp_sl"] = pd.to_numeric(data["y_true_tp_sl"], errors="coerce")
    data = data[(data["y_pred"].notna())]
    if data.empty:
        return pd.DataFrame(columns=["decile", "count", "mean_true", "mean_pred", "low", "high"])
    # Deterministic rank percent (0..1]
    order = data["y_pred"].rank(method="first", pct=True)
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    labels = list(range(1, int(n_bins) + 1))
    dec = pd.cut(order, bins=bins, include_lowest=True, labels=labels)
    data = data.assign(__decile=dec)
    # Compute stats only on rows with full horizon and available label
    mask_eval = (data.get("has_full_horizon", True) == True) & data["y_true_tp_sl"].notna()  # noqa: E712
    eval_df = data[mask_eval].copy()
    grouped = eval_df.groupby("__decile", dropna=False).agg(
        count=("y_true_tp_sl", "count"),
        mean_true=("y_true_tp_sl", "mean"),
        mean_pred=("y_pred", "mean"),
    )
    # Low/high boundaries from all rows per decile
    bounds = data.groupby("__decile", dropna=False).agg(low=("y_pred", "min"), high=("y_pred", "max"))
    out = grouped.join(bounds, how="outer")
    out.index = out.index.astype("Int64")
    full_index = pd.Index(labels, name="decile")
    out = out.reindex(full_index).reset_index()
    return out


def _build_decile_figure(stats: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not stats.empty:
        x = stats["decile"].astype(str)
        # Bars: observed event rate (mean_true)
        fig.add_bar(name="Observed (TP before SL)", x=x, y=stats["mean_true"], marker_color="#2ca02c", opacity=0.85)
        # Line: average predicted score
        fig.add_trace(
            go.Scatter(
                name="Predicted",
                x=x,
                y=stats["mean_pred"],
                mode="lines+markers",
                line=dict(color="#ff7f0e"),
            )
        )
    fig.update_layout(
        height=360,
        margin=dict(l=0, r=0, t=30, b=30),
        template="plotly_white",
        title="Decile Calibration (TP +4% before SL −2% in 24h)",
        xaxis_title="y_pred decile (low → high)",
        yaxis_title="rate / probability",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(range=[-0.05, 1.05]),
    )
    return fig


# ===== App =====


def main() -> None:
    st.set_page_config(page_title="Classifier Dashboard (24h)", layout="wide")
    st.title("Classifier Dashboard: 24h Event Performance")

    # Sidebar: data sources and filters
    with st.sidebar:
        st.subheader("Data Sources")
        ohlcv_path = st.text_input("OHLCV DuckDB path", value=DEFAULT_OHLCV_DB)
        classifier_path = st.text_input("Classifier DuckDB path", value=DEFAULT_CLASSIFIER_DB)
        refresh = st.button("Refresh data", type="primary")

        st.subheader("Date Range")
        today = datetime.utcnow().date()
        start_default = today - timedelta(days=180)
        start_date, end_date = st.date_input(
            "Window",
            value=(start_default, today),
            max_value=today,
        )

        st.subheader("Window Settings")
        auc_window_days = st.number_input("Rolling AUC window (days)", min_value=7, max_value=365, value=90)
        trade_window_days = st.number_input("Rolling trade window (days)", min_value=7, max_value=365, value=90)

    if refresh:
        cached_load_ohlcv.clear()
        cached_load_predictions.clear()
        cached_load_model_metrics.clear()
        cached_load_feature_importance.clear()
        st.toast("Cleared cache and reloaded.")

    # Load
    try:
        ohlcv_df = cached_load_ohlcv(ohlcv_path)
    except DataLoadError as exc:
        st.error(f"Failed to load OHLCV: {exc}")
        return

    try:
        preds_all = cached_load_predictions(classifier_path)
    except DataLoadError as exc:
        st.error(f"Failed to load predictions: {exc}")
        return

    # Basic checks
    if preds_all.empty:
        st.warning("No predictions found.")
        return

    # Derive run_id from model_path (folder name starting with 'run_')
    def _extract_run_id(path_str: str) -> str:
        try:
            p = Path(str(path_str))
            # prefer parent folders starting with run_
            for part in reversed(p.parts):
                if part.startswith("run_"):
                    return part
            # fallback to parent folder name
            return p.parent.name if p.parent.name else p.name
        except Exception:
            return str(path_str)

    if not preds_all.empty and "model_path" in preds_all.columns:
        preds_all = preds_all.copy()
        preds_all["run_id"] = preds_all["model_path"].astype(str).apply(_extract_run_id)
    else:
        preds_all["run_id"] = ""

    # Model/feature selection by run_id only
    run_options = sorted([r for r in preds_all["run_id"].dropna().unique().tolist() if isinstance(r, str)])
    # Default run: most recent created_at (or timestamp) within all rows
    default_run = ""
    if run_options:
        df_nonnull = preds_all.dropna(subset=["run_id"]).copy()
        if "created_at" in df_nonnull.columns and df_nonnull["created_at"].notna().any():
            grp = df_nonnull.groupby("run_id")["created_at"].max().sort_values()
        else:
            grp = df_nonnull.groupby("run_id")["timestamp"].max().sort_values()
        if not grp.empty:
            default_run = grp.index[-1]
    feature_options = sorted([f for f in preds_all["feature_key"].dropna().unique().tolist() if isinstance(f, str)])

    with st.sidebar:
        st.subheader("Filters")
        selected_run = st.selectbox(
            "Model (run)", options=run_options, index=(run_options.index(default_run) if default_run in run_options else 0)
        )
        selected_feature_key = st.selectbox(
            "Feature key (optional)", options=["(any)"] + feature_options, index=0
        )

    # Date bounds
    start_ts, end_ts = _timestamp_bounds(start_date, end_date)

    # Filter to model + optional feature_key + date range
    df_model = preds_all[preds_all["run_id"] == selected_run].copy()
    if selected_feature_key != "(any)":
        df_model = df_model[df_model["feature_key"] == selected_feature_key]
    df_model = df_model.sort_values("timestamp")
    df_model = _filter_by_range(df_model, start_ts, end_ts)

    if df_model.empty:
        st.warning("No predictions for the selected filters and date range.")
        return

    # Compute y_true over all rows (24h horizon)
    ytrue_frame = _compute_ytrue_all(df_model, ohlcv_df, horizon_h=24)

    # Default threshold band = decile 8's low to decile 9's high (i.e., ~p70..p90)
    # Prefer decile edges computed via qcut; fallback to p70..p90 quantiles.
    edges_defaults = _compute_decile_edges(ytrue_frame["y_pred"], n_bins=10)
    p70 = _quantile_default(ytrue_frame["y_pred"], 0.70)
    p90 = _quantile_default(ytrue_frame["y_pred"], 0.90)
    d8_low = None
    d9_high = None
    if edges_defaults is not None and len(edges_defaults) >= 10:  # expect 11 edges (0..10) for 10 deciles
        try:
            d8_low = float(edges_defaults[7])  # decile 8: edges[7]..edges[8]
            d9_high = float(edges_defaults[9])  # decile 9 high edge
        except Exception:
            d8_low, d9_high = None, None
    if d8_low is None or not np.isfinite(d8_low):
        d8_low = float(p70) if np.isfinite(p70) else 0.5
    if d9_high is None or not np.isfinite(d9_high):
        d9_high = float(p90) if np.isfinite(p90) else d8_low
    if d9_high <= d8_low:
        # fallback to span up to max observed value
        d9_high = float(pd.to_numeric(ytrue_frame["y_pred"], errors="coerce").max())

    with st.sidebar:
        st.subheader("Signal threshold band")
        min_pred = float(pd.to_numeric(ytrue_frame["y_pred"], errors="coerce").min())
        max_pred = float(pd.to_numeric(ytrue_frame["y_pred"], errors="coerce").max())
        lower_bound = st.number_input(
            "Lower bound",
            min_value=float(min_pred),
            max_value=float(max_pred),
            value=float(d8_low),
            step=0.001,
            format="%0.6f",
        )
        upper_bound = st.number_input(
            "Upper bound",
            min_value=float(min_pred),
            max_value=float(max_pred),
            value=float(d9_high),
            step=0.001,
            format="%0.6f",
        )
        if upper_bound < lower_bound:
            st.warning("Upper bound is less than lower bound; swapping values for processing.")
            lower_bound, upper_bound = float(min(lower_bound, upper_bound)), float(max(lower_bound, upper_bound))
        st.caption(f"Defaults based on deciles: d8 low≈p70 to d9 high≈p90 → [{d8_low:.4f}, {d9_high:.4f}]")

    # Build trades (triggered signals within band [lower_bound, upper_bound])
    # First compute metrics for all rows, then filter by band
    trades_all = compute_signal_metrics(
        df_model,
        ohlcv_df,
        horizon_hours=24,
        direction_threshold=float("-inf"),  # include all
        include_long=True,
        include_short=False,
    )
    trades = trades_all[
        (pd.to_numeric(trades_all["y_pred"], errors="coerce") >= float(lower_bound))
        & (pd.to_numeric(trades_all["y_pred"], errors="coerce") <= float(upper_bound))
    ].sort_values("timestamp")

    # Rolling AUC on all predictions using y_true = TP(4%) before SL(2%) within 24h
    # Compute barrier-based labels exclusively for AUC
    auc_labels = _compute_tp_before_sl_labels(
        ytrue_frame[["timestamp", "y_pred"]], ohlcv_df, horizon_h=24, up_frac=0.04, dn_frac=0.02
    )
    ytrue_for_auc = ytrue_frame.merge(auc_labels, on="timestamp", how="left")
    # Ensure only one 'y_true' column exists: remove any prior y_true then rename tp/sl label
    if "y_true" in ytrue_for_auc.columns:
        ytrue_for_auc = ytrue_for_auc.drop(columns=["y_true"])
    ytrue_for_auc = ytrue_for_auc.rename(columns={"y_true_tp_sl": "y_true"})
    # Drop rows without full horizon for y_true
    if "has_full_horizon_y" in ytrue_for_auc.columns:
        ytrue_for_auc.loc[~ytrue_for_auc["has_full_horizon_y"].fillna(False), "y_true"] = np.nan
    # Build rolling AUC
    auc_df_all = _compute_rolling_auc(ytrue_for_auc[["timestamp", "y_pred", "y_true"]], window_days=int(auc_window_days), min_obs=100)
    auc_display = _filter_by_range(auc_df_all, start_ts, end_ts)

    # Rolling trade stats on triggered signals
    trade_stats = _compute_rolling_trade_stats(trades, window_days=int(trade_window_days))
    trade_stats_display = _filter_by_range(trade_stats, start_ts, end_ts)
    # Rolling stats on all bars
    all_stats = _compute_rolling_all_bars_stats(ytrue_frame, window_days=int(trade_window_days))
    all_stats_display = _filter_by_range(all_stats, start_ts, end_ts)
    # Rolling strategy stats on trades (TP/SL else natural close)
    strat_stats = _compute_rolling_strategy_stats(
        trades, ohlcv_df, window_days=int(trade_window_days), horizon_h=24, up_frac=0.04, dn_frac=0.02
    )
    strat_stats_display = _filter_by_range(strat_stats, start_ts, end_ts)

    # Top: market + signals
    ohlcv_display = _filter_by_range(ohlcv_df, start_ts, end_ts)
    market_fig = _build_market_figure(ohlcv_display, trades)
    st.plotly_chart(market_fig, use_container_width=True)

    # Rolling AUC
    if not auc_display.empty and auc_display["auc"].notna().any():
        st.plotly_chart(_build_line_figure(auc_display, "auc", title="Rolling AUC (all predictions)", y_range=(-0.05, 1.05)), use_container_width=True)
    else:
        st.caption("Rolling AUC requires enough samples with both classes in-window.")

    # Rolling Win-Rate (overlay: trades vs all bars)
    if (not trade_stats_display.empty and trade_stats_display["win_rate"].notna().any()) or (
        not all_stats_display.empty and all_stats_display["win_rate"].notna().any()
    ):
        fig_wr = _build_overlay_line_figure(
            [
                (trade_stats_display, "win_rate", "Trades"),
                (all_stats_display, "win_rate", "All bars"),
            ],
            title="Rolling Win-Rate (trades vs all bars)",
            y_range=(-0.05, 1.05),
        )
        st.plotly_chart(fig_wr, use_container_width=True)
    else:
        st.caption("No data to compute rolling win-rate for trades or all bars in the window.")

    # Rolling Returns (avg + total) — overlay trades (nat close), trades (strategy), and all bars
    cols = st.columns(2)
    with cols[0]:
        if (
            (not trade_stats_display.empty and trade_stats_display["avg_return"].notna().any())
            or (not strat_stats_display.empty and strat_stats_display["avg_return_strategy"].notna().any())
            or (not all_stats_display.empty and all_stats_display["avg_return"].notna().any())
        ):
            fig_avg = _build_overlay_line_figure(
                [
                    (trade_stats_display, "avg_return", "Trades (nat close)"),
                    (strat_stats_display, "avg_return_strategy", "Trades (strategy)"),
                    (all_stats_display, "avg_return", "All bars"),
                ],
                title="Rolling Average 24h Return",
            )
            st.plotly_chart(fig_avg, use_container_width=True)
        else:
            st.caption("No rolling average return for trades or all bars in the window.")
    with cols[1]:
        if (
            (not trade_stats_display.empty and trade_stats_display["total_return"].notna().any())
            or (not strat_stats_display.empty and strat_stats_display["total_return_strategy"].notna().any())
            or (not all_stats_display.empty and all_stats_display["total_return"].notna().any())
        ):
            fig_total = _build_overlay_line_figure(
                [
                    (trade_stats_display, "total_return", "Trades (nat close)"),
                    (strat_stats_display, "total_return_strategy", "Trades (strategy)"),
                    (all_stats_display, "total_return", "All bars"),
                ],
                title="Rolling Total 24h Return",
            )
            st.plotly_chart(fig_total, use_container_width=True)
        else:
            st.caption("No rolling total return for trades or all bars in the window.")

    # ===== Trade Stats (counts + win/lose distribution) =====
    st.subheader("Trade Stats")
    trades_in_window = _filter_by_range(trades, start_ts, end_ts)
    realized_trades = trades_in_window[trades_in_window["has_full_horizon"] == True]  # noqa: E712
    n_total_triggered = int(len(trades_in_window))
    n_realized = int(len(realized_trades))
    n_pending = int(n_total_triggered - n_realized)
    wins = int((realized_trades.get("is_correct") == True).sum())  # noqa: E712
    losses = int((realized_trades.get("is_correct") == False).sum())  # noqa: E712
    win_rate = (wins / (wins + losses)) if (wins + losses) > 0 else np.nan
    realized_returns = pd.to_numeric(realized_trades.get("forward_return"), errors="coerce").dropna()
    avg_return_val = float(realized_returns.mean()) if not realized_returns.empty else np.nan
    total_return_val = float(realized_returns.sum()) if not realized_returns.empty else np.nan

    col_stats_left, col_stats_right = st.columns([1, 2])
    with col_stats_left:
        st.metric("Trades (realized, 24h)", f"{n_realized}")
        st.caption(f"Triggered: {n_total_triggered} | Pending: {n_pending}")
        if np.isfinite(win_rate):
            st.metric("Win rate", f"{win_rate*100:.2f}%")
        else:
            st.metric("Win rate", "n/a")
        if np.isfinite(avg_return_val):
            st.metric("Avg 24h return (realized)", f"{avg_return_val*100:.2f}%")
        else:
            st.metric("Avg 24h return (realized)", "n/a")
        if np.isfinite(total_return_val):
            st.metric("Total 24h return (sum)", f"{total_return_val*100:.2f}%")
        else:
            st.metric("Total 24h return (sum)", "n/a")
        # Strategy returns (TP +4% / SL -2% else natural close)
        strat_ret_all = _compute_strategy_return_tp_sl(trades_in_window["timestamp"], ohlcv_df, horizon_h=24, up_frac=0.04, dn_frac=0.02)
        strat_ret_realized = strat_ret_all.loc[realized_trades.index] if not realized_trades.empty else pd.Series([], dtype=float)
        if not strat_ret_realized.dropna().empty:
            st.metric("Strategy avg return (realized)", f"{strat_ret_realized.dropna().mean()*100:.2f}%")
            st.metric("Strategy cum return (sum)", f"{strat_ret_realized.dropna().sum()*100:.2f}%")
        else:
            st.metric("Strategy avg return (realized)", "n/a")
            st.metric("Strategy cum return (sum)", "n/a")
        # All tradable bars (all prediction rows in window)
        all_bars_window = _filter_by_range(ytrue_frame, start_ts, end_ts)
        n_tradable = int(len(all_bars_window))
        st.metric("Tradable bars (all)", f"{n_tradable}")
        realized_all = all_bars_window[all_bars_window["has_full_horizon"] == True]  # noqa: E712
        avg_ret_all = pd.to_numeric(realized_all.get("forward_return"), errors="coerce").dropna()
        if not avg_ret_all.empty:
            st.metric("Avg 24h return (all bars)", f"{avg_ret_all.mean()*100:.2f}%")
        else:
            st.metric("Avg 24h return (all bars)", "n/a")
        # TP/SL hits across all bars
        if not realized_all.empty and {"max_future_high", "min_future_low", "close"}.issubset(realized_all.columns):
            entry_c = pd.to_numeric(realized_all["close"], errors="coerce")
            max_up_all = pd.to_numeric(realized_all["max_future_high"], errors="coerce")
            min_dn_all = pd.to_numeric(realized_all["min_future_low"], errors="coerce")
            tp_hits_all = int(((max_up_all / entry_c) - 1.0 >= 0.04).sum())
            sl_hits_all = int(((min_dn_all / entry_c) - 1.0 <= -0.02).sum())
            st.caption(f"All bars — TP≥+4%: {tp_hits_all} | SL≤-2%: {sl_hits_all}")
        # TP/SL hit counts (intra-horizon extremes)
        if not realized_trades.empty and {"max_future_high", "min_future_low", "close"}.issubset(realized_trades.columns):
            entry_close = pd.to_numeric(realized_trades["close"], errors="coerce")
            max_up = pd.to_numeric(realized_trades["max_future_high"], errors="coerce")
            min_dn = pd.to_numeric(realized_trades["min_future_low"], errors="coerce")
            tp_hits = int(((max_up / entry_close) - 1.0 >= 0.04).sum())
            sl_hits = int(((min_dn / entry_close) - 1.0 <= -0.02).sum())
            st.caption(f"TP≥+4% hits: {tp_hits} | SL≤-2% hits: {sl_hits}")
    with col_stats_right:
        # Return distribution by outcome (Win/Lose), colored accordingly
        ret_win = pd.to_numeric(realized_trades.loc[realized_trades["is_correct"] == True, "forward_return"], errors="coerce")  # noqa: E712
        ret_lose = pd.to_numeric(realized_trades.loc[realized_trades["is_correct"] == False, "forward_return"], errors="coerce")  # noqa: E712
        has_any = (ret_win.notna().any() or ret_lose.notna().any())
        if has_any:
            hist_fig = go.Figure()
            if ret_win.notna().any():
                hist_fig.add_trace(
                    go.Histogram(
                        x=ret_win * 100.0,
                        name="Win",
                        marker_color="#2ca02c",
                        opacity=0.55,
                        nbinsx=50,
                    )
                )
            if ret_lose.notna().any():
                hist_fig.add_trace(
                    go.Histogram(
                        x=ret_lose * 100.0,
                        name="Lose",
                        marker_color="#d62728",
                        opacity=0.55,
                        nbinsx=50,
                    )
                )
            hist_fig.update_layout(
                barmode="overlay",
                height=260,
                margin=dict(l=0, r=0, t=30, b=30),
                template="plotly_white",
                title="Return Distribution by Outcome (realized trades)",
                xaxis_title="24h return (%)",
                yaxis_title="count",
            )
            # Zero line
            hist_fig.add_shape(type="line", x0=0, x1=0, y0=0, y1=1, xref="x", yref="paper", line=dict(color="#9ca3af", dash="dash"))
            st.plotly_chart(hist_fig, use_container_width=True)
        else:
            st.caption("No realized trades with returns available to plot distribution.")

    st.divider()

    # ===== Cumulative Return (sum, $1 per trade) =====
    st.subheader("Cumulative Return (sum, $1 per trade)")
    # All bars: natural close 24h (full-horizon rows only)
    all_bars_window = _filter_by_range(ytrue_frame, start_ts, end_ts)
    realized_all = all_bars_window[all_bars_window["has_full_horizon"] == True]  # noqa: E712
    df_all_cum = _build_cum_pnl_df(realized_all["timestamp"], realized_all["forward_return"]) if not realized_all.empty else pd.DataFrame(columns=["timestamp","cum_return"])
    # Trades nat close: realized trades only
    df_trades_cum = _build_cum_pnl_df(realized_trades["timestamp"], realized_trades["forward_return"]) if not realized_trades.empty else pd.DataFrame(columns=["timestamp","cum_return"])
    # Trades strategy: include all trades with computed strategy return (TP/SL or natural close) in-window
    strat_all_series = _compute_strategy_return_tp_sl(trades_in_window["timestamp"], ohlcv_df, horizon_h=24, up_frac=0.04, dn_frac=0.02)
    df_strat_cum = _build_cum_pnl_df(trades_in_window["timestamp"], strat_all_series)

    traces_cum: List[Tuple[pd.DataFrame, str, str]] = []
    if not df_trades_cum.empty:
        traces_cum.append((df_trades_cum.rename(columns={"cum_return": "Trades (nat close)"}), "Trades (nat close)", "Trades (nat close)"))
    if not df_strat_cum.empty:
        traces_cum.append((df_strat_cum.rename(columns={"cum_return": "Trades (strategy)"}), "Trades (strategy)", "Trades (strategy)"))
    if not df_all_cum.empty:
        traces_cum.append((df_all_cum.rename(columns={"cum_return": "All bars"}), "All bars", "All bars"))

    # Build figure
    fig_cum = go.Figure()
    colors = {"Trades (nat close)": "#1f77b4", "Trades (strategy)": "#ff7f0e", "All bars": "#2ca02c"}
    for df_line, y_col_name, label in traces_cum:
        if df_line.empty:
            continue
        fig_cum.add_trace(
            go.Scatter(
                x=df_line["timestamp"],
                y=df_line[label] if label in df_line.columns else df_line.get("cum_return", pd.Series(dtype=float)),
                mode="lines",
                name=label,
                line=dict(color=colors.get(label, None)),
            )
        )
    fig_cum.update_layout(
        height=360,
        margin=dict(l=0, r=0, t=30, b=30),
        template="plotly_white",
        title="Cumulative Return ($1 per trade)",
        xaxis_title="Timestamp",
        yaxis_title="Cumulative return (sum)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # ===== Selected Model Information =====
    st.subheader("Selected Model Information")
    st.caption(selected_run)

    # Resolve a representative model_path for artifacts under the selected run
    model_path_for_artifacts = ""
    if not df_model.empty and "model_path" in df_model.columns:
        non_null_paths = df_model.dropna(subset=["model_path"]).copy()
        if not non_null_paths.empty:
            if "timestamp" in non_null_paths.columns:
                non_null_paths = non_null_paths.sort_values("timestamp")
            model_path_for_artifacts = str(non_null_paths["model_path"].iloc[-1])
    metrics_df = cached_load_model_metrics(model_path_for_artifacts)
    fi_df = cached_load_feature_importance(model_path_for_artifacts)

    # Quick meta from predictions
    meta = preds_all[preds_all["run_id"] == selected_run]
    if not meta.empty:
        t_min = pd.to_datetime(meta["timestamp"]).min()
        t_max = pd.to_datetime(meta["timestamp"]).max()
        n_rows = len(meta)
        st.caption(f"Predictions: {n_rows:,} rows | Range: {t_min} → {t_max} UTC")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("Metrics")
        if not metrics_df.empty:
            # Show top numeric metrics
            display = metrics_df.copy()
            # Sort: try to prioritize auc-like first if present
            priority = display["metric"].str.contains("auc", case=False, na=False)
            display = pd.concat([display[priority], display[~priority]], axis=0)
            st.dataframe(display, use_container_width=True, height=260)
            src = display.get("__source_path__")
            if src is not None and not src.dropna().empty:
                st.caption(f"Metrics source: {src.dropna().iloc[0]}")
        else:
            st.caption("No metrics.json/csv found next to the model.")

    with col_b:
        st.markdown("Feature Importance")
        if not fi_df.empty:
            st.plotly_chart(_build_feature_importance_fig(fi_df, top_n=20), use_container_width=True)
            src = fi_df.get("__source_path__")
            if src is not None and not src.dropna().empty:
                st.caption(f"FI source: {src.dropna().iloc[0]}")
        else:
            st.caption("No feature_importance.csv found next to the model.")

    # y_pred distribution for current window (selected model/feature_key)
    preds_in_window = df_model.copy()
    s_pred = pd.to_numeric(preds_in_window.get("y_pred"), errors="coerce").dropna()
    if not s_pred.empty:
        fig_pred_dist = go.Figure()
        fig_pred_dist.add_trace(
            go.Histogram(
                x=s_pred,
                nbinsx=50,
                name="y_pred",
                marker_color="#1f77b4",
                opacity=0.8,
            )
        )
        # Vertical band lines at current bounds
        for x_val, color in [(float(lower_bound), "#ef553b"), (float(upper_bound), "#ef553b")]:
            fig_pred_dist.add_shape(
                type="line",
                x0=x_val,
                x1=x_val,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(color=color, width=2, dash="dash"),
            )
        fig_pred_dist.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=30, b=30),
            template="plotly_white",
            title="y_pred Distribution (current window)",
            xaxis_title="y_pred",
            yaxis_title="count",
            showlegend=False,
        )
        st.plotly_chart(fig_pred_dist, use_container_width=True)
        # Quick summary stats
        q80 = float(s_pred.quantile(0.80))
        q90 = float(s_pred.quantile(0.90))
        q95 = float(s_pred.quantile(0.95))
        share_band = float(((s_pred >= float(lower_bound)) & (s_pred <= float(upper_bound))).mean())
        st.caption(
            f"p80={q80:.4f} | p90={q90:.4f} | p95={q95:.4f} | share in band: {share_band*100:.2f}%"
        )

    # Decile calibration over the full time range for the selected run (TP +4% before SL −2% in 24h)
    df_run_full = preds_all[preds_all["run_id"] == selected_run][["timestamp", "y_pred"]].copy()
    decile_labels_df = _compute_tp_before_sl_labels(
        df_run_full, ohlcv_df, horizon_h=24, up_frac=0.04, dn_frac=0.02
    )
    decile_join = df_run_full.merge(decile_labels_df, on="timestamp", how="left")
    # Rank-based, size-balanced deciles across the full run
    dec_stats = _rank_decile_stats(decile_join, n_bins=10)
    if not dec_stats.empty:
        st.plotly_chart(_build_decile_figure(dec_stats), use_container_width=True)
        # Show decile boundaries (low/high) under the plot
        bounds_view = dec_stats[["decile", "low", "high", "count"]].copy()
        # Pretty formatting
        bounds_view["decile"] = bounds_view["decile"].astype(int)
        bounds_view["low"] = bounds_view["low"].astype(float)
        bounds_view["high"] = bounds_view["high"].astype(float)
        st.dataframe(bounds_view, use_container_width=True, height=280)

    st.divider()

    # ===== All Models' Predictions Over Time =====
    st.subheader("All Models — Predictions Over Time")
    all_runs = [r for r in preds_all["run_id"].dropna().unique().tolist() if isinstance(r, str)]
    default_runs = all_runs  # show all by default
    selected_runs = st.multiselect("Runs to display", options=all_runs, default=default_runs)
    preds_display = _filter_by_range(preds_all, start_ts, end_ts)
    if selected_runs:
        fig_all = _build_all_models_predictions_fig(
            preds_display[preds_display["run_id"].isin(selected_runs)], models=selected_runs
        )
        st.plotly_chart(fig_all, use_container_width=True)
    else:
        st.caption("Select one or more runs to display time series of y_pred.")

    # Footer context
    st.caption(
        "Signals triggered where y_pred is within the selected band (long-only). "
        "AUC uses TP(+4%) before SL(−2%) within 24h as ground truth. "
        "Rolling windows are time-based with minimum sample requirements for AUC."
    )


if __name__ == "__main__":
    main()
