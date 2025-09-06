#!/usr/bin/env python3
"""
Simulate the MFE/MAE hedging strategy described in strategies/mfe_mae_hedging.md.

Approach per doc:
- For each entry bar i, open equal-size long and short.
- Over next H bars (default 24), detect first cross of MFE up-target or MAE down-target.
- Close the losing leg at the hit price; keep the winning leg and distribute exits evenly over remaining bars.
- Net PnL is the sum of scale-out profits relative to the base cost price until reversal or horizon end.

Inputs:
- OHLCV CSV with timestamp and open/high/low/close.
- MFE and MAE signals CSVs containing timestamp and exp_ret_avg.

Outputs:
- CSV with per-entry results: timestamp, entry_price, mfe_price, mae_price, first_hit, hit_ts, bars_to_end,
  pnl_estimate, pnl_long, pnl_short.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd


ReturnType = Literal["percent", "log", "abs"]
TiePolicy = Literal["zero", "mfe", "mae"]


def _load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Prefer 'timestamp' if present, else 'time'
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    elif "time" in df.columns:
        ts = pd.to_datetime(df["time"], errors="coerce", utc=True)
    else:
        raise ValueError("OHLCV CSV must have 'timestamp' or 'time' column")
    df["timestamp"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    # Normalize column names
    colmap = {}
    for c in df.columns:
        lc = str(c).lower()
        if lc in {"open", "high", "low", "close"}:
            colmap[c] = lc
    df = df.rename(columns=colmap)
    required = {"open", "high", "low", "close", "timestamp"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"OHLCV CSV missing columns: {sorted(missing)}")
    out = df[["timestamp", "open", "high", "low", "close"]].dropna(subset=["timestamp"]).reset_index(drop=True)
    return out


def _load_signals(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["timestamp"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        raise ValueError(f"Signals CSV for {label} must contain 'timestamp'")
    if "exp_ret_avg" not in df.columns:
        raise ValueError(f"Signals CSV for {label} must contain 'exp_ret_avg'")
    out = df[["timestamp", "exp_ret_avg"]].rename(columns={"exp_ret_avg": f"exp_ret_avg_{label}"})
    return out


def _convert_return_to_price(entry_price: float, ret_value: float, ret_type: ReturnType) -> float:
    if ret_type == "percent":
        return float(entry_price * (1.0 + ret_value))
    if ret_type == "log":
        return float(entry_price * np.exp(ret_value))
    if ret_type == "abs":
        return float(entry_price + ret_value)
    raise ValueError(f"Unsupported ret_type: {ret_type}")


@dataclass
class SimulationConfig:
    horizon: int
    ret_type: ReturnType
    tie_policy: TiePolicy
    dollar_size: float


def _find_first_hits(
    highs: np.ndarray,
    lows: np.ndarray,
    start_idx: int,
    end_idx: int,
    mfe_price: float,
    mae_price: float,
) -> Tuple[Optional[int], Optional[int]]:
    """Return first indices where MFE up-target and MAE down-target are hit.

    Returns (idx_mfe, idx_mae) relative to the provided arrays, or (None, None).
    """
    idx_mfe: Optional[int] = None
    idx_mae: Optional[int] = None
    for t in range(start_idx + 1, end_idx + 1):
        if idx_mfe is None and highs[t] >= mfe_price:
            idx_mfe = t
        if idx_mae is None and lows[t] <= mae_price:
            idx_mae = t
        if idx_mfe is not None and idx_mae is not None:
            break
    return idx_mfe, idx_mae


def _compute_scaleout_pnl_long(
    closes: np.ndarray,
    base_cost: float,
    first_hit_idx: int,
    end_idx: int,
) -> float:
    """Scale-out PnL for a long with equal fractions over remaining bars, with early reversal.

    Early reversal: if the low crosses below base_cost on a subsequent bar, the position is closed immediately.
    This function expects the caller to stop on reversal and pass the end_idx accordingly; here we compute PnL
    from closes only over the active window.
    """
    R = end_idx - first_hit_idx
    if R <= 0:
        return 0.0
    frac = 1.0 / float(R)
    pnl = 0.0
    for t in range(first_hit_idx + 1, end_idx + 1):
        pnl += frac * (float(closes[t]) - base_cost)
    return float(pnl)


def _compute_scaleout_pnl_short(
    closes: np.ndarray,
    base_cost: float,
    first_hit_idx: int,
    end_idx: int,
) -> float:
    R = end_idx - first_hit_idx
    if R <= 0:
        return 0.0
    frac = 1.0 / float(R)
    pnl = 0.0
    for t in range(first_hit_idx + 1, end_idx + 1):
        pnl += frac * (base_cost - float(closes[t]))
    return float(pnl)


def _apply_early_reversal_cutoff(
    highs: np.ndarray,
    lows: np.ndarray,
    base_cost: float,
    first_hit: str,
    first_hit_idx: int,
    end_idx: int,
) -> int:
    """Return the adjusted end index if early reversal occurs; else return end_idx.

    - After MFE (long), reversal occurs if low[t] <= base_cost.
    - After MAE (short), reversal occurs if high[t] >= base_cost.
    The close for the reversal bar is excluded per doc.
    """
    if first_hit == "MFE":
        for t in range(first_hit_idx + 1, end_idx + 1):
            if lows[t] <= base_cost:
                return t - 1
        return end_idx
    if first_hit == "MAE":
        for t in range(first_hit_idx + 1, end_idx + 1):
            if highs[t] >= base_cost:
                return t - 1
        return end_idx
    return end_idx


def simulate(df: pd.DataFrame, cfg: SimulationConfig) -> pd.DataFrame:
    timestamps = df["timestamp"].to_numpy()
    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    exp_mfe = df["exp_ret_avg_mfe"].to_numpy(dtype=float)
    exp_mae = df["exp_ret_avg_mae"].to_numpy(dtype=float)

    n = len(df)
    rows = []
    H = int(cfg.horizon)
    for i in range(0, n - H):
        entry_ts = timestamps[i]
        entry_price = float(closes[i])
        quantity = float(cfg.dollar_size) / entry_price if entry_price > 0 else 0.0
        # Clip returns by sign convention
        r_mfe = max(float(exp_mfe[i]), 0.0)
        r_mae = min(float(exp_mae[i]), 0.0)

        mfe_price = _convert_return_to_price(entry_price, r_mfe, cfg.ret_type)
        mae_price = _convert_return_to_price(entry_price, r_mae, cfg.ret_type)

        end_idx = i + H
        idx_mfe, idx_mae = _find_first_hits(highs, lows, i, end_idx, mfe_price, mae_price)

        first_hit: str = "NONE"
        hit_idx: Optional[int] = None
        if idx_mfe is not None and idx_mae is not None:
            if idx_mfe < idx_mae:
                first_hit, hit_idx = "MFE", idx_mfe
            elif idx_mae < idx_mfe:
                first_hit, hit_idx = "MAE", idx_mae
            else:
                # Same bar hit
                if cfg.tie_policy == "mfe":
                    first_hit, hit_idx = "MFE", idx_mfe
                elif cfg.tie_policy == "mae":
                    first_hit, hit_idx = "MAE", idx_mae
                else:
                    # zero: PnL = 0, no base cost
                    first_hit, hit_idx = "NONE", None
        elif idx_mfe is not None:
            first_hit, hit_idx = "MFE", idx_mfe
        elif idx_mae is not None:
            first_hit, hit_idx = "MAE", idx_mae

        pnl_long = 0.0
        pnl_short = 0.0
        pnl_estimate = 0.0
        hit_ts: Optional[pd.Timestamp] = None
        bars_to_end = H
        early_reversal = False
        missed_upper = 0.0
        missed_lower = 0.0

        if first_hit == "MFE" and hit_idx is not None:
            base_cost = mfe_price
            hit_ts = timestamps[hit_idx]
            bars_to_end = end_idx - hit_idx
            adj_end_idx = _apply_early_reversal_cutoff(highs, lows, base_cost, first_hit, hit_idx, end_idx)
            bars_to_end = adj_end_idx - hit_idx if adj_end_idx >= hit_idx else 0
            early_reversal = adj_end_idx < end_idx
            pnl_long = _compute_scaleout_pnl_long(closes, base_cost, hit_idx, adj_end_idx)
            pnl_estimate = pnl_long
            if early_reversal:
                # Additional missed PnL bounds from adj_end_idx+1 .. end_idx with fraction 1/R_total
                R_total = end_idx - hit_idx
                if R_total > 0 and adj_end_idx < end_idx:
                    frac = 1.0 / float(R_total)
                    for t in range(adj_end_idx + 1, end_idx + 1):
                        missed_upper += frac * (float(highs[t]) - base_cost)  # optimistic for long
                        missed_lower += frac * (float(lows[t]) - base_cost)   # pessimistic for long
        elif first_hit == "MAE" and hit_idx is not None:
            base_cost = mae_price
            hit_ts = timestamps[hit_idx]
            bars_to_end = end_idx - hit_idx
            adj_end_idx = _apply_early_reversal_cutoff(highs, lows, base_cost, first_hit, hit_idx, end_idx)
            bars_to_end = adj_end_idx - hit_idx if adj_end_idx >= hit_idx else 0
            early_reversal = adj_end_idx < end_idx
            pnl_short = _compute_scaleout_pnl_short(closes, base_cost, hit_idx, adj_end_idx)
            pnl_estimate = pnl_short
            if early_reversal:
                R_total = end_idx - hit_idx
                if R_total > 0 and adj_end_idx < end_idx:
                    frac = 1.0 / float(R_total)
                    for t in range(adj_end_idx + 1, end_idx + 1):
                        missed_upper += frac * (base_cost - float(lows[t]))   # optimistic for short
                        missed_lower += frac * (base_cost - float(highs[t]))  # pessimistic for short
        else:
            # NONE: PnL ~ 0 by doc
            pass

        rows.append(
            {
                "timestamp": entry_ts,
                "entry_price": entry_price,
                "quantity": float(quantity),
                "mfe_price": mfe_price,
                "mae_price": mae_price,
                "first_hit": first_hit,
                "hit_ts": hit_ts,
                "bars_to_end": int(bars_to_end),
                "pnl_estimate": float(pnl_estimate),
                "pnl_long": float(pnl_long),
                "pnl_short": float(pnl_short),
                "early_reversal": bool(early_reversal),
                "missed_pnl_upper": float(missed_upper),
                "missed_pnl_lower": float(missed_lower),
                "pnl_estimate_usd": float(pnl_estimate * quantity),
                "pnl_long_usd": float(pnl_long * quantity),
                "pnl_short_usd": float(pnl_short * quantity),
                "missed_pnl_upper_usd": float(missed_upper * quantity),
                "missed_pnl_lower_usd": float(missed_lower * quantity),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate MFE/MAE hedging using hourly OHLCV and expected returns")
    parser.add_argument("--ohlcv-csv", type=Path, required=True, help="Path to OHLCV CSV with timestamp, open, high, low, close")
    parser.add_argument("--signals-mfe", type=Path, required=True, help="Path to MFE signals CSV with exp_ret_avg and timestamp")
    parser.add_argument("--signals-mae", type=Path, required=True, help="Path to MAE signals CSV with exp_ret_avg and timestamp")
    parser.add_argument("--horizon", type=int, default=24, help="Number of bars to look ahead (default 24)")
    parser.add_argument("--ret-type", choices=["percent", "log", "abs"], default="percent", help="Return interpretation for exp_ret_avg")
    parser.add_argument("--tie-policy", choices=["zero", "mfe", "mae"], default="zero", help="If both targets hit on same bar")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path for per-entry results")
    parser.add_argument("--dollar-size", type=float, default=1000.0, help="Dollar notional per leg at entry; quantity=dollar_size/close")
    args = parser.parse_args()

    ohlcv = _load_ohlcv(args.ohlcv_csv)
    mfe = _load_signals(args.signals_mfe, label="mfe")
    mae = _load_signals(args.signals_mae, label="mae")

    # Align all by timestamp with inner joins to ensure consistent rows
    merged = ohlcv.merge(mfe, on="timestamp", how="inner").merge(mae, on="timestamp", how="inner")
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    cfg = SimulationConfig(horizon=int(args.horizon), ret_type=args.ret_type, tie_policy=args.tie_policy, dollar_size=float(args.dollar_size))  # type: ignore[arg-type]
    results = simulate(merged, cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)
    print(f"Wrote simulation results: {args.out}")


if __name__ == "__main__":
    main()


