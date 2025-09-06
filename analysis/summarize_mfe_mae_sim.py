#!/usr/bin/env python3
"""
Summarize MFE/MAE hedge simulation results.

Reads the CSV produced by strategies/simulate_mfe_mae_hedge.py and prints:
- Total PnL (USD), average PnL per trade, win/loss/flat rates
- Trading frequency and counts by first_hit (MFE/MAE/NONE)
- Breakdown of PnL by first_hit and by magnitude buckets
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def load_sim(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize timestamp if present
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["timestamp"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return df


def summarize(
    df: pd.DataFrame,
    pnl_col: str,
    horizon: int,
    leverage: float,
    buffer: float,
    dollar_size: float | None,
) -> str:
    out_lines: List[str] = []

    if pnl_col not in df.columns:
        # Fallback to non-USD column name
        if pnl_col.endswith("_usd"):
            alt = pnl_col[:-4]
            if alt in df.columns:
                pnl_col = alt
            else:
                raise KeyError(f"Neither '{pnl_col}' nor '{alt}' present in columns: {list(df.columns)}")
        else:
            raise KeyError(f"Missing column '{pnl_col}' in data")

    total = len(df)
    traded_mask = df["first_hit"].astype(str) != "NONE"
    traded = int(traded_mask.sum())
    freq = traded / total if total else 0.0

    pnl = pd.to_numeric(df[pnl_col], errors="coerce").fillna(0.0)
    pnl_trades = pnl[traded_mask]
    total_pnl = float(pnl.sum())
    avg_pnl_trade = float(pnl_trades.mean()) if traded > 0 else 0.0

    wins = int((pnl_trades > 0).sum())
    losses = int((pnl_trades < 0).sum())
    flats = traded - wins - losses
    win_rate = wins / traded if traded else 0.0
    loss_rate = losses / traded if traded else 0.0

    out_lines.append("=== Overview ===")
    out_lines.append(f"rows={total}  traded={traded}  freq={freq:.3f}")
    out_lines.append(f"total_{pnl_col}={total_pnl:.2f}  avg_per_trade={avg_pnl_trade:.2f}")
    out_lines.append(f"win_rate={win_rate:.3f}  loss_rate={loss_rate:.3f}  flats={flats}")

    # Counts by first_hit
    counts = df["first_hit"].astype(str).value_counts().rename_axis("first_hit").to_frame("count")
    sums = df.groupby(df["first_hit"].astype(str))[pnl_col].sum().rename("sum_pnl")
    by_hit = counts.join(sums, how="left").fillna(0.0)
    out_lines.append("\n=== By first_hit ===")
    out_lines.append(by_hit.to_string())

    # Magnitude buckets on traded PnL
    bins = [-np.inf, -200, -100, -50, -10, -1e-6, 1e-6, 10, 50, 100, 200, np.inf]
    labels = [
        "<=-200", "(-200,-100]", "(-100,-50]", "(-50,-10]", "(-10,0)", "≈0", "(0,10]", "(10,50]", "(50,100]", "(100,200]", ">200",
    ]
    cats = pd.cut(pnl_trades.to_numpy(), bins=bins, labels=labels, include_lowest=True)
    bucket_counts = pd.Series(cats, dtype="category").value_counts().sort_index()
    out_lines.append("\n=== PnL buckets (traded only) ===")
    out_lines.append(bucket_counts.to_string())

    # Zero-PnL breakdown: early reversal vs other
    if "early_reversal" in df.columns:
        traded_ers = df.loc[traded_mask, "early_reversal"].astype(bool)
        zero_mask = pnl_trades.abs() <= 1e-12
        zero_total = int(zero_mask.sum())
        zero_er = int((zero_mask & traded_ers.values).sum())
        zero_other = zero_total - zero_er
        out_lines.append("\n=== Zero PnL breakdown (traded only) ===")
        out_lines.append(f"zero_total={zero_total}  early_reversal={zero_er}  other={zero_other}")
        # Missed PnL bounds due to early reversal (USD if available)
        upper_col = "missed_pnl_upper_usd" if f"{pnl_col}_usd" == pnl_col or pnl_col.endswith("_usd") else "missed_pnl_upper"
        lower_col = "missed_pnl_lower_usd" if f"{pnl_col}_usd" == pnl_col or pnl_col.endswith("_usd") else "missed_pnl_lower"
        if upper_col in df.columns and lower_col in df.columns:
            er_mask_full = (df["first_hit"].astype(str) != "NONE") & df["early_reversal"].astype(bool)
            missed_upper_sum = float(pd.to_numeric(df.loc[er_mask_full, upper_col], errors="coerce").sum())
            missed_lower_sum = float(pd.to_numeric(df.loc[er_mask_full, lower_col], errors="coerce").sum())
            out_lines.append("missed_pnl_bounds (sum over early reversals):")
            out_lines.append(f"  upper_bound_sum={missed_upper_sum:.2f}  lower_bound_sum={missed_lower_sum:.2f}")

    # Capital estimate
    out_lines.append("\n=== Capital estimate ===")
    # Determine per-leg dollar size if available (CLI overrides)
    S: float | None = None
    if dollar_size is not None and dollar_size > 0:
        S = float(dollar_size)
        source = "cli"
    elif "quantity" in df.columns and "entry_price" in df.columns:
        qty = pd.to_numeric(df["quantity"], errors="coerce")
        ep = pd.to_numeric(df["entry_price"], errors="coerce")
        est = (qty * ep).dropna()
        if not est.empty:
            S = float(est.median())
            source = "inferred_from_quantity"
        else:
            source = "per_unit"
    else:
        source = "per_unit"

    H = int(horizon)
    L = float(leverage) if leverage > 0 else np.inf
    if S is not None:
        per_entry_notional = 2.0 * S
        peak_notional = per_entry_notional * H
        margin = peak_notional / L
        buffered = margin * (1.0 + float(buffer))
        out_lines.append(f"sizing_source={source}  S_per_leg={S:,.2f}")
        out_lines.append(f"per_entry_notional={per_entry_notional:,.2f}  peak_notional≈{peak_notional:,.2f}")
        out_lines.append(f"leverage={leverage:g}  margin≈{margin:,.2f}  with_buffer≈{buffered:,.2f} (buffer={buffer:.2f})")
    else:
        ep = pd.to_numeric(df["entry_price"], errors="coerce").dropna()
        if ep.empty:
            out_lines.append("entry_price missing; cannot estimate capital")
        else:
            p_max = float(ep.max())
            p_mean = float(ep.mean())
            per_entry_notional_max = 2.0 * p_max
            per_entry_notional_mean = 2.0 * p_mean
            peak_notional_max = per_entry_notional_max * H
            peak_notional_mean = per_entry_notional_mean * H
            margin_max = peak_notional_max / L
            margin_mean = peak_notional_mean / L
            out_lines.append("sizing_source=per_unit (1 long + 1 short)")
            out_lines.append(
                f"per_entry_notional≈2*price  max≈{per_entry_notional_max:,.2f}  mean≈{per_entry_notional_mean:,.2f}"
            )
            out_lines.append(
                f"peak_notional  max≈{peak_notional_max:,.2f}  mean≈{peak_notional_mean:,.2f}  (H={H})"
            )
            out_lines.append(
                f"leverage={leverage:g}  margin  max≈{margin_max:,.2f}  mean≈{margin_mean:,.2f}  (no buffer)"
            )

    return "\n".join(out_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MFE/MAE hedge simulation CSV")
    parser.add_argument("--sim-csv", type=Path, required=True, help="Path to simulation CSV (mfe_mae_sim_*.csv)")
    parser.add_argument("--pnl-col", type=str, default="pnl_estimate_usd", help="PnL column to summarize (default pnl_estimate_usd)")
    parser.add_argument("--horizon", type=int, default=24, help="Horizon in bars (concurrent entries if trading every bar)")
    parser.add_argument("--leverage", type=float, default=10.0, help="Leverage used for margin estimate")
    parser.add_argument("--buffer", type=float, default=0.25, help="Additional buffer ratio on top of margin (e.g., 0.25)")
    parser.add_argument("--dollar-size", type=float, default=None, help="Optional per-leg dollar size S; overrides inference/per-unit")
    parser.add_argument("--out", type=Path, required=False, help="Optional path to write the textual summary")
    args = parser.parse_args()

    df = load_sim(args.sim_csv)
    report = summarize(df, args.pnl_col, args.horizon, args.leverage, args.buffer, args.dollar_size)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        print(f"Wrote summary: {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()


