#!/usr/bin/env python3
"""
Compute expected-return metrics, probability-of-positive, diagnostics, and basic
signals from merged quantile prediction CSVs.

Usage:
  python model/compute_signals_from_quantiles.py \
    --pred-csv "/Volumes/Extreme SSD/trading_data/cex/models/BINANCE_BTCUSDT.P, 60/diagnosis/y_logret_168h/pred_test.csv" 
    --method avg 
    --threshold-csvs "/Volumes/Extreme SSD/trading_data/cex/models/BINANCE_BTCUSDT.P, 60/diagnosis/y_logret_168h/pred_train.csv" "/Volumes/Extreme SSD/trading_data/cex/models/BINANCE_BTCUSDT.P, 60/diagnosis/y_logret_168h/pred_val.csv" 
    --prob-thresholds 0.55 0.45 
    --out "/Volumes/Extreme SSD/trading_data/cex/models/BINANCE_BTCUSDT.P, 60/diagnosis/y_logret_168h/pred_test_signals.csv"

Minimal expected-returns only output (handy for MFE/MAE targets):
  python model/compute_signals_from_quantiles.py \
    --pred-csv /abs/merged/pred_val.csv \
    --exp-only \
    --out /abs/merged/pred_val_exp.csv

This will read the merged predictions (with columns timestamp,y_true,pred_q05..q95),
compute metrics and optional threshold-based signals, and write an augmented CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd

from strategies.quantile_signals import (
    DEFAULT_QCOLS,
    add_basic_signals,
    add_core_metrics,
    compute_static_thresholds,
)


def _infer_out_path(pred_csv: Path) -> Path:
    stem = pred_csv.stem
    if stem.endswith("_signals"):
        return pred_csv
    return pred_csv.with_name(f"{stem}_signals{pred_csv.suffix}")


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize timestamp if present
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["timestamp"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute signals from merged quantile predictions.")
    parser.add_argument("--pred-csv", type=Path, required=True, help="Path to merged predictions (pred_*.csv)")
    parser.add_argument("--method", choices=["avg", "conservative"], default="avg", help="Expected-return method for signal_exp")
    parser.add_argument("--exp-zero-band", type=float, default=1e-6, help="Deadband around zero for expected-return signal")
    parser.add_argument("--threshold-csvs", type=Path, nargs="*", help="Optional list of CSVs (e.g., merged train and val) for static thresholds over the union")
    parser.add_argument("--prob-thresholds", type=float, nargs=2, metavar=("TAU_LONG", "TAU_SHORT"), default=(0.55, 0.45), help="Probability thresholds for long/short")
    parser.add_argument("--out", type=Path, required=False, help="Output CSV path; defaults to <pred>_signals.csv")
    parser.add_argument("--exp-only", action="store_true", help="Only compute expected returns and write minimal columns")
    args = parser.parse_args()

    pred_csv: Path = args.pred_csv.resolve()
    out_csv: Path = args.out.resolve() if args.out else _infer_out_path(pred_csv)

    df = _read_csv(pred_csv)

    # Compute core metrics with a monotonic fix applied for computations
    df_metrics = add_core_metrics(df, DEFAULT_QCOLS)

    # Optional static thresholds from union of provided CSVs
    quantile_thresholds: Tuple[float, float] | None = None
    if args.threshold_csvs:
        frames: List[pd.DataFrame] = []
        for p in args.threshold_csvs:
            p = Path(p).resolve()
            if not p.exists():
                raise FileNotFoundError(f"Threshold CSV not found: {p}")
            frames.append(_read_csv(p))
        # Compute percentiles: 90th for pred_q90 long, 10th for pred_q10 short
        th = compute_static_thresholds(frames, ["pred_q90", "pred_q10"], [0.90, 0.10])
        thr_q90 = float(th.get("pred_q90@0.9000", float("nan")))
        thr_q10 = float(th.get("pred_q10@0.1000", float("nan")))
        if pd.notna(thr_q90) and pd.notna(thr_q10):
            quantile_thresholds = (thr_q90, thr_q10)

    # If only expected returns are requested, emit a minimal CSV and exit
    if args.exp_only:
        cols_to_keep: list[str] = []
        if "timestamp" in df_metrics.columns:
            cols_to_keep.append("timestamp")
        for c in ["exp_ret_avg", "exp_ret_conservative"]:
            if c in df_metrics.columns:
                cols_to_keep.append(c)
        if not cols_to_keep:
            raise RuntimeError("No expected return columns found to write.")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_metrics[cols_to_keep].to_csv(out_csv, index=False)
        print(f"Wrote expected returns: {out_csv}")
        return

    # Add basic signals
    df_signals = add_basic_signals(
        df_metrics,
        selected_method=args.method,
        exp_zero_band=float(args.exp_zero_band),
        prob_thresholds=(float(args.prob_thresholds[0]), float(args.prob_thresholds[1])),
        quantile_thresholds=quantile_thresholds,
        qcols=DEFAULT_QCOLS,
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_signals.to_csv(out_csv, index=False)
    print(f"Wrote signals: {out_csv}")


if __name__ == "__main__":
    main()


