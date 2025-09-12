#!/usr/bin/env python3
"""
Orchestrate a single LightGBM inference run for the latest hourly bar.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from datetime import datetime

import pandas as pd

from run.data_loader import load_ohlcv_csv, HistoryRequirement, ensure_min_history
from run.lookbacks_builder import build_latest_lookbacks
from run.features_builder import compute_latest_features_from_lookbacks
from run.model_io_lgbm import load_lgbm_model, predict_latest_row
from run.persist_duckdb import ensure_tables, write_prediction, write_features_latest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hourly LightGBM inference on latest bar")
    parser.add_argument('--input-csv', required=True, help='Hourly OHLCV CSV path')
    parser.add_argument('--dataset', required=True, help='Dataset label (e.g., BINANCE_BTCUSDT.P, 60)')
    parser.add_argument('--model-path', default=None, help='Path to a run dir containing model.txt')
    parser.add_argument('--model-root', default=None, help='Root dir with run_*; latest model.txt will be used')
    parser.add_argument('--duckdb', required=True, help='DuckDB database file path')
    parser.add_argument('--debug-dir', default=None, help='Directory to write debug artifacts')
    parser.add_argument('--buffer-hours', type=int, default=6, help='Extra hours on top of 30d minimum')
    parser.add_argument('--timeframes', nargs='+', default=['1H','4H','12H','1D'], help='Timeframes to use')
    args = parser.parse_args()

    # 1) Load data
    df = load_ohlcv_csv(args.input_csv)

    # 2) Ensure history coverage (30d + buffer)
    history = HistoryRequirement(required_hours=30*24, buffer_hours=args.buffer_hours)
    df_trimmed, latest_ts = ensure_min_history(df, hours_required=history.total_required_hours)

    # 3) Build lookbacks at latest bar
    lookbacks = build_latest_lookbacks(df_trimmed, window_hours=history.total_required_hours, timeframes=args.timeframes)

    # 4) Compute features for latest bar
    features_row = compute_latest_features_from_lookbacks(lookbacks)

    # 5) Load model & predict
    booster, run_dir = load_lgbm_model(model_root=args.model_root, model_path=args.model_path)
    y_pred = predict_latest_row(booster, features_row)

    # 6) Persist prediction and (optional) features
    db_path = Path(args.duckdb)
    ensure_tables(db_path)
    write_prediction(db_path, {
        'dataset': args.dataset,
        'timestamp': features_row['timestamp'].iloc[0],
        'model_run': str(run_dir),
        'y_pred': y_pred,
    })

    # Write features row for debugging/auditing (optional but useful)
    try:
        write_features_latest(db_path, features_row)
    except Exception:
        pass

    # 7) Debug artifacts
    if args.debug_dir:
        dbg_dir = Path(args.debug_dir)
        dbg_dir.mkdir(parents=True, exist_ok=True)
        ds_tag = args.dataset.replace(' ', '_').replace(',', '')
        bar_ts = pd.Timestamp(features_row['timestamp'].iloc[0]).strftime('%Y%m%d_%H%M%S')
        run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        features_row.to_csv(dbg_dir / f"features_{ds_tag}_{bar_ts}_{run_ts}.csv", index=False)
        pd.DataFrame({
            'timestamp': [features_row['timestamp'].iloc[0]],
            'y_pred': [y_pred],
            'model_run': [str(run_dir)],
            'run_ts': [run_ts],
        }).to_csv(
            dbg_dir / f"prediction_{ds_tag}_{bar_ts}_{run_ts}.csv", index=False
        )

    print(f"OK: {args.dataset} @ {features_row['timestamp'].iloc[0]} -> y_pred={y_pred:.6f}")


if __name__ == '__main__':
    main()


