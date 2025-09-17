#!/usr/bin/env python3
"""
Orchestrate a single LightGBM inference run for the latest hourly bar.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from run.data_loader import load_ohlcv_csv, HistoryRequirement, ensure_min_history, validate_hourly_continuity
from run.lookbacks_builder import build_latest_lookbacks, trim_lookbacks_to_base_window, validate_lookbacks_exact
from run.features_builder import compute_latest_features_from_lookbacks, validate_features_for_model
from run.model_io_lgbm import load_lgbm_model, predict_latest_row
from run.predictions_table import PredictionRow, connect as predictions_connect, insert_predictions
from run.features_table import FeatureRow, connect as features_connect, ensure_table as ensure_features_table, upsert_feature_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hourly LightGBM inference on latest bar")
    parser.add_argument('--input-csv', required=True, help='Hourly OHLCV CSV path')
    parser.add_argument('--dataset', required=True, help='Dataset label (e.g., BINANCE_BTCUSDT.P, 60)')
    parser.add_argument('--model-path', default=None, help='Path to a run dir containing model.txt')
    parser.add_argument('--model-root', default=None, help='Root dir with run_*; latest model.txt will be used')
    parser.add_argument('--duckdb', required=True, help='DuckDB database file path')
    parser.add_argument('--feature-key', required=True, help='Feature snapshot key associated with this prediction run')
    parser.add_argument('--debug-dir', default=None, help='Directory to write debug artifacts')
    parser.add_argument('--buffer-hours', type=int, default=6, help='Extra hours on top of 30d minimum')
    parser.add_argument('--timeframes', nargs='+', default=['1H','4H','12H','1D'], help='Timeframes to use')
    args = parser.parse_args()

    # 1) Load data
    df = load_ohlcv_csv(args.input_csv)

    # 2) Ensure history coverage (30d + buffer)
    history = HistoryRequirement(required_hours=30*24, buffer_hours=args.buffer_hours)
    df_trimmed, latest_ts = ensure_min_history(df, hours_required=history.total_required_hours)
    validate_hourly_continuity(df_trimmed, end_ts=latest_ts, required_hours=history.total_required_hours)

    # 3) Build lookbacks at latest bar
    lookbacks = build_latest_lookbacks(df_trimmed, window_hours=history.total_required_hours, timeframes=args.timeframes)
    # 3b) Trim lookbacks to base window (no buffer influence on features)
    lookbacks = trim_lookbacks_to_base_window(lookbacks, base_hours=30*24)
    validate_lookbacks_exact(lookbacks, base_hours=30*24, end_ts=latest_ts)

    # 4) Compute features for latest bar
    features_row = compute_latest_features_from_lookbacks(lookbacks)

    # 5) Load model & predict
    booster, run_dir = load_lgbm_model(model_root=args.model_root, model_path=args.model_path)
    model_path = None
    if args.model_path:
        mp = Path(args.model_path)
        if mp.exists():
            model_path = mp if mp.is_file() else mp / "model.txt"
    if model_path is None:
        model_path = Path(run_dir) / "model.txt"
    # strict validation using model feature schema
    validate_features_for_model(booster, features_row)
    y_pred = predict_latest_row(booster, features_row)

    # 6) Persist prediction and (optional) features
    db_path = Path(args.duckdb)
    payload = {
        'timestamp': features_row['timestamp'].iloc[0],
        'model_path': str(model_path),
        'y_pred': y_pred,
        'feature_key': args.feature_key,
    }
    row = PredictionRow.from_payload(payload)
    with predictions_connect(db_path) as con:
        insert_predictions(con, [row])

    feature_cols = list(booster.feature_name())
    feature_series = features_row.iloc[0][feature_cols]
    feature_row = FeatureRow.from_series(
        args.feature_key,
        pd.Timestamp(features_row['timestamp'].iloc[0]),
        feature_series,
    )
    with features_connect(db_path) as con:
        ensure_features_table(con)
        upsert_feature_rows(con, [feature_row])

    # 7) Debug artifacts
    if args.debug_dir:
        dbg_dir = Path(args.debug_dir)
        dbg_dir.mkdir(parents=True, exist_ok=True)
        ds_tag = args.dataset.replace(' ', '_').replace(',', '')
        bar_ts = pd.Timestamp(features_row['timestamp'].iloc[0]).strftime('%Y%m%d_%H%M%S')
        run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        features_row.to_csv(dbg_dir / f"features_{ds_tag}_{bar_ts}_{run_ts}.csv", index=False)
        pd.DataFrame({'timestamp': [features_row['timestamp'].iloc[0]], 'y_pred': [y_pred], 'model_run': [str(run_dir)], 'run_ts': [run_ts]}).to_csv(
            dbg_dir / f"prediction_{ds_tag}_{bar_ts}_{run_ts}.csv", index=False
        )

    print(f"OK: {args.dataset} @ {features_row['timestamp'].iloc[0]} -> y_pred={y_pred:.6f}")


if __name__ == '__main__':
    main()
