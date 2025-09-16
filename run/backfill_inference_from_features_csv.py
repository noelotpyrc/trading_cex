#!/usr/bin/env python3
from __future__ import annotations

"""
Backfill predictions by scoring a precomputed features CSV (e.g., features_all_tf.csv).

Features CSV is expected to contain a 'timestamp' column and arbitrary feature
columns (may include extra columns such as targets y_* which will be ignored).

The script loads a LightGBM model (latest in --model-root or the one at --model-path),
aligns features to the model's schema, predicts for all rows, writes an output
predictions CSV, and optionally persists to DuckDB's predictions table.

Example:
  python run/backfill_inference_from_features_csv.py \
    --features-csv "/Volumes/Extreme SSD/trading_data/cex/lookbacks/binance_btcusdt_perp_1h/features_all_tf.csv" \
    --dataset "BINANCE_BTCUSDT.P, 60" \
    --model-root "/Volumes/Extreme SSD/trading_data/cex/models/BINANCE_BTCUSDT.P, 60" \
    --output-csv "/Volumes/Extreme SSD/trading_data/cex/lookbacks/binance_btcusdt_perp_1h/predictions_from_features.csv" \
    --duckdb "/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_ohlcv.duckdb"
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.lgbm_inference import predict_from_csv, resolve_model_file
from run.persist_duckdb import ensure_tables, write_prediction


def _default_output_path(features_csv: Path) -> Path:
    return features_csv.with_name('predictions_from_features.csv')


def main() -> None:
    parser = argparse.ArgumentParser(description='Predict from a features CSV and optionally persist to DuckDB')
    parser.add_argument('--features-csv', required=True, help='Path to features CSV (must include timestamp)')
    parser.add_argument('--dataset', required=True, help='Dataset label stored in predictions')
    parser.add_argument('--model-root', default=None, help='Model root containing run_* dirs')
    parser.add_argument('--model-path', default=None, help='Explicit run dir or model.txt path')
    parser.add_argument('--output-csv', default=None, help='Output predictions CSV path (default: predictions_from_features.csv beside input)')
    parser.add_argument('--duckdb', default=None, help='DuckDB path to persist predictions (optional)')
    parser.add_argument('--timestamp-column', default='timestamp', help='Name of timestamp column in features CSV')
    args = parser.parse_args()

    features_path = Path(args.features_csv)
    if not features_path.exists():
        raise FileNotFoundError(f'Features CSV not found: {features_path}')

    out_csv: Optional[Path] = Path(args.output_csv) if args.output_csv else _default_output_path(features_path)

    # Resolve model run dir (for metadata when persisting)
    run_dir_str = ''
    try:
        mf, run_dir = resolve_model_file(model_root=args.model_root, model_path=args.model_path)
        run_dir_str = str(run_dir)
    except Exception:
        # Allow predict_from_csv to resolve; run_dir metadata may be blank when writing to DB
        run_dir_str = str(args.model_path or args.model_root or '')

    # Predict via shared inference helper (handles alignment and optional write)
    result = predict_from_csv(
        input_csv=str(features_path),
        model_root=str(args.model_root) if args.model_root else None,
        model_path=str(args.model_path) if args.model_path else None,
        output_csv=str(out_csv) if out_csv is not None else None,
        timestamp_column=str(args.timestamp_column),
        merge_input=False,
    )

    n = len(result)
    print(f'Predictions computed: rows={n}')
    if out_csv is not None:
        print(f'Predictions CSV: {out_csv}')

    # Optionally persist to DuckDB
    if args.duckdb:
        db_path = Path(args.duckdb)
        ensure_tables(db_path)
        # Expect timestamp column present in result due to timestamp_column flag
        if args.timestamp_column not in result.columns:
            # If predict_from_csv returned without timestamp, attach from input
            inp = pd.read_csv(features_path, usecols=[args.timestamp_column])
            result.insert(0, args.timestamp_column, inp[args.timestamp_column].values)

        for i in range(n):
            ts = pd.to_datetime(result.iloc[i][args.timestamp_column], errors='coerce')
            y = float(result.iloc[i]['y_pred'])
            write_prediction(db_path, {
                'dataset': args.dataset,
                'timestamp': ts,
                'model_run': run_dir_str,
                'y_pred': y,
            })
        print(f'Persisted {n} predictions to DuckDB: {db_path}')

    print('backfill_inference_from_features_csv OK')


if __name__ == '__main__':
    main()
