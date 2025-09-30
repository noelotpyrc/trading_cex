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
  --dataset "binance_btcusdt_perp_1h" \  
  --model-path "/Volumes/Extreme SSD/trading_data/cex/models/run/binance_btcusdt_perp_1h" \  
  --output-csv "/Volumes/Extreme SSD/trading_data/cex/inference/binance_btcusdt_perp_1h/prediction/backfill_predictions.csv" \
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.lgbm_inference import predict_from_csv, resolve_model_file, load_booster, align_features_for_booster
from run.predictions_table import PredictionRow, connect as predictions_connect, insert_predictions
from run.features_table import FeatureRow, connect as features_connect, ensure_table as ensure_features_table, upsert_feature_rows


def _default_output_path(features_csv: Path) -> Path:
    return features_csv.with_name('predictions_from_features.csv')


def main() -> None:
    parser = argparse.ArgumentParser(description='Predict from a features CSV and optionally persist to DuckDB')
    parser.add_argument('--features-csv', required=True, help='Path to features CSV (must include timestamp)')
    parser.add_argument('--dataset', required=True, help='Dataset label stored in predictions')
    parser.add_argument('--model-root', default=None, help='Model root containing run_* dirs')
    parser.add_argument('--model-path', default=None, help='Explicit run dir or model.txt path')
    parser.add_argument('--feature-key', required=True, help='Feature snapshot key to associate with predictions/features')
    parser.add_argument('--output-csv', default=None, help='Output predictions CSV path (default: predictions_from_features.csv beside input)')
    parser.add_argument('--duckdb', default=None, help='DuckDB path to persist predictions (optional)')
    parser.add_argument('--timestamp-column', default='timestamp', help='Name of timestamp column in features CSV')
    args = parser.parse_args()

    features_path = Path(args.features_csv)
    if not features_path.exists():
        raise FileNotFoundError(f'Features CSV not found: {features_path}')

    if args.feature_key and not args.timestamp_column:
        raise SystemExit('--feature-key requires --timestamp-column to be set')

    out_csv: Optional[Path] = Path(args.output_csv) if args.output_csv else _default_output_path(features_path)

    # Resolve model run dir (for metadata when persisting)
    model_path_str = ''
    model_file: Optional[Path] = None
    try:
        mf, run_dir = resolve_model_file(model_root=args.model_root, model_path=args.model_path)
        model_file = mf
        if args.model_path:
            mp = Path(args.model_path)
            if mp.exists():
                model_path_str = str(mp if mp.is_file() else mp / 'model.txt')
        if not model_path_str:
            model_path_str = str(mf)
    except Exception:
        # Allow predict_from_csv to resolve; run_dir metadata may be blank when writing to DB
        fallback = args.model_path or args.model_root or ''
        model_path_str = str(fallback)

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
        # Expect timestamp column present in result due to timestamp_column flag
        if args.timestamp_column not in result.columns:
            # If predict_from_csv returned without timestamp, attach from input
            inp = pd.read_csv(features_path, usecols=[args.timestamp_column])
            result.insert(0, args.timestamp_column, inp[args.timestamp_column].values)

        rows = []
        for i in range(n):
            ts = pd.to_datetime(result.iloc[i][args.timestamp_column], errors='coerce')
            rows.append(PredictionRow.from_payload({
                'timestamp': ts,
                'model_path': model_path_str,
                'y_pred': float(result.iloc[i]['y_pred']),
                'feature_key': args.feature_key,
            }))
        with predictions_connect(db_path) as con:
            insert_predictions(con, rows)
        print(f'Persisted {n} predictions to DuckDB: {db_path}')

        if model_file is None:
            raise SystemExit('Unable to resolve model file; cannot persist features')
        raw_features = pd.read_csv(features_path)
        if args.timestamp_column not in raw_features.columns:
            raise SystemExit(f"Timestamp column '{args.timestamp_column}' not found in features CSV")
        timestamps = pd.to_datetime(raw_features[args.timestamp_column], errors='coerce')
        features_only = raw_features.drop(columns=[args.timestamp_column])
        booster = load_booster(model_file)
        aligned = align_features_for_booster(features_only, booster)
        feature_rows = []
        for i in range(len(aligned)):
            ts = timestamps.iloc[i]
            if pd.isna(ts):
                continue
            feature_rows.append(FeatureRow.from_series(args.feature_key, ts, aligned.iloc[i]))
        if feature_rows:
            with features_connect(db_path) as con:
                ensure_features_table(con)
                upsert_feature_rows(con, feature_rows)
            print(f"Persisted {len(feature_rows)} feature rows to DuckDB under key '{args.feature_key}'")
        else:
            print('No valid timestamps found for feature persistence; skipped')

    print('backfill_inference_from_features_csv OK')


if __name__ == '__main__':
    main()
