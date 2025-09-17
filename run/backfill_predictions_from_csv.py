#!/usr/bin/env python3
"""Backfill predictions from a CSV into the production DuckDB table."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from predictions_table import PredictionRow, connect as predictions_connect, insert_predictions, ensure_table

DEFAULT_INPUT = Path("/Volumes/Extreme SSD/trading_data/cex/inference/binance_btcusdt_perp_1h/prediction/backfill_predictions.csv")
DEFAULT_OUTPUT_DB = Path("/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_prediction.duckdb")
DEFAULT_MODEL_PATH = Path("/Volumes/Extreme SSD/trading_data/cex/models/run/binance_btcusdt_perp_1h/y_logret_168h")
DEFAULT_CUTOFF = pd.Timestamp("2025-03-21 04:00:00", tz='UTC')
DEFAULT_FEATURE_KEY = "binance_btcusdt_perp_1h__backfill_v1"


def load_predictions(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if 'timestamp' not in df.columns:
        raise ValueError("Input CSV must contain a 'timestamp' column")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    df = df.dropna(subset=['timestamp'])
    return df


def filter_predictions(df: pd.DataFrame, cutoff_ts: pd.Timestamp) -> pd.DataFrame:
    # Ensure cutoff is timezone-aware for comparison, then convert to naive UTC
    cutoff_ts = pd.Timestamp(cutoff_ts).tz_convert('UTC') if cutoff_ts.tzinfo else pd.Timestamp(cutoff_ts).tz_localize('UTC')
    filtered = df[df['timestamp'] > cutoff_ts]
    filtered = filtered.copy()
    filtered['timestamp'] = filtered['timestamp'].dt.tz_convert(None)
    filtered = filtered.sort_values('timestamp')
    return filtered


def to_prediction_rows(df: pd.DataFrame, model_path: str, feature_key: str) -> Iterable[PredictionRow]:
    if 'y_pred' not in df.columns:
        raise ValueError("Input CSV must contain a 'y_pred' column")
    for _, row in df.iterrows():
        yield PredictionRow.from_payload({
            'timestamp': row['timestamp'],
            'y_pred': float(row['y_pred']),
            'model_path': model_path,
            'feature_key': feature_key,
        })


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill predictions into DuckDB from a CSV")
    parser.add_argument('--input-csv', type=Path, default=DEFAULT_INPUT, help='Source predictions CSV')
    parser.add_argument('--duckdb', type=Path, default=DEFAULT_OUTPUT_DB, help='Target DuckDB path')
    parser.add_argument('--model-path', type=Path, default=DEFAULT_MODEL_PATH, help='Model path to record with predictions')
    parser.add_argument('--feature-key', default=DEFAULT_FEATURE_KEY, help='Feature snapshot key associated with these predictions')
    parser.add_argument('--cutoff-ts', default=str(DEFAULT_CUTOFF), help='Only keep predictions after this timestamp (UTC)')
    args = parser.parse_args()

    df = load_predictions(args.input_csv)
    cutoff = pd.Timestamp(args.cutoff_ts)
    filtered = filter_predictions(df, cutoff)

    if filtered.empty:
        print('No predictions after cutoff; nothing to insert')
        return

    model_path_str = str(args.model_path)
    feature_key = str(args.feature_key)

    rows = list(to_prediction_rows(filtered, model_path_str, feature_key))

    with predictions_connect(args.duckdb) as con:
        ensure_table(con)
        inserted = insert_predictions(con, rows)

    print(f'Inserted {inserted} predictions into {args.duckdb}')


if __name__ == '__main__':
    main()
