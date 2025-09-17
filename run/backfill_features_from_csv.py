#!/usr/bin/env python3
"""Backfill feature snapshots from a CSV into the DuckDB features table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.lgbm_inference import align_features_for_booster, load_booster, resolve_model_file
from run.features_table import FeatureRow, connect as features_connect, ensure_table as ensure_features_table, upsert_feature_rows

DEFAULT_FEATURES_CSV = Path("/Volumes/Extreme SSD/trading_data/cex/lookbacks/binance_btcusdt_perp_1h/features_all_tf.csv")
DEFAULT_DUCKDB = Path("/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_features.duckdb")
DEFAULT_MODEL_PATH = Path("/Volumes/Extreme SSD/trading_data/cex/models/run/binance_btcusdt_perp_1h/y_logret_168h")
DEFAULT_TIMESTAMP_COLUMN = "timestamp"
DEFAULT_FEATURE_KEY = "binance_btcusdt_perp_1h__backfill_v1"


def _load_features_csv(path: Path, timestamp_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Features CSV not found: {path}")
    df = pd.read_csv(path)
    if timestamp_column not in df.columns:
        raise ValueError(f"CSV missing required timestamp column '{timestamp_column}'")
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors="coerce", utc=True)
    df = df.dropna(subset=[timestamp_column])
    return df


def _filter_by_cutoff(df: pd.DataFrame, timestamp_column: str, cutoff: Optional[pd.Timestamp]) -> pd.DataFrame:
    if cutoff is None:
        return df
    cutoff_utc = pd.Timestamp(cutoff)
    if cutoff_utc.tzinfo is None:
        cutoff_utc = cutoff_utc.tz_localize("UTC")
    else:
        cutoff_utc = cutoff_utc.tz_convert("UTC")
    filtered = df[df[timestamp_column] > cutoff_utc].copy()
    return filtered


def _build_feature_rows(
    df: pd.DataFrame,
    feature_key: str,
    timestamp_column: str,
    aligned_features: pd.DataFrame,
) -> List[FeatureRow]:
    rows: List[FeatureRow] = []
    timestamps = df[timestamp_column].reset_index(drop=True)
    for idx in range(len(aligned_features)):
        ts = timestamps.iloc[idx]
        if pd.isna(ts):
            continue
        rows.append(FeatureRow.from_series(feature_key, ts, aligned_features.iloc[idx]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill features into DuckDB from a CSV snapshot")
    parser.add_argument('--features-csv', type=Path, default=DEFAULT_FEATURES_CSV, help='Path to features CSV used during inference')
    parser.add_argument('--duckdb', type=Path, default=DEFAULT_DUCKDB, help='DuckDB database that stores the features table')
    parser.add_argument('--model-path', type=Path, default=DEFAULT_MODEL_PATH, help='Path to model directory or model.txt for alignment')
    parser.add_argument('--feature-key', default=DEFAULT_FEATURE_KEY, help='Feature snapshot key to tag stored rows')
    parser.add_argument('--timestamp-column', default=DEFAULT_TIMESTAMP_COLUMN, help='Name of timestamp column in the features CSV')
    parser.add_argument('--cutoff-ts', default=None, help='Optional UTC timestamp; keep rows strictly after this value')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Number of rows to insert per batch')
    args = parser.parse_args()

    df = _load_features_csv(args.features_csv, args.timestamp_column)
    if args.cutoff_ts:
        df = _filter_by_cutoff(df, args.timestamp_column, pd.Timestamp(args.cutoff_ts))
    if df.empty:
        print('No feature rows to backfill after filtering; exiting')
        return

    model_file, _ = resolve_model_file(model_path=args.model_path)
    booster = load_booster(model_file)

    feature_frame = df.drop(columns=[args.timestamp_column])
    aligned = align_features_for_booster(feature_frame, booster)

    rows = _build_feature_rows(df, args.feature_key, args.timestamp_column, aligned)
    if not rows:
        print('No valid feature rows constructed; exiting')
        return

    total_written = 0
    batch: List[FeatureRow] = []
    with features_connect(args.duckdb) as con:
        ensure_features_table(con)
        for row in rows:
            batch.append(row)
            if len(batch) >= args.chunk_size:
                total_written += upsert_feature_rows(con, batch)
                batch.clear()
        if batch:
            total_written += upsert_feature_rows(con, batch)

    print(f"Inserted/updated {total_written} feature rows into {args.duckdb} under key '{args.feature_key}'")


if __name__ == '__main__':
    main()
