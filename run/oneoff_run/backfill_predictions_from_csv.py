#!/usr/bin/env python3
"""Backfill predictions from a CSV into the production DuckDB table."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from predictions_table import (
    TABLE_NAME,
    PredictionRow,
    connect as predictions_connect,
    ensure_table,
    insert_predictions,
)

DEFAULT_INPUT = Path("/Volumes/Extreme SSD/trading_data/cex/inference/binance_btcusdt_perp_1h/prediction/backfill_predictions.csv")
DEFAULT_OUTPUT_DB = Path("/Volumes/Extreme SSD/trading_data/cex/db/binance_btcusdt_perp_prediction.duckdb")
DEFAULT_MODEL_PATH = Path("/Volumes/Extreme SSD/trading_data/cex/models/run/binance_btcusdt_perp_1h/y_logret_168h")
DEFAULT_CUTOFF = pd.Timestamp("2025-03-21 04:00:00", tz="UTC")
DEFAULT_FEATURE_KEY = "binance_btcusdt_perp_1h__backfill_v1"


def load_predictions(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise ValueError("Input CSV must contain a 'timestamp' column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])
    return df


def _normalise_timestamp(ts: str | pd.Timestamp | None) -> pd.Timestamp | None:
    if ts is None:
        return None
    if isinstance(ts, str) and ts.strip().lower() == "none":
        return None
    parsed = pd.Timestamp(ts)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    return parsed


def filter_predictions(
    df: pd.DataFrame,
    *,
    min_ts: pd.Timestamp | None,
    max_ts: pd.Timestamp | None,
) -> pd.DataFrame:
    if min_ts is not None and max_ts is not None and min_ts > max_ts:
        raise ValueError("min_ts cannot be after max_ts")

    filtered = df
    if min_ts is not None:
        filtered = filtered[filtered["timestamp"] >= min_ts]
    if max_ts is not None:
        filtered = filtered[filtered["timestamp"] <= max_ts]

    filtered = filtered.copy()
    filtered["timestamp"] = filtered["timestamp"].dt.tz_convert(None)
    filtered = filtered.sort_values("timestamp")
    return filtered


def drop_existing_predictions(
    df: pd.DataFrame,
    *,
    con,
    model_path: str,
    feature_key: str,
) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0

    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    existing_df = con.execute(
        f"""
        SELECT ts
        FROM {TABLE_NAME}
        WHERE ts BETWEEN ? AND ?
          AND model_path = ?
          AND feature_key = ?
        """,
        [ts_min.to_pydatetime(), ts_max.to_pydatetime(), model_path, feature_key],
    ).fetch_df()

    if existing_df.empty:
        return df, 0

    existing_ts = set(pd.to_datetime(existing_df["ts"]))
    deduped = df[~df["timestamp"].isin(existing_ts)].copy()
    skipped = len(df) - len(deduped)
    return deduped, skipped


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
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT, help="Source predictions CSV")
    parser.add_argument("--duckdb", type=Path, default=DEFAULT_OUTPUT_DB, help="Target DuckDB path")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Model path to record with predictions")
    parser.add_argument(
        "--feature-key",
        default=DEFAULT_FEATURE_KEY,
        help="Feature snapshot key associated with these predictions",
    )
    parser.add_argument(
        "--start-ts",
        dest="start_ts",
        default=str(DEFAULT_CUTOFF),
        help="Earliest timestamp (inclusive, UTC) to include. Use 'none' to remove the lower bound.",
    )
    parser.add_argument(
        "--cutoff-ts",
        dest="start_ts",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--end-ts",
        dest="end_ts",
        default=None,
        help="Latest timestamp (inclusive, UTC) to include. Use 'none' to include current rows.",
    )
    args = parser.parse_args()

    df = load_predictions(args.input_csv)
    min_ts = _normalise_timestamp(args.start_ts)
    max_ts = _normalise_timestamp(args.end_ts)
    model_path_str = str(args.model_path)
    feature_key = str(args.feature_key)

    with predictions_connect(args.duckdb) as con:
        ensure_table(con)
        filtered = filter_predictions(df, min_ts=min_ts, max_ts=max_ts)
        filtered, skipped = drop_existing_predictions(
            filtered,
            con=con,
            model_path=model_path_str,
            feature_key=feature_key,
        )

        if filtered.empty:
            if skipped:
                print(
                    "All predictions in requested window already exist; nothing to insert"
                )
            else:
                print("No predictions in requested window; nothing to insert")
            return

        rows = list(to_prediction_rows(filtered, model_path_str, feature_key))

        inserted = insert_predictions(con, rows)

    if min_ts or max_ts:
        range_msg = []
        if min_ts:
            range_msg.append(f"start={min_ts.isoformat()}")
        if max_ts:
            range_msg.append(f"end={max_ts.isoformat()}")
        range_str = " (" + ", ".join(range_msg) + ")"
    else:
        range_str = ""
    extra = f"; skipped {skipped} duplicates" if 'skipped' in locals() and skipped else ""
    print(f"Inserted {inserted} predictions into {args.duckdb}{range_str}{extra}")


if __name__ == '__main__':
    main()
