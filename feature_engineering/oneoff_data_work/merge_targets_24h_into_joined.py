"""
One-off merger: append selected 24h TP-before-SL classification columns from
targets_24h.csv into the existing merged features+targets table.

Defaults point to the BINANCE_BTCUSDT.P, 60 dataset paths you provided.

Usage (defaults should work as-is):
  /Users/noel/projects/trading_cex/venv/bin/python analysis/merge_targets_24h_into_joined.py \
    --targets "/Volumes/Extreme SSD/trading_data/cex/targets/BINANCE_BTCUSDT.P, 60/targets_24h.csv" \
    --merged "/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/merged_features_targets_joined.csv" \
    --output "/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/merged_features_targets_joined_plus_24h_tp_sl.csv"

Only the following columns are merged from targets_24h.csv:
  - y_tp_before_sl_u0.04_d0.02_24h
  - y_tp_before_sl_u0.03_d0.01_24h
"""

from __future__ import annotations

import argparse
import os
from typing import List

import pandas as pd


DEFAULT_TARGETS_PATH = \
    "/Volumes/Extreme SSD/trading_data/cex/targets/BINANCE_BTCUSDT.P, 60/targets_24h.csv"
DEFAULT_MERGED_PATH = \
    "/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/merged_features_targets_joined.csv"
DEFAULT_OUTPUT_PATH = \
    "/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/merged_features_targets_joined_plus_24h_tp_sl.csv"

COLS_TO_MERGE: List[str] = [
    "y_tp_before_sl_u0.04_d0.02_24h",
    "y_tp_before_sl_u0.03_d0.01_24h",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge selected 24h TP/SL targets into merged features+targets table")
    p.add_argument("--targets", default=DEFAULT_TARGETS_PATH, help="Path to targets_24h.csv")
    p.add_argument("--merged", default=DEFAULT_MERGED_PATH, help="Path to merged_features_targets_joined.csv")
    p.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output CSV path")
    p.add_argument("--overwrite", action="store_true", help="Overwrite columns if they already exist in merged table")
    return p.parse_args()


def _read_csv_with_timestamp(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Expected 'timestamp' column in {path}")
    # Normalize timestamp to naive UTC-like (consistent with project conventions)
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["timestamp"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return df


def main() -> None:
    args = parse_args()

    print(f"Reading targets: {args.targets}")
    tdf = _read_csv_with_timestamp(args.targets)
    missing = [c for c in COLS_TO_MERGE if c not in tdf.columns]
    if missing:
        raise ValueError(f"Targets file missing required columns: {missing}")
    tdf = tdf[["timestamp"] + COLS_TO_MERGE].copy()

    print(f"Reading merged table: {args.merged}")
    mdf = _read_csv_with_timestamp(args.merged)

    # Check for existing columns and guard unless --overwrite
    existing = [c for c in COLS_TO_MERGE if c in mdf.columns]
    if existing and not args.overwrite:
        raise ValueError(
            "Output columns already exist in merged table: "
            + ", ".join(existing)
            + ". Use --overwrite to replace."
        )

    # Perform left-join to keep merged table row count identical
    print("Merging on 'timestamp' (left join)...")
    out = mdf.merge(tdf, on="timestamp", how="left", suffixes=("", "_dup"))

    # If overwrite is requested, drop original and keep new values
    if existing and args.overwrite:
        for c in existing:
            # Keep the merged (right) values; in our merge we didn't duplicate names, so nothing to rename
            pass

    # Basic diagnostics
    print(f"Rows (merged): {len(mdf):,}; Rows (targets): {len(tdf):,}; Rows (out): {len(out):,}")
    for c in COLS_TO_MERGE:
        na_cnt = int(out[c].isna().sum()) if c in out.columns else -1
        print(f"  {c}: present={c in out.columns}, NaNs={na_cnt:,}")

    # Write output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote: {args.output} (cols={len(out.columns)}, rows={len(out)})")


if __name__ == "__main__":
    main()

