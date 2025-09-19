"""
Pick a single random timestamp after a cutoff and compare the two
'current_bar_with_lags_*.csv' files for:
  - whether the row exists in both
  - missing columns between the files
  - high-level row summary (counts of NaNs across common columns)

This avoids per-column statistics and large dumps.

Example:
  python run/inspect_compare_current_bar_lags_one_row.py \
    --dir-a "/Volumes/Extreme SSD/trading_data/cex/lookbacks/binance_btcusdt_perp_1h" \
    --dir-b "/Volumes/Extreme SSD/trading_data/cex/lookbacks/BINANCE_BTCUSDT.P, 60" \
    --cutoff "2025-03-21 04:00:00" \
    --until "2025-08-12" \
    --seed 42
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Optional

import numpy as np
import pandas as pd


def _find_latest_csv(dir_path: str, pattern: str = "current_bar_with_lags_*.csv") -> str:
    paths = glob.glob(os.path.join(dir_path, pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching '{pattern}' in {dir_path}")
    paths.sort(key=lambda p: os.path.getmtime(p))
    return paths[-1]


def _load_features_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Try to locate timestamp column or index
    ts_col: Optional[str] = None
    for cand in ["timestamp", "time", "datetime", "date"]:
        if cand in df.columns:
            ts_col = cand
            break
    if ts_col is None:
        # Assume first column is the index if no explicit timestamp column
        ts_col = df.columns[0]
    # Parse as UTC-aware then drop tz to ensure consistent naive-UTC alignment
    ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    # For Series, use .dt to manipulate tz
    if hasattr(ts, "dt"):
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    else:
        # DatetimeIndex path
        ts = ts.tz_convert("UTC").tz_localize(None)
    df = df.drop(columns=[ts_col])
    df.index = pd.DatetimeIndex(ts)
    df = df.sort_index()
    return df


def main():
    parser = argparse.ArgumentParser(description="Compare a single row between two current-bar-with-lags CSVs and report missing columns")
    parser.add_argument("--dir-a", default="/Volumes/Extreme SSD/trading_data/cex/lookbacks/binance_btcusdt_perp_1h")
    parser.add_argument("--dir-b", default="/Volumes/Extreme SSD/trading_data/cex/lookbacks/BINANCE_BTCUSDT.P, 60")
    parser.add_argument("--file-a", help="Explicit CSV path (overrides auto-detect in dir-a)")
    parser.add_argument("--file-b", help="Explicit CSV path (overrides auto-detect in dir-b)")
    parser.add_argument("--pattern", default="current_bar_with_lags_*.csv")
    parser.add_argument("--cutoff", default="2025-03-21 04:00:00")
    parser.add_argument("--until", default="2025-08-12", help="Upper bound (inclusive). Accepts date or datetime.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-missing-max", type=int, default=50, help="Max missing column names to print from each side")
    parser.add_argument("--tol", type=float, default=1e-9, help="Absolute tolerance for value comparison")
    parser.add_argument("--print-diffs", type=int, default=25, help="Print up to N differing columns (largest abs diff first)")
    args = parser.parse_args()

    file_a = args.file_a or _find_latest_csv(args.dir_a, args.pattern)
    file_b = args.file_b or _find_latest_csv(args.dir_b, args.pattern)

    print("A file:", file_a)
    print("B file:", file_b)

    df_a = _load_features_csv(file_a)
    df_b = _load_features_csv(file_b)

    cutoff_ts = pd.to_datetime(args.cutoff)
    inter = df_a.index.intersection(df_b.index)
    until_ts = pd.to_datetime(args.until)
    inter = inter[(inter >= cutoff_ts) & (inter <= until_ts)]
    print(f"Intersection rows within window: {len(inter)}")
    # Debug: show index ranges
    print("A index range:", df_a.index.min(), "->", df_a.index.max(), "(tz=naive UTC)")
    print("B index range:", df_b.index.min(), "->", df_b.index.max(), "(tz=naive UTC)")
    if len(inter) == 0:
        print("No overlapping timestamps after cutoff; nothing to compare.")
        return

    rng = np.random.default_rng(args.seed)
    ts = pd.Timestamp(rng.choice(inter.values)) if len(inter) > 0 else None
    print("Sampled timestamp:", ts)

    # Column comparison
    cols_a = set(df_a.columns)
    cols_b = set(df_b.columns)
    common_cols = sorted(cols_a & cols_b)
    only_a = sorted(cols_a - cols_b)
    only_b = sorted(cols_b - cols_a)

    print(f"Columns | A: {len(cols_a)} | B: {len(cols_b)} | common: {len(common_cols)} | A-only: {len(only_a)} | B-only: {len(only_b)}")
    if only_a:
        print(f"A-only (showing up to {args.print_missing_max}):")
        for name in only_a[: args.print_missing_max]:
            print("  -", name)
        if len(only_a) > args.print_missing_max:
            print(f"  ... (+{len(only_a) - args.print_missing_max} more)")
    if only_b:
        print(f"B-only (showing up to {args.print_missing_max}):")
        for name in only_b[: args.print_missing_max]:
            print("  -", name)
        if len(only_b) > args.print_missing_max:
            print(f"  ... (+{len(only_b) - args.print_missing_max} more)")

    # Row presence and NaN counts (across common columns only)
    row_a = df_a.loc[ts] if ts in df_a.index else None
    row_b = df_b.loc[ts] if ts in df_b.index else None
    print("Row present | A:", row_a is not None, "| B:", row_b is not None)
    if (row_a is not None) and (row_b is not None) and common_cols:
        a_common = pd.to_numeric(row_a[common_cols], errors="coerce")
        b_common = pd.to_numeric(row_b[common_cols], errors="coerce")
        nan_a = int(a_common.isna().sum())
        nan_b = int(b_common.isna().sum())
        print(f"NaN counts on sampled row (common columns): A={nan_a}, B={nan_b}, total_common={len(common_cols)}")

        # Value comparison on non-NaN pairs
        mask_valid = (~a_common.isna()) & (~b_common.isna())
        diffs = (a_common - b_common).abs()
        diffs_valid = diffs[mask_valid]
        mismatches = diffs_valid[diffs_valid > args.tol]
        matched_cnt = int(mask_valid.sum() - mismatches.size)
        print(f"Value check | valid_pairs={int(mask_valid.sum())}, matched_within_tol={matched_cnt}, mismatched={int(mismatches.size)}, tol={args.tol}")

        if mismatches.size > 0 and args.print_diffs > 0:
            top = mismatches.sort_values(ascending=False).head(args.print_diffs)
            print(f"Top {len(top)} differing columns (abs diff desc):")
            for col, d in top.items():
                va = a_common[col]
                vb = b_common[col]
                print(f"  - {col}: A={va} | B={vb} | |diff|={d}")


if __name__ == "__main__":
    main()
