#!/usr/bin/env python3
"""Sanity-check derived hit-flag targets against the original targets.csv.

This script compares target columns present in a derived file (e.g.,
`targets_with_hit_flags.csv`) against the canonical `targets.csv` to ensure
consistency in timestamps, absence of unexpected NaNs, and basic statistic
alignment. Optionally inspects overlap with log-return thresholds to verify
that hit flags correspond to underlying MFE/MAE movements.

Usage example:

    ./venv/bin/python feature_engineering/oneoff_data_work/check_target_hits.py \
        --original /Volumes/Extreme\ SSD/trading_data/cex/training/\
            BINANCE_BTCUSDT.P,\ 60/feature_store/targets.csv \
        --derived /Volumes/Extreme\ SSD/trading_data/cex/training/\
            BINANCE_BTCUSDT.P,\ 60/feature_store/targets_with_hit_flags.csv \
        --sample 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True, help="Path to the baseline targets.csv")
    parser.add_argument("--derived", type=Path, required=True, help="Path to the derived targets file with hit flags")
    parser.add_argument("--sample", type=int, default=5, help="Number of rows to randomly sample for spot checks")
    parser.add_argument("--hit-prefix", default="y_hit_", help="Prefix identifying derived binary hit columns")
    parser.add_argument("--alpha", type=float, default=0.02, help="Return threshold for cross-checking log returns (optional)")
    return parser.parse_args()


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"CSV {path} missing 'timestamp' column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def _find_hit_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    return [c for c in df.columns if c.startswith(prefix)]


def _check_timestamp_alignment(original: pd.DataFrame, derived: pd.DataFrame) -> None:
    if len(original) != len(derived):
        raise AssertionError(f"Row count mismatch: original={len(original)}, derived={len(derived)}")
    if not original["timestamp"].equals(derived["timestamp"]):
        mismatches = (original["timestamp"] != derived["timestamp"])
        diff_indices = list(np.where(mismatches)[0][:10])
        raise AssertionError(f"Timestamp misalignment at indices: {diff_indices}")


def _summarize_hits(derived_hits: pd.DataFrame) -> pd.DataFrame:
    summary = derived_hits.agg(["count", "mean", "std"]).T
    summary = summary.rename(columns={"count": "non_nan", "mean": "hit_rate", "std": "std"})
    summary["non_nan"] = summary["non_nan"].astype(int)
    return summary


def _check_na_counts(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    return df[list(columns)].isna().sum()


def _parse_hit_meta(hit_col: str) -> Optional[Tuple[str, str, float]]:
    # expected pattern: y_hit_{direction}_{horizon}_ge_{threshold}
    parts = hit_col.split("_")
    if len(parts) < 6:
        return None
    direction = parts[2]
    horizon = parts[3]
    try:
        threshold = float(parts[-1])
    except ValueError:
        return None
    return direction, horizon, threshold


def _cross_check_returns(original: pd.DataFrame, derived: pd.DataFrame, hits: List[str], threshold: float) -> pd.DataFrame:
    logret_cols = [c for c in original.columns if c.startswith("y_logret_")]
    if not logret_cols:
        return pd.DataFrame()

    results = []
    for hit_col in hits:
        # Example naming: y_hit_up_24h_ge_0.03
        meta = _parse_hit_meta(hit_col)
        if not meta:
            continue
        direction, horizon, _ = meta
        candidate_logret_cols = [c for c in logret_cols if f"_{horizon}" in c]
        if not candidate_logret_cols:
            continue

        logret_col = candidate_logret_cols[0]
        hit_series = derived[hit_col]
        returns = original[logret_col]

        high_return = returns >= np.log(1 + threshold)
        low_return = returns <= np.log(1 - threshold)

        results.append(
            {
                "hit_col": hit_col,
                "horizon_logret": logret_col,
                "hit_rate": float(hit_series.mean(skipna=True)),
                "pct_high_return": float(high_return.mean()),
                "pct_low_return": float(low_return.mean()),
                "hit_and_high": float(((hit_series == 1) & high_return).mean()),
            }
        )

    return pd.DataFrame(results)


def _cross_check_mfe_mae(original: pd.DataFrame, derived: pd.DataFrame, hits: List[str]) -> pd.DataFrame:
    mfe_cols = {c for c in original.columns if c.startswith("y_mfe_")}
    mae_cols = {c for c in original.columns if c.startswith("y_mae_")}
    if not mfe_cols and not mae_cols:
        return pd.DataFrame()

    rows: List[dict] = []
    for hit_col in hits:
        meta = _parse_hit_meta(hit_col)
        if not meta:
            continue
        direction, horizon, threshold = meta

        mfe_col = f"y_mfe_{horizon}"
        mae_col = f"y_mae_{horizon}"

        if direction == "up":
            if mfe_col not in mfe_cols:
                continue
            excursion = original[mfe_col]
            mask = derived[hit_col] == 1
            rows.append(
                {
                    "hit_col": hit_col,
                    "excursion_col": mfe_col,
                    "threshold": threshold,
                    "median_excursion_when_hit": float(excursion[mask].median(skipna=True)),
                    "min_excursion_when_hit": float(excursion[mask].min(skipna=True)),
                    "pct_excursion_above_threshold": float((excursion >= threshold).mean()),
                    "pct_hit_implies_excursion": float(((excursion >= threshold) & mask).mean()),
                }
            )
        elif direction == "down":
            if mae_col not in mae_cols:
                continue
            excursion = original[mae_col]
            mask = derived[hit_col] == 1
            rows.append(
                {
                    "hit_col": hit_col,
                    "excursion_col": mae_col,
                    "threshold": threshold,
                    "median_excursion_when_hit": float(excursion[mask].median(skipna=True)),
                    "min_excursion_when_hit": float(excursion[mask].max(skipna=True)),
                    "pct_excursion_below_threshold": float((excursion <= -threshold).mean()),
                    "pct_hit_implies_excursion": float(((excursion <= -threshold) & mask).mean()),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    original = _load_csv(args.original)
    derived = _load_csv(args.derived)

    _check_timestamp_alignment(original, derived)

    hit_cols = _find_hit_columns(derived, args.hit_prefix)
    if not hit_cols:
        print("No hit columns found in derived file.")
        sys.exit(0)

    print(f"Found {len(hit_cols)} derived hit columns.")

    na_counts = _check_na_counts(derived, hit_cols)
    if na_counts.sum() > 0:
        print("Warning: NaNs detected in derived hit columns:")
        print(na_counts[na_counts > 0].sort_values(ascending=False).head(20))

    summary = _summarize_hits(derived[hit_cols])
    print("\nHit column summary (non-NaN count, mean, std):")
    print(summary.sort_values("hit_rate", ascending=False).head(20))

    cross_checks = _cross_check_returns(original, derived, hit_cols, args.alpha)
    if not cross_checks.empty:
        print("\nCross-check against log returns (approximate thresholding):")
        print(cross_checks.head(20))
    else:
        print("\nNo log-return columns available for cross-checks.")

    mfe_mae_checks = _cross_check_mfe_mae(original, derived, hit_cols)
    if not mfe_mae_checks.empty:
        print("\nCross-check against MFE/MAE excursions:")
        print(mfe_mae_checks.head(20))
    else:
        print("\nNo MFE/MAE columns available for cross-checks.")

    if args.sample > 0:
        sample_df = pd.concat([original, derived[hit_cols]], axis=1).sample(n=min(args.sample, len(original)), random_state=123)
        print("\nSample rows:")
        print(sample_df.head(args.sample))


if __name__ == "__main__":
    main()

