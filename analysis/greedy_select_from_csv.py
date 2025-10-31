#!/usr/bin/env python3
"""
Greedy feature selection from a feature CSV (no MLflow).

This script:
- Loads a feature matrix from a CSV file.
- (Optional) filters rows by a time window using the 'timestamp' column.
- Reads an initial feature list (JSON array) to seed the greedy process.
- Runs target-free greedy de-duplication using absolute Spearman correlation,
  with a per-family cap (families defined by stripping timeframe suffixes).
- (Optional) greedily adds more features from the rest of X in unsupervised
  quality order, enforcing the same tau and family cap.
- Saves the selected features to a JSON file.

Usage example:
  python analysis/greedy_select_from_csv.py \
    --features-csv /path/to/features.csv \
    --initial-features-json configs/feature_lists/handpicked.json \
    --start-ts 2023-01-01T00:00:00 --end-ts 2025-08-01T00:00:00 \
    --tau 0.90 --cap-per-family 2 --min-overlap 200 \
    --include-rest --target-total 120 --nan-penalty 2.0 \
    --output configs/feature_lists/handpicked_selected.json

Notes:
- Default exclusion: base names ending with '_all_tf_normalized' are skipped.
  Disable with --no-default-exclude or add more via --exclude-suffix.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd


def _add_repo_root_to_path() -> None:
    from pathlib import Path as _P
    root = _P.cwd()
    for _ in range(6):
        if (root / "utils").exists():
            break
        if root.parent == root:
            break
        root = root.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_add_repo_root_to_path()

try:
    from utils.greedy_feature_selector import (
        GreedyParams,
        greedy_select,
        greedy_expand_from_rest,
    )
except Exception as e:  # pragma: no cover
    raise SystemExit(
        f"Failed to import selector utilities. Ensure you run from repo or set PYTHONPATH. Error: {e}"
    )


def load_feature_matrix(
    features_csv: Path,
    *,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(features_csv)
    # Optional time-window filter by 'timestamp'
    if 'timestamp' in df.columns and (start_ts or end_ts):
        ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        ts = ts.dt.tz_convert('UTC').dt.tz_localize(None)
        df = df.copy()
        df['timestamp'] = ts
        if start_ts:
            try:
                st = pd.Timestamp(start_ts)
                df = df[df['timestamp'] >= st]
            except Exception:
                pass
        if end_ts:
            try:
                et = pd.Timestamp(end_ts)
                df = df[df['timestamp'] <= et]
            except Exception:
                pass

    # Build X: drop timestamp and target-like columns; coerce numeric
    drop_cols = [c for c in df.columns if str(c) == 'timestamp' or str(c).startswith('y_')]
    X = df.drop(columns=drop_cols, errors='ignore')
    X = X.apply(pd.to_numeric, errors='coerce')
    return df, X


def main() -> None:
    ap = argparse.ArgumentParser(description="Greedy feature selection from a CSV (no MLflow)")
    ap.add_argument("--features-csv", type=Path, required=True, help="Path to features CSV")
    ap.add_argument("--initial-features-json", type=Path, required=True, help="JSON array of initial feature names")
    ap.add_argument("--start-ts", type=str, default=None, help="ISO start timestamp to filter rows (by 'timestamp')")
    ap.add_argument("--end-ts", type=str, default=None, help="ISO end timestamp to filter rows (by 'timestamp')")

    # Greedy params
    ap.add_argument("--tau", type=float, default=0.90, help="Absolute Spearman correlation threshold")
    ap.add_argument("--cap-per-family", type=int, default=2, help="Max kept per base family")
    ap.add_argument("--min-overlap", type=int, default=200, help="Min pairwise overlap to accept rho")
    ap.add_argument("--known-tfs", nargs="*", default=["1H", "4H", "12H", "1D"], help="Timeframe suffix tokens")
    ap.add_argument("--exclude-suffix", action="append", default=None, help="Suffix to exclude (repeatable)")
    ap.add_argument("--no-default-exclude", action="store_true", help="Do not exclude '_all_tf_normalized' by default")

    # Expand-from-rest options
    ap.add_argument("--include-rest", action="store_true", help="After seeding, greedily add from rest of X")
    ap.add_argument("--target-total", type=int, default=None, help="Optional total feature target when including rest")
    ap.add_argument("--nan-penalty", type=float, default=1.0, help="Exponent for (1-NaN_rate) in rest ordering")
    ap.add_argument("--restrict-features-json", type=Path, default=None, help="Whitelisted features for rest additions")

    # Output
    ap.add_argument("--output", type=Path, required=True, help="Path to write selected features JSON list")
    ap.add_argument("--print-limit", type=int, default=50, help="Print only first N selected")

    args = ap.parse_args()

    # Load data
    _, X = load_feature_matrix(args.features_csv, start_ts=args.start_ts, end_ts=args.end_ts)
    # Load initial order
    try:
        initial_order = json.loads(args.initial_features_json.read_text())
    except Exception as e:
        raise SystemExit(f"Failed to read initial features JSON: {e}")

    # Keep only those present in X
    initial_order = [c for c in initial_order if c in X.columns]
    if not initial_order:
        raise SystemExit("Initial feature list has no overlap with CSV columns.")

    # Exclusion suffixes
    exclude_suffixes = args.exclude_suffix or []
    if not args.no_default_exclude and "_all_tf_normalized" not in exclude_suffixes:
        exclude_suffixes.append("_all_tf_normalized")

    params = GreedyParams(
        tau=args.tau,
        cap_per_family=args.cap_per_family,
        min_overlap=args.min_overlap,
        known_tfs=args.known_tfs,
        exclude_suffixes=exclude_suffixes,
    )

    # Greedy prune the seed list
    keep = greedy_select(X, initial_order, params)

    # Optionally expand from the rest (restricted to whitelist if provided)
    restrict_list = None
    if args.restrict_features_json:
        try:
            restrict_list = json.loads(args.restrict_features_json.read_text())
        except Exception as e:
            print(f"Warning: failed to read restrict list JSON: {e}")

    if args.include_rest:
        keep = greedy_expand_from_rest(
            X,
            keep,
            params,
            order_known_tfs=args.known_tfs,
            nan_penalty=args.nan_penalty,
            target_total=args.target_total,
            restrict_features=restrict_list,
        )

    # Write selected
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(keep, indent=2))

    # Print summary
    n = len(keep)
    print(f"Selected features: {n}")
    if n:
        for f in keep[: max(0, int(args.print_limit))]:
            print(f)
        if n > int(args.print_limit):
            print(f"... ({n - int(args.print_limit)} more)")


if __name__ == "__main__":  # pragma: no cover
    main()

