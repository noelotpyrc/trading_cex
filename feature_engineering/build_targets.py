"""
Process OHLCV data and generate forward-window target variables.

Usage examples:
  # Basic: 1H data, horizons 3,6,12,24 bars, TP/SL pairs 1.5%/1.0%
  ./venv/bin/python feature_engineering/build_targets.py \
    --input /path/to/ohlcv.parquet \
    --base-dir '/Volumes/Extreme SSD/trading_data/cex/targets' \
    --dataset 'BINANCE_BTCUSDT.P, 60' \
    --output targets.csv \
    --freq 1H \
    --horizons 3 6 12 24 \
    --barriers 0.015:0.01

Notes:
  - Labels are leakage-safe: only use (t+1..t+H] forward data and entry at t (or t+1 open).
  - Uses only OHLCV-forward window (no fees/slippage/ATR scaling).
  - Output includes a 'timestamp' column when the input has a DatetimeIndex or a 'timestamp' column;
    otherwise a 'row' column is included. Labels may be NaN near the end where forward data is insufficient.
  - Default write format is CSV. To write Parquet, use an output path ending with .parquet (requires pyarrow/fastparquet).
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Local imports
from targets import (
    TargetGenerationConfig,
    generate_targets_for_row,
    extract_forward_window,
)
from utils import validate_ohlcv_data


def _read_ohlcv(input_path: str) -> pd.DataFrame:
    lower = input_path.lower()
    if lower.endswith(".parquet") or lower.endswith(".pq"):
        df = pd.read_parquet(input_path)
    elif lower.endswith(".csv"):
        df = pd.read_csv(input_path)
    else:
        raise ValueError("Unsupported input format. Use .parquet or .csv")

    # Normalize columns to lower-case OHLCV if possible
    colmap: Dict[str, str] = {}
    for c in df.columns:
        lc = str(c).lower()
        if lc in {"open", "high", "low", "close", "volume"}:
            colmap[c] = lc
    df = df.rename(columns=colmap)

    # Use 'timestamp' or 'time' column as index if present
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            if "timestamp" in df.columns:
                ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
                df["timestamp"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)
                df = df.set_index("timestamp")
            elif "time" in df.columns:
                ts = pd.to_datetime(df["time"], errors="coerce", utc=True)
                df["timestamp"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)
                df = df.drop(columns=["time"])
                df = df.set_index("timestamp")
        except Exception:
            # keep as-is if conversion fails
            pass
    return df


def _parse_barrier_pairs(pairs: Optional[List[str]]) -> Optional[List[Tuple[float, float]]]:
    if not pairs:
        return None
    out: List[Tuple[float, float]] = []
    for p in pairs:
        if ":" not in p:
            raise ValueError(f"Barrier pair must be 'tp:sl', got '{p}'")
        a, b = p.split(":", 1)
        tp = float(a)
        sl = float(b)
        if tp <= 0.0 or sl <= 0.0:
            raise ValueError(f"Barrier values must be > 0, got '{p}'")
        out.append((tp, sl))
    return out


def _horizon_labels_for_freq(freq: Optional[str], horizons_bars: List[int]) -> Optional[Dict[int, str]]:
    if not freq:
        return None
    f = str(freq).upper()
    labels: Dict[int, str] = {}
    if f.endswith("H"):
        for h in horizons_bars:
            labels[h] = f"{h}h"
    elif f in ("T", "MIN", "MINUTE") or f.endswith("T"):
        # Pandas alias: 'T' is minute
        for h in horizons_bars:
            labels[h] = f"{h}min"
    elif f.endswith("D"):
        for h in horizons_bars:
            labels[h] = f"{h}d"
    else:
        # Fallback: bars
        for h in horizons_bars:
            labels[h] = f"{h}b"
    return labels


def process_targets(
    input_path: str,
    output_path: str,
    horizons_bars: List[int],
    barrier_pairs: Optional[List[Tuple[float, float]]],
    *,
    freq: Optional[str] = None,
    entry: str = "close_t",
    tie_policy: str = "conservative",
    log_every: int = 500,
) -> pd.DataFrame:
    df = _read_ohlcv(input_path)
    if not validate_ohlcv_data(df):
        raise ValueError("Input OHLCV data is invalid. Ensure columns open, high, low, close, volume.")

    horizons_bars_sorted = sorted(set(int(x) for x in horizons_bars))
    horizon_labels = _horizon_labels_for_freq(freq, horizons_bars_sorted)
    cfg = TargetGenerationConfig(
        horizons_bars=horizons_bars_sorted,
        barrier_pairs=barrier_pairs,
        tie_policy=tie_policy,  # type: ignore[arg-type]
        horizon_labels=horizon_labels,
        include_returns=True,
        include_mfe_mae=True,
        include_barriers=bool(barrier_pairs),
        log_returns=True,
    )

    max_h = max(horizons_bars_sorted) if horizons_bars_sorted else 0
    rows: List[Dict[str, float]] = []
    idx_values: List[pd.Timestamp | int] = []

    # Ensure numeric dtype
    df = df.copy()
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    n = len(df)
    for i in range(n):
        # Entry price selection
        if entry == "close_t":
            entry_price = float(df['close'].iloc[i])
        elif entry == "open_t1":
            if i + 1 < n and np.isfinite(df['open'].iloc[i + 1]):
                entry_price = float(df['open'].iloc[i + 1])
            else:
                entry_price = np.nan
        else:
            raise ValueError("entry must be one of {'close_t','open_t1'}")

        fwd = extract_forward_window(df[['open','high','low','close','volume']], i, max_h)
        res = generate_targets_for_row(fwd, entry_price, cfg)
        rows.append(res)
        idx_values.append(df.index[i] if isinstance(df.index, pd.DatetimeIndex) else i)

        if log_every and ((i + 1) % log_every == 0 or (i + 1) == n):
            print(f"Processed {i+1}/{n} rows")

    out_df = pd.DataFrame(rows, index=idx_values)
    # Always include an explicit 'timestamp' column when possible
    if isinstance(df.index, pd.DatetimeIndex):
        out_df.index.name = 'timestamp'
        out_df = out_df.reset_index()
    else:
        out_df.index.name = 'row'
        # materialize index as a 'row' column
        out_df = out_df.reset_index()
        # If original data had a 'timestamp' column, include it explicitly
        if 'timestamp' in df.columns:
            try:
                ts_col = pd.to_datetime(df['timestamp'], errors='coerce')
                # Ensure 'timestamp' is the first column
                if 'timestamp' in out_df.columns:
                    out_df = out_df.drop(columns=['timestamp'])
                out_df.insert(0, 'timestamp', ts_col.values)
            except Exception:
                # Keep only 'row' when timestamp cannot be parsed
                pass

    # Save (CSV default; Parquet only if explicitly requested)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lower = output_path.lower()
    if lower.endswith('.parquet') or lower.endswith('.pq'):
        out_df.to_parquet(output_path, index=False)
    else:
        if not lower.endswith('.csv'):
            # enforce CSV by default if no recognized suffix
            output_path = output_path + '.csv'
        out_df.to_csv(output_path, index=False)
    print(f"Wrote targets: {output_path} (rows={len(out_df)}, cols={len(out_df.columns)})")
    return out_df


def main():
    parser = argparse.ArgumentParser(description='Process OHLCV to forward-window target variables')
    parser.add_argument('--input', required=True, help='Path to OHLCV file (.parquet or .csv)')
    parser.add_argument('--output', required=True, help='Output file name or path (.csv default, .parquet optional)')
    parser.add_argument('--base-dir', default="/Volumes/Extreme SSD/trading_data/cex/targets", help='Base directory for outputs (used when --dataset is provided)')
    parser.add_argument('--dataset', default=None, help='Dataset folder name under base dir (e.g., BINANCE_BTCUSDT.P, 60)')
    parser.add_argument('--freq', default=None, help='Native bar frequency (e.g., 1H, 15T, 1D) for horizon labels')
    parser.add_argument('--horizons', nargs='+', type=int, default=[3, 6, 12, 24], help='Horizons in bars')
    parser.add_argument('--barriers', nargs='*', default=None, help="Barrier pairs as 'tp:sl' (e.g., 0.015:0.01)")
    parser.add_argument('--entry', choices=['close_t', 'open_t1'], default='close_t', help='Entry price convention')
    parser.add_argument('--tie-policy', choices=['conservative', 'proximity_to_open'], default='conservative')
    parser.add_argument('--log-every', type=int, default=500)
    args = parser.parse_args()

    barrier_pairs = _parse_barrier_pairs(args.barriers)
    # Compose output path similar to process_lookbacks: base_dir/dataset/output
    if args.dataset:
        out_path = os.path.join(args.base_dir, args.dataset, args.output)
    else:
        out_path = args.output
    process_targets(
        input_path=args.input,
        output_path=out_path,
        horizons_bars=args.horizons,
        barrier_pairs=barrier_pairs,
        freq=args.freq,
        entry=args.entry,
        tie_policy=args.tie_policy,
        log_every=args.log_every,
    )


if __name__ == '__main__':
    main()


