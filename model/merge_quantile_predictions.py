#!/usr/bin/env python3
"""
Merge quantile prediction CSVs from multiple run directories into a single
pred_val.csv or pred_test.csv suitable for plotting with plot_predictions_interactive.py.

Usage:
  python model/merge_quantile_predictions.py \
    --run-dirs /abs/run_dir_q05 /abs/run_dir_q10 ... \
    --split test \
    --out-dir /abs/merged_output_dir

This will write /abs/merged_output_dir/pred_test.csv (or pred_val.csv) with columns:
  timestamp (if present), y_true, pred_q05, pred_q10, ..., pred_q95
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _read_alpha_from_config(run_dir: Path) -> float | None:
    # Prefer pipeline_config.json (guaranteed by pipeline)
    cfg_path = run_dir / 'pipeline_config.json'
    if cfg_path.exists():
        with cfg_path.open('r') as f:
            cfg = json.load(f)
        obj = (cfg.get('target') or {}).get('objective') or {}
        if (obj.get('name') or '').lower() == 'quantile':
            params = obj.get('params') or {}
            alpha = params.get('alpha')
            if alpha is not None:
                return float(alpha)
    # Fallback: run_metadata.json may also contain full config
    meta_path = run_dir / 'run_metadata.json'
    if meta_path.exists():
        with meta_path.open('r') as f:
            meta = json.load(f)
        cfg = meta.get('config') or {}
        obj = (cfg.get('target') or {}).get('objective') or {}
        if (obj.get('name') or '').lower() == 'quantile':
            params = obj.get('params') or {}
            alpha = params.get('alpha')
            if alpha is not None:
                return float(alpha)
    return None


def _format_q_label(alpha: float) -> str:
    # Map e.g. 0.05 -> 'q05', 0.5 -> 'q50', 0.95 -> 'q95'
    pct = int(round(alpha * 100))
    return f"q{pct:02d}"


def _load_pred_csv(run_dir: Path, split: str) -> pd.DataFrame:
    csv_path = run_dir / f'pred_{split}.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {csv_path}")
    df = pd.read_csv(csv_path)
    # Normalize timestamp if present
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        df['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    else:
        df['__row_index__'] = range(len(df))
    return df


def merge_quantile_runs(run_dirs: List[Path], split: str) -> pd.DataFrame:
    # Load all and align by timestamp if present; otherwise by row index
    frames: List[Tuple[str, pd.DataFrame]] = []
    for rd in run_dirs:
        alpha = _read_alpha_from_config(rd)
        if alpha is None:
            raise ValueError(f"Could not determine quantile alpha for run_dir: {rd}")
        qlabel = _format_q_label(alpha)
        df = _load_pred_csv(rd, split)
        # Rename y_pred to pred_qXX and keep y_true
        if 'y_pred' not in df.columns or 'y_true' not in df.columns:
            raise ValueError(f"pred_{split}.csv in {rd} must contain 'y_true' and 'y_pred'")
        df = df[['timestamp', 'y_true', 'y_pred']] if 'timestamp' in df.columns else df[['__row_index__', 'y_true', 'y_pred']]
        df = df.rename(columns={'y_pred': f'pred_{qlabel}'})
        frames.append((qlabel, df))

    # Sort by quantile for stable column order
    frames.sort(key=lambda t: int(t[0][1:]))

    # Start with the first frame as base
    base = frames[0][1].copy()
    for qlabel, df in frames[1:]:
        if 'timestamp' in base.columns:
            base = base.merge(df, on=['timestamp', 'y_true'], how='inner')
        else:
            base = base.merge(df, on=['__row_index__', 'y_true'], how='inner')

    # Clean helper column
    if '__row_index__' in base.columns:
        base = base.drop(columns=['__row_index__'])

    return base


def main() -> None:
    parser = argparse.ArgumentParser(description='Merge quantile prediction CSVs from multiple run dirs.')
    parser.add_argument('--run-dirs', nargs='+', type=Path, required=True, help='Paths to run directories to merge (quantile runs).')
    parser.add_argument('--split', choices=['val', 'test'], default='test', help='Which split to merge (pred_val or pred_test).')
    parser.add_argument('--out-dir', type=Path, required=True, help='Output directory to write merged pred_<split>.csv')
    args = parser.parse_args()

    run_dirs = [p.resolve() for p in args.run_dirs]
    for p in run_dirs:
        if not p.exists():
            raise FileNotFoundError(f"Run directory not found: {p}")

    merged = merge_quantile_runs(run_dirs, args.split)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / f'pred_{args.split}.csv'
    merged.to_csv(out_csv, index=False)
    print(f"Wrote merged predictions: {out_csv}")


if __name__ == '__main__':
    main()


