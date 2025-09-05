#!/usr/bin/env python3
"""
Inspect hyperparameter tuning results and report the best iteration for the
best params (lowest best_mean) per quantile, scoped by target and input_data.

Usage:
  python model/inspect_tuning_best_iterations.py \
    --models-root "/abs/models_root" \
    --target y_logret_24h \
    --input-data "/abs/training/merged_features_targets.csv" \
    [--metric pinball_loss] [--out /abs/report.csv]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


def _read_basic_config(run_dir: Path) -> Tuple[Optional[str], Optional[str], Optional[float], Optional[str]]:
    """Return (target_variable, objective_name_lower, alpha, input_data_path)."""
    cfg_path = run_dir / 'pipeline_config.json'
    if cfg_path.exists():
        try:
            with cfg_path.open('r') as f:
                cfg = json.load(f)
            target = ((cfg.get('target') or {}).get('variable'))
            objective_name = ((cfg.get('target') or {}).get('objective') or {}).get('name')
            alpha = ((cfg.get('target') or {}).get('objective') or {}).get('params') or {}
            alpha_val = alpha.get('alpha')
            input_data = cfg.get('input_data')
            return target, (objective_name.lower() if isinstance(objective_name, str) else None), (float(alpha_val) if alpha_val is not None else None), str(input_data) if input_data is not None else None
        except Exception:
            pass
    # Fallback to run_metadata.json
    meta_path = run_dir / 'run_metadata.json'
    if meta_path.exists():
        try:
            with meta_path.open('r') as f:
                meta = json.load(f)
            cfg = meta.get('config') or {}
            target = ((cfg.get('target') or {}).get('variable'))
            objective_name = ((cfg.get('target') or {}).get('objective') or {}).get('name')
            alpha = ((cfg.get('target') or {}).get('objective') or {}).get('params') or {}
            alpha_val = alpha.get('alpha')
            input_data = cfg.get('input_data')
            return target, (objective_name.lower() if isinstance(objective_name, str) else None), (float(alpha_val) if alpha_val is not None else None), str(input_data) if input_data is not None else None
        except Exception:
            pass
    return None, None, None, None


def _format_q_label(alpha: float) -> str:
    pct = int(round(alpha * 100))
    return f"q{pct:02d}"


def discover_runs(models_root: Path, target: str, input_data: str) -> List[Path]:
    if not models_root.exists() or not models_root.is_dir():
        raise FileNotFoundError(f"models_root does not exist or is not a directory: {models_root}")
    matches: List[Path] = []
    for child in sorted(models_root.iterdir()):
        if not child.is_dir():
            continue
        cfg_target, obj_name, alpha, cfg_input = _read_basic_config(child)
        if cfg_target != target or obj_name != 'quantile' or cfg_input != input_data:
            continue
        if not (child / 'tuning_trials.csv').exists():
            continue
        matches.append(child)
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description='Report best_iteration for best params per quantile run.')
    parser.add_argument('--models-root', type=Path, required=True, help='Root directory containing run_* folders')
    parser.add_argument('--target', type=str, required=True, help='Target variable to match (e.g., y_logret_24h)')
    parser.add_argument('--input-data', type=str, required=True, help='Exact input_data path to match')
    parser.add_argument('--metric', type=str, default='pinball_loss', help='Metric column to consider from tuning_trials.csv (default: pinball_loss)')
    parser.add_argument('--out', type=Path, required=False, help='Optional CSV output path for the report')
    args = parser.parse_args()

    runs = discover_runs(args.models_root.resolve(), args.target, str(args.input_data))
    if not runs:
        raise SystemExit('No matching quantile runs found for the given models_root/target/input_data')

    records = []
    for rd in runs:
        target, obj, alpha, cfg_input = _read_basic_config(rd)
        if alpha is None:
            continue
        qlabel = _format_q_label(alpha)
        trials_path = rd / 'tuning_trials.csv'
        try:
            trials = pd.read_csv(trials_path)
        except Exception as e:
            print(f"Warning: could not read {trials_path}: {e}")
            continue
        # Filter by metric if column present
        dfm = trials.copy()
        if 'metric' in dfm.columns:
            dfm = dfm[dfm['metric'] == args.metric]
        if dfm.empty:
            continue
        # Choose best by minimal best_mean
        best_idx = dfm['best_mean'].idxmin()
        row = dfm.loc[best_idx]
        records.append({
            'quantile': qlabel,
            'alpha': float(alpha),
            'run_dir': rd.name,
            'trial': int(row.get('trial')) if not pd.isna(row.get('trial')) else None,
            'best_iteration': int(row.get('best_iteration')) if not pd.isna(row.get('best_iteration')) else None,
            'best_mean': float(row.get('best_mean')) if not pd.isna(row.get('best_mean')) else None,
            'best_stdv': float(row.get('best_stdv')) if not pd.isna(row.get('best_stdv')) else None,
            'n_folds': int(row.get('n_folds')) if not pd.isna(row.get('n_folds')) else None,
        })

    if not records:
        raise SystemExit('No records collected; check metric name and runs.')

    report = pd.DataFrame(records).sort_values(['alpha']).reset_index(drop=True)
    # Console print
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(report.to_string(index=False))

    # Optional CSV
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.out, index=False)
        print(f"Wrote report: {args.out}")


if __name__ == '__main__':
    main()


