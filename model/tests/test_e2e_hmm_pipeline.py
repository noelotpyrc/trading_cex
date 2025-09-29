#!/usr/bin/env python3
from __future__ import annotations

"""
E2E test for model/run_hmm_pipeline.py

Uses sampled rows from the external hmm_features.csv as input, writes outputs
to a temporary folder under the project, and verifies core artifacts exist.

Skips gracefully if hmmlearn is not installed or the source CSV is missing.
"""

import json
import os
import sys
import glob
from datetime import datetime
from pathlib import Path
import subprocess


def _proj_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    root = _proj_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Check hmmlearn availability
    try:
        import hmmlearn  # noqa: F401
    except ModuleNotFoundError:
        print('SKIP: hmmlearn not installed')
        return

    # Choose python executable
    venv_python = root / 'venv' / 'bin' / 'python'
    PY = str(venv_python) if venv_python.exists() else sys.executable

    # Source CSV path (combined features file)
    src = Path('/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/hmm_features.csv')
    if not src.exists():
        print(f'SKIP: source features CSV not found: {src}')
        return

    # Read header to pick columns
    import pandas as pd
    try:
        df = pd.read_csv(src)
    except Exception as e:
        print(f'SKIP: failed to read source CSV: {e}')
        return

    if 'timestamp' not in df.columns:
        print("SKIP: source CSV missing 'timestamp' column")
        return

    # Prefer v1 columns if available; else v2; else pick any few numeric feature columns
    v1 = ['close_logret_current_1H', 'log_volume_delta_current_1H', 'close_parkinson_20_1H']
    v2 = ['close_ret_zscore_20_1H', 'volume_rvol_20_1H', 'close_over_vwap_1H', 'close_log_ratio_vwap_1H', 'high_low_range_pct_current_1H', 'close_open_pct_current_1H']
    cols_present_v1 = [c for c in v1 if c in df.columns]
    cols_present_v2 = [c for c in v2 if c in df.columns]

    feature_cols = cols_present_v1 or cols_present_v2
    if not feature_cols:
        # Fallback: pick first 3 numeric columns excluding timestamp
        num_cols = [c for c in df.columns if c != 'timestamp']
        feature_cols = num_cols[:3]
    if not feature_cols:
        print('SKIP: no usable feature columns found in source CSV')
        return

    # Sample smaller subset to speed up e2e
    keep = ['timestamp'] + feature_cols
    df_small = df[keep].dropna()
    if len(df_small) > 2000:
        df_small = df_small.iloc[-2000:].reset_index(drop=True)

    # Persist under project .tmp folder
    ts_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    tmp_dir = root / '.tmp' / f'hmm_e2e_{ts_tag}'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    sample_csv = tmp_dir / 'hmm_features_sample.csv'
    df_small.to_csv(sample_csv, index=False)

    # Build a mock prep_metadata.json to align HMM train-only fitting with a
    # realistic LightGBM-style split, without depending on external files.
    # We'll use a 70/15/15 time-ordered split over the sampled rows.
    ts_series = pd.to_datetime(df_small['timestamp'], errors='coerce', utc=True)
    ts_naive = ts_series.dt.tz_convert('UTC').dt.tz_localize(None)
    df_small['timestamp'] = ts_naive
    n = len(df_small)
    i1 = int(n * 0.70)
    i2 = int(n * 0.85)
    split_timestamps = {
        'train': df_small.iloc[:i1]['timestamp'].astype(str).tolist(),
        'val': df_small.iloc[i1:i2]['timestamp'].astype(str).tolist(),
        'test': df_small.iloc[i2:]['timestamp'].astype(str).tolist(),
    }
    mock_meta = {
        'split_timestamps': split_timestamps
    }
    mock_meta_path = tmp_dir / 'mock_prep_metadata.json'
    with open(mock_meta_path, 'w') as f:
        json.dump(mock_meta, f)
    cfg = {
        'input_data': str(sample_csv),
        'output_dir': str(tmp_dir / 'hmm_out'),
        'features': {
            'columns': feature_cols,
        },
        'split': {'prep_metadata': str(mock_meta_path)},
        'model': {
            'state_grid': [2, 3],
            'covariance_type': 'diag',
            'n_iter': 50,
            'tol': 1e-3,
            'random_state': 42,
        }
    }
    cfg_path = tmp_dir / 'hmm_cfg.json'
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f)

    # Run pipeline
    script = root / 'model' / 'run_hmm_pipeline.py'
    cmd = [PY, str(script), '--config', str(cfg_path), '--log-level', 'INFO']
    print('Running:', ' '.join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print('ERROR: pipeline failed:', e)
        raise

    # Verify outputs
    out_root = Path(cfg['output_dir'])
    runs = sorted(glob.glob(str(out_root / 'run_*')))
    assert runs, 'No run_* directory created'
    run_dir = Path(runs[-1])
    expect = ['model.joblib', 'scaler.joblib', 'config.json', 'metrics.json', 'regimes.csv']
    for name in expect:
        p = run_dir / name
        assert p.exists(), f'Missing artifact: {p}'

    # regimes length should equal input length
    reg = pd.read_csv(run_dir / 'regimes.csv')
    assert len(reg) == len(df_small), 'regimes length mismatch'
    # split map should be saved
    split_map_path = run_dir / 'split_map.csv'
    assert split_map_path.exists(), 'Missing split_map.csv'
    sm = pd.read_csv(split_map_path)
    assert 'split' in sm.columns and 'timestamp' in sm.columns
    uniq = set(sm['split'].unique().tolist())
    assert uniq.issubset({'train','val','test'}), f'Unexpected split labels: {uniq}'
    # If we used prep_metadata, ensure the config recorded it and at least one label exists
    with open(run_dir / 'config.json','r') as f:
        saved_cfg = json.load(f)
    assert 'prep_metadata' in saved_cfg.get('split', {}), 'prep_metadata not recorded in saved config'
    assert len(sm) == len(reg), 'split_map and regimes length mismatch'
    print(f'Artifacts persisted under: {tmp_dir}')
    print(f'Artifacts persisted under: {tmp_dir}')

    print('hmm_pipeline e2e test OK')


if __name__ == '__main__':
    main()
