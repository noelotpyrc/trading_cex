#!/usr/bin/env python3
"""
HMM training pipeline runner (lean, config-driven), similar style to run_lgbm_pipeline.py.

Usage:
  python model/run_hmm_pipeline.py --config configs/model_configs/hmm_v1_1h.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import shutil
from sklearn.preprocessing import StandardScaler

# Hydra support removed – legacy CLI only
DictConfig = Any  # type: ignore
OmegaConf = None  # type: ignore
hydra = None  # type: ignore


def setup_logging(log_level: str = "INFO", logfile: Path | None = None) -> None:
    """Initialize logging to stdout and optional logfile."""
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(logfile)))
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )


def load_config(path: Path) -> Dict[str, Any]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, 'r') as f:
        cfg = json.load(f)

    for key in ['input_data', 'output_dir', 'split', 'model']:
        if key not in cfg:
            raise KeyError(f"Missing config key: {key}")

    cfg['input_data'] = Path(cfg['input_data'])
    cfg['output_dir'] = Path(cfg['output_dir'])

    feature_cfg = cfg.get('features')
    include = None
    if isinstance(feature_cfg, dict):
        include = feature_cfg.get('include')
    elif isinstance(feature_cfg, str):
        feature_path = Path(feature_cfg)
        if not feature_path.is_absolute():
            feature_path = Path(__file__).resolve().parent.parent / feature_path
        with open(feature_path, 'r') as f_list:
            include = json.load(f_list)
    elif feature_cfg is None:
        default_list = Path(__file__).resolve().parent.parent / 'configs' / 'feature_lists' / 'binance_btcusdt_p60_hmm_1h.json'
        with open(default_list, 'r') as f_list:
            include = json.load(f_list)
    else:
        raise ValueError("features must be dict or path to JSON list")

    cfg['features'] = {'include': include}

    split = cfg['split']
    split.setdefault('train_ratio', 0.7)
    split.setdefault('val_ratio', 0.15)
    split.setdefault('test_ratio', 0.15)
    model = cfg['model']
    model.setdefault('covariance_type', 'diag')
    model.setdefault('n_iter', 200)
    model.setdefault('tol', 1e-3)
    model.setdefault('random_state', 42)
    # Optional numeric stability and sticky init
    model.setdefault('reg_covar', None)
    model.setdefault('sticky_diag', None)
    # Normalize state_grid from min/max range if provided
    sg = model.get('state_grid')
    if isinstance(sg, dict):
        kmin = int(sg.get('min'))
        kmax = int(sg.get('max'))
        if kmin < 1 or kmax < kmin:
            raise ValueError(f"Invalid state_grid range: {sg}")
        model['state_grid'] = list(range(kmin, kmax + 1))

    # Selection defaults (HMM-appropriate)
    sel = cfg.get('selection') or {}
    sel.setdefault('criterion', 'bic')  # 'icl' preferred for stricter selection; default 'bic' for backward comp
    sel.setdefault('restarts', 1)
    sel.setdefault('delta_threshold', None)  # e.g., 10.0 for elbow rule
    sel.setdefault('min_state_occupancy_pct', 0.0)  # e.g., 0.02 to require >=2% per state
    sel.setdefault('cv_folds', 0)  # blocked CV folds on train (>=2 to enable)
    sel.setdefault('one_std_rule', False)  # if cv enabled, pick smallest K within 1 std err of best
    cfg['selection'] = sel
    return cfg


def select_feature_columns(df_cols: List[str], cfg: Dict[str, Any]) -> List[str]:
    feats = cfg['features']
    include = feats.get('include')
    if not include:
        raise ValueError("features.include must provide a non-empty list of columns")
    present = [c for c in include if c in df_cols]
    if not present:
        raise ValueError("None of the requested feature columns are present in the input data")
    return present


def load_features(path: Path, selected_cols: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'timestamp' not in df.columns:
        raise ValueError("Features CSV must include 'timestamp'")
    ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    df['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    keep = ['timestamp'] + [c for c in selected_cols if c in df.columns]
    df = df[keep].dropna().sort_values('timestamp').reset_index(drop=True)
    return df


def time_split_indices(ts: pd.Series, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("ratios must sum to 1.0")
    n = len(ts)
    i1 = int(n * train_ratio)
    i2 = int(n * (train_ratio + val_ratio))
    train_idx = np.arange(0, i1)
    val_idx = np.arange(i1, i2)
    test_idx = np.arange(i2, n)
    return train_idx, val_idx, test_idx


def load_split_indices_from_meta(ts: pd.Series, meta_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load split indices from a LightGBM prepare_splits prep_metadata.json file.

    Matches by exact string representation of timestamps to avoid tz pitfalls.
    """
    meta = json.load(open(meta_path, 'r'))
    split_ts = meta.get('split_timestamps', {})
    train_set = set(split_ts.get('train', []) or [])
    val_set = set(split_ts.get('val', []) or [])
    test_set = set(split_ts.get('test', []) or [])
    ts_str = ts.astype(str).tolist()
    train_idx = np.array([i for i, s in enumerate(ts_str) if s in train_set], dtype=int)
    val_idx = np.array([i for i, s in enumerate(ts_str) if s in val_set], dtype=int)
    test_idx = np.array([i for i, s in enumerate(ts_str) if s in test_set], dtype=int)
    return train_idx, val_idx, test_idx


def hmm_param_count(n_states: int, n_features: int, covariance_type: str = 'diag') -> int:
    startprob = n_states - 1
    transmat = n_states * (n_states - 1)
    means = n_states * n_features
    if covariance_type == 'full':
        cov = n_states * (n_features * (n_features + 1) // 2)
    else:
        cov = n_states * n_features
    return startprob + transmat + means + cov


def _sticky_transmat(n_states: int, diag_weight: float) -> np.ndarray:
    # Create a transition matrix with high self-transition probability
    diag_weight = float(min(max(diag_weight, 0.0), 0.999))
    off = (1.0 - diag_weight)
    if n_states > 1:
        off_each = off / (n_states - 1)
    else:
        off_each = 0.0
    T = np.full((n_states, n_states), off_each, dtype=float)
    np.fill_diagonal(T, diag_weight)
    return T


def fit_and_score(
    X: np.ndarray,
    n_states: int,
    covariance_type: str,
    n_iter: int,
    tol: float,
    random_state: int,
    *,
    reg_covar: float | None = None,
    sticky_diag: float | None = None,
) -> Tuple[GaussianHMM, float, float, float, float, float, List[float]]:
    # Configure model
    kwargs: Dict[str, Any] = dict(
        n_components=int(n_states),
        covariance_type=covariance_type,
        n_iter=int(n_iter),
        tol=float(tol),
        random_state=int(random_state),
    )
    if reg_covar is not None:
        # hmmlearn uses min_covar
        kwargs['min_covar'] = float(reg_covar)
    # If we want to preserve custom transmat_, exclude 't' from init_params
    init_params = 'stmc'
    if sticky_diag is not None:
        init_params = 'smc'
    model = GaussianHMM(init_params=init_params, **kwargs)
    if sticky_diag is not None:
        model.transmat_ = _sticky_transmat(int(n_states), float(sticky_diag))
    # Fit
    model.fit(X)
    # Train log-likelihood
    ll = float(model.score(X))
    # Information criteria
    p = hmm_param_count(int(n_states), X.shape[1], covariance_type)
    n = X.shape[0]
    bic = -2.0 * ll + p * np.log(n)
    aic = -2.0 * ll + 2.0 * p
    # Classification entropy and ICL
    gamma = model.predict_proba(X)
    eps = 1e-12
    H = float(-np.sum(gamma * np.log(np.clip(gamma, eps, 1.0))))
    icl = bic - 2.0 * H
    # Expected state occupancy (fraction)
    occ = list(np.sum(gamma, axis=0) / float(n))
    return model, ll, bic, aic, icl, H, occ


def select_n_states(
    X_train: np.ndarray,
    X_val: np.ndarray,
    model_cfg: Dict[str, Any],
    sel_cfg: Dict[str, Any],
) -> Tuple[int, Dict[str, Any]]:
    # If CV is enabled, use blocked CV over the train window to choose K
    cv_folds = int(sel_cfg.get('cv_folds') or 0)
    one_se = bool(sel_cfg.get('one_std_rule') or False)
    grid = model_cfg.get('state_grid') or [model_cfg.get('n_states', 3)]
    cov = model_cfg.get('covariance_type', 'diag')
    n_iter = model_cfg.get('n_iter', 200)
    tol = model_cfg.get('tol', 1e-3)
    rs = model_cfg.get('random_state', 42)
    reg_covar = model_cfg.get('reg_covar')
    sticky_diag = model_cfg.get('sticky_diag')

    restarts = int(sel_cfg.get('restarts', 1))
    crit_name = str(sel_cfg.get('criterion', 'bic')).lower()
    delta_thr = sel_cfg.get('delta_threshold')
    min_occ = float(sel_cfg.get('min_state_occupancy_pct') or 0.0)

    if cv_folds and cv_folds > 1 and X_train.shape[0] > cv_folds:
        N = X_train.shape[0]
        fold_size = N // cv_folds
        folds = []
        for i in range(cv_folds):
            start = i * fold_size
            end = (i + 1) * fold_size if i < cv_folds - 1 else N
            val_idx = np.arange(start, end)
            train_idx = np.concatenate([np.arange(0, start), np.arange(end, N)]) if start > 0 else np.arange(end, N)
            folds.append((train_idx, val_idx))

        per_k_cv: List[Dict[str, Any]] = []
        for k in grid:
            fold_lls: List[float] = []
            fold_recs: List[Dict[str, Any]] = []
            for (tr_idx, va_idx) in folds:
                Xtr = X_train[tr_idx]
                Xva = X_train[va_idx]
                best_for_fold = None
                for r in range(max(1, restarts)):
                    seed = int(rs) + r
                    m, ll_tr, bic, aic, icl, H, occ = fit_and_score(
                        Xtr, k, cov, n_iter, tol, seed, reg_covar=reg_covar, sticky_diag=sticky_diag,
                    )
                    if min_occ > 0.0 and any(o < min_occ for o in occ):
                        continue
                    ll_val = float(m.score(Xva)) if len(Xva) else float('nan')
                    rec = {'n_states': int(k), 'train_ll': ll_tr, 'val_ll': ll_val, 'bic': bic, 'aic': aic, 'icl': icl, 'entropy': H, 'occupancy': occ, 'seed': seed}
                    # Select restart by validation likelihood (higher is better); tie-break by train LL
                    if best_for_fold is None:
                        best_for_fold = rec
                    else:
                        curr_v = rec['val_ll']
                        best_v = best_for_fold['val_ll']
                        curr_v = curr_v if np.isfinite(curr_v) else -np.inf
                        best_v = best_v if np.isfinite(best_v) else -np.inf
                        if (curr_v > best_v) or (curr_v == best_v and rec['train_ll'] > best_for_fold['train_ll']):
                            best_for_fold = rec
                if best_for_fold is not None:
                    fold_lls.append(best_for_fold['val_ll'])
                    fold_recs.append(best_for_fold)
            if fold_lls:
                mean_ll = float(np.mean(fold_lls))
                std_ll = float(np.std(fold_lls, ddof=1)) if len(fold_lls) > 1 else 0.0
                per_k_cv.append({'n_states': int(k), 'cv_mean_ll': mean_ll, 'cv_std_ll': std_ll, 'folds': len(fold_lls)})

        if per_k_cv:
            # Select by max cv_mean_ll; apply one-std-error rule if enabled
            best = max(per_k_cv, key=lambda r: r['cv_mean_ll'])
            if one_se:
                se = best['cv_std_ll'] / np.sqrt(best['folds']) if best['folds'] > 0 else 0.0
                threshold = best['cv_mean_ll'] - se
                # choose smallest K with mean >= threshold
                candidates = sorted([r for r in per_k_cv if r['cv_mean_ll'] >= threshold], key=lambda r: int(r['n_states']))
                if candidates:
                    chosen_cv = candidates[0]
                else:
                    chosen_cv = best
            else:
                chosen_cv = best
            return int(chosen_cv['n_states']), {'grid': per_k_cv, 'chosen': chosen_cv, 'criterion': 'cv_ll', 'cv_folds': cv_folds, 'one_std_rule': one_se, 'restarts': restarts}

    per_k: List[Dict[str, Any]] = []
    for k in grid:
        best_for_k = None
        for r in range(max(1, restarts)):
            seed = int(rs) + r
            m, ll_tr, bic, aic, icl, H, occ = fit_and_score(
                X_train, k, cov, n_iter, tol, seed, reg_covar=reg_covar, sticky_diag=sticky_diag,
            )
            # Reject by occupancy threshold
            if min_occ > 0.0 and any(o < min_occ for o in occ):
                continue
            ll_val = float(m.score(X_val)) if len(X_val) > 0 else float('nan')
            rec = {
                'n_states': int(k), 'train_ll': ll_tr, 'val_ll': ll_val,
                'bic': bic, 'aic': aic, 'icl': icl, 'entropy': H, 'occupancy': occ,
                'seed': seed,
            }
            # Keep best per K by primary criterion then by train_ll
            def _crit(r):
                return r.get(crit_name, np.inf)
            if (best_for_k is None) or (_crit(rec) < _crit(best_for_k)) or (_crit(rec) == _crit(best_for_k) and rec['train_ll'] > best_for_k['train_ll']):
                best_for_k = rec
        if best_for_k is not None:
            per_k.append(best_for_k)

    if not per_k:
        # Fallback: try without occupancy filter
        for k in grid:
            m, ll_tr, bic, aic, icl, H, occ = fit_and_score(X_train, k, cov, n_iter, tol, rs, reg_covar=reg_covar, sticky_diag=sticky_diag)
            ll_val = float(m.score(X_val)) if len(X_val) > 0 else float('nan')
            per_k.append({'n_states': int(k), 'train_ll': ll_tr, 'val_ll': ll_val, 'bic': bic, 'aic': aic, 'icl': icl, 'entropy': H, 'occupancy': occ, 'seed': rs})

    # Apply delta-threshold elbow if configured
    chosen = None
    if delta_thr is not None and len(per_k) >= 2:
        per_k_sorted = sorted(per_k, key=lambda r: int(r['n_states']))
        crit_vals = [r.get(crit_name, np.inf) for r in per_k_sorted]
        ks = [int(r['n_states']) for r in per_k_sorted]
        # improvements: previous - current (positive means improvement)
        improvements = [np.inf]
        for i in range(1, len(crit_vals)):
            improvements.append(crit_vals[i-1] - crit_vals[i])
        elbow_idx = None
        for i in range(1, len(improvements)):
            if improvements[i] < float(delta_thr):
                elbow_idx = i - 1
                break
        if elbow_idx is not None and elbow_idx >= 0:
            chosen_k = ks[elbow_idx]
            chosen = next(r for r in per_k_sorted if int(r['n_states']) == chosen_k)

    # If not chosen by elbow, choose by primary criterion then by train_ll
    if chosen is None:
        chosen = sorted(per_k, key=lambda r: (r.get(crit_name, np.inf), -r['train_ll']))[0]

    return int(chosen['n_states']), {'grid': per_k, 'chosen': chosen, 'criterion': crit_name, 'delta_threshold': delta_thr, 'min_state_occupancy_pct': min_occ, 'restarts': restarts}


def evaluate(model: GaussianHMM, X: np.ndarray, ts: pd.Series) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    states = model.predict(X)
    post = model.predict_proba(X)
    out = pd.DataFrame({'timestamp': pd.to_datetime(ts), 'state': states})
    for k in range(post.shape[1]):
        out[f'p_state_{k}'] = post[:, k]
    # Minimal diagnostics
    diag = {
        'n_states': int(model.n_components),
        'covariance_type': str(model.covariance_type),
        'transmat': model.transmat_.tolist(),
        'startprob': model.startprob_.tolist(),
        'means': model.means_.tolist(),
    }
    return out, diag


def save_artifacts(out_dir: Path, model: GaussianHMM, scaler: StandardScaler, config: Dict[str, Any], metrics: Dict[str, Any], regimes_df: pd.DataFrame, diagnostics: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / 'model.joblib')
    joblib.dump(scaler, out_dir / 'scaler.joblib')
    def _json_safe(x):
        from pathlib import Path as _P
        if isinstance(x, _P):
            return str(x)
        if isinstance(x, dict):
            return {k: _json_safe(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_json_safe(v) for v in x]
        return x
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(_json_safe(config), f, indent=2)
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    regimes_df.to_csv(out_dir / 'regimes.csv', index=False)
    with open(out_dir / 'diagnostics.json', 'w') as f:
        json.dump(diagnostics, f, indent=2)


def run_hmm_pipeline(cfg: Dict[str, Any], log_level: str = "INFO") -> Path:
    # First-stage logging to stdout only; will add file handler once run_dir is known
    setup_logging(log_level)

    # Load and select columns
    tmp_df = pd.read_csv(cfg['input_data'], nrows=1)
    feature_cols = select_feature_columns(list(tmp_df.columns), cfg)
    df = load_features(cfg['input_data'], feature_cols)
    logging.info('Loaded features: rows=%d cols=%d', len(df), len(feature_cols))

    # Prepare run directory and attach file logging
    ts_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    cols_tag = f"cols{len(feature_cols)}"
    run_dir = cfg['output_dir'] / f"run_{ts_tag}_{cols_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # Reconfigure logging to also write to file
    file_handler = logging.FileHandler(str(run_dir / 'hmm_pipeline.log'))
    file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    logging.info('Run directory: %s', run_dir)

    # Split: prefer prepared metadata (LightGBM) if provided
    split_cfg = cfg.get('split', {})
    tr = va = te = None
    meta_path = None
    existing_dir = split_cfg.get('existing_dir')
    if existing_dir:
        meta_path = Path(existing_dir) / 'prep_metadata.json'
    meta_path = split_cfg.get('prep_metadata') or meta_path
    if meta_path:
        meta_path = Path(meta_path)
        if not meta_path.exists():
            raise FileNotFoundError(f"prep_metadata.json not found at: {meta_path}")
        tr, va, te = load_split_indices_from_meta(df['timestamp'], meta_path)
        logging.info('Loaded split indices from %s | sizes train/val/test: %d/%d/%d', meta_path, len(tr), len(va), len(te))
    else:
        tr, va, te = time_split_indices(df['timestamp'], cfg['split']['train_ratio'], cfg['split']['val_ratio'], cfg['split']['test_ratio'])
    X_train = df.iloc[tr][feature_cols].values
    X_val = df.iloc[va][feature_cols].values
    X_test = df.iloc[te][feature_cols].values
    ts_all = df['timestamp']

    # Scale: fit on train only to prevent leakage
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val) if len(X_val) else X_val
    X_test_s = scaler.transform(X_test) if len(X_test) else X_test

    # Select n_states
    n_states_cfg = cfg['model'].get('n_states')
    if n_states_cfg is None:
        # If using prepared splits, default to train-only selection by passing empty val unless selection one-std is enabled
        use_val = False if meta_path else True
        X_val_for_sel = X_val_s if (use_val and len(X_val_s)) else np.zeros((0, X_train_s.shape[1]))
        n_best, sel = select_n_states(X_train_s, X_val_for_sel, cfg['model'], cfg.get('selection', {}))
        cfg['model']['n_states'] = int(n_best)
        logging.info('Selected n_states=%d using %s over grid (restarts=%s, delta_thr=%s, min_occ=%.4f)', n_best, sel.get('criterion'), sel.get('restarts'), sel.get('delta_threshold'), sel.get('min_state_occupancy_pct') or 0.0)
        # Persist selection grid for diagnosis
        try:
            import pandas as _pd
            import numpy as _np
            grid = sel.get('grid', []) or []
            if grid:
                grid_df = _pd.DataFrame(grid)
                crit = sel.get('criterion')
                if crit == 'cv_ll':
                    # Higher is better
                    grid_df['criterion_value'] = grid_df.get('cv_mean_ll', _pd.Series([np.nan]*len(grid_df)))
                    grid_df = grid_df.sort_values('n_states')
                    grid_df['delta'] = grid_df['criterion_value'].diff()
                else:
                    # Lower is better (e.g., bic, icl)
                    if crit in grid_df.columns:
                        grid_df['criterion_value'] = grid_df[crit]
                    else:
                        grid_df['criterion_value'] = np.nan
                    grid_df = grid_df.sort_values('n_states')
                    # improvement = prev - current (positive is better)
                    grid_df['delta'] = grid_df['criterion_value'].shift(1) - grid_df['criterion_value']
                # Expand occupancy list if present
                if 'occupancy' in grid_df.columns:
                    grid_df['occupancy_min'] = grid_df['occupancy'].apply(lambda x: float(min(x)) if isinstance(x, (list, tuple)) and len(x) else np.nan)
                grid_df.to_csv(run_dir / 'selection_grid.csv', index=False)
                logging.info('Saved selection grid to %s', run_dir / 'selection_grid.csv')
        except Exception as e:
            logging.warning('Failed to persist selection grid: %s', e)
    else:
        sel = {'grid': [{'n_states': int(n_states_cfg)}], 'chosen': {'n_states': int(n_states_cfg)}}

    # Final fit set: default train+val if val exists, else train; can be overridden
    final_fit_on = (cfg.get('final_fit') or {}).get('on') or cfg['model'].get('final_fit_on')
    if final_fit_on is None:
        final_fit_on = 'train_val' if len(X_val_s) else 'train'
    if str(final_fit_on).lower() == 'train':
        X_final = X_train_s
    elif str(final_fit_on).lower() in ('train_val', 'train+val'):
        X_final = np.vstack([X_train_s, X_val_s]) if len(X_val_s) else X_train_s
    else:
        # Fallback to train_val for unknown values
        X_final = np.vstack([X_train_s, X_val_s]) if len(X_val_s) else X_train_s
    logging.info('Final fit on: %s (rows=%d)', final_fit_on, X_final.shape[0])
    reg_covar = cfg['model'].get('reg_covar')
    sticky_diag = cfg['model'].get('sticky_diag')
    model, ll_train_final, bic_final, aic_final, icl_final, H_final, occ_final = fit_and_score(
        X_final,
        cfg['model']['n_states'],
        cfg['model']['covariance_type'],
        cfg['model']['n_iter'],
        cfg['model']['tol'],
        cfg['model']['random_state'],
        reg_covar=reg_covar,
        sticky_diag=sticky_diag,
    )
    ll_test = float(model.score(X_test_s)) if len(X_test_s) else float('nan')

    # Evaluate on full series for regimes (apply-only on val/test to avoid leakage)
    X_all_s = scaler.transform(df[feature_cols].values)
    regimes_df, diagnostics = evaluate(model, X_all_s, ts_all)

    # Persist split map alongside regimes for audit
    split_map = pd.DataFrame({'timestamp': ts_all.astype(str), 'split': ['train'] * len(ts_all)})
    split_map.loc[va, 'split'] = 'val'
    split_map.loc[te, 'split'] = 'test'

    # Metrics
    metrics = {
        'n_states': int(cfg['model']['n_states']),
        'bic_final': bic_final,
        'aic_final': aic_final,
        'icl_final': icl_final,
        'entropy_final': H_final,
        'train_final_ll': ll_train_final,
        'test_ll': ll_test,
        'selection': sel,
        'rows': len(df),
        'n_features': len(feature_cols),
        'feature_cols': feature_cols,
        'final_occupancy': occ_final,
    }

    # Save
    save_artifacts(run_dir, model, scaler, cfg, metrics, regimes_df, diagnostics)
    # write split map
    split_map.to_csv(run_dir / 'split_map.csv', index=False)
    logging.info('Saved artifacts to %s', run_dir)

    # Also persist key diagnostics to the top-level output_dir for convenience
    try:
        out_root = Path(cfg['output_dir'])
        out_root.mkdir(parents=True, exist_ok=True)
        run_name = run_dir.name
        log_src = run_dir / 'hmm_pipeline.log'
        grid_src = run_dir / 'selection_grid.csv'
        if log_src.exists():
            dst = out_root / f"{run_name}_hmm_pipeline.log"
            shutil.copy2(log_src, dst)
            logging.info('Copied log to %s', dst)
        if grid_src.exists():
            dst = out_root / f"{run_name}_selection_grid.csv"
            shutil.copy2(grid_src, dst)
            logging.info('Copied selection grid to %s', dst)
    except Exception as e:
        logging.warning('Failed to copy diagnostics to output_dir: %s', e)

    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description='Train a Gaussian HMM on v1/v2 features (config-driven)')
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--log-level', default='INFO')
    args = ap.parse_args()

    # First-stage logging to stdout only; will add file handler once run_dir is known
    setup_logging(args.log_level)
    cfg = load_config(args.config)
    logging.info('Loaded config: %s', args.config)

    run_hmm_pipeline(cfg, args.log_level)
    logging.info("HMM pipeline completed successfully")


if __name__ == '__main__':
    main()
