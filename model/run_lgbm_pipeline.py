#!/usr/bin/env python3
"""
LightGBM training pipeline runner.

Executes the complete LightGBM model training workflow from a single config file:
1. Data preparation (train/val/test split)
2. Hyperparameter tuning 
3. LightGBM model training with best parameters
4. Model persistence and evaluation

Usage:
    python model/run_lgbm_pipeline.py --config configs/model_configs/btc_24h_logret.json
"""

import argparse
import json
import csv
import logging
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import pandas as pd
import numpy as np
import lightgbm as lgb
import shutil
from sklearn.model_selection import TimeSeriesSplit


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate configuration file and normalize objective schema."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Validate required keys
    required_keys = ['input_data', 'output_dir', 'target', 'split', 'model']
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: {key}")
    
    # Validate model type is LightGBM
    if config['model'].get('type', '').lower() != 'lgbm':
        raise ValueError("This pipeline only supports LightGBM models (model.type should be 'lgbm')")

    # Normalize objective schema
    # Supported LightGBM objectives (regression-focused):
    #   'regression', 'regression_l1', 'huber', 'fair', 'poisson', 'quantile',
    #   'mape', 'gamma', 'tweedie'
    target_config = config['target']
    obj_cfg = target_config.get('objective')
    if obj_cfg is None:
        raise ValueError("target.objective must be provided.")

    normalized_obj: Dict[str, Any]
    # Backward compatibility with {"type":"quantiles","quantiles":[0.05]}
    if isinstance(obj_cfg, dict) and obj_cfg.get('type') == 'quantiles':
        qs = obj_cfg.get('quantiles', [])
        if not isinstance(qs, list) or len(qs) != 1:
            raise ValueError("For quantile objective, provide exactly one quantile in 'quantiles'.")
        normalized_obj = {'name': 'quantile', 'params': {'alpha': float(qs[0])}}
    elif isinstance(obj_cfg, dict) and 'name' in obj_cfg:
        normalized_obj = {'name': str(obj_cfg['name']).lower(), 'params': obj_cfg.get('params', {})}
    elif isinstance(obj_cfg, str):
        normalized_obj = {'name': obj_cfg.lower(), 'params': {}}
    else:
        raise ValueError("target.objective must be either a string 'objective', or {name: ..., params: {...}}")

    # Basic validation
    allowed_objectives = {
        'regression', 'regression_l1', 'huber', 'fair', 'poisson', 'quantile',
        'mape', 'gamma', 'tweedie'
    }
    if normalized_obj['name'] not in allowed_objectives:
        raise ValueError(f"Unsupported LightGBM objective: {normalized_obj['name']}")
    if normalized_obj['name'] == 'quantile':
        alpha = normalized_obj['params'].get('alpha')
        if alpha is None:
            raise ValueError("Quantile objective requires params.alpha (e.g., 0.05)")
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError("Quantile alpha must be in (0, 1)")

    # Store normalized objective back
    config['target']['objective'] = normalized_obj
    
    # Convert paths to Path objects
    config['input_data'] = Path(config['input_data'])
    config['output_dir'] = Path(config['output_dir'])
    # Optional training splits root directory (for reusable splits across runs)
    training_splits_dir = config.get('training_splits_dir')
    if training_splits_dir is None:
        training_splits_dir = config['output_dir'] / 'training_splits'
    config['training_splits_dir'] = Path(training_splits_dir)
    # Optional existing splits directory
    split_cfg = config.get('split', {}) or {}
    if 'existing_dir' in split_cfg and split_cfg['existing_dir']:
        split_cfg['existing_dir'] = Path(split_cfg['existing_dir'])
        config['split'] = split_cfg
    
    return config


def prepare_training_data(config: Dict[str, Any]) -> Path:
    """
    Prepare training data splits using programmatic API from prepare_training_data.py.
    
    Returns:
        Path to the prepared data directory
    """
    # Import local module robustly regardless of CWD
    try:
        # Try as package import (namespace package)
        from model.prepare_training_data import prepare_splits  # type: ignore
    except Exception:
        # Fallback: add script directory to sys.path and import module directly
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from prepare_training_data import prepare_splits  # type: ignore

    logging.info("Starting data preparation...")

    target_name = config['target']['variable']
    split_cfg = config.get('split', {}) or {}
    splits_root: Path = config['training_splits_dir']
    splits_root.mkdir(parents=True, exist_ok=True)

    # Reuse existing splits if provided
    existing_dir = split_cfg.get('existing_dir')
    if existing_dir is not None:
        if not existing_dir.exists():
            raise FileNotFoundError(f"Configured split.existing_dir does not exist: {existing_dir}")
        expected = [f"X_{s}.csv" for s in ('train', 'val', 'test')] + [f"y_{s}.csv" for s in ('train', 'val', 'test')]
        missing = [p for p in expected if not (existing_dir / p).exists()]
        if missing:
            raise FileNotFoundError(f"Existing splits missing required files: {missing} under {existing_dir}")
        prepared_dir = existing_dir
    else:
        # Generate new splits into timestamped folder under training_splits_dir
        ts = config.get('_run_ts') or datetime.now().strftime("%Y%m%d_%H%M%S")
        base_prepared_out = splits_root / f"prepared_{ts}"
        prepared_dir = prepare_splits(
            input_path=config['input_data'],
            output_dir=base_prepared_out,
            target=target_name,
            train_ratio=float(split_cfg.get('train_ratio', 0.7)),
            val_ratio=float(split_cfg.get('val_ratio', 0.15)),
            test_ratio=float(split_cfg.get('test_ratio', 0.15)),
            cutoff_start=split_cfg.get('cutoff_start'),
            cutoff_mid=split_cfg.get('cutoff_mid'),
        )

    logging.info(f"Data preparation completed. Output: {prepared_dir}")
    return prepared_dir


def tune_hyperparameters(config: Dict[str, Any], data_dir: Path) -> Dict[str, Any]:
    """
    Perform hyperparameter tuning based on config specification.
    
    Args:
        config: Pipeline configuration
        data_dir: Path to prepared training data
        
    Returns:
        Dictionary of best hyperparameters
    """
    logging.info("Starting hyperparameter tuning...")
    
    model_config = config['model']
    tuning_method = model_config.get('hyperparameter_tuning_method', None)
    search_space = model_config.get('hyperparameter_search_space', {})

    if not tuning_method or not search_space:
        best_params = model_config.get('params', {})
        logging.info("No hyperparameter tuning requested; using model.params as best_params")
        return best_params

    tuning_method = tuning_method.lower()
    if tuning_method == 'bayesian':
        best_params = _tune_bayesian(config, data_dir, search_space)
    elif tuning_method == 'grid':
        best_params = _tune_grid_search(config, data_dir, search_space)
    else:
        logging.warning(f"Unknown tuning method '{tuning_method}', falling back to grid search")
        best_params = _tune_grid_search(config, data_dir, search_space)

    logging.info(f"Best parameters: {best_params}")
    return best_params


def _read_split(data_dir: Path, split: str, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = pd.read_csv(data_dir / f"X_{split}.csv")
    y = pd.read_csv(data_dir / f"y_{split}.csv")
    if target in y.columns:
        y_series = y[target]
    else:
        y_series = y.iloc[:, 0]
    return X, y_series.astype(float)


def _primary_metric_for_objective(objective_name: str, config_metrics: Optional[List[str]]) -> str:
    # Use user-provided metrics; otherwise select sensible default per objective
    if config_metrics and len(config_metrics) > 0:
        return config_metrics[0]
    if objective_name == 'quantile':
        return 'pinball_loss'
    # default regression
    return 'rmse'


def _evaluate_metric(name: str, y_true: np.ndarray, y_pred: np.ndarray, alpha: Optional[float] = None) -> float:
    if name == 'rmse':
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    if name == 'mae' or name == 'l1':
        return float(np.mean(np.abs(y_true - y_pred)))
    if name == 'pinball_loss':
        if alpha is None:
            raise ValueError("pinball_loss requires alpha for quantile evaluation")
        diff = y_true - y_pred
        return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))
    # Fallback to rmse
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _lgb_metric_name(primary: str) -> str:
    # Map our primary metric names to LightGBM metric strings
    if primary == 'mae' or primary == 'l1':
        return 'l1'
    if primary == 'pinball_loss':
        return 'quantile'
    return primary  # rmse, mape, tweedie, etc.


def _make_run_dir(config: Dict[str, Any], prepared_dir: Path) -> Path:
    ts = config.get('_run_ts') or datetime.now().strftime("%Y%m%d_%H%M%S")
    target_name = config['target']['variable']
    objective_name = config['target']['objective']['name']
    # Place all non-split outputs in a single timestamped folder directly under output_dir
    run_dir = config['output_dir'] / f"run_{ts}_lgbm_{target_name}_{objective_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _get_cv_config(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cv = model_cfg.get('cv', {}) or {}
    method = str(cv.get('method', 'expanding')).lower()
    n_folds = int(cv.get('n_folds', 3))
    fold_val_size = cv.get('fold_val_size', 0.2)  # fraction (0,1] or absolute int
    gap = int(cv.get('gap', 0))  # number of rows between train end and val start
    log_period = int(cv.get('log_period', 0))  # 0 disables per-iter logging
    return {
        'method': method,
        'n_folds': max(1, n_folds),
        'fold_val_size': fold_val_size,
        'gap': max(0, gap),
        # Allow CV to have its own training schedule; fall back handled by callers
        'num_boost_round': int(cv.get('num_boost_round', 1000)),
        'early_stopping_rounds': int(cv.get('early_stopping_rounds', 100)),
        'log_period': max(0, log_period),
    }


def _build_time_series_folds(
    n_rows: int,
    n_folds: int,
    fold_val_size: float | int,
    gap: int,
    method: str,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_rows < 10:
        raise ValueError("Not enough rows for time-series CV; need >= 10")
    # Determine absolute test_size per fold
    if isinstance(fold_val_size, float) and 0 < fold_val_size < 1.0:
        test_size = max(1, int(round(n_rows * fold_val_size)))
    else:
        test_size = max(1, int(fold_val_size))

    # Configure sklearn TimeSeriesSplit
    max_train_size = None
    if method == 'rolling':
        # Use a conservative fixed train window if requested; leave None for expanding
        # Heuristic: ensure at least 3x test_size in train window if possible
        candidate = n_rows - (n_folds * (test_size + gap))
        if candidate > 3 * test_size:
            max_train_size = candidate

    splitter = TimeSeriesSplit(
        n_splits=max(1, n_folds),
        test_size=test_size,
        gap=max(0, gap),
        max_train_size=max_train_size,
    )
    indices = np.arange(n_rows)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for train_idx, val_idx in splitter.split(indices):
        folds.append((train_idx, val_idx))
    if not folds:
        raise ValueError("Failed to construct CV folds via TimeSeriesSplit; adjust cv settings.")
    return folds


def _append_tuning_log_row(csv_path: Path, row: Dict[str, Any]) -> None:
    """Append a single tuning record row to a CSV file, creating header on first write.

    The CSV is intended to capture per-trial/combination CV summary metrics so runs are inspectable.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    # Normalize params to JSON string for readability
    if isinstance(row.get('params'), (dict, list)):
        row = {**row, 'params': json.dumps(row['params'], separators=(',', ':'))}
    # Ensure all fieldnames present consistently
    fieldnames = [
        'method', 'trial', 'metric', 'best_iteration', 'best_mean', 'best_stdv',
        'n_folds', 'fold_val_size', 'gap', 'objective', 'params'
    ]
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in fieldnames})


def _fit_once(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    objective_name: str,
    objective_params: Dict[str, Any],
    params: Dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int,
) -> Tuple[lgb.Booster, Dict[str, float], Dict[str, np.ndarray]]:
    feature_names = list(X_train.columns)
    train_set = lgb.Dataset(X_train, label=y_train.values, feature_name=feature_names, free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val.values, feature_name=feature_names, reference=train_set, free_raw_data=False)

    # Provide a default metric to enable early stopping
    def _default_metric_for_objective(name: str) -> List[str] | str:
        if name == 'quantile':
            return 'quantile'
        if name in {'regression', 'huber', 'fair'}:
            return ['rmse', 'l1']
        if name == 'regression_l1':
            return 'l1'
        if name in {'poisson', 'gamma', 'tweedie', 'mape'}:
            return name  # let LightGBM handle if available; otherwise it will fallback internally
        return 'rmse'

    lgb_params = {
        'objective': objective_name,
        'metric': _default_metric_for_objective(objective_name),
        **params,
    }
    # Objective-specific additions
    if objective_name == 'quantile':
        lgb_params['alpha'] = float(objective_params['alpha'])

    booster = lgb.train(
        lgb_params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[val_set],
        valid_names=['val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, first_metric_only=True),
            lgb.log_evaluation(period=200),
        ],
    )

    preds = {
        'train': booster.predict(X_train, num_iteration=booster.best_iteration),
        'val': booster.predict(X_val, num_iteration=booster.best_iteration),
    }

    metrics = {}
    # Primary metric for selection computed on validation
    primary = _primary_metric_for_objective(objective_name, None)
    alpha = objective_params.get('alpha') if objective_name == 'quantile' else None
    metrics[primary] = _evaluate_metric(primary, y_val.values, preds['val'], alpha=alpha)

    return booster, metrics, preds


def _load_timestamps(data_dir: Path) -> Dict[str, List[str]]:
    meta_path = data_dir / 'prep_metadata.json'
    if not meta_path.exists():
        return {'train': [], 'val': [], 'test': []}
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return meta.get('split_timestamps', {'train': [], 'val': [], 'test': []})


def _tune_grid_search(config: Dict[str, Any], data_dir: Path, search_space: Dict[str, Any]) -> Dict[str, Any]:
    target = config['target']['variable']
    obj = config['target']['objective']
    objective_name = obj['name']
    objective_params = obj.get('params', {})

    # Read train split only; CV folds are built within this set
    X_full_train, y_full_train = _read_split(data_dir, 'train', target)

    base_params = config['model'].get('params', {})
    # Use CV-specific schedule if provided; otherwise fall back to base training schedule
    cv_cfg = _get_cv_config(config['model'])
    num_boost_round = int(cv_cfg.get('num_boost_round', base_params.get('num_boost_round', 1000)))
    early_stopping_rounds = int(cv_cfg.get('early_stopping_rounds', base_params.get('early_stopping_rounds', 100)))

    cv_cfg = _get_cv_config(config['model'])
    folds = _build_time_series_folds(
        n_rows=len(X_full_train),
        n_folds=cv_cfg['n_folds'],
        fold_val_size=cv_cfg['fold_val_size'],
        gap=cv_cfg['gap'],
        method=cv_cfg['method'],
    )

    keys = list(search_space.keys())
    values_lists = [search_space[k] for k in keys]
    best_params = base_params.copy()
    best_score = float('inf')
    primary = _primary_metric_for_objective(objective_name, config['model'].get('eval_metrics'))

    # Prepare full training dataset once for lgb.cv
    feature_names = list(X_full_train.columns)
    lgb_dataset = lgb.Dataset(X_full_train, label=y_full_train.values, feature_name=feature_names, free_raw_data=False)

    # Map metric to LightGBM metric name
    lgb_metric = _lgb_metric_name(primary)

    log_csv = data_dir / 'tuning_trials.csv'
    trial_idx = 0
    for combo in product(*values_lists):
        trial_params = base_params.copy()
        for k, v in zip(keys, combo):
            trial_params[k] = v

        lgb_params = {
            'objective': objective_name,
            'metric': lgb_metric,
            **trial_params,
        }
        if objective_name == 'quantile':
            lgb_params['alpha'] = float(objective_params['alpha'])

        # Use LightGBM's built-in CV with sklearn splitter
        logging.info(f"[grid] Trial {trial_idx} params: {json.dumps(trial_params, sort_keys=True)}")
        logging.info(f"[grid] Trial {trial_idx}: starting lgb.cv with num_boost_round={num_boost_round}, early_stopping_rounds={early_stopping_rounds}, folds={cv_cfg['n_folds']}")
        callbacks_list = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, first_metric_only=True),
        ]
        if int(cv_cfg.get('log_period', 0)) > 0:
            callbacks_list.append(lgb.log_evaluation(period=int(cv_cfg['log_period'])))
        cv_results = lgb.cv(
            params=lgb_params,
            train_set=lgb_dataset,
            folds=folds,
            num_boost_round=num_boost_round,
            callbacks=callbacks_list,
        )
        # cv returns dict of lists like 'rmse-mean'; infer key
        mean_key = f"{lgb_metric}-mean"
        if mean_key not in cv_results:
            # fallback: take first key ending with -mean
            mean_keys = [k for k in cv_results.keys() if k.endswith('-mean')]
            if not mean_keys:
                continue
            mean_key = mean_keys[0]
        # Determine best iteration and stats
        best_iter_idx = int(np.argmin(cv_results[mean_key]))
        std_key = mean_key.replace('-mean', '-stdv')
        best_mean = float(cv_results[mean_key][best_iter_idx])
        best_stdv = float(cv_results.get(std_key, [np.nan] * (best_iter_idx + 1))[best_iter_idx])
        best_cv = best_mean
        # Log this trial
        _append_tuning_log_row(log_csv, {
            'method': 'grid',
            'trial': trial_idx,
            'metric': primary,
            'best_iteration': best_iter_idx + 1,
            'best_mean': best_mean,
            'best_stdv': best_stdv,
            'n_folds': cv_cfg['n_folds'],
            'fold_val_size': cv_cfg['fold_val_size'],
            'gap': cv_cfg['gap'],
            'objective': objective_name,
            'params': trial_params,
        })
        trial_idx += 1
        if best_cv < best_score:
            best_score = best_cv
            best_params = trial_params

    return best_params


def _tune_bayesian(config: Dict[str, Any], data_dir: Path, search_space: Dict[str, Any]) -> Dict[str, Any]:
    """Bayesian optimization using optuna if available; falls back to grid when missing."""
    try:
        import optuna  # type: ignore
    except Exception:
        logging.warning("Optuna not installed; falling back to grid search for tuning")
        return _tune_grid_search(config, data_dir, search_space)

    target = config['target']['variable']
    obj = config['target']['objective']
    objective_name = obj['name']
    objective_params = obj.get('params', {})

    X_full_train, y_full_train = _read_split(data_dir, 'train', target)

    base_params = config['model'].get('params', {})
    # Use CV-specific schedule if provided; otherwise fall back to base training schedule
    cv_cfg = _get_cv_config(config['model'])
    num_boost_round = int(cv_cfg.get('num_boost_round', base_params.get('num_boost_round', 1000)))
    early_stopping_rounds = int(cv_cfg.get('early_stopping_rounds', base_params.get('early_stopping_rounds', 100)))

    primary = _primary_metric_for_objective(objective_name, config['model'].get('eval_metrics'))

    # Convert list-based search space to optuna suggest space (categorical)
    keys = list(search_space.keys())
    values_lists = [search_space[k] for k in keys]

    cv_cfg = _get_cv_config(config['model'])
    folds = _build_time_series_folds(
        n_rows=len(X_full_train),
        n_folds=cv_cfg['n_folds'],
        fold_val_size=cv_cfg['fold_val_size'],
        gap=cv_cfg['gap'],
        method=cv_cfg['method'],
    )

    # Prepare dataset for lgb.cv
    feature_names = list(X_full_train.columns)
    lgb_dataset = lgb.Dataset(X_full_train, label=y_full_train.values, feature_name=feature_names, free_raw_data=False)
    lgb_metric = _lgb_metric_name(primary)

    log_csv = data_dir / 'tuning_trials.csv'

    def objective(trial: "optuna.Trial") -> float:
        trial_params = base_params.copy()
        for k, choices in zip(keys, values_lists):
            trial_params[k] = trial.suggest_categorical(k, choices)

        lgb_params = {
            'objective': objective_name,
            'metric': lgb_metric,
            **trial_params,
        }
        if objective_name == 'quantile':
            lgb_params['alpha'] = float(objective_params['alpha'])

        logging.info(f"[bayes] Trial {trial.number} params: {json.dumps(trial_params, sort_keys=True)}")
        logging.info(f"[bayes] Trial {trial.number}: starting lgb.cv with num_boost_round={num_boost_round}, early_stopping_rounds={early_stopping_rounds}, folds={cv_cfg['n_folds']}")
        callbacks_list = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, first_metric_only=True),
        ]
        if int(cv_cfg.get('log_period', 0)) > 0:
            callbacks_list.append(lgb.log_evaluation(period=int(cv_cfg['log_period'])))
        cv_results = lgb.cv(
            params=lgb_params,
            train_set=lgb_dataset,
            folds=folds,
            num_boost_round=num_boost_round,
            callbacks=callbacks_list,
        )
        mean_key = f"{lgb_metric}-mean"
        if mean_key not in cv_results:
            mean_keys = [k for k in cv_results.keys() if k.endswith('-mean')]
            if not mean_keys:
                return float('inf')
            mean_key = mean_keys[0]
        best_iter_idx = int(np.argmin(cv_results[mean_key]))
        std_key = mean_key.replace('-mean', '-stdv')
        best_mean = float(cv_results[mean_key][best_iter_idx])
        best_stdv = float(cv_results.get(std_key, [np.nan] * (best_iter_idx + 1))[best_iter_idx])
        # Log this optuna trial
        _append_tuning_log_row(log_csv, {
            'method': 'bayesian',
            'trial': trial.number,
            'metric': primary,
            'best_iteration': best_iter_idx + 1,
            'best_mean': best_mean,
            'best_stdv': best_stdv,
            'n_folds': cv_cfg['n_folds'],
            'fold_val_size': cv_cfg['fold_val_size'],
            'gap': cv_cfg['gap'],
            'objective': objective_name,
            'params': trial_params,
        })
        return best_mean

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=min(50, max(10, len(list(product(*values_lists))) // 2)))
    best_params = base_params.copy()
    best_params.update(study.best_params)
    return best_params


 


def train_model(config: Dict[str, Any], data_dir: Path, best_params: Dict[str, Any]) -> Tuple[lgb.Booster, Dict[str, float], Path]:
    """Train LightGBM model and return (booster, metrics, run_dir)."""
    logging.info("Starting LightGBM model training...")

    target = config['target']['variable']
    obj = config['target']['objective']
    objective_name = obj['name']
    objective_params = obj.get('params', {})

    # Read splits
    X_train, y_train = _read_split(data_dir, 'train', target)
    X_val, y_val = _read_split(data_dir, 'val', target)
    X_test, y_test = _read_split(data_dir, 'test', target)

    # Tweedie/Poisson require non-negative labels
    if objective_name in {'tweedie', 'poisson'}:
        if (y_train < 0).any() or (y_val < 0).any() or (y_test < 0).any():
            logging.warning(f"Objective {objective_name} expects non-negative labels; applying abs(y) for training/metrics")
            y_train = y_train.abs()
            y_val = y_val.abs()
            y_test = y_test.abs()

    num_boost_round = int(best_params.get('num_boost_round', config['model'].get('params', {}).get('num_boost_round', 1000)))
    early_stopping_rounds = int(best_params.get('early_stopping_rounds', config['model'].get('params', {}).get('early_stopping_rounds', 100)))

    booster, _, preds = _fit_once(
        X_train, y_train, X_val, y_val,
        objective_name, objective_params,
        best_params, num_boost_round, early_stopping_rounds,
    )

    # Evaluate on all splits
    primary = _primary_metric_for_objective(objective_name, config['model'].get('eval_metrics'))
    alpha = objective_params.get('alpha') if objective_name == 'quantile' else None
    metrics = {
        f'{primary}_train': _evaluate_metric(primary, y_train.values, preds['train'], alpha=alpha),
        f'{primary}_val': _evaluate_metric(primary, y_val.values, preds['val'], alpha=alpha),
    }
    preds_test = booster.predict(X_test, num_iteration=booster.best_iteration)
    metrics[f'{primary}_test'] = _evaluate_metric(primary, y_test.values, preds_test, alpha=alpha)

    # Persist artifacts
    run_dir = _make_run_dir(config, data_dir)
    # Save model
    model_path = run_dir / 'model.txt'
    booster.save_model(str(model_path))
    # Save predictions
    ts = _load_timestamps(data_dir)
    def save_preds(name: str, y_true: pd.Series, y_pred: np.ndarray, ts_list: List[str]) -> None:
        out = pd.DataFrame({'y_true': y_true.values, 'y_pred': y_pred})
        if ts_list and len(ts_list) == len(out):
            out.insert(0, 'timestamp', ts_list)
        out.to_csv(run_dir / f'pred_{name}.csv', index=False)

    # For train/val we already have predictions from _fit_once
    save_preds('train', y_train, preds['train'], ts.get('train', []))
    save_preds('val', y_val, preds['val'], ts.get('val', []))
    save_preds('test', y_test, preds_test, ts.get('test', []))

    # Save metrics
    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Feature importances
    fi_gain = booster.feature_importance(importance_type='gain')
    fi_split = booster.feature_importance(importance_type='split')
    fi_df = pd.DataFrame({
        'feature': list(X_train.columns),
        'importance_gain': fi_gain,
        'importance_split': fi_split,
    })
    fi_df.sort_values('importance_gain', ascending=False).to_csv(run_dir / 'feature_importance.csv', index=False)

    logging.info(f"LightGBM training completed. Metrics: {metrics}")
    return booster, metrics, run_dir


def _train_lgbm(config: Dict[str, Any], data_dir: Path, target: str, params: Dict[str, Any]) -> Tuple[lgb.Booster, Dict[str, float], Path]:
    # Kept as wrapper for compatibility if called elsewhere
    return train_model(config, data_dir, params)




def persist_results(config: Dict[str, Any], run_dir: Path, metrics: Dict[str, float], best_params: Dict[str, Any], data_dir: Path) -> Path:
    """
    Save model, metadata, and results to output directory.
    
    Args:
        config: Pipeline configuration
        model: Trained model object
        metrics: Evaluation metrics
        best_params: Best hyperparameters used
        
    Returns:
        Path to the saved model directory
    """
    logging.info("Persisting pipeline metadata into run_dir...")
    
    # Save metadata
    metadata = {
        'config': config,
        'best_params': best_params,
        'metrics': metrics,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'target': config['target']['variable'],
        'model_type': config['model']['type'],
    }
    
    # Convert Path objects to strings for JSON serialization
    metadata_json = json.loads(json.dumps(metadata, default=str))
    
    with open(run_dir / 'run_metadata.json', 'w') as f:
        json.dump(metadata_json, f, indent=2)
    # Also persist individual convenience files
    with open(run_dir / 'pipeline_config.json', 'w') as f:
        json.dump(json.loads(json.dumps(config, default=str)), f, indent=2)
    with open(run_dir / 'best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    with open(run_dir / 'paths.json', 'w') as f:
        json.dump({
            'input_data': str(config['input_data']),
            'prepared_data_dir': str(data_dir),
            'training_splits_dir': str(config.get('training_splits_dir', '')),
            'output_dir': str(config['output_dir'])
        }, f, indent=2)
    # Copy tuning log and prep metadata for full reproducibility
    tuning_csv = data_dir / 'tuning_trials.csv'
    if tuning_csv.exists():
        shutil.copy2(tuning_csv, run_dir / 'tuning_trials.csv')
    src_meta = data_dir / 'prep_metadata.json'
    if src_meta.exists():
        shutil.copy2(src_meta, run_dir / 'prep_metadata.json')
    logging.info(f"Run metadata saved to: {run_dir}")
    return run_dir


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description="Run unified training pipeline")
    parser.add_argument('--config', type=Path, required=True,
                       help='Path to pipeline configuration JSON file')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logging.info("Starting training pipeline...")
    
    try:
        # Load configuration
        config = load_config(args.config)
        logging.info(f"Loaded config from: {args.config}")
        
        # Step 1: Prepare training data
        data_dir = prepare_training_data(config)
        
        # Step 2: Hyperparameter tuning
        tuned_best_params = tune_hyperparameters(config, data_dir)
        # Expose tuned params for persistence/metadata
        config['_tuned_best_params'] = tuned_best_params

        # Choose final params source: tuned best vs fixed config params
        use_best_for_final = bool(config['model'].get('use_best_params_for_final', True))
        final_params = tuned_best_params if use_best_for_final else config['model'].get('params', {})
        
        # Step 3: Train model with chosen final params
        model, metrics, run_dir = train_model(config, data_dir, final_params)
        
        # Step 4: Persist pipeline metadata into the same run_dir
        run_dir = persist_results(config, run_dir, metrics, final_params, data_dir)
        
        logging.info(f"Pipeline completed successfully! Results in: {run_dir}")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()