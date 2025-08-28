import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb


DEFAULT_MODELS_ROOT = Path('/Volumes/Extreme SSD/trading_data/cex/models')


@dataclass
class RunConfig:
    quantiles: List[float]
    learning_rate: float
    num_leaves: int
    max_depth: int
    min_data_in_leaf: int
    feature_fraction: float
    bagging_fraction: float
    bagging_freq: int
    lambda_l1: float
    lambda_l2: float
    num_boost_round: int
    early_stopping_rounds: int
    seed: int
    # Regression (point estimate) settings
    train_regression: bool
    regression_objective: str  # one of: 'regression', 'regression_l1', 'huber', 'fair', 'tweedie'
    tweedie_variance_power: float | None
    # Quantile-specific overrides (optional)
    quantile_learning_rate: float | None
    quantile_num_leaves: int | None
    quantile_min_data_in_leaf: int | None
    quantile_feature_fraction: float | None
    quantile_bagging_fraction: float | None
    # General knobs
    min_gain_to_split: float
    target_scale: float


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _read_split(data_dir: Path, split: str, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = pd.read_csv(data_dir / f"X_{split}.csv")
    y = pd.read_csv(data_dir / f"y_{split}.csv")
    # y is a one-column dataframe with header [target]
    if target not in y.columns:
        # fallback to first column
        y_series = y.iloc[:, 0]
    else:
        y_series = y[target]
    return X, y_series.astype(float)


def _load_timestamps(data_dir: Path) -> Dict[str, List[str]]:
    meta_path = data_dir / 'prep_metadata.json'
    if not meta_path.exists():
        return {'train': [], 'val': [], 'test': []}
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return meta.get('split_timestamps', {'train': [], 'val': [], 'test': []})


def _make_output_dir(prepared_dir: Path, target: str, output_dir: Path | None) -> Path:
    """
    Compose output directory as:
      <root>/<dataset_name>/lgbm_<target>/run_<timestamp>
    - root: provided via --output-dir or defaults to DEFAULT_MODELS_ROOT
    - dataset_name: inferred from prepared_dir path (parent of prepared folder)
    """
    # Determine root
    root = Path(output_dir) if output_dir is not None else DEFAULT_MODELS_ROOT

    # Infer dataset name from prepared_dir path
    # Expecting: .../training/<dataset_name>/prepared_<target>
    dataset_name = prepared_dir.parent.name if prepared_dir.parent != prepared_dir else 'unknown_dataset'

    subdir = f"lgbm_{target}"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = root / dataset_name / subdir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def train_and_evaluate(
    data_dir: Path,
    target: str,
    cfg: RunConfig,
    output_dir: Path | None,
) -> None:
    # Load splits
    X_train, y_train = _read_split(data_dir, 'train', target)
    X_val, y_val = _read_split(data_dir, 'val', target)
    X_test, y_test = _read_split(data_dir, 'test', target)

    # Keep feature names
    feature_names = list(X_train.columns)

    # Create output directory
    run_dir = _make_output_dir(data_dir, target, output_dir)

    # Save config
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(asdict(cfg), f, indent=2)

    # Optional target scaling for stability (train-time only)
    scale = max(cfg.target_scale, 1e-12)
    y_train_scaled = y_train.values * scale
    y_val_scaled = y_val.values * scale

    # Training datasets (for quantile models)
    lgb_train = lgb.Dataset(X_train, label=y_train_scaled, feature_name=feature_names, free_raw_data=False)
    lgb_val = lgb.Dataset(X_val, label=y_val_scaled, reference=lgb_train, feature_name=feature_names, free_raw_data=False)

    timestamps = _load_timestamps(data_dir)

    # Aggregate predictions across quantiles
    preds_train: Dict[str, np.ndarray] = {}
    preds_val: Dict[str, np.ndarray] = {}
    preds_test: Dict[str, np.ndarray] = {}
    metrics: Dict[str, Dict[str, float]] = {}

    # Optional baseline regression model (point estimate)
    if cfg.train_regression:
        # Prepare objective and labels for regression path
        reg_objective = cfg.regression_objective
        reg_metric = ['rmse', 'l1'] if reg_objective != 'tweedie' else 'tweedie'

        # Tweedie expects non-negative labels; train on abs(y) if chosen
        if reg_objective == 'tweedie':
            reg_y_train = np.abs(y_train.values) * scale
            reg_y_val = np.abs(y_val.values) * scale
        else:
            reg_y_train = y_train_scaled
            reg_y_val = y_val_scaled

        lgb_train_reg = lgb.Dataset(
            X_train,
            label=reg_y_train,
            feature_name=feature_names,
            free_raw_data=False,
        )
        lgb_val_reg = lgb.Dataset(
            X_val,
            label=reg_y_val,
            reference=lgb_train_reg,
            feature_name=feature_names,
            free_raw_data=False,
        )

        reg_params = {
            'objective': reg_objective,
            'metric': reg_metric,
            'learning_rate': cfg.learning_rate,
            'num_leaves': cfg.num_leaves,
            'max_depth': cfg.max_depth,
            'min_data_in_leaf': cfg.min_data_in_leaf,
            'feature_fraction': cfg.feature_fraction,
            'bagging_fraction': cfg.bagging_fraction,
            'bagging_freq': cfg.bagging_freq,
            'lambda_l1': cfg.lambda_l1,
            'lambda_l2': cfg.lambda_l2,
            'min_gain_to_split': cfg.min_gain_to_split,
            'verbosity': -1,
            'seed': cfg.seed,
        }

        if reg_objective == 'tweedie':
            # Default to 1.5 if not provided
            var_power = cfg.tweedie_variance_power if cfg.tweedie_variance_power is not None else 1.5
            reg_params['tweedie_variance_power'] = var_power

        booster_reg = lgb.train(
            reg_params,
            lgb_train_reg,
            num_boost_round=cfg.num_boost_round,
            valid_sets=[lgb_val_reg],
            valid_names=['val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=cfg.early_stopping_rounds, first_metric_only=True),
                lgb.log_evaluation(period=200),
            ],
        )

        # Save model
        model_path_reg = run_dir / "lgbm_reg.txt"
        booster_reg.save_model(str(model_path_reg))

        # Predictions
        preds_train['reg'] = booster_reg.predict(X_train, num_iteration=booster_reg.best_iteration) / scale
        preds_val['reg'] = booster_reg.predict(X_val, num_iteration=booster_reg.best_iteration) / scale
        preds_test['reg'] = booster_reg.predict(X_test, num_iteration=booster_reg.best_iteration) / scale

        # Metrics
        # Evaluate against absolute target if tweedie was used
        if reg_objective == 'tweedie':
            y_train_eval = np.abs(y_train.values)
            y_val_eval = np.abs(y_val.values)
            y_test_eval = np.abs(y_test.values)
        else:
            y_train_eval = y_train.values
            y_val_eval = y_val.values
            y_test_eval = y_test.values

        metrics['reg'] = {
            'rmse_train': rmse(y_train_eval, preds_train['reg']),
            'rmse_val': rmse(y_val_eval, preds_val['reg']),
            'rmse_test': rmse(y_test_eval, preds_test['reg']),
            'mae_train': mae(y_train_eval, preds_train['reg']),
            'mae_val': mae(y_val_eval, preds_val['reg']),
            'mae_test': mae(y_test_eval, preds_test['reg']),
        }

        # Feature importance
        fi_gain = booster_reg.feature_importance(importance_type='gain')
        fi_split = booster_reg.feature_importance(importance_type='split')
        fi_df = pd.DataFrame({'feature': feature_names, 'importance_gain': fi_gain, 'importance_split': fi_split})
        fi_df.sort_values('importance_gain', ascending=False).to_csv(run_dir / 'feature_importance_reg.csv', index=False)

    for q in cfg.quantiles:
        # Apply quantile-specific overrides if provided
        q_lr = cfg.quantile_learning_rate if cfg.quantile_learning_rate is not None else cfg.learning_rate
        q_leaves = cfg.quantile_num_leaves if cfg.quantile_num_leaves is not None else cfg.num_leaves
        q_min_leaf = cfg.quantile_min_data_in_leaf if cfg.quantile_min_data_in_leaf is not None else cfg.min_data_in_leaf
        q_feat_frac = cfg.quantile_feature_fraction if cfg.quantile_feature_fraction is not None else cfg.feature_fraction
        q_bagg_frac = cfg.quantile_bagging_fraction if cfg.quantile_bagging_fraction is not None else cfg.bagging_fraction

        params = {
            'objective': 'quantile',
            'metric': 'quantile',
            'alpha': q,
            'learning_rate': q_lr,
            'num_leaves': q_leaves,
            'max_depth': cfg.max_depth,
            'min_data_in_leaf': q_min_leaf,
            'feature_fraction': q_feat_frac,
            'bagging_fraction': q_bagg_frac,
            'bagging_freq': cfg.bagging_freq,
            'lambda_l1': cfg.lambda_l1,
            'lambda_l2': cfg.lambda_l2,
            'min_gain_to_split': cfg.min_gain_to_split,
            'verbosity': -1,
            'seed': cfg.seed,
        }

        booster = lgb.train(
            params,
            lgb_train,
            num_boost_round=cfg.num_boost_round,
            valid_sets=[lgb_val],
            valid_names=['val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=cfg.early_stopping_rounds, first_metric_only=True),
                lgb.log_evaluation(period=200),
            ],
        )

        q_tag = f"q{int(round(q * 100)):02d}"

        # Save model
        model_path = run_dir / f"lgbm_{q_tag}.txt"
        booster.save_model(str(model_path))

        # Predictions
        preds_train[q_tag] = booster.predict(X_train, num_iteration=booster.best_iteration) / scale
        preds_val[q_tag] = booster.predict(X_val, num_iteration=booster.best_iteration) / scale
        preds_test[q_tag] = booster.predict(X_test, num_iteration=booster.best_iteration) / scale

        # Metrics
        metrics[q_tag] = {
            'pinball_train': pinball_loss(y_train.values, preds_train[q_tag], q),
            'pinball_val': pinball_loss(y_val.values, preds_val[q_tag], q),
            'pinball_test': pinball_loss(y_test.values, preds_test[q_tag], q),
        }

        # Feature importance
        fi_gain = booster.feature_importance(importance_type='gain')
        fi_split = booster.feature_importance(importance_type='split')
        fi_df = pd.DataFrame({'feature': feature_names, 'importance_gain': fi_gain, 'importance_split': fi_split})
        fi_df.sort_values('importance_gain', ascending=False).to_csv(run_dir / f'feature_importance_{q_tag}.csv', index=False)

    # Save predictions per split
    def save_preds(name: str, X: pd.DataFrame, y: pd.Series, ts_list: List[str], store: Dict[str, np.ndarray]) -> None:
        out = pd.DataFrame({'y_true': y.values})
        for k, arr in store.items():
            out[f'pred_{k}'] = arr
        if ts_list and len(ts_list) == len(out):
            out.insert(0, 'timestamp', ts_list)
        out.to_csv(run_dir / f'pred_{name}.csv', index=False)

    save_preds('train', X_train, y_train, timestamps.get('train', []), preds_train)
    save_preds('val', X_val, y_val, timestamps.get('val', []), preds_val)
    save_preds('test', X_test, y_test, timestamps.get('test', []), preds_test)

    # Save metrics
    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Training complete. Artifacts saved to: {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Train LightGBM quantile regression models for a target')
    parser.add_argument('--data-dir', type=Path, required=True, help='Directory containing prepared X_/y_ CSVs and prep_metadata.json')
    parser.add_argument('--target', type=str, default='y_logret_24h')
    parser.add_argument('--quantiles', type=float, nargs='+', default=[0.05, 0.25, 0.75, 0.95])
    parser.add_argument('--no-regression', action='store_true', help='Disable baseline regression model training')
    parser.add_argument('--regression-objective', choices=['regression', 'regression_l1', 'huber', 'fair', 'tweedie'], default='regression', help='Baseline regression objective')
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--quantile-learning-rate', type=float, default=None, help='Override learning rate for quantile models')
    parser.add_argument('--num-leaves', type=int, default=256)
    parser.add_argument('--quantile-num-leaves', type=int, default=None, help='Override num_leaves for quantile models')
    parser.add_argument('--max-depth', type=int, default=-1)
    parser.add_argument('--min-data-in-leaf', type=int, default=20)
    parser.add_argument('--quantile-min-data-in-leaf', type=int, default=None, help='Override min_data_in_leaf for quantile models')
    parser.add_argument('--feature-fraction', type=float, default=1)
    parser.add_argument('--quantile-feature-fraction', type=float, default=None, help='Override feature_fraction for quantile models')
    parser.add_argument('--bagging-fraction', type=float, default=1)
    parser.add_argument('--quantile-bagging-fraction', type=float, default=None, help='Override bagging_fraction for quantile models')
    parser.add_argument('--bagging-freq', type=int, default=1)
    parser.add_argument('--lambda-l1', type=float, default=0.0)
    parser.add_argument('--lambda-l2', type=float, default=0.0)
    parser.add_argument('--num-boost-round', type=int, default=8000)
    parser.add_argument('--early-stopping-rounds', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-gain-to-split', type=float, default=0.0)
    parser.add_argument('--target-scale', type=float, default=1000, help='Multiply y by this during training, divide predictions back by it')
    parser.add_argument('--output-dir', type=Path, default=None, help='Root output dir (defaults to /Volumes/Extreme SSD/trading_data/cex/models). Subdir is <dataset>/lgbm_<target>/run_<ts>')
    args = parser.parse_args()

    cfg = RunConfig(
        quantiles=args.quantiles,
        learning_rate=args.learning_rate,
        quantile_learning_rate=args.quantile_learning_rate,
        num_leaves=args.num_leaves,
        quantile_num_leaves=args.quantile_num_leaves,
        max_depth=args.max_depth,
        min_data_in_leaf=args.min_data_in_leaf,
        quantile_min_data_in_leaf=args.quantile_min_data_in_leaf,
        feature_fraction=args.feature_fraction,
        quantile_feature_fraction=args.quantile_feature_fraction,
        bagging_fraction=args.bagging_fraction,
        quantile_bagging_fraction=args.quantile_bagging_fraction,
        bagging_freq=args.bagging_freq,
        lambda_l1=args.lambda_l1,
        lambda_l2=args.lambda_l2,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        seed=args.seed,
        train_regression=(not args.no_regression),
        regression_objective=args.regression_objective,
        tweedie_variance_power=None,
        min_gain_to_split=args.min_gain_to_split,
        target_scale=args.target_scale,
    )

    train_and_evaluate(
        data_dir=args.data_dir,
        target=args.target,
        cfg=cfg,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()


