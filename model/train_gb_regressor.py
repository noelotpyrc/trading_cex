import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


DEFAULT_MODELS_ROOT = Path('/Volumes/Extreme SSD/trading_data/cex/models')


@dataclass
class RunConfig:
    learning_rate: float
    n_estimators: int
    max_depth: int
    min_samples_leaf: int
    subsample: float
    random_state: int
    target_scale: float


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _read_split(data_dir: Path, split: str, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = pd.read_csv(data_dir / f"X_{split}.csv")
    y = pd.read_csv(data_dir / f"y_{split}.csv")
    if target not in y.columns:
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


def _make_output_dir(prepared_dir: Path, target: str, output_root: Path | None) -> Path:
    root = Path(output_root) if output_root is not None else DEFAULT_MODELS_ROOT
    dataset_name = prepared_dir.parent.name if prepared_dir.parent != prepared_dir else 'unknown_dataset'
    subdir = f"gbr_{target}"
    run_dir = root / dataset_name / subdir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def train_and_evaluate(data_dir: Path, target: str, cfg: RunConfig, output_root: Path | None) -> None:
    X_train, y_train = _read_split(data_dir, 'train', target)
    X_val, y_val = _read_split(data_dir, 'val', target)
    X_test, y_test = _read_split(data_dir, 'test', target)

    timestamps = _load_timestamps(data_dir)
    feature_names = list(X_train.columns)

    run_dir = _make_output_dir(data_dir, target, output_root)

    with open(run_dir / 'config.json', 'w') as f:
        json.dump(asdict(cfg), f, indent=2)

    scale = max(cfg.target_scale, 1e-12)
    y_train_s = y_train.values * scale
    y_val_s = y_val.values * scale

    model = GradientBoostingRegressor(
        loss='squared_error',
        learning_rate=cfg.learning_rate,
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        subsample=cfg.subsample,
        random_state=cfg.random_state,
        max_features=None,
        verbose=1,
    )

    # Fit on train only; monitor on val manually if desired
    model.fit(X_train, y_train_s)

    # Save model
    import joblib
    joblib.dump(model, run_dir / 'gbr_model.joblib')

    def pred(df: pd.DataFrame) -> np.ndarray:
        return model.predict(df) / scale

    preds_train = pred(X_train)
    preds_val = pred(X_val)
    preds_test = pred(X_test)

    def save_preds(name: str, y_true: pd.Series, preds: np.ndarray, ts_list: List[str]) -> None:
        out = pd.DataFrame({'y_true': y_true.values, 'pred_reg': preds})
        if ts_list and len(ts_list) == len(out):
            out.insert(0, 'timestamp', ts_list)
        out.to_csv(run_dir / f'pred_{name}.csv', index=False)

    save_preds('train', y_train, preds_train, timestamps.get('train', []))
    save_preds('val', y_val, preds_val, timestamps.get('val', []))
    save_preds('test', y_test, preds_test, timestamps.get('test', []))

    metrics = {
        'reg': {
            'rmse_train': rmse(y_train.values, preds_train),
            'rmse_val': rmse(y_val.values, preds_val),
            'rmse_test': rmse(y_test.values, preds_test),
            'mae_train': mae(y_train.values, preds_train),
            'mae_val': mae(y_val.values, preds_val),
            'mae_test': mae(y_test.values, preds_test),
        }
    }
    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Simple feature importance (GBR uses decision trees)
    fi = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})
    fi.sort_values('importance', ascending=False).to_csv(run_dir / 'feature_importance_gbr.csv', index=False)

    print(f"Training complete. Artifacts saved to: {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Train sklearn GradientBoostingRegressor for a target')
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--target', type=str, default='y_logret_24h')
    parser.add_argument('--learning-rate', type=float, default=0.03)
    parser.add_argument('--n-estimators', type=int, default=6000)
    parser.add_argument('--max-depth', type=int, default=3)
    parser.add_argument('--min-samples-leaf', type=int, default=20)
    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--target-scale', type=float, default=1.0)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()

    cfg = RunConfig(
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        subsample=args.subsample,
        random_state=args.random_state,
        target_scale=args.target_scale,
    )

    train_and_evaluate(
        data_dir=args.data_dir,
        target=args.target,
        cfg=cfg,
        output_root=args.output_dir,
    )


if __name__ == '__main__':
    main()


