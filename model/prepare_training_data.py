import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SplitConfig:
    train_ratio: float
    val_ratio: float
    test_ratio: float
    cutoff_dates: Optional[Tuple[Optional[str], Optional[str]]] = None


@dataclass
class PrepMetadata:
    input_path: str
    num_rows_before: int
    num_rows_after: int
    num_features_before: int
    num_features_after: int
    target_column: str
    split_strategy: str
    split_params: Dict[str, str]
    dropped_constant_columns: List[str]
    dropped_na_rows: int
    split_timestamps: Dict[str, List[str]]
    split_timestamp_ranges: Dict[str, Dict[str, Optional[str]]]


def _load_merged(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        df['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    return df


def _clean_dataframe(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, List[str], int]:
    cols_to_numeric = [c for c in df.columns if c not in ('timestamp', target_col)]
    for c in cols_to_numeric:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna(axis=1, how='all')

    constant_cols: List[str] = []
    for c in [c for c in df.columns if c not in ('timestamp', target_col)]:
        series = df[c]
        if series.nunique(dropna=True) <= 1:
            constant_cols.append(c)
    if constant_cols:
        df = df.drop(columns=constant_cols)

    before_rows = len(df)
    # Only enforce non-NA on features (exclude all y_* leakage columns) plus the selected target
    feature_cols_no_y = [c for c in df.columns if c != 'timestamp' and not c.startswith('y_')]
    df = df.dropna(axis=0, how='any', subset=feature_cols_no_y + [target_col])
    dropped_na_rows = before_rows - len(df)

    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)

    return df, constant_cols, dropped_na_rows


def _time_based_split(df: pd.DataFrame, cfg: SplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if 'timestamp' not in df.columns:
        n = len(df)
        n_train = int(n * cfg.train_ratio)
        n_val = int(n * cfg.val_ratio)
        train = df.iloc[:n_train]
        val = df.iloc[n_train:n_train + n_val]
        test = df.iloc[n_train + n_val:]
        return train, val, test

    if cfg.cutoff_dates and (cfg.cutoff_dates[0] or cfg.cutoff_dates[1]):
        start, mid = cfg.cutoff_dates
        ts = df['timestamp']
        if start:
            train = df[ts < pd.to_datetime(start)]
            remain = df[ts >= pd.to_datetime(start)]
        else:
            train = pd.DataFrame(columns=df.columns)
            remain = df
        if mid:
            val = remain[remain['timestamp'] < pd.to_datetime(mid)]
            test = remain[remain['timestamp'] >= pd.to_datetime(mid)]
        else:
            n_remain = len(remain)
            n_val = int(n_remain * cfg.val_ratio / (cfg.val_ratio + cfg.test_ratio))
            val = remain.iloc[:n_val]
            test = remain.iloc[n_val:]
        return train, val, test

    n = len(df)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)
    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]
    return train, val, test


def _write_outputs(out_dir: Path, target_col: str, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        # Exclude timestamp and all y_* columns from X to avoid leakage
        x_cols = [c for c in df.columns if c != 'timestamp' and not c.startswith('y_')]
        X = df[x_cols]
        y = df[target_col].astype(float)
        return X, y

    for name, part in [('train', train), ('val', val), ('test', test)]:
        X, y = split_xy(part)
        X.to_csv(out_dir / f'X_{name}.csv', index=False)
        y.to_csv(out_dir / f'y_{name}.csv', index=False, header=[target_col])


def prepare_splits(
    input_path: Path,
    output_dir: Path,
    target: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    cutoff_start: Optional[str] = None,
    cutoff_mid: Optional[str] = None,
) -> Path:
    """Programmatic API to prepare train/val/test splits.

    Returns the final output directory where X_*/y_* and prep_metadata.json are written.
    """
    merged = _load_merged(input_path)
    num_rows_before, num_cols_before = merged.shape

    if target not in merged.columns:
        raise ValueError(f"Target column '{target}' not found. Available: {len(merged.columns)} columns")

    cleaned, dropped_constants, dropped_na_rows = _clean_dataframe(merged.copy(), target_col=target)
    num_rows_after, num_cols_after = cleaned.shape

    split_cfg = SplitConfig(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        cutoff_dates=(cutoff_start, cutoff_mid),
    )
    train, val, test = _time_based_split(cleaned, split_cfg)

    final_out_dir = output_dir.parent / f"{output_dir.name}_{target}"

    _write_outputs(final_out_dir, target, train, val, test)

    def _ts_list(df: pd.DataFrame) -> List[str]:
        if 'timestamp' in df.columns:
            return df['timestamp'].astype(str).tolist()
        return []

    def _ts_range(df: pd.DataFrame) -> Dict[str, Optional[str]]:
        if 'timestamp' in df.columns and len(df) > 0:
            return {
                'min': str(df['timestamp'].min()),
                'max': str(df['timestamp'].max()),
            }
        return {'min': None, 'max': None}

    meta = PrepMetadata(
        input_path=str(input_path),
        num_rows_before=num_rows_before,
        num_rows_after=num_rows_after,
        num_features_before=num_cols_before,
        num_features_after=num_cols_after,
        target_column=target,
        split_strategy='cutoff' if (cutoff_start or cutoff_mid) else 'ratio_time_order',
        split_params={
            'train_ratio': str(train_ratio),
            'val_ratio': str(val_ratio),
            'test_ratio': str(test_ratio),
            'cutoff_start': str(cutoff_start),
            'cutoff_mid': str(cutoff_mid),
        },
        dropped_constant_columns=dropped_constants,
        dropped_na_rows=dropped_na_rows,
        split_timestamps={
            'train': _ts_list(train),
            'val': _ts_list(val),
            'test': _ts_list(test),
        },
        split_timestamp_ranges={
            'train': _ts_range(train),
            'val': _ts_range(val),
            'test': _ts_range(test),
        },
    )
    with open(final_out_dir / 'prep_metadata.json', 'w') as f:
        json.dump(asdict(meta), f, indent=2, default=str)

    return final_out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare ML training data from merged features-targets')
    parser.add_argument('--input', type=Path, required=True, help='Path to merged_features_targets.csv')
    parser.add_argument('--output-dir', type=Path, required=False, default=Path('/Volumes/Extreme SSD/trading_data/cex/training/BINANCE_BTCUSDT.P, 60/prepared'), help='Base output directory for prepared splits (target suffix will be appended)')
    parser.add_argument('--target', type=str, required=False, default='y_logret_24h', help='Target column to predict')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)
    parser.add_argument('--cutoff-start', type=str, default=None, help='Optional timestamp cutoff for train end (e.g., 2024-12-31)')
    parser.add_argument('--cutoff-mid', type=str, default=None, help='Optional timestamp cutoff to split val/test (e.g., 2025-06-01)')
    args = parser.parse_args()

    final_out_dir = prepare_splits(
        input_path=args.input,
        output_dir=args.output_dir,
        target=args.target,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        cutoff_start=args.cutoff_start,
        cutoff_mid=args.cutoff_mid,
    )

    print(f"Prepared data written to: {final_out_dir}")
    # sizes are already available in metadata; re-load to print
    meta = json.load(open(final_out_dir / 'prep_metadata.json', 'r'))
    print(
        f"Train/Val/Test sizes: {len(meta['split_timestamps']['train'])}/"
        f"{len(meta['split_timestamps']['val'])}/"
        f"{len(meta['split_timestamps']['test'])}"
    )


if __name__ == '__main__':
    main()


