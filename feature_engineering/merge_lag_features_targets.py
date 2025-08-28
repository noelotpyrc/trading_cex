import argparse
import glob
import os
from pathlib import Path
from typing import Optional

import pandas as pd


def _normalize_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    if 'timestamp' in data.columns:
        ts = pd.to_datetime(data['timestamp'], errors='coerce', utc=True)
        data['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    elif 'Unnamed: 0' in data.columns:
        data = data.rename(columns={'Unnamed: 0': 'timestamp'})
        ts = pd.to_datetime(data['timestamp'], errors='coerce', utc=True)
        data['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    elif 'index' in data.columns:
        data = data.rename(columns={'index': 'timestamp'})
        ts = pd.to_datetime(data['timestamp'], errors='coerce', utc=True)
        data['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    elif isinstance(data.index, pd.DatetimeIndex):
        data = data.reset_index().rename(columns={'index': 'timestamp'})
        ts = pd.to_datetime(data['timestamp'], errors='coerce', utc=True)
        data['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    else:
        first_col = data.columns[0]
        if first_col != 'timestamp':
            maybe_ts = pd.to_datetime(data[first_col], errors='coerce', utc=True)
            if maybe_ts.notna().any():
                data = data.rename(columns={first_col: 'timestamp'})
                data['timestamp'] = maybe_ts.dt.tz_convert('UTC').dt.tz_localize(None)

    if 'timestamp' not in data.columns:
        raise ValueError("Could not infer a 'timestamp' column from input data")

    data = data.dropna(subset=['timestamp'])
    data = data[~data['timestamp'].duplicated(keep='last')]
    data = data.sort_values('timestamp')
    return data


def _read_csv_with_timestamp(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _normalize_timestamp_column(df)


def _find_latest_lags_csv(features_dir: Path) -> Optional[Path]:
    candidates = sorted(
        features_dir.glob('current_bar_with_lags_*.csv'),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description='Merge lag features CSV with targets CSV on timestamp (drop rows with missing values)')
    parser.add_argument('--features-dir', type=Path, default=Path('/Volumes/Extreme SSD/trading_data/cex/lookbacks/BINANCE_BTCUSDT.P, 60'), help='Directory containing lag features CSVs')
    parser.add_argument('--lags-path', type=Path, default=None, help='Explicit path to a current_bar_with_lags_*.csv')
    parser.add_argument('--targets-dir', type=Path, default=Path('/Volumes/Extreme SSD/trading_data/cex/targets/BINANCE_BTCUSDT.P, 60'), help='Directory containing targets.csv')
    parser.add_argument('--targets-file', type=str, default='targets.csv', help='Targets filename within targets-dir')
    parser.add_argument('--output', type=Path, default=None, help='Output path (default: training-dir/dataset/merged_lags_targets.csv)')
    parser.add_argument('--training-dir', type=Path, default=Path('/Volumes/Extreme SSD/trading_data/cex/training'), help='Training base dir for default output')
    parser.add_argument('--dataset', type=str, default='BINANCE_BTCUSDT.P, 60', help='Dataset subfolder name under training-dir')
    parser.add_argument('--output-name', type=str, default='merged_lags_targets.csv', help='Default output filename when --output not specified')
    args = parser.parse_args()

    lags_path: Optional[Path] = args.lags_path
    if lags_path is None:
        lags_path = _find_latest_lags_csv(args.features_dir)
        if lags_path is None:
            raise FileNotFoundError(f"No current_bar_with_lags_*.csv found in {args.features_dir}")

    targets_path = args.targets_dir / args.targets_file
    print(f"Reading lag features: {lags_path}")
    print(f"Reading targets: {targets_path}")

    lags_df = _read_csv_with_timestamp(lags_path)
    targets_df = _read_csv_with_timestamp(targets_path)

    print(f"lags_df: rows={len(lags_df)}, cols={lags_df.shape[1]}")
    print(f"targets_df: rows={len(targets_df)}, cols={targets_df.shape[1]}")

    merged = pd.merge(
        lags_df,
        targets_df,
        on='timestamp',
        how='inner',
        validate='one_to_one',
    ).sort_values('timestamp')
    print(f"merged (pre-NA drop): rows={len(merged)}, cols={merged.shape[1]}")

    merged_clean = merged.dropna(axis=0, how='any').reset_index(drop=True)
    print(f"merged (post-NA drop): rows={len(merged_clean)}, cols={merged_clean.shape[1]}")

    out_path: Path
    if args.output is None:
        out_path = args.training_dir / args.dataset / args.output_name
        print(f"Output (default): training_dir={args.training_dir} dataset='{args.dataset}' file='{args.output_name}'")
    else:
        out_path = Path(args.output)
        print(f"Output (explicit): {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lower = str(out_path).lower()
    if lower.endswith('.parquet') or lower.endswith('.pq'):
        merged_clean.to_parquet(out_path)
    else:
        if not lower.endswith('.csv'):
            out_path = out_path.with_suffix('.csv')
        merged_clean.to_csv(out_path, index=False)
    print(f"Wrote merged dataset: {out_path}")


if __name__ == '__main__':
    main()


