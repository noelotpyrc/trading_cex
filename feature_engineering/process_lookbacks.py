"""
Process stored lookback PKLs and compute features using core functions (excluding get_lags).

Usage:
  python process_lookbacks.py \
    --dataset 'BINANCE_BTCUSDT.P, 60' \
    --base-dir '/Volumes/Extreme SSD/trading_data/cex/lookbacks' \
    --timeframes 1H 4H 12H 1D \
    --output features.parquet

Notes:
  - Excludes get_lags and dependent volume lag helper; computes the rest with defaults.
  - Keys returned by core functions are column-aware; we append timeframe suffix.
"""

import os
import sys
import time
import argparse
from typing import Dict
import pandas as pd
import numpy as np

# Import core functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core_functions import (
    calculate_price_differences,
    calculate_log_transform,
    calculate_percentage_changes,
    calculate_cumulative_returns,
    calculate_zscore,
    calculate_sma,
    calculate_ema,
    calculate_wma,
    calculate_ma_crossovers,
    calculate_ma_distance,
    calculate_macd,
    calculate_volume_ma,
    calculate_rsi,
    calculate_stochastic,
    calculate_cci,
    calculate_roc,
    calculate_williams_r,
    calculate_ultimate_oscillator,
    calculate_mfi,
    calculate_historical_volatility,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_volatility_ratio,
    calculate_parkinson_volatility,
    calculate_garman_klass_volatility,
    calculate_obv,
    calculate_vwap,
    calculate_adl,
    calculate_chaikin_oscillator,
    calculate_volume_roc,
    calculate_rolling_percentiles,
    calculate_distribution_features,
    calculate_autocorrelation,
    calculate_hurst_exponent,
    calculate_entropy,
    calculate_price_volume_ratios,
    calculate_candle_patterns,
    calculate_typical_price,
    calculate_ohlc_average,
    calculate_volatility_adjusted_returns,
    calculate_time_features,
    calculate_rolling_extremes,
    calculate_dominant_cycle,
    # Additional
    calculate_adx,
    calculate_rogers_satchell_volatility,
    calculate_yang_zhang_volatility,
    calculate_rvol,
    calculate_donchian_distance,
    calculate_aroon,
    calculate_return_zscore,
    calculate_atr_normalized_distance,
    calculate_roll_spread,
    calculate_amihud_illiquidity,
    calculate_turnover_zscore,
    calculate_ljung_box_pvalue,
    calculate_permutation_entropy,
    calculate_ou_half_life,
    calculate_var_cvar,
    calculate_spectral_entropy,
)


def _append_tf_suffix(d: Dict[str, float], tf: str) -> Dict[str, float]:
    return {f"{k}_{tf}": v for k, v in d.items()}


def compute_features_one(lb: pd.DataFrame, tf: str, skip_slow: bool = False) -> Dict[str, float]:
    """Compute a broad set of features for a single lookback window (one timeframe).

    Excludes get_lags; uses defaults for most params; handles missing data gracefully.
    """
    if lb is None or lb.empty:
        return {}
    # Ensure lower-case columns
    cols = {c: c for c in lb.columns}
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c not in lb.columns:
            # try case-insensitive
            for cc in lb.columns:
                if str(cc).lower() == c:
                    cols[cc] = c
    lb = lb.rename(columns=cols)

    open_s = lb.get('open', pd.Series(dtype=float))
    high_s = lb.get('high', pd.Series(dtype=float))
    low_s = lb.get('low', pd.Series(dtype=float))
    close_s = lb.get('close', pd.Series(dtype=float))
    volume_s = lb.get('volume', pd.Series(dtype=float))

    out: Dict[str, float] = {}

    # Basic diffs and transforms
    out.update(calculate_price_differences(open_s, high_s, low_s, close_s, 0))
    out.update(calculate_log_transform(close_s, 'close'))
    out.update(calculate_percentage_changes(close_s, 0, 'close'))
    out.update(calculate_cumulative_returns(close_s, [5, 10, 20], 'close'))
    out.update(calculate_zscore(close_s, 20, 'close'))

    # Moving averages and crossovers
    out.update(calculate_sma(close_s, 20, 'close'))
    out.update(calculate_ema(close_s, 12, 'close'))
    out.update(calculate_wma(close_s, 20, 'close'))
    out.update(calculate_ma_crossovers(close_s, 5, 20, 'close'))

    # MA distance using SMA20
    sma20 = out.get('close_sma_20')
    if sma20 is not None and not np.isnan(sma20) and len(close_s) > 0:
        out.update(calculate_ma_distance(float(close_s.iloc[-1]), float(sma20), 'close', 'sma20'))
    else:
        out.update({'close_ma_distance_sma20': np.nan, 'close_ma_distance_pct_sma20': np.nan})

    # MACD
    out.update(calculate_macd(close_s, 12, 26, 9, 'close'))

    # Volume basics
    out.update(calculate_volume_ma(volume_s, 20, 'volume'))

    # Oscillators
    out.update(calculate_rsi(close_s, 14, 'close'))
    out.update(calculate_stochastic(high_s, low_s, close_s, 14, 3, 'close'))
    out.update(calculate_cci(high_s, low_s, close_s, 20, 'close'))
    out.update(calculate_roc(close_s, 10, 'close'))
    out.update(calculate_williams_r(high_s, low_s, close_s, 14, 'close'))
    out.update(calculate_ultimate_oscillator(high_s, low_s, close_s, [7, 14, 28], [4, 2, 1], 'close'))
    out.update(calculate_mfi(high_s, low_s, close_s, volume_s, 14, 'close'))

    # Volatility family
    out.update(calculate_historical_volatility(close_s, 20, 'close'))
    out.update(calculate_atr(high_s, low_s, close_s, 14, 'close'))
    out.update(calculate_bollinger_bands(close_s, 20, 2.0, 'close'))
    out.update(calculate_volatility_ratio(close_s, 5, 50, 'close'))
    out.update(calculate_parkinson_volatility(high_s, low_s, 20, 'close'))
    out.update(calculate_garman_klass_volatility(high_s, low_s, open_s, close_s, 20, 'close'))

    # Volume-price integration
    out.update(calculate_obv(close_s, volume_s, 'close'))
    out.update(calculate_vwap(high_s, low_s, close_s, volume_s, 'close'))
    if not skip_slow:
        out.update(calculate_adl(high_s, low_s, close_s, volume_s, 'close'))
        out.update(calculate_chaikin_oscillator(high_s, low_s, close_s, volume_s, 3, 10, 'close'))
    out.update(calculate_volume_roc(volume_s, 10, 'volume'))

    # Statistical
    out.update(calculate_rolling_percentiles(close_s, 20, [25, 50, 75], 'close'))
    out.update(calculate_distribution_features(close_s, 30, 'close'))
    out.update(calculate_autocorrelation(close_s, 1, 30, 'close'))
    # Always include (vectorized/lightweight)
    out.update(calculate_hurst_exponent(close_s, 100, 'close'))
    out.update(calculate_entropy(close_s, 20, 'close'))

    # Ratios and candle
    out.update(calculate_price_volume_ratios(close_s, high_s, volume_s, 'close', 'high'))
    out.update(calculate_candle_patterns(open_s, high_s, low_s, close_s, 'close'))
    out.update(calculate_typical_price(high_s, low_s, close_s, 'close'))
    out.update(calculate_ohlc_average(open_s, high_s, low_s, close_s, 'close'))

    # Volatility-adjusted returns using ATR14
    atr_val = out.get('close_atr_14')
    if atr_val is not None and not np.isnan(atr_val) and len(close_s) > 1:
        out.update(calculate_volatility_adjusted_returns(close_s, float(atr_val), 'close', 'atr14'))
    else:
        out.update({'close_vol_adj_return_atr14': np.nan})

    # Time & extremes
    # (time features use the last timestamp of window)
    if len(lb.index) > 0:
        out.update(calculate_time_features(pd.Timestamp(lb.index[-1])))
    out.update(calculate_rolling_extremes(close_s, 20, 'close'))
    # Dominant cycle uses FFT; keep it
    out.update(calculate_dominant_cycle(close_s, 50, 'close'))

    # Additional trend/vol/liquidity
    out.update(calculate_adx(high_s, low_s, close_s, 14, 'close'))
    out.update(calculate_rogers_satchell_volatility(high_s, low_s, open_s, close_s, 20, 'close'))
    out.update(calculate_yang_zhang_volatility(open_s, high_s, low_s, close_s, 20, 'close'))
    out.update(calculate_rvol(volume_s, 20, 'volume'))
    out.update(calculate_donchian_distance(high_s, low_s, close_s, 20, 'close'))
    out.update(calculate_aroon(high_s, low_s, 14, 'close'))
    out.update(calculate_return_zscore(close_s, 20, 'close'))

    # ATR-normalized distance from SMA20 if available
    if sma20 is not None and not np.isnan(sma20) and atr_val is not None and not np.isnan(atr_val) and len(close_s) > 0:
        out.update(calculate_atr_normalized_distance(float(close_s.iloc[-1]), float(sma20), float(atr_val), 'close', 'sma20'))
    else:
        out.update({'close_dist_sma20_atr': np.nan})

    # Liquidity/stats
    out.update(calculate_roll_spread(close_s, 20, 'close'))
    out.update(calculate_amihud_illiquidity(close_s, volume_s, 20, 'close'))
    out.update(calculate_turnover_zscore(close_s, volume_s, 20, 'turnover'))
    # Ljung–Box and OU half-life are vectorized/light-weight; keep them
    out.update(calculate_ljung_box_pvalue(close_s, 5, 100, 'close'))
    if not skip_slow:
        out.update(calculate_permutation_entropy(close_s, 50, 3, 'close'))
    out.update(calculate_ou_half_life(close_s, 100, 'close'))
    out.update(calculate_var_cvar(close_s, 50, 0.05, 'close'))
    # Spectral entropy uses FFT; keep it
    out.update(calculate_spectral_entropy(close_s, 50, 'close'))

    return _append_tf_suffix(out, tf)


def process_dataset(base_dir: str, dataset: str, timeframes: list[str], skip_slow: bool = False, log_every: int = 500) -> pd.DataFrame:
    folder = os.path.join(base_dir, dataset)
    # Load stores
    stores = {}
    for tf in timeframes:
        pkl = os.path.join(folder, f'lookbacks_{tf}.pkl')
        if not os.path.exists(pkl):
            raise FileNotFoundError(f"Missing lookback file: {pkl}")
        stores[tf] = pd.read_pickle(pkl)
    base_index = stores[timeframes[0]]['base_index']

    rows = []
    start_time = time.time()
    last_log = start_time
    for i, ts in enumerate(base_index):
        ts_key = pd.Timestamp(ts).strftime('%Y%m%d_%H%M%S')
        feat_row: Dict[str, float] = {}
        for tf in timeframes:
            lb = stores[tf]['rows'].get(ts_key)
            feats_tf = compute_features_one(lb, tf, skip_slow=skip_slow)
            feat_row.update(feats_tf)
        feat_row['timestamp'] = pd.Timestamp(ts)
        rows.append(feat_row)

        if log_every and ((i + 1) % log_every == 0 or (i + 1) == len(base_index)):
            now = time.time()
            elapsed = now - start_time
            batch_elapsed = now - last_log
            speed = (i + 1) / elapsed if elapsed > 0 else float('inf')
            batch_speed = log_every / batch_elapsed if batch_elapsed > 0 else float('inf')
            remaining = len(base_index) - (i + 1)
            eta = remaining / speed if speed > 0 else float('inf')
            print(f"Processed {i+1}/{len(base_index)} rows | avg {speed:.2f} rows/s | last {batch_speed:.2f} rows/s | ETA ~{eta/60:.1f} min")
            last_log = now

    df_feat = pd.DataFrame(rows).set_index('timestamp')
    return df_feat


def main():
    parser = argparse.ArgumentParser(description='Process lookback PKLs to feature table')
    parser.add_argument('--dataset', required=True, help='Dataset folder name under base dir (e.g., BINANCE_BTCUSDT.P, 60)')
    parser.add_argument('--base-dir', default='/Volumes/Extreme SSD/trading_data/cex/lookbacks', help='Base directory containing lookbacks')
    parser.add_argument('--timeframes', nargs='+', default=['1H', '4H', '12H', '1D'])
    parser.add_argument('--output', default='features.parquet', help='Output file (.parquet or .csv)')
    parser.add_argument('--skip-slow', action='store_true', help='Skip slower features to speed up processing')
    parser.add_argument('--log-every', type=int, default=500, help='Log progress every N rows')
    args = parser.parse_args()

    df_feat = process_dataset(args.base_dir, args.dataset, args.timeframes, skip_slow=args.skip_slow, log_every=args.log_every)
    out_path = os.path.join(args.base_dir, args.dataset, args.output)
    if out_path.lower().endswith('.csv'):
        df_feat.to_csv(out_path, index=True)
    else:
        df_feat.to_parquet(out_path)
    print(f"Wrote features: {out_path} (rows={len(df_feat)}, cols={len(df_feat.columns)})")


if __name__ == '__main__':
    main()


