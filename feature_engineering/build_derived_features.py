#!/usr/bin/env python3
"""
Generate all derived features from raw unified data.

Input: btcusdt_perp_1h_unified_raw.csv
Output: btcusdt_perp_1h_features.csv

Based on feature_generation_plan.md (~127 features).
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import primitives
from feature_engineering.primitives import (
    parkinson_volatility, historical_volatility, rsi, adx, zscore
)

# Import derived features
from feature_engineering.derived_features import (
    # State Variables
    price_vwap_distance_zscore, price_ema_distance_zscore, price_roc_over_volatility,
    # Momentum - Price
    return_autocorr, variance_ratio,
    # Momentum - OI
    oi_price_accel_product, oi_price_momentum,
    # Momentum - Flow
    taker_imb_cvd_slope, taker_imb_zscore, relative_volume, trade_count_lead_price_corr,
    # Mean-Reversion - Price
    pullback_slope_ema, pullback_slope_vwap, mean_cross_rate_ema, mean_cross_rate_vwap,
    # Mean-Reversion - OI/Premium/Spot
    oi_zscore, oi_ema_distance_zscore, premium_zscore, long_short_ratio_zscore,
    spot_vol_zscore, avg_trade_size_zscore, taker_imb_price_corr, avg_trade_size_price_corr,
    # Regime
    efficiency_avg, vol_ratio, oi_volatility, oi_vol_ratio, cvar_var_ratio, tail_skewness,
    premium_vol_ratio, spot_dominance_zscore, spot_dom_vol_ratio,
    # Interactions
    displacement_speed_product, range_chop_interaction, range_stretch_interaction,
    scaled_acceleration, oi_price_ratio_spread, spot_vol_price_corr, oi_vol_price_corr,
    spot_dom_price_corr, spot_dom_oi_corr, trade_count_oi_corr, trade_count_spot_dom_corr,
    relative_amihud, oi_volume_efficiency, oi_volume_efficiency_signed
)

# === Configuration ===
INPUT_PATH = Path("/Volumes/Extreme SSD/trading_data/cex/ohlvc/btcusdt_perp_1h_unified_raw.csv")
OUTPUT_PATH = Path("/Volumes/Extreme SSD/trading_data/cex/ohlvc/btcusdt_perp_1h_derived_features.csv")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time features."""
    ts = pd.to_datetime(df['timestamp'])
    
    # Hour of day (0-23)
    hour = ts.dt.hour
    df['hour_of_day_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_of_day_cos'] = np.cos(2 * np.pi * hour / 24)
    
    # Day of week (0-6)
    dow = ts.dt.dayofweek
    df['day_of_week_sin'] = np.sin(2 * np.pi * dow / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * dow / 7)
    
    # Week of month (1-5)
    wom = (ts.dt.day - 1) // 7 + 1
    df['week_of_month_sin'] = np.sin(2 * np.pi * wom / 5)
    df['week_of_month_cos'] = np.cos(2 * np.pi * wom / 5)
    
    # Month of year (1-12)
    moy = ts.dt.month
    df['month_of_year_sin'] = np.sin(2 * np.pi * moy / 12)
    df['month_of_year_cos'] = np.cos(2 * np.pi * moy / 12)
    
    return df


def add_primitive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add primitive features (volatility, RSI, ADX)."""
    close = df['close']
    high = df['high']
    low = df['low']
    
    # Parkinson volatility
    df['parkinson_volatility_24'] = parkinson_volatility(high, low, 24)
    df['parkinson_volatility_168'] = parkinson_volatility(high, low, 168)
    
    # Historical volatility
    df['historical_volatility_24'] = historical_volatility(close, 24)
    df['historical_volatility_168'] = historical_volatility(close, 168)
    
    # RSI
    df['rsi_24'] = rsi(close, 24)
    df['rsi_168'] = rsi(close, 168)
    df['rsi_720'] = rsi(close, 720)
    
    # ADX
    df['adx_24'] = adx(high, low, close, 24)
    df['adx_168'] = adx(high, low, close, 168)
    df['adx_720'] = adx(high, low, close, 720)
    
    return df


def add_state_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add state variable features."""
    o, h, l, c, v = df['open'], df['high'], df['low'], df['close'], df['volume']
    
    # Price VWAP distance z-score
    df['price_vwap_distance_zscore_24_168'] = price_vwap_distance_zscore(o, h, l, c, v, 24, 168)
    df['price_vwap_distance_zscore_168_168'] = price_vwap_distance_zscore(o, h, l, c, v, 168, 168)
    df['price_vwap_distance_zscore_720_168'] = price_vwap_distance_zscore(o, h, l, c, v, 720, 168)
    
    # Price EMA distance z-score
    df['price_ema_distance_zscore_24_168'] = price_ema_distance_zscore(c, 24, 168)
    df['price_ema_distance_zscore_24_720'] = price_ema_distance_zscore(c, 24, 720)
    
    # Price ROC over volatility
    df['price_roc_over_volatility_24_24'] = price_roc_over_volatility(c, h, l, 24, 24)
    df['price_roc_over_volatility_24_168'] = price_roc_over_volatility(c, h, l, 24, 168)
    df['price_roc_over_volatility_24_720'] = price_roc_over_volatility(c, h, l, 24, 720)
    
    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum persistence features."""
    c = df['close']
    oi = df['sum_open_interest']
    spot_vol = df['spot_volume']
    taker_buy = df['spot_taker_buy_volume']
    num_trades = df['spot_num_trades']
    vol = df['volume']
    ts = df['timestamp']
    
    # Price-based momentum
    df['return_autocorr_48'] = return_autocorr(c, 48)
    df['return_autocorr_168'] = return_autocorr(c, 168)
    df['variance_ratio_24_48'] = variance_ratio(c, 24, 48)
    df['variance_ratio_24_168'] = variance_ratio(c, 24, 168)
    df['variance_ratio_24_720'] = variance_ratio(c, 24, 720)
    
    # OI-based momentum
    df['oi_price_accel_product_168'] = oi_price_accel_product(oi, c, 168)
    df['oi_price_momentum_168'] = oi_price_momentum(oi, c, 168)
    
    # Flow-based momentum
    df['taker_imb_cvd_slope_24'] = taker_imb_cvd_slope(taker_buy, spot_vol, 24)
    df['taker_imb_cvd_slope_168'] = taker_imb_cvd_slope(taker_buy, spot_vol, 168)
    df['taker_imb_zscore_168'] = taker_imb_zscore(taker_buy, spot_vol, 168)
    df['relative_volume_7'] = relative_volume(vol, ts, 7)
    df['relative_volume_14'] = relative_volume(vol, ts, 14)
    df['relative_volume_30'] = relative_volume(vol, ts, 30)
    df['trade_count_lead_price_corr_24_168'] = trade_count_lead_price_corr(num_trades, c, 24, 168)
    
    return df


def add_mean_reversion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add mean-reversion strength features."""
    o, h, l, c, v = df['open'], df['high'], df['low'], df['close'], df['volume']
    oi = df['sum_open_interest']
    prem = df['premium_idx_close']
    ls = df['long_short_ratio']
    spot_vol = df['spot_volume']
    quote_vol = df['spot_volume'] * df['spot_close']  # approximate quote volume
    num_trades = df['spot_num_trades']
    taker_buy = df['spot_taker_buy_volume']
    
    # Pullback slope - EMA
    for ema_span in [24, 168, 720]:
        for window in [48, 168]:
            df[f'pullback_slope_ema_{ema_span}_{window}'] = pullback_slope_ema(c, ema_span, window)
    
    # Pullback slope - VWAP
    for vwap_window in [24, 168, 720]:
        for window in [48, 168]:
            df[f'pullback_slope_vwap_{vwap_window}_{window}'] = pullback_slope_vwap(o, h, l, c, v, vwap_window, window)
    
    # Mean cross rate - EMA
    for ema_span in [24, 168, 720]:
        for window in [48, 168]:
            df[f'mean_cross_rate_ema_{ema_span}_{window}'] = mean_cross_rate_ema(c, ema_span, window)
    
    # Mean cross rate - VWAP
    for vwap_window in [24, 168, 720]:
        for window in [48, 168]:
            df[f'mean_cross_rate_vwap_{vwap_window}_{window}'] = mean_cross_rate_vwap(o, h, l, c, v, vwap_window, window)
    
    # OI-based reversion
    df['oi_zscore_168'] = oi_zscore(oi, 168)
    df['oi_ema_distance_zscore_24_168'] = oi_ema_distance_zscore(oi, 24, 168)
    df['oi_ema_distance_zscore_168_168'] = oi_ema_distance_zscore(oi, 168, 168)
    df['oi_ema_distance_zscore_720_168'] = oi_ema_distance_zscore(oi, 720, 168)
    
    # Premium/L/S/Spot reversion
    df['premium_zscore_168'] = premium_zscore(prem, 168)
    df['long_short_ratio_zscore_48'] = long_short_ratio_zscore(ls, 48)
    df['long_short_ratio_zscore_168'] = long_short_ratio_zscore(ls, 168)
    df['spot_vol_zscore_168'] = spot_vol_zscore(spot_vol, 168)
    df['avg_trade_size_zscore_48'] = avg_trade_size_zscore(quote_vol, num_trades, 48)
    df['avg_trade_size_zscore_168'] = avg_trade_size_zscore(quote_vol, num_trades, 168)
    
    # Flow reversion
    df['taker_imb_price_corr_168'] = taker_imb_price_corr(taker_buy, spot_vol, c, 168)
    df['avg_trade_size_price_corr_168'] = avg_trade_size_price_corr(quote_vol, num_trades, c, 168)
    
    return df


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add regime indicator features."""
    o, h, l, c = df['open'], df['high'], df['low'], df['close']
    oi = df['sum_open_interest']
    prem = df['premium_idx_close']
    spot_vol = df['spot_volume']
    perp_vol = df['volume']
    
    # Price-based regime
    df['efficiency_avg_24'] = efficiency_avg(o, h, l, c, 24)
    df['efficiency_avg_168'] = efficiency_avg(o, h, l, c, 168)
    df['vol_ratio_24_168'] = vol_ratio(h, l, 24, 168)
    df['vol_ratio_24_720'] = vol_ratio(h, l, 24, 720)
    
    # OI regime
    df['oi_volatility_168'] = oi_volatility(oi, 168)
    df['oi_vol_ratio_24_168'] = oi_vol_ratio(oi, 24, 168)
    
    # Tail regime
    df['cvar_var_ratio_168'] = cvar_var_ratio(c, 168)
    df['cvar_var_ratio_720'] = cvar_var_ratio(c, 720)
    df['tail_skewness_168'] = tail_skewness(c, 168)
    df['tail_skewness_720'] = tail_skewness(c, 720)
    
    # Premium regime
    df['premium_vol_ratio_24_48'] = premium_vol_ratio(prem, 24, 48)
    df['premium_vol_ratio_24_168'] = premium_vol_ratio(prem, 24, 168)
    
    # Spot regime
    df['spot_dominance_zscore_168'] = spot_dominance_zscore(spot_vol, perp_vol, 168)
    df['spot_dom_vol_ratio_24_168'] = spot_dom_vol_ratio(spot_vol, perp_vol, 24, 168)
    
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features."""
    o, h, l, c, v = df['open'], df['high'], df['low'], df['close'], df['volume']
    oi = df['sum_open_interest']
    spot_vol = df['spot_volume']
    perp_vol = df['volume']
    num_trades = df['spot_num_trades']
    
    # Core physics interactions
    df['displacement_speed_product_168_24'] = displacement_speed_product(o, h, l, c, v, 168, 24)
    df['displacement_speed_product_168_48'] = displacement_speed_product(o, h, l, c, v, 168, 48)
    df['displacement_speed_product_720_24'] = displacement_speed_product(o, h, l, c, v, 720, 24)
    df['displacement_speed_product_720_48'] = displacement_speed_product(o, h, l, c, v, 720, 48)
    
    df['range_chop_interaction_24'] = range_chop_interaction(h, l, o, c, 24)
    df['range_chop_interaction_168'] = range_chop_interaction(h, l, o, c, 168)
    
    df['range_stretch_interaction_168_24'] = range_stretch_interaction(o, h, l, c, v, 168, 24)
    df['range_stretch_interaction_168_168'] = range_stretch_interaction(o, h, l, c, v, 168, 168)
    df['range_stretch_interaction_720_24'] = range_stretch_interaction(o, h, l, c, v, 720, 24)
    df['range_stretch_interaction_720_168'] = range_stretch_interaction(o, h, l, c, v, 720, 168)
    
    # Utility
    df['scaled_acceleration_24'] = scaled_acceleration(c, 24)
    df['scaled_acceleration_168'] = scaled_acceleration(c, 168)
    
    # OI-Price interactions
    df['oi_price_ratio_spread_24_168'] = oi_price_ratio_spread(oi, c, 24, 168)
    
    # Spot-Price interactions
    df['spot_vol_price_corr_168'] = spot_vol_price_corr(spot_vol, c, 168)
    df['oi_vol_price_corr_168'] = oi_vol_price_corr(oi, c, 168)
    df['spot_dom_price_corr_24'] = spot_dom_price_corr(spot_vol, perp_vol, c, 24)
    df['spot_dom_price_corr_168'] = spot_dom_price_corr(spot_vol, perp_vol, c, 168)
    df['spot_dom_oi_corr_24'] = spot_dom_oi_corr(spot_vol, perp_vol, oi, 24)
    df['spot_dom_oi_corr_168'] = spot_dom_oi_corr(spot_vol, perp_vol, oi, 168)
    
    # Trade interactions
    df['trade_count_oi_corr_168'] = trade_count_oi_corr(num_trades, oi, 168)
    df['trade_count_spot_dom_corr_168'] = trade_count_spot_dom_corr(num_trades, spot_vol, perp_vol, 168)
    
    # Amihud / OI efficiency
    df['relative_amihud_168'] = relative_amihud(c, v, 168)
    df['oi_volume_efficiency_24_168'] = oi_volume_efficiency(oi, v, 24, 168)
    
    eff_pos, eff_neg = oi_volume_efficiency_signed(oi, v, 48, 168)
    df['oi_volume_efficiency_signed_pos_48_168'] = eff_pos
    df['oi_volume_efficiency_signed_neg_48_168'] = eff_neg
    
    eff_pos2, eff_neg2 = oi_volume_efficiency_signed(oi, v, 168, 168)
    df['oi_volume_efficiency_signed_pos_168_168'] = eff_pos2
    df['oi_volume_efficiency_signed_neg_168_168'] = eff_neg2
    
    return df


def main():
    print("=" * 60)
    print("Feature Generation Script")
    print("=" * 60)
    
    # Load raw data
    print(f"\nLoading data from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"  Rows: {len(df):,}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # === Step 1: Forward-fill zeros BEFORE filtering ===
    # Zeros in raw data indicate missing/gap values - forward-fill them
    # EXCEPT premium columns (zeros are valid for premium)
    # NaN values are kept as-is
    print("\nStep 1: Forward-filling zeros for numeric columns (except premium)...")
    
    # Columns to skip zero-fill (zeros are valid values)
    skip_zero_fill = ['premium_idx_open', 'premium_idx_high', 'premium_idx_low', 'premium_idx_close']
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col in skip_zero_fill:
            continue
        zero_count = (df[col] == 0).sum()
        if zero_count > 0:
            # Track original NaN positions
            original_nan_mask = df[col].isna()
            
            # Replace zeros with NaN, forward-fill
            df[col] = df[col].replace(0, np.nan).ffill()
            
            # Restore original NaN positions
            df.loc[original_nan_mask, col] = np.nan
            
            print(f"    {col}: forward-filled {zero_count} zeros")
    
    # === Step 2: Generate features on FULL continuous time series ===
    # NO filtering here - keep all rows to avoid time discontinuities
    print("\nStep 2: Generating features...")
    
    print("  Time features...")
    df = add_time_features(df)
    
    print("  Primitive features...")
    df = add_primitive_features(df)
    
    print("  State features...")
    df = add_state_features(df)
    
    print("  Momentum features...")
    df = add_momentum_features(df)
    
    print("  Mean-reversion features...")
    df = add_mean_reversion_features(df)
    
    print("  Regime features...")
    df = add_regime_features(df)
    
    print("  Interaction features...")
    df = add_interaction_features(df)
    
    # Clean inf values
    print("\nStep 3: Cleaning inf values...")
    for col in df.select_dtypes(include=[np.number]).columns:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            print(f"  {col}: {inf_count} inf values replaced with NaN")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    # === Step 4: Filter by time range ===
    # Filter to date range where OI data is available (2021-12-01 onwards)
    # End at 2025-08-31 to exclude incomplete data
    print("\nStep 4: Filtering by time range...")
    start_date = pd.Timestamp('2021-12-01')
    end_date = pd.Timestamp('2025-08-31 23:59:59')
    
    rows_before = len(df)
    df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)].copy()
    df = df.reset_index(drop=True)
    print(f"    Date range: {start_date.date()} to {end_date.date()}")
    print(f"    Removed {rows_before - len(df):,} rows outside range")
    print(f"    Rows after filter: {len(df):,}")
    
    # Summary
    feature_cols = [c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                                        'sum_open_interest', 'sum_open_interest_value',
                                                        'long_short_ratio', 'long_account', 'short_account',
                                                        'premium_idx_open', 'premium_idx_high', 'premium_idx_low', 'premium_idx_close',
                                                        'spot_open', 'spot_high', 'spot_low', 'spot_close',
                                                        'spot_volume', 'spot_num_trades', 'spot_taker_buy_volume']]
    print(f"\nGenerated {len(feature_cols)} features")
    
    # Save
    print(f"\nSaving to: {OUTPUT_PATH}")
    df.to_csv(OUTPUT_PATH, index=False)
    print("Done!")
    
    return df


if __name__ == "__main__":
    main()
