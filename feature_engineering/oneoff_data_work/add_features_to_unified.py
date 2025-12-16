#!/usr/bin/env python3
"""
Add derived features to the unified BTC/USDT perp dataset.

Input: btcusdt_perp_1h_unified.csv (merged OHLCV + metrics + premium)
Output: btcusdt_perp_1h_unified_with_features.csv

Features added:
- OI Z-Score (168h rolling window)
- OI ROC EMA (momentum)
- OI Volatility (sign * diff, like Parkinson)
- OI/Price Scaled Acceleration (raw diff / volatility)
- Scaled Accel Interactions (divergence, product, lead/lag)
- OI EMA distance and slope
- OI-Price EMA interactions (momentum, divergence, ratio spread)
- Premium Z-Score

Usage:
    python feature_engineering/oneoff_data_work/add_features_to_unified.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# === Configuration ===
INPUT_PATH = Path("/Volumes/Extreme SSD/trading_data/cex/ohlvc/btcusdt_perp_1h_unified.csv")
OUTPUT_PATH = Path("/Volumes/Extreme SSD/trading_data/cex/ohlvc/btcusdt_perp_1h_unified_with_features.csv")


def log_df_status(df: pd.DataFrame, stage: str, key_cols: list):
    """Log DataFrame status at a given stage."""
    print(f"\n  --- {stage} ---")
    print(f"  Rows: {len(df):,}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    for col in key_cols:
        if col in df.columns:
            na_count = df[col].isna().sum()
            print(f"  {col}: {na_count} NaN ({100*na_count/len(df):.2f}%)")


def add_oi_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add OI and Price derived features with logging.
    
    Features:
    - OI Z-Score (168h rolling window)
    - OI ROC EMA (momentum)
    - OI Volatility (sign * diff, measures magnitude of change like Parkinson)
    - OI/Price Scaled Acceleration (raw diff / volatility, regime-normalized)
    - Scaled Accel Interactions (divergence, product, lead/lag)
    - OI EMA distance and slope
    - OI-Price momentum and divergence
    """
    print("\n" + "="*60)
    print("Adding OI & Price Features")
    print("="*60)
    
    df = df.copy()
    oi = df['sum_open_interest']
    price = df['close']
    
    # === OI Rate of Change ===
    print("\n  Computing OI pct_change(1)...")
    oi_roc_1h = oi.pct_change(1, fill_method=None)
    print(f"    oi_roc_1h NaN: {oi_roc_1h.isna().sum()}")
    
    # === Price Rate of Change ===
    print("\n  Computing Price pct_change(1)...")
    price_roc_1h = price.pct_change(1, fill_method=None)
    print(f"    price_roc_1h NaN: {price_roc_1h.isna().sum()}")
    
    # === OI Z-Score (168h rolling window) ===
    print("\n  Computing OI Z-Score (168h)...")
    oi_mean_168h = oi.rolling(168, min_periods=24).mean()
    oi_std_168h = oi.rolling(168, min_periods=24).std()
    df['oi_zscore_168h'] = (oi - oi_mean_168h) / oi_std_168h
    print(f"    oi_zscore_168h NaN: {df['oi_zscore_168h'].isna().sum()}")
    
    # === OI ROC EMA (momentum) ===
    print("\n  Computing OI ROC EMA (168h)...")
    df['oi_roc_ema_168h'] = oi_roc_1h.ewm(span=168, adjust=False).mean()
    print(f"    oi_roc_ema_168h NaN: {df['oi_roc_ema_168h'].isna().sum()}")
    
    # === OI Volatility (sign * diff - measures magnitude like Parkinson) ===
    print("\n  Computing OI Volatility (sign * diff)...")
    oi_volatility = np.sign(oi_roc_1h) * oi_roc_1h.diff(1)
    print(f"    oi_volatility raw NaN: {oi_volatility.isna().sum()}")
    df['oi_volatility_24h'] = oi_volatility.ewm(span=24, adjust=False).mean()
    df['oi_volatility_168h'] = oi_volatility.ewm(span=168, adjust=False).mean()
    print(f"    oi_volatility_24h NaN: {df['oi_volatility_24h'].isna().sum()}")
    print(f"    oi_volatility_168h NaN: {df['oi_volatility_168h'].isna().sum()}")
    
    # === OI Acceleration (raw diff, intermediate - not saved to output) ===
    print("\n  Computing OI Acceleration (raw diff)...")
    oi_accel = oi_roc_1h.diff(1)
    oi_accel_24h = oi_accel.ewm(span=24, adjust=False).mean()
    oi_accel_168h = oi_accel.ewm(span=168, adjust=False).mean()
    print(f"    oi_accel_24h NaN: {oi_accel_24h.isna().sum()}")
    print(f"    oi_accel_168h NaN: {oi_accel_168h.isna().sum()}")
    
    # === Price Volatility (sign * diff) ===
    print("\n  Computing Price Volatility (sign * diff)...")
    price_volatility = np.sign(price_roc_1h) * price_roc_1h.diff(1)
    price_volatility_24h = price_volatility.ewm(span=24, adjust=False).mean()
    price_volatility_168h = price_volatility.ewm(span=168, adjust=False).mean()
    print(f"    price_volatility_24h NaN: {price_volatility_24h.isna().sum()}")
    print(f"    price_volatility_168h NaN: {price_volatility_168h.isna().sum()}")
    
    # === Price Acceleration (raw diff, intermediate) ===
    print("\n  Computing Price Acceleration (raw diff)...")
    price_accel = price_roc_1h.diff(1)
    price_accel_24h = price_accel.ewm(span=24, adjust=False).mean()
    price_accel_168h = price_accel.ewm(span=168, adjust=False).mean()
    print(f"    price_accel_24h NaN: {price_accel_24h.isna().sum()}")
    print(f"    price_accel_168h NaN: {price_accel_168h.isna().sum()}")
    
    # === Scaled Acceleration (accel / volatility) ===
    # Edge case: replace 0 volatility with NaN to avoid division by zero
    print("\n  Computing Scaled Acceleration (accel / volatility)...")
    oi_vol_24h_safe = df['oi_volatility_24h'].replace(0, np.nan)
    oi_vol_168h_safe = df['oi_volatility_168h'].replace(0, np.nan)
    price_vol_24h_safe = price_volatility_24h.replace(0, np.nan)
    price_vol_168h_safe = price_volatility_168h.replace(0, np.nan)
    
    df['oi_accel_scaled_24h'] = oi_accel_24h / oi_vol_24h_safe
    df['oi_accel_scaled_168h'] = oi_accel_168h / oi_vol_168h_safe
    df['price_accel_scaled_24h'] = price_accel_24h / price_vol_24h_safe
    df['price_accel_scaled_168h'] = price_accel_168h / price_vol_168h_safe
    
    # Log stats and extreme values
    print(f"    oi_accel_scaled_24h NaN: {df['oi_accel_scaled_24h'].isna().sum()}")
    print(f"    oi_accel_scaled_168h NaN: {df['oi_accel_scaled_168h'].isna().sum()}")
    print(f"    price_accel_scaled_24h NaN: {df['price_accel_scaled_24h'].isna().sum()}")
    print(f"    price_accel_scaled_168h NaN: {df['price_accel_scaled_168h'].isna().sum()}")
    
    # Check for inf values (can happen if volatility is near-zero)
    for col in ['oi_accel_scaled_24h', 'oi_accel_scaled_168h', 
                'price_accel_scaled_24h', 'price_accel_scaled_168h']:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            print(f"    WARNING: {col} has {inf_count} inf values (will be cleaned later)")
    
    # === Scaled Accel Interactions ===
    print("\n  Computing Scaled Accel Interactions...")
    
    # 1. Divergence: OI accel - Price accel (positive = OI accelerating faster)
    df['oi_price_accel_div_24h'] = df['oi_accel_scaled_24h'] - df['price_accel_scaled_24h']
    df['oi_price_accel_div_168h'] = df['oi_accel_scaled_168h'] - df['price_accel_scaled_168h']
    print(f"    oi_price_accel_div_24h NaN: {df['oi_price_accel_div_24h'].isna().sum()}")
    print(f"    oi_price_accel_div_168h NaN: {df['oi_price_accel_div_168h'].isna().sum()}")
    
    # 2. Product: positive = both accelerating same direction
    df['oi_price_accel_product_168h'] = df['oi_accel_scaled_168h'] * df['price_accel_scaled_168h']
    print(f"    oi_price_accel_product_168h NaN: {df['oi_price_accel_product_168h'].isna().sum()}")
    
    # 3. Lead/Lag: does OI lead price?
    df['oi_accel_vs_price_lag1h'] = df['oi_accel_scaled_168h'] - df['price_accel_scaled_168h'].shift(1)
    df['price_accel_vs_oi_lag1h'] = df['price_accel_scaled_168h'] - df['oi_accel_scaled_168h'].shift(1)
    print(f"    oi_accel_vs_price_lag1h NaN: {df['oi_accel_vs_price_lag1h'].isna().sum()}")
    print(f"    price_accel_vs_oi_lag1h NaN: {df['price_accel_vs_oi_lag1h'].isna().sum()}")
    
    # === OI EMA features ===
    print("\n  Computing OI EMA distance and slope...")
    oi_ema_168h = oi.ewm(span=168, adjust=False).mean()
    df['oi_ema_distance_168h'] = (oi - oi_ema_168h) / oi_ema_168h
    df['oi_ema_slope_168h'] = oi_ema_168h.pct_change(1, fill_method=None)
    print(f"    oi_ema_distance_168h NaN: {df['oi_ema_distance_168h'].isna().sum()}")
    print(f"    oi_ema_slope_168h NaN: {df['oi_ema_slope_168h'].isna().sum()}")
    
    # === OI-Price EMA Interactions ===
    print("\n  Computing OI-Price EMA interactions...")
    price_ema_168h = price.ewm(span=168, adjust=False).mean()
    price_ema_distance_168h = (price - price_ema_168h) / price_ema_168h
    price_ema_slope_168h = price_ema_168h.pct_change(1, fill_method=None)
    
    # Momentum: OI slope × Price slope (positive = aligned)
    df['oi_price_momentum_168h'] = df['oi_ema_slope_168h'] * price_ema_slope_168h
    # Divergence: OI distance - Price distance
    df['oi_price_divergence_168h'] = df['oi_ema_distance_168h'] - price_ema_distance_168h
    
    # Ratio Spread: OI EMA ratio - Price EMA ratio
    oi_ema_24h = oi.ewm(span=24, adjust=False).mean()
    price_ema_24h = price.ewm(span=24, adjust=False).mean()
    oi_ema_ratio_24_168 = oi_ema_24h / oi_ema_168h
    price_ema_ratio_24_168 = price_ema_24h / price_ema_168h
    df['oi_price_ratio_spread'] = oi_ema_ratio_24_168 - price_ema_ratio_24_168
    
    print(f"    oi_price_momentum_168h NaN: {df['oi_price_momentum_168h'].isna().sum()}")
    print(f"    oi_price_divergence_168h NaN: {df['oi_price_divergence_168h'].isna().sum()}")
    print(f"    oi_price_ratio_spread NaN: {df['oi_price_ratio_spread'].isna().sum()}")
    
    return df


def add_premium_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Premium Index derived features with logging."""
    print("\n" + "="*60)
    print("Adding Premium Features")
    print("="*60)
    
    df = df.copy()
    premium = df['premium_idx_close']
    
    print("\n  Computing premium z-score (rolling 168h, min_periods=24)...")
    premium_mean = premium.rolling(168, min_periods=24).mean()
    premium_std = premium.rolling(168, min_periods=24).std()
    df['premium_zscore_168h'] = (premium - premium_mean) / premium_std
    print(f"    premium_zscore_168h NaN: {df['premium_zscore_168h'].isna().sum()}")
    
    return df


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean derived features with logging."""
    print("\n" + "="*60)
    print("Cleaning Features")
    print("="*60)
    
    df = df.copy()
    
    feature_cols = [
        # OI z-score and volatility
        'oi_zscore_168h',
        'oi_roc_ema_168h',
        'oi_volatility_24h', 'oi_volatility_168h',
        'oi_ema_distance_168h', 'oi_ema_slope_168h',
        # Scaled acceleration
        'oi_accel_scaled_24h', 'oi_accel_scaled_168h',
        'price_accel_scaled_24h', 'price_accel_scaled_168h',
        # Scaled accel interactions
        'oi_price_accel_div_24h', 'oi_price_accel_div_168h',
        'oi_price_accel_product_168h',
        'oi_accel_vs_price_lag1h', 'price_accel_vs_oi_lag1h',
        # OI-Price EMA interactions
        'oi_price_momentum_168h', 'oi_price_divergence_168h',
        'oi_price_ratio_spread',
        # Premium
        'premium_zscore_168h',
    ]
    
    # Replace inf with NaN
    print("\n  Replacing inf values with NaN...")
    for col in feature_cols:
        if col in df.columns:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                print(f"    {col}: {inf_count} inf values replaced")
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    # Clean extreme EMA distance/slope
    print("\n  Cleaning extreme EMA distance/slope values (>±50%)...")
    for col in ['oi_ema_distance_168h', 'oi_ema_slope_168h']:
        if col in df.columns:
            extreme = ((df[col] <= -0.5) | (df[col] >= 0.5)).sum()
            if extreme > 0:
                print(f"    {col}: {extreme} extreme values replaced with NaN")
                df[col] = df[col].where((df[col] > -0.5) & (df[col] < 0.5), np.nan)
    
    return df


def main():
    """Main function with detailed logging."""
    print("\n" + "#"*60)
    print("# FEATURE DERIVATION - DETAILED LOG")
    print("#"*60)
    
    # Load
    print("\n" + "="*60)
    print("STEP 1: Load Unified Dataset")
    print("="*60)
    
    df = pd.read_csv(INPUT_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    base_cols = ['sum_open_interest', 'count_long_short_ratio', 'premium_idx_close', 'y_logret_24h']
    log_df_status(df, "Loaded data", base_cols)
    
    # STEP 2: Add Premium features FIRST (on full data from 2020-01)
    # This ensures the rolling window has full warmup before OI data starts
    print("\n" + "="*60)
    print("STEP 2: Add Premium Features (on full data)")
    print("="*60)
    df = add_premium_features(df)
    print(f"\n  premium_zscore_168h NaN: {df['premium_zscore_168h'].isna().sum()}")
    
    # STEP 3: Filter to OI data
    print("\n" + "="*60)
    print("STEP 3: Filter to OI Data")
    print("="*60)
    
    rows_before = len(df)
    df = df[df['sum_open_interest'].notna()].copy()
    df = df.reset_index(drop=True)
    print(f"\n  Removed {rows_before - len(df):,} rows without OI data")
    log_df_status(df, "After OI filter", base_cols + ['premium_zscore_168h'])
    
    # STEP 4: Filter date range
    print("\n" + "="*60)
    print("STEP 4: Filter Date Range (up to 2025-10-31)")
    print("="*60)
    
    cutoff_date = pd.Timestamp('2025-10-31 23:59:59', tz='UTC')
    rows_before = len(df)
    df = df[df['timestamp'] <= cutoff_date].copy()
    df = df.reset_index(drop=True)
    print(f"\n  Removed {rows_before - len(df):,} rows after cutoff")
    log_df_status(df, "After date filter", base_cols + ['premium_zscore_168h'])
    
    # STEP 5: Add OI & Price features (now on filtered data)
    df = add_oi_price_features(df)
    df = clean_features(df)
    
    # STEP 6: Remove warmup rows (only for OI features now)
    print("\n" + "="*60)
    print("STEP 6: Remove Warmup Rows")
    print("="*60)
    
    # Find first row where all features are non-null
    feature_cols = [
        # OI z-score and volatility
        'oi_zscore_168h',
        'oi_roc_ema_168h',
        'oi_volatility_24h', 'oi_volatility_168h',
        'oi_ema_distance_168h', 'oi_ema_slope_168h',
        # Scaled acceleration
        'oi_accel_scaled_24h', 'oi_accel_scaled_168h',
        'price_accel_scaled_24h', 'price_accel_scaled_168h',
        # Scaled accel interactions
        'oi_price_accel_div_24h', 'oi_price_accel_div_168h',
        'oi_price_accel_product_168h',
        'oi_accel_vs_price_lag1h', 'price_accel_vs_oi_lag1h',
        # OI-Price EMA interactions
        'oi_price_momentum_168h', 'oi_price_divergence_168h',
        'oi_price_ratio_spread',
        # Premium
        'premium_zscore_168h',
    ]
    
    print("\n  Checking first few rows for NaN (warmup):")
    for i in range(min(30, len(df))):
        na_count = df.loc[i, feature_cols].isna().sum()
        if na_count > 0:
            print(f"    Row {i}: {na_count} NaN features")
        else:
            print(f"    Row {i}: ALL FEATURES VALID - this is first valid row")
            break
    
    # Remove warmup rows (24 now: oi_zscore has min_periods=24)
    # Premium already had full warmup from 2020-01 before filtering
    warmup_rows = 24
    print(f"\n  Removing first {warmup_rows} warmup rows (oi_zscore min_periods)...")
    df = df.iloc[warmup_rows:].reset_index(drop=True)
    
    # FINAL SUMMARY
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    all_cols = base_cols + feature_cols
    log_df_status(df, "Final dataset", all_cols)
    
    # Check for any remaining NaN
    print("\n  Any remaining NaN in derived features?")
    for col in feature_cols:
        na = df[col].isna().sum()
        if na > 0:
            print(f"    WARNING: {col} has {na} NaN!")
        else:
            print(f"    {col}: 0 NaN ✓")
    
    # Save
    print(f"\n  Saving to: {OUTPUT_PATH}")
    df.to_csv(OUTPUT_PATH, index=False)
    print("  Done!")
    
    return df


if __name__ == "__main__":
    main()
