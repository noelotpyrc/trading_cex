#!/usr/bin/env python3
"""
Merge OI features (unified) with technical features into a single modeling-ready file.

Input Files:
1. Unified features: /Volumes/Extreme SSD/trading_data/cex/ohlvc/btcusdt_perp_1h_unified_with_features.csv
2. Tech features: /Volumes/Extreme SSD/trading_data/cex/training/binance_btcusdt_perp_1h_original/feature_store/features.csv

Output:
- Merged feature file with selected features only
- time_week_of_month_1H generated on the fly

Usage:
    python feature_engineering/oneoff_data_work/merge_features_for_modeling.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# === Configuration ===
UNIFIED_PATH = Path("/Volumes/Extreme SSD/trading_data/cex/ohlvc/btcusdt_perp_1h_unified_with_features.csv")
TECH_FEATURES_PATH = Path("/Volumes/Extreme SSD/trading_data/cex/training/binance_btcusdt_perp_1h_original/feature_store/features.csv")
OUTPUT_PATH = Path("/Volumes/Extreme SSD/trading_data/cex/training/binance_btcusdt_perp_1h_original/feature_store/merged_oi_tech_features.csv")

# Selected features for modeling
OI_FEATURES = [
    'count_long_short_ratio',
    'premium_idx_close',
    'oi_volatility_24h',
    'oi_volatility_168h',
    'oi_ema_distance_168h',
    'oi_accel_scaled_24h',
    'oi_accel_scaled_168h',
    'price_accel_scaled_24h',
    'price_accel_scaled_168h',
    'oi_price_accel_div_24h',
    'oi_price_accel_div_168h',
    'oi_price_accel_product_168h',
    'oi_accel_vs_price_lag1h',
    'price_accel_vs_oi_lag1h',
    'oi_price_momentum_168h',
    'oi_price_divergence_168h',
    'oi_price_ratio_spread',
    'premium_zscore_168h',
]

TECH_FEATURES = [
    'close_parkinson_20_1H',
    'close_hv_20_12H',
    'close_atr_14_1D',
    'close_var_5_50_12H',
    'close_cvar_5_50_4H',
    'close_rsi_14_12H',
    'close_adx_14_12H',
    'close_cum_return_10_1D',
    'close_macd_histogram_12_26_9_1D',
    'close_ma_cross_diff_5_20_12H',
    'volume_roc_10_12H',
    'time_day_of_week_1H',
    'time_month_of_year_1H',
]

# time_week_of_month_1H will be generated on the fly


def log_section(title: str):
    """Print section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def log_df_stats(df: pd.DataFrame, name: str, feature_cols: list = None):
    """Log DataFrame statistics."""
    print(f"\n  [{name}]")
    print(f"    Rows: {len(df):,}")
    print(f"    Columns: {len(df.columns)}")
    
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'])
        print(f"    Date range: {ts.min()} to {ts.max()}")
    
    if feature_cols:
        print(f"\n    Feature NaN counts:")
        for col in feature_cols:
            if col in df.columns:
                na_count = df[col].isna().sum()
                na_pct = 100 * na_count / len(df)
                status = "⚠️" if na_count > 0 else "✓"
                print(f"      {status} {col}: {na_count:,} ({na_pct:.2f}%)")
            else:
                print(f"      ❌ {col}: MISSING FROM DATAFRAME")


def load_unified_features() -> pd.DataFrame:
    """Load unified OI features."""
    log_section("STEP 1: Load Unified OI Features")
    
    print(f"\n  Reading: {UNIFIED_PATH}")
    df = pd.read_csv(UNIFIED_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    
    # Check which OI features are present
    present_oi = [c for c in OI_FEATURES if c in df.columns]
    missing_oi = [c for c in OI_FEATURES if c not in df.columns]
    
    print(f"\n  OI features found: {len(present_oi)}/{len(OI_FEATURES)}")
    if missing_oi:
        print(f"  ⚠️ Missing OI features: {missing_oi}")
    
    # Select only needed columns
    keep_cols = ['timestamp'] + present_oi
    df = df[keep_cols].copy()
    
    log_df_stats(df, "Unified features", present_oi)
    
    return df


def load_tech_features() -> pd.DataFrame:
    """Load technical features."""
    log_section("STEP 2: Load Technical Features")
    
    print(f"\n  Reading: {TECH_FEATURES_PATH}")
    df = pd.read_csv(TECH_FEATURES_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    
    # Check which tech features are present
    present_tech = [c for c in TECH_FEATURES if c in df.columns]
    missing_tech = [c for c in TECH_FEATURES if c not in df.columns]
    
    print(f"\n  Tech features found: {len(present_tech)}/{len(TECH_FEATURES)}")
    if missing_tech:
        print(f"  ⚠️ Missing tech features: {missing_tech}")
    
    # Select only needed columns
    keep_cols = ['timestamp'] + present_tech
    df = df[keep_cols].copy()
    
    log_df_stats(df, "Tech features", present_tech)
    
    return df


def merge_features(df_oi: pd.DataFrame, df_tech: pd.DataFrame) -> pd.DataFrame:
    """Merge OI and tech features on timestamp."""
    log_section("STEP 3: Merge Features")
    
    # Check timestamp overlap
    oi_start, oi_end = df_oi['timestamp'].min(), df_oi['timestamp'].max()
    tech_start, tech_end = df_tech['timestamp'].min(), df_tech['timestamp'].max()
    
    print(f"\n  OI range:   {oi_start} to {oi_end}")
    print(f"  Tech range: {tech_start} to {tech_end}")
    
    overlap_start = max(oi_start, tech_start)
    overlap_end = min(oi_end, tech_end)
    print(f"  Overlap:    {overlap_start} to {overlap_end}")
    
    # Merge on timestamp (inner join)
    print("\n  Performing inner join on timestamp...")
    merged = df_oi.merge(df_tech, on='timestamp', how='inner')
    merged = merged.sort_values('timestamp').reset_index(drop=True)
    
    # Log join statistics
    oi_rows = len(df_oi)
    tech_rows = len(df_tech)
    merged_rows = len(merged)
    
    oi_lost = oi_rows - merged_rows
    tech_lost = tech_rows - merged_rows
    
    print(f"\n  Join Statistics:")
    print(f"    OI rows:     {oi_rows:,}")
    print(f"    Tech rows:   {tech_rows:,}")
    print(f"    Merged rows: {merged_rows:,}")
    print(f"    OI rows not matched:   {oi_lost:,} ({100*oi_lost/oi_rows:.2f}%)")
    print(f"    Tech rows not matched: {tech_lost:,} ({100*tech_lost/tech_rows:.2f}%)")
    
    if oi_lost > 0:
        # Find which OI timestamps were not matched
        oi_only = df_oi[~df_oi['timestamp'].isin(merged['timestamp'])]
        print(f"\n  ⚠️ OI-only timestamps sample: {oi_only['timestamp'].head().tolist()}")
    
    return merged


def generate_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate time_week_of_month_1H on the fly."""
    log_section("STEP 4: Generate Time Features")
    
    print("\n  Generating time_week_of_month_1H...")
    df['time_week_of_month_1H'] = (df['timestamp'].dt.day - 1) // 7 + 1
    
    print(f"    Value distribution:")
    dist = df['time_week_of_month_1H'].value_counts().sort_index()
    for week, count in dist.items():
        print(f"      Week {week}: {count:,}")
    
    return df


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate merged data and handle missing values."""
    log_section("STEP 5: Validate and Clean Data")
    
    all_features = OI_FEATURES + TECH_FEATURES + ['time_week_of_month_1H']
    present_features = [c for c in all_features if c in df.columns]
    
    # Check NaN counts
    print("\n  Checking for missing values in features...")
    total_na = 0
    features_with_na = []
    
    for col in present_features:
        na_count = df[col].isna().sum()
        if na_count > 0:
            na_pct = 100 * na_count / len(df)
            print(f"    ⚠️ {col}: {na_count:,} NaN ({na_pct:.2f}%)")
            total_na += na_count
            features_with_na.append(col)
    
    if total_na == 0:
        print("    ✓ No missing values in any feature!")
    else:
        print(f"\n  Total NaN cells: {total_na:,}")
        print(f"  Features with NaN: {len(features_with_na)}")
    
    # Check for inf values
    print("\n  Checking for inf values...")
    inf_found = False
    for col in present_features:
        if df[col].dtype in ['float64', 'float32']:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                print(f"    ⚠️ {col}: {inf_count:,} inf values")
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                inf_found = True
    
    if not inf_found:
        print("    ✓ No inf values found!")
    
    # Drop rows with any NaN in features (if any)
    rows_before = len(df)
    df_clean = df.dropna(subset=present_features)
    rows_after = len(df_clean)
    
    if rows_before > rows_after:
        print(f"\n  ⚠️ Dropped {rows_before - rows_after:,} rows with NaN values")
    else:
        print("\n  ✓ No rows dropped")
    
    return df_clean


def save_output(df: pd.DataFrame):
    """Save merged features to output file."""
    log_section("STEP 6: Save Output")
    
    # Final column order
    all_features = OI_FEATURES + TECH_FEATURES + ['time_week_of_month_1H']
    present_features = [c for c in all_features if c in df.columns]
    output_cols = ['timestamp'] + present_features
    
    df_out = df[output_cols].copy()
    
    print(f"\n  Output file: {OUTPUT_PATH}")
    print(f"  Rows: {len(df_out):,}")
    print(f"  Feature columns: {len(present_features)}")
    
    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    df_out.to_csv(OUTPUT_PATH, index=False)
    print("  ✓ Saved successfully!")
    
    # Print final feature list
    print("\n  Features included:")
    for i, col in enumerate(present_features, 1):
        print(f"    {i:2}. {col}")


def main():
    """Main function."""
    print("\n" + "#" * 60)
    print("# MERGE FEATURES FOR MODELING")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 60)
    
    # Load data
    df_oi = load_unified_features()
    df_tech = load_tech_features()
    
    # Merge
    df_merged = merge_features(df_oi, df_tech)
    
    # Generate time features
    df_merged = generate_time_features(df_merged)
    
    # Validate and clean
    df_clean = validate_and_clean(df_merged)
    
    # Save
    save_output(df_clean)
    
    # Final summary
    log_section("FINAL SUMMARY")
    all_features = OI_FEATURES + TECH_FEATURES + ['time_week_of_month_1H']
    present_features = [c for c in all_features if c in df_clean.columns]
    
    print(f"\n  Total features: {len(present_features)}")
    print(f"  Total rows: {len(df_clean):,}")
    print(f"  Date range: {df_clean['timestamp'].min()} to {df_clean['timestamp'].max()}")
    print("\n  Done!")


if __name__ == "__main__":
    main()
