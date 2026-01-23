"""
VWAP Momentum Strategy - Fast Version with Precomputed Features

This is an optimized version that reads precomputed features from the CSV:
- parkinson_30, parkinson_1440, parkinson_ratio
- volume_sma_20, volume_sma_30, volume_ratio_20, volume_ratio_30
- vwap_60
- intrabar_return, vwap_gap_15
- cross_direction_15

Entry Logic:
- Bar 0: Cross detected
- Bars 1-4: Measure momentum (Filters A, B, C)
- Bar 5: Check avg trade size z-score (Filter H)
- Bar 6: Enter at OPEN if ALL filters pass

Exit Logic:
- Checkpoints at bars after entry (configurable)
- Stop loss (optional)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def check_momentum_filters(
    cross_idx: int,
    current_idx: int,
    is_long: bool,
    intrabar_returns: np.ndarray,
    vwap_gaps: np.ndarray,
    opens: np.ndarray,
    closes: np.ndarray,
    gap_widening_required: int = 3,
    min_cumulative_move_bps: float = 5.0,
    enable_filter_a: bool = True,
    enable_filter_b: bool = True,
    enable_filter_c: bool = True,
    entry_bar: int = 5,
) -> tuple[bool, dict]:
    """
    Check if momentum filters pass for entry.
    Uses precomputed intrabar_returns and vwap_gaps.

    Args:
        enable_filter_a: Enable intrabar returns filter (3 of 4 rule)
        enable_filter_b: Enable gap widening filter
        enable_filter_c: Enable cumulative move filter
        entry_bar: Entry bar offset (default 6)
    """
    if current_idx < cross_idx + entry_bar:
        return False, {}

    bar_indices = [cross_idx + i for i in range(5)]

    # Filter A: Intrabar returns for bars 1, 2, 3, 4 (precomputed)
    bar_returns = [intrabar_returns[bar_indices[i]] for i in range(1, 5)]

    if is_long:
        positive_returns = sum(1 for r in bar_returns if r > 0)
        filter_a_pass = positive_returns >= 3 if enable_filter_a else True
    else:
        negative_returns = sum(1 for r in bar_returns if r < 0)
        filter_a_pass = negative_returns >= 3 if enable_filter_a else True

    # Filter B: Gap widening for bars 1, 2, 3, 4 (using precomputed vwap_gap)
    gaps = [vwap_gaps[bar_indices[i]] for i in range(5)]

    gap_diffs = []
    for i in range(1, 5):
        gap_diff = gaps[i] - gaps[i-1]
        if not is_long:
            gap_diff = -gap_diff
        gap_diffs.append(gap_diff)

    positive_gap_diffs = sum(1 for gd in gap_diffs if gd > 0)
    filter_b_pass = positive_gap_diffs >= gap_widening_required if enable_filter_b else True

    # Filter C: Cumulative move from bar 1 open to bar 4 close
    bar1_open = opens[bar_indices[1]]
    bar4_close = closes[bar_indices[4]]
    cumulative_return_pct = (bar4_close / bar1_open - 1) * 100

    if is_long:
        filter_c_pass = cumulative_return_pct >= (min_cumulative_move_bps / 100) if enable_filter_c else True
    else:
        filter_c_pass = cumulative_return_pct <= -(min_cumulative_move_bps / 100) if enable_filter_c else True

    details = {
        'intrabar_returns': bar_returns,
        'positive_returns': positive_returns if is_long else negative_returns,
        'filter_a_pass': filter_a_pass,
        'gaps': gaps,
        'gap_diffs': gap_diffs,
        'positive_gap_diffs': positive_gap_diffs,
        'filter_b_pass': filter_b_pass,
        'bar1_open': bar1_open,
        'bar4_close': bar4_close,
        'cumulative_return_pct': cumulative_return_pct,
        'filter_c_pass': filter_c_pass,
    }

    return filter_a_pass and filter_b_pass and filter_c_pass, details


def precalculate_crosses(df: pd.DataFrame, cross_col: str = 'cross_direction_15') -> pd.DataFrame:
    """
    Pre-calculate cross information for all bars using specified cross column.
    """
    df = df.copy()

    latest_cross_idx = None
    latest_cross_dir = None

    cross_indices = []
    cross_directions = []

    cross_values = df[cross_col].values

    for i in range(len(df)):
        # Store latest cross info from BEFORE this bar
        cross_indices.append(latest_cross_idx)
        cross_directions.append(latest_cross_dir)

        # Check if bar i itself is a cross (from precomputed column)
        cross_dir = cross_values[i]
        if pd.notna(cross_dir):
            latest_cross_idx = i
            latest_cross_dir = cross_dir

    df['latest_cross_index'] = cross_indices
    df['latest_cross_direction'] = cross_directions

    return df


def run_strategy(
    df: pd.DataFrame,
    entry_bar: int = 6,
    checkpoint_bars: list = [10, 15, 20],
    fee_pct: float = 0.03,
    profit_adjustment_pct: float = 0.0,
    gap_widening_required: int = 3,
    min_cumulative_move_bps: float = 5.0,
    parkinson_threshold: float = 1.5,
    enable_cooldown: bool = False,
    cooldown_count: int = 2,
    stop_loss_pct: float = None,
    volume_ratio_threshold: float = None,
    volume_ratio_col: str = 'volume_ratio_30',
    enable_vwap_alignment: bool = False,
    vwap_alignment_col: str = 'vwap_60',
    taker_imb_long_col: str = None,
    taker_imb_short_col: str = None,
    avg_trade_size_zscore_threshold: float = None,
    avg_trade_size_zscore_col: str = 'avg_trade_size_zscore_60',
    avg_trade_size_zscore_peak: bool = False,
    enable_filter_a: bool = True,
    enable_filter_b: bool = True,
    enable_filter_c: bool = True,
    vwap_window: int = 15,  # VWAP window for cross detection (15, 30, 60, etc.)
    direction_filter: str = 'both',  # 'both', 'long', or 'short'
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run VWAP momentum strategy using precomputed features.

    Args:
        df: DataFrame with precomputed features (from process_data.py)
        entry_bar: Enter N bars after cross (default 6, was 5)
        checkpoint_bars: Bars after entry to check exit conditions
        fee_pct: Trading fee percentage
        profit_adjustment_pct: Adjustment for profitability check
        gap_widening_required: Number of gap diffs required (2 or 3)
        min_cumulative_move_bps: Minimum cumulative move in bps
        parkinson_threshold: Parkinson ratio threshold for entry
        enable_cooldown: Enable cooldown after stop loss
        cooldown_count: Number of crosses to skip after stop loss
        stop_loss_pct: Stop loss percentage (negative, e.g., -0.1)
        volume_ratio_threshold: Volume ratio threshold (uses precomputed column)
        volume_ratio_col: Which volume ratio column to use
        enable_vwap_alignment: Require close aligned with VWAP (higher timeframe)
        vwap_alignment_col: Which VWAP column to use for alignment (vwap_60, vwap_240, etc)
        taker_imb_long_col: Taker imbalance column for longs (requires >= 0), None to disable
        taker_imb_short_col: Taker imbalance column for shorts (requires < 0), None to disable
        avg_trade_size_zscore_threshold: Min avg trade size z-score for entry (e.g., 0.5), None to disable
        avg_trade_size_zscore_col: Which z-score column to use (avg_trade_size_zscore_60/240/1440)
        avg_trade_size_zscore_peak: If True, require bar 5's z-score to be highest among bars 0-5
        enable_filter_a: Enable intrabar returns filter (3 of 4 rule)
        enable_filter_b: Enable gap widening filter
        enable_filter_c: Enable cumulative move filter
        vwap_window: VWAP window size for cross detection (15, 30, 60, etc.)
        verbose: Print progress
    """
    if entry_bar != 6:
        print(f"WARNING: This strategy is designed for entry_bar=6")

    # Select column names based on VWAP window
    # Try HLC variants first (for VWAP-15), then OHLC/4 variants
    if vwap_window == 15:
        # Prefer HLC/3 for VWAP-15 (legacy behavior)
        if 'cross_direction_15_hlc' in df.columns:
            cross_col = 'cross_direction_15_hlc'
            vwap_gap_col = 'vwap_gap_15_hlc'
            vwap_col = 'vwap_15_hlc'
        else:
            cross_col = 'cross_direction_15'
            vwap_gap_col = 'vwap_gap_15'
            vwap_col = 'vwap_15'
    else:
        # For other windows, use OHLC/4 naming convention
        cross_col = f'cross_direction_{vwap_window}'
        vwap_gap_col = f'vwap_gap_{vwap_window}'
        vwap_col = f'vwap_{vwap_window}'

    if verbose:
        print(f"Running VWAP Momentum Strategy (Fast Version)...")
        print(f"  VWAP window: {vwap_window} bars (using {cross_col})")
        print(f"  Direction: {direction_filter.upper()}")
        print(f"  Entry: Bar {entry_bar} after cross")
        print(f"  Filters:")
        if enable_filter_a:
            print(f"    A: 3+ out of 4 intrabar returns agree with direction")
        else:
            print(f"    A: DISABLED")
        if enable_filter_b:
            print(f"    B: {gap_widening_required}+ out of 4 gap diffs show widening")
        else:
            print(f"    B: DISABLED")
        if enable_filter_c:
            print(f"    C: Cumulative move >= {min_cumulative_move_bps} bps")
        else:
            print(f"    C: DISABLED")
        print(f"    D: Parkinson ratio >= {parkinson_threshold}")
        if volume_ratio_threshold is not None:
            print(f"    E: {volume_ratio_col} >= {volume_ratio_threshold}")
        if enable_vwap_alignment:
            print(f"    F: Close aligned with {vwap_alignment_col}")
        if taker_imb_long_col:
            print(f"    G (long): {taker_imb_long_col} >= 0")
        if taker_imb_short_col:
            print(f"    G (short): {taker_imb_short_col} < 0")
        if avg_trade_size_zscore_threshold is not None:
            print(f"    H: {avg_trade_size_zscore_col} >= {avg_trade_size_zscore_threshold}")
        if avg_trade_size_zscore_peak:
            print(f"    H: {avg_trade_size_zscore_col} at bar 5 is highest among bars 0-5")
        print(f"  Checkpoints: {checkpoint_bars}")
        print(f"  Stop Loss: {stop_loss_pct}%" if stop_loss_pct else "  Stop Loss: DISABLED")
        print(f"  Fee: {fee_pct}%")
        print()

    # Prepare data
    df_sorted = df.sort_values('datetime_utc').reset_index(drop=True)

    # Validate required columns exist (no on-the-fly computation)
    required_cols = ['parkinson_ratio', 'intrabar_return', vwap_gap_col, cross_col]
    if volume_ratio_threshold is not None:
        required_cols.append(volume_ratio_col)
    if enable_vwap_alignment:
        required_cols.append(vwap_alignment_col)
    if taker_imb_long_col:
        required_cols.append(taker_imb_long_col)
    if taker_imb_short_col:
        required_cols.append(taker_imb_short_col)
    if avg_trade_size_zscore_threshold is not None or avg_trade_size_zscore_peak:
        required_cols.append(avg_trade_size_zscore_col)

    missing = [c for c in required_cols if c not in df_sorted.columns]
    if missing:
        raise ValueError(f"Missing precomputed columns: {missing}. Run process_data.py first.")

    # Pre-calculate crosses
    if verbose:
        print("Processing cross events...")
    df_sorted = precalculate_crosses(df_sorted, cross_col=cross_col)

    if verbose:
        print(f"Data ready: {len(df_sorted):,} bars")
        print(f"Parkinson ratio >= {parkinson_threshold}: "
              f"{(df_sorted['parkinson_ratio'] >= parkinson_threshold).mean() * 100:.1f}%")
        print()

    # Convert to numpy arrays for faster access
    n_bars = len(df_sorted)
    opens = df_sorted['open'].values
    closes = df_sorted['close'].values
    highs = df_sorted['high'].values
    lows = df_sorted['low'].values
    datetimes = df_sorted['datetime_utc'].values
    cross_indices = df_sorted['latest_cross_index'].values
    cross_directions = df_sorted['latest_cross_direction'].values

    # Precomputed features
    parkinson_ratios = df_sorted['parkinson_ratio'].values
    intrabar_returns = df_sorted['intrabar_return'].values
    vwap_gaps = df_sorted[vwap_gap_col].values

    if volume_ratio_threshold is not None:
        volume_ratios = df_sorted[volume_ratio_col].values
    if enable_vwap_alignment:
        vwap_alignment = df_sorted[vwap_alignment_col].values
    if taker_imb_long_col:
        taker_imb_long = df_sorted[taker_imb_long_col].values
    if taker_imb_short_col:
        taker_imb_short = df_sorted[taker_imb_short_col].values
    if avg_trade_size_zscore_threshold is not None or avg_trade_size_zscore_peak:
        avg_trade_size_zscore = df_sorted[avg_trade_size_zscore_col].values

    checkpoint_set = set(checkpoint_bars)
    max_checkpoint = checkpoint_bars[-1]

    # Initialize state
    position = None
    trades = []
    skip_crosses_count = 0
    filter_stats = {
        'total_potential_entries': 0,
        'filter_abc_passed': 0,
        'parkinson_passed': 0,
        'skipped_due_to_parkinson': 0,
        'skipped_due_to_cooldown': 0,
        'skipped_due_to_volume': 0,
        'skipped_due_to_vwap_alignment': 0,
        'skipped_due_to_taker_imb': 0,
        'skipped_due_to_avg_trade_size': 0,
    }

    # Process each bar
    for i in range(n_bars):
        # =================================================================
        # ENTRY CHECK
        # =================================================================
        if position is None:
            cross_idx = cross_indices[i]
            cross_dir = cross_directions[i]

            if not pd.isna(cross_idx):
                bars_since_cross = i - int(cross_idx)

                if bars_since_cross == entry_bar:
                    is_long = (cross_dir == 'above')

                    # Direction filter
                    if direction_filter == 'long' and not is_long:
                        continue
                    if direction_filter == 'short' and is_long:
                        continue

                    filters_pass, _ = check_momentum_filters(
                        cross_idx=int(cross_idx),
                        current_idx=i,
                        is_long=is_long,
                        intrabar_returns=intrabar_returns,
                        vwap_gaps=vwap_gaps,
                        opens=opens,
                        closes=closes,
                        gap_widening_required=gap_widening_required,
                        min_cumulative_move_bps=min_cumulative_move_bps,
                        enable_filter_a=enable_filter_a,
                        enable_filter_b=enable_filter_b,
                        enable_filter_c=enable_filter_c,
                        entry_bar=entry_bar,
                    )

                    filter_stats['total_potential_entries'] += 1

                    if filters_pass:
                        filter_stats['filter_abc_passed'] += 1

                        # Filter D: Parkinson (use bar 5 = i-1 when entry_bar=6)
                        if parkinson_ratios[i - 1] < parkinson_threshold:
                            filter_stats['skipped_due_to_parkinson'] += 1
                            continue

                        filter_stats['parkinson_passed'] += 1

                        # Filter E: Volume ratio
                        if volume_ratio_threshold is not None:
                            # Avg volume ratio of bars 1-4
                            bar_indices_1_4 = [int(cross_idx) + j for j in range(1, 5)]
                            avg_vol_ratio = np.mean([volume_ratios[idx] for idx in bar_indices_1_4])
                            if pd.isna(avg_vol_ratio) or avg_vol_ratio < volume_ratio_threshold:
                                filter_stats['skipped_due_to_volume'] += 1
                                continue

                        # Filter F: VWAP alignment (higher timeframe)
                        if enable_vwap_alignment:
                            close_price = closes[i - 1]
                            vwap_val = vwap_alignment[i - 1]
                            if pd.isna(vwap_val):
                                filter_stats['skipped_due_to_vwap_alignment'] += 1
                                continue
                            if is_long and close_price <= vwap_val:
                                filter_stats['skipped_due_to_vwap_alignment'] += 1
                                continue
                            elif not is_long and close_price >= vwap_val:
                                filter_stats['skipped_due_to_vwap_alignment'] += 1
                                continue

                        # Filter G: Taker imbalance alignment
                        if is_long and taker_imb_long_col:
                            imb_val = taker_imb_long[i - 1]
                            if pd.isna(imb_val) or imb_val <= 0:
                                filter_stats['skipped_due_to_taker_imb'] += 1
                                continue
                        elif not is_long and taker_imb_short_col:
                            imb_val = taker_imb_short[i - 1]
                            if pd.isna(imb_val) or imb_val >= 0:
                                filter_stats['skipped_due_to_taker_imb'] += 1
                                continue

                        # Filter H: Avg trade size z-score
                        if avg_trade_size_zscore_threshold is not None:
                            # Threshold mode: bar 5's z-score must be >= threshold
                            ats_zscore = avg_trade_size_zscore[i - 1]  # i is bar 6, i-1 is bar 5
                            if pd.isna(ats_zscore) or ats_zscore < avg_trade_size_zscore_threshold:
                                filter_stats['skipped_due_to_avg_trade_size'] += 1
                                continue

                        if avg_trade_size_zscore_peak:
                            # Peak mode: bar 5's z-score must be highest among bars 0-5
                            cross_idx_int = int(cross_idx)
                            bar5_zscore = avg_trade_size_zscore[cross_idx_int + 5]  # bar 5
                            if pd.isna(bar5_zscore):
                                filter_stats['skipped_due_to_avg_trade_size'] += 1
                                continue
                            # Get z-scores for bars 0-4
                            is_peak = True
                            for offset in range(5):  # bars 0, 1, 2, 3, 4
                                other_zscore = avg_trade_size_zscore[cross_idx_int + offset]
                                if pd.notna(other_zscore) and other_zscore >= bar5_zscore:
                                    is_peak = False
                                    break
                            if not is_peak:
                                filter_stats['skipped_due_to_avg_trade_size'] += 1
                                continue

                        # Cooldown check
                        if enable_cooldown and skip_crosses_count > 0:
                            skip_crosses_count -= 1
                            filter_stats['skipped_due_to_cooldown'] += 1
                            continue

                        # ENTER at OPEN
                        position = {
                            'entry_bar_index': i,
                            'entry_price': opens[i],
                            'entry_time': datetimes[i],
                            'is_long': is_long,
                            'cross_bar_index': int(cross_idx),
                            'last_checkpoint_close': None,
                            'checkpoint_index': 0
                        }

        # =================================================================
        # EXIT CHECK
        # =================================================================
        if position is not None:
            bars_held = i - position['entry_bar_index']

            # Stop loss check
            if stop_loss_pct is not None:
                if position['is_long']:
                    stop_price = position['entry_price'] * (1 + stop_loss_pct / 100)
                    if lows[i] <= stop_price:
                        pnl_pct = stop_loss_pct - fee_pct
                        trades.append({
                            'entry_time': position['entry_time'],
                            'exit_time': datetimes[i],
                            'direction': 'long',
                            'entry_price': position['entry_price'],
                            'exit_price': stop_price,
                            'bars_held': bars_held,
                            'exit_reason': 'stop_loss',
                            'pnl_pct': pnl_pct
                        })
                        position = None
                        if enable_cooldown:
                            skip_crosses_count = cooldown_count
                        continue
                else:
                    stop_price = position['entry_price'] * (1 - stop_loss_pct / 100)
                    if highs[i] >= stop_price:
                        pnl_pct = stop_loss_pct - fee_pct
                        trades.append({
                            'entry_time': position['entry_time'],
                            'exit_time': datetimes[i],
                            'direction': 'short',
                            'entry_price': position['entry_price'],
                            'exit_price': stop_price,
                            'bars_held': bars_held,
                            'exit_reason': 'stop_loss',
                            'pnl_pct': pnl_pct
                        })
                        position = None
                        if enable_cooldown:
                            skip_crosses_count = cooldown_count
                        continue

            # Checkpoint check
            if bars_held in checkpoint_set:
                checkpoint_idx = checkpoint_bars.index(bars_held)
                current_close = closes[i]

                exit_triggered = False
                exit_reason = None

                if position['is_long']:
                    profit_threshold = position['entry_price'] * (1 + profit_adjustment_pct / 100)
                    is_profitable = current_close > profit_threshold
                else:
                    profit_threshold = position['entry_price'] * (1 - profit_adjustment_pct / 100)
                    is_profitable = current_close < profit_threshold

                if not is_profitable:
                    exit_triggered = True
                    exit_reason = f'checkpoint_{bars_held}_not_profitable'

                if not exit_triggered and checkpoint_idx > 0 and position['last_checkpoint_close'] is not None:
                    if position['is_long']:
                        moved_against = current_close < position['last_checkpoint_close']
                    else:
                        moved_against = current_close > position['last_checkpoint_close']

                    if moved_against:
                        exit_triggered = True
                        exit_reason = f'checkpoint_{bars_held}_moved_against'

                if not exit_triggered and bars_held == max_checkpoint:
                    exit_triggered = True
                    exit_reason = f'checkpoint_{bars_held}_max'

                if exit_triggered:
                    if position['is_long']:
                        pnl_pct = (current_close / position['entry_price'] - 1) * 100 - fee_pct
                    else:
                        pnl_pct = (position['entry_price'] / current_close - 1) * 100 - fee_pct

                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': datetimes[i],
                        'direction': 'long' if position['is_long'] else 'short',
                        'entry_price': position['entry_price'],
                        'exit_price': current_close,
                        'bars_held': bars_held,
                        'exit_reason': exit_reason,
                        'pnl_pct': pnl_pct
                    })
                    position = None
                else:
                    position['last_checkpoint_close'] = current_close
                    position['checkpoint_index'] = checkpoint_idx + 1

    # Force close any open position
    if position is not None:
        exit_price = closes[n_bars - 1]
        bars_held = (n_bars - 1) - position['entry_bar_index']

        if position['is_long']:
            pnl_pct = (exit_price / position['entry_price'] - 1) * 100 - fee_pct
        else:
            pnl_pct = (position['entry_price'] / exit_price - 1) * 100 - fee_pct

        trades.append({
            'entry_time': position['entry_time'],
            'exit_time': datetimes[n_bars - 1],
            'direction': 'long' if position['is_long'] else 'short',
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'bars_held': bars_held,
            'exit_reason': 'end_of_data',
            'pnl_pct': pnl_pct
        })

    trades_df = pd.DataFrame(trades)

    if verbose:
        print(f"Simulation complete: {len(trades_df)} trades")
        print(f"\nFilter Statistics:")
        print(f"  Total potential entries: {filter_stats['total_potential_entries']:,}")
        if filter_stats['total_potential_entries'] > 0:
            print(f"  Filters A+B+C passed: {filter_stats['filter_abc_passed']:,} "
                  f"({filter_stats['filter_abc_passed']/filter_stats['total_potential_entries']*100:.1f}%)")
        if filter_stats['filter_abc_passed'] > 0:
            print(f"  Parkinson passed: {filter_stats['parkinson_passed']:,} "
                  f"({filter_stats['parkinson_passed']/filter_stats['filter_abc_passed']*100:.1f}%)")
        print(f"  Skipped (Parkinson): {filter_stats['skipped_due_to_parkinson']:,}")
        print(f"  Skipped (Volume): {filter_stats['skipped_due_to_volume']:,}")
        print(f"  Skipped (VWAP align): {filter_stats['skipped_due_to_vwap_alignment']:,}")
        print(f"  Skipped (Taker imb): {filter_stats['skipped_due_to_taker_imb']:,}")
        print(f"  Skipped (Avg trade size): {filter_stats['skipped_due_to_avg_trade_size']:,}")
        print(f"  Skipped (Cooldown): {filter_stats['skipped_due_to_cooldown']:,}")
        print()

    return trades_df


def analyze_results(trades_df: pd.DataFrame, initial_capital: float = 100_000, max_position: float = 100_000):
    """Analyze strategy performance."""
    if len(trades_df) == 0:
        print("No trades to analyze")
        return

    print("=" * 70)
    print("PERFORMANCE ANALYSIS")
    print("=" * 70)

    total_trades = len(trades_df)
    win_trades = len(trades_df[trades_df['pnl_pct'] > 0])
    loss_trades = len(trades_df[trades_df['pnl_pct'] <= 0])
    win_rate = win_trades / total_trades * 100

    print(f"Total Trades: {total_trades:,}")
    print(f"Wins: {win_trades:,} | Losses: {loss_trades:,}")
    print(f"Win Rate: {win_rate:.2f}%")
    print()

    avg_pnl = trades_df['pnl_pct'].mean()
    avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if win_trades > 0 else 0
    avg_loss = trades_df[trades_df['pnl_pct'] <= 0]['pnl_pct'].mean() if loss_trades > 0 else 0

    print(f"Average PnL: {avg_pnl:.4f}%")
    print(f"Average Win: {avg_win:.4f}%")
    print(f"Average Loss: {avg_loss:.4f}%")
    print()

    print("Exit Reason Breakdown:")
    for reason, count in trades_df['exit_reason'].value_counts().items():
        pct = count / total_trades * 100
        avg_pnl_reason = trades_df[trades_df['exit_reason'] == reason]['pnl_pct'].mean()
        print(f"  {reason}: {count:,} ({pct:.1f}%) - Avg PnL: {avg_pnl_reason:.4f}%")
    print()

    # Cumulative PnL
    capital = initial_capital
    for pnl_pct in trades_df.sort_values('entry_time')['pnl_pct']:
        position_size = min(capital, max_position)
        capital += position_size * (pnl_pct / 100)

    total_roi = (capital - initial_capital) / initial_capital * 100
    print(f"Total ROI: {total_roi:.2f}%")
    print(f"Final Capital: ${capital:,.2f}")
    print()

    return trades_df.sort_values('entry_time')


def load_data(data_path: Path, year: int = None) -> pd.DataFrame:
    """Load and optionally filter data by year."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])

    if year:
        df = df[df['datetime_utc'].dt.year == year].copy()
        print(f"Filtered to {year}: {len(df):,} bars")
    else:
        print(f"Loaded {len(df):,} bars")

    return df


if __name__ == "__main__":
    data_path = Path("/Volumes/Extreme SSD/trading_data/cex/ohlvc/binance_btcusdt_perp_1m/BTCUSDT-1m-vwap-sessions.csv")

    # Test with 2024 data
    df = load_data(data_path, year=2024)

    # Test with VWAP-15 (default, uses precomputed columns)
    print("\n" + "="*80)
    print("Testing VWAP-15 (precomputed)")
    print("="*80)
    trades_df = run_strategy(
        df=df,
        entry_bar=6,
        checkpoint_bars=[120],
        fee_pct=0.03,
        gap_widening_required=3,
        min_cumulative_move_bps=15.0,
        parkinson_threshold=1.5,
        stop_loss_pct=-0.1,
        volume_ratio_threshold=1.0,
        volume_ratio_col='volume_ratio_30',
        vwap_window=15,  # Use VWAP-15 for cross detection
        verbose=True
    )
    analyze_results(trades_df)

    # Test with VWAP-30 (precomputed)
    print("\n" + "="*80)
    print("Testing VWAP-30 (precomputed)")
    print("="*80)
    trades_df_30 = run_strategy(
        df=df,
        entry_bar=6,
        checkpoint_bars=[120],
        fee_pct=0.03,
        gap_widening_required=3,
        min_cumulative_move_bps=15.0,
        parkinson_threshold=1.5,
        stop_loss_pct=-0.1,
        volume_ratio_threshold=1.0,
        volume_ratio_col='volume_ratio_30',
        vwap_window=30,  # Use VWAP-30 for cross detection
        verbose=True
    )
    analyze_results(trades_df_30)

    # Test with VWAP-60 (computed on-the-fly from vwap_60 column)
    print("\n" + "="*80)
    print("Testing VWAP-60 (computed on-the-fly)")
    print("="*80)
    trades_df_60 = run_strategy(
        df=df,
        entry_bar=6,
        checkpoint_bars=[120],
        fee_pct=0.03,
        gap_widening_required=3,
        min_cumulative_move_bps=15.0,
        parkinson_threshold=1.5,
        stop_loss_pct=-0.1,
        volume_ratio_threshold=1.0,
        volume_ratio_col='volume_ratio_30',
        vwap_window=60,  # Use VWAP-60 (computed on-the-fly)
        verbose=True
    )
    analyze_results(trades_df_60)

