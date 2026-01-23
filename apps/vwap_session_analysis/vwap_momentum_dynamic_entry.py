"""
VWAP Momentum Strategy - Dynamic Entry Version

Dynamic entry logic:
- Bar 0: Cross detected
- Bars 1-5: Check each bar's close for entry conditions:
  - Z-score >= 0.5 at that bar
  - Cumulative move from bar 1 open to that bar's close >= 10 bps
- Enter at NEXT bar's OPEN when conditions first met
- Filters A/B disabled, D/E/F/G still apply
"""

import pandas as pd
import numpy as np
from pathlib import Path


def run_dynamic_strategy(
    df: pd.DataFrame,
    min_entry_bar: int = 2,  # Earliest entry (bar after bar 1)
    max_entry_bar: int = 6,  # Latest entry (bar after bar 5)
    checkpoint_bars: list = [120],
    fee_pct: float = 0.03,
    min_cumulative_move_bps: float = 10.0,
    zscore_threshold: float = 0.5,
    parkinson_threshold: float = 1.5,
    stop_loss_pct: float = None,
    volume_ratio_threshold: float = None,
    volume_ratio_col: str = 'volume_ratio_30',
    taker_imb_long_col: str = None,
    taker_imb_short_col: str = None,
    avg_trade_size_zscore_col: str = 'avg_trade_size_zscore_60',
    use_hlc_vwap: bool = True,
    direction_filter: str = 'both',  # 'both', 'long', or 'short'
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run VWAP momentum strategy with dynamic entry.

    Entry triggers when FIRST bar (1-5) meets:
    - Z-score >= threshold
    - Cumulative move >= min_bps
    Then enter at next bar's OPEN.
    """

    # Select column names based on VWAP formula
    if use_hlc_vwap:
        cross_col = 'cross_direction_15_hlc'
        vwap_formula = 'HLC/3'
    else:
        cross_col = 'cross_direction_15'
        vwap_formula = 'OHLC/4'

    if verbose:
        print(f"Running VWAP Momentum Strategy (Dynamic Entry)...")
        print(f"  VWAP formula: {vwap_formula}")
        print(f"  Direction: {direction_filter.upper()}")
        print(f"  Dynamic Entry: Check bars 1-5, enter when z-score >= {zscore_threshold} AND cum_bps >= {min_cumulative_move_bps}")
        print(f"  Entry range: Bar {min_entry_bar} to Bar {max_entry_bar}")
        print(f"  Filters:")
        print(f"    A: DISABLED")
        print(f"    B: DISABLED")
        print(f"    C: Cumulative move >= {min_cumulative_move_bps} bps (dynamic)")
        print(f"    D: Parkinson ratio >= {parkinson_threshold}")
        if volume_ratio_threshold is not None:
            print(f"    E: {volume_ratio_col} >= {volume_ratio_threshold}")
        if taker_imb_long_col:
            print(f"    G (long): {taker_imb_long_col} >= 0")
        if taker_imb_short_col:
            print(f"    G (short): {taker_imb_short_col} < 0")
        print(f"    H: {avg_trade_size_zscore_col} >= {zscore_threshold} (dynamic)")
        print(f"  Checkpoints: {checkpoint_bars}")
        print(f"  Stop Loss: {stop_loss_pct}%" if stop_loss_pct else "  Stop Loss: DISABLED")
        print(f"  Fee: {fee_pct}%")
        print()

    # Prepare data
    df_sorted = df.sort_values('datetime_utc').reset_index(drop=True)

    # Validate required columns
    required_cols = ['parkinson_ratio', cross_col, avg_trade_size_zscore_col]
    if volume_ratio_threshold is not None:
        required_cols.append(volume_ratio_col)
    if taker_imb_long_col:
        required_cols.append(taker_imb_long_col)
    if taker_imb_short_col:
        required_cols.append(taker_imb_short_col)

    missing = [c for c in required_cols if c not in df_sorted.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Pre-calculate crosses
    if verbose:
        print("Processing cross events...")

    latest_cross_idx = None
    latest_cross_dir = None
    cross_indices = []
    cross_directions = []
    cross_values = df_sorted[cross_col].values

    for i in range(len(df_sorted)):
        cross_indices.append(latest_cross_idx)
        cross_directions.append(latest_cross_dir)
        cross_dir = cross_values[i]
        if pd.notna(cross_dir):
            latest_cross_idx = i
            latest_cross_dir = cross_dir

    df_sorted['latest_cross_index'] = cross_indices
    df_sorted['latest_cross_direction'] = cross_directions

    if verbose:
        print(f"Data ready: {len(df_sorted):,} bars")
        print()

    # Convert to numpy arrays
    n_bars = len(df_sorted)
    opens = df_sorted['open'].values
    closes = df_sorted['close'].values
    highs = df_sorted['high'].values
    lows = df_sorted['low'].values
    datetimes = df_sorted['datetime_utc'].values
    cross_indices_arr = df_sorted['latest_cross_index'].values
    cross_directions_arr = df_sorted['latest_cross_direction'].values

    parkinson_ratios = df_sorted['parkinson_ratio'].values
    zscores = df_sorted[avg_trade_size_zscore_col].values

    if volume_ratio_threshold is not None:
        volume_ratios = df_sorted[volume_ratio_col].values
    if taker_imb_long_col:
        taker_imb_long = df_sorted[taker_imb_long_col].values
    if taker_imb_short_col:
        taker_imb_short = df_sorted[taker_imb_short_col].values

    checkpoint_set = set(checkpoint_bars)
    max_checkpoint = checkpoint_bars[-1]

    # Initialize state
    position = None
    trades = []

    # Track which crosses we've already evaluated
    processed_crosses = set()

    filter_stats = {
        'total_crosses': 0,
        'dynamic_condition_met': 0,
        'parkinson_passed': 0,
        'skipped_parkinson': 0,
        'skipped_volume': 0,
        'skipped_taker_imb': 0,
        'skipped_no_trigger': 0,
        'entry_bar_distribution': {2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
    }

    # Process each bar
    for i in range(n_bars):
        # =================================================================
        # ENTRY CHECK
        # =================================================================
        if position is None:
            cross_idx = cross_indices_arr[i]
            cross_dir = cross_directions_arr[i]

            if not pd.isna(cross_idx):
                cross_idx_int = int(cross_idx)
                bars_since_cross = i - cross_idx_int

                # Only process if we're in the entry window and haven't processed this cross
                if min_entry_bar <= bars_since_cross <= max_entry_bar and cross_idx_int not in processed_crosses:

                    is_long = (cross_dir == 'above')

                    # Direction filter
                    if direction_filter == 'long' and not is_long:
                        processed_crosses.add(cross_idx_int)
                        continue
                    if direction_filter == 'short' and is_long:
                        processed_crosses.add(cross_idx_int)
                        continue

                    bar1_open = opens[cross_idx_int + 1] if cross_idx_int + 1 < n_bars else None

                    if bar1_open is None:
                        continue

                    # Check each bar from 1 to (bars_since_cross - 1) for trigger
                    # We enter at bar i, so we check up to bar (i-1) which is (bars_since_cross - 1) bars after cross
                    trigger_bar = None

                    for check_offset in range(1, bars_since_cross):  # bars 1, 2, ..., up to current-1
                        check_idx = cross_idx_int + check_offset
                        if check_idx >= n_bars:
                            break

                        # Get z-score at this bar
                        zscore_val = zscores[check_idx]
                        if pd.isna(zscore_val) or zscore_val < zscore_threshold:
                            continue

                        # Calculate cumulative move from bar 1 open to this bar's close
                        bar_close = closes[check_idx]
                        cum_return_pct = (bar_close / bar1_open - 1) * 100
                        cum_bps = cum_return_pct * 100  # convert to bps

                        # Check direction
                        if is_long:
                            if cum_bps >= min_cumulative_move_bps:
                                trigger_bar = check_offset
                                break
                        else:
                            if cum_bps <= -min_cumulative_move_bps:
                                trigger_bar = check_offset
                                break

                    # If no trigger found yet, skip
                    if trigger_bar is None:
                        if bars_since_cross == max_entry_bar:
                            # Last chance, mark as processed
                            processed_crosses.add(cross_idx_int)
                            filter_stats['skipped_no_trigger'] += 1
                        continue

                    # We have a trigger! Now check other filters
                    filter_stats['total_crosses'] += 1
                    filter_stats['dynamic_condition_met'] += 1
                    processed_crosses.add(cross_idx_int)

                    # Entry bar is trigger_bar + 1 (enter at OPEN of next bar after trigger)
                    actual_entry_bar = trigger_bar + 1
                    entry_idx = cross_idx_int + actual_entry_bar

                    if entry_idx != i:
                        # We're not at the right entry bar yet
                        continue

                    # Filter D: Parkinson (use bar before entry)
                    if parkinson_ratios[i - 1] < parkinson_threshold:
                        filter_stats['skipped_parkinson'] += 1
                        continue

                    filter_stats['parkinson_passed'] += 1

                    # Filter E: Volume ratio
                    if volume_ratio_threshold is not None:
                        vol_ratio = volume_ratios[i - 1]
                        if pd.isna(vol_ratio) or vol_ratio < volume_ratio_threshold:
                            filter_stats['skipped_volume'] += 1
                            continue

                    # Filter G: Taker imbalance
                    if is_long and taker_imb_long_col:
                        imb_val = taker_imb_long[i - 1]
                        if pd.isna(imb_val) or imb_val < 0:
                            filter_stats['skipped_taker_imb'] += 1
                            continue
                    elif not is_long and taker_imb_short_col:
                        imb_val = taker_imb_short[i - 1]
                        if pd.isna(imb_val) or imb_val >= 0:
                            filter_stats['skipped_taker_imb'] += 1
                            continue

                    # ENTER at OPEN
                    filter_stats['entry_bar_distribution'][actual_entry_bar] += 1

                    position = {
                        'entry_bar_index': i,
                        'entry_price': opens[i],
                        'entry_time': datetimes[i],
                        'is_long': is_long,
                        'cross_bar_index': cross_idx_int,
                        'trigger_bar': trigger_bar,
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
                            'pnl_pct': pnl_pct,
                            'trigger_bar': position['trigger_bar'],
                        })
                        position = None
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
                            'pnl_pct': pnl_pct,
                            'trigger_bar': position['trigger_bar'],
                        })
                        position = None
                        continue

            # Checkpoint check
            if bars_held in checkpoint_set:
                checkpoint_idx = checkpoint_bars.index(bars_held)
                current_close = closes[i]

                exit_triggered = False
                exit_reason = None

                if position['is_long']:
                    is_profitable = current_close > position['entry_price']
                else:
                    is_profitable = current_close < position['entry_price']

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
                        'pnl_pct': pnl_pct,
                        'trigger_bar': position['trigger_bar'],
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
            'pnl_pct': pnl_pct,
            'trigger_bar': position['trigger_bar'],
        })

    trades_df = pd.DataFrame(trades)

    if verbose:
        print(f"Simulation complete: {len(trades_df)} trades")
        print(f"\nFilter Statistics:")
        print(f"  Dynamic condition met: {filter_stats['dynamic_condition_met']:,}")
        print(f"  Parkinson passed: {filter_stats['parkinson_passed']:,}")
        print(f"  Skipped (no trigger): {filter_stats['skipped_no_trigger']:,}")
        print(f"  Skipped (Parkinson): {filter_stats['skipped_parkinson']:,}")
        print(f"  Skipped (Volume): {filter_stats['skipped_volume']:,}")
        print(f"  Skipped (Taker imb): {filter_stats['skipped_taker_imb']:,}")
        print(f"\nEntry Bar Distribution:")
        for bar, count in sorted(filter_stats['entry_bar_distribution'].items()):
            if count > 0:
                print(f"  Bar {bar}: {count} trades")
        print()

    return trades_df


def analyze_results(trades_df: pd.DataFrame):
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

    # By trigger bar
    if 'trigger_bar' in trades_df.columns:
        print("Performance by Trigger Bar:")
        for bar in sorted(trades_df['trigger_bar'].unique()):
            subset = trades_df[trades_df['trigger_bar'] == bar]
            wins = len(subset[subset['pnl_pct'] > 0])
            total_pnl = subset['pnl_pct'].sum()
            print(f"  Bar {bar}: {len(subset)} trades, {wins} wins ({wins/len(subset)*100:.1f}%), PnL: {total_pnl:.2f}%")
        print()

    # Cumulative PnL
    capital = 100_000
    max_position = 100_000
    for pnl_pct in trades_df.sort_values('entry_time')['pnl_pct']:
        position_size = min(capital, max_position)
        capital += position_size * (pnl_pct / 100)

    total_roi = (capital - 100_000) / 100_000 * 100
    print(f"Total ROI: {total_roi:.2f}%")
    print(f"Final Capital: ${capital:,.2f}")
    print()


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

    for year in [2023, 2024, 2025]:
        print(f"\n{'='*70}")
        print(f"YEAR {year}")
        print(f"{'='*70}")

        df = load_data(data_path, year=year)

        # Use year-specific parkinson threshold
        parkinson = 1.8 if year in [2023, 2024] else 1.5

        trades_df = run_dynamic_strategy(
            df=df,
            checkpoint_bars=[120] if year != 2024 else [90],
            parkinson_threshold=parkinson,
            stop_loss_pct=-0.1,
            volume_ratio_threshold=1.0,
            min_cumulative_move_bps=10.0,
            zscore_threshold=0.5,
            taker_imb_long_col=None,
            taker_imb_short_col=None,
            verbose=True
        )

        analyze_results(trades_df)
