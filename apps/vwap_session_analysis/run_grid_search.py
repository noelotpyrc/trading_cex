"""
Grid search for VWAP momentum strategy parameters across 2023/2024/2025 data.

Parameters:
- parkinson_threshold: [1.5, 1.8]
- volume_ratio_threshold: [1.0]
- checkpoint_bars: [90, 120]
- min_cumulative_move_bps: [15, 20]
- taker_imb_long_col: [None, 'taker_imb_ema_15', 'taker_imb_ema_30']
- taker_imb_short_col: [None, 'taker_imb_ema_15', 'taker_imb_ema_30']
- stop_loss_pct: [-0.1, -0.25, -0.35]
- vwap_window: [15, 30]
"""

import pandas as pd
from pathlib import Path
from itertools import product
import time
import argparse

from vwap_momentum_parkinson_fast_vwap_window import run_strategy, load_data


def run_grid_search_for_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Run grid search for a single year's data."""
    
    # Parameter grid
    parkinson_thresholds = [1.5]
    volume_ratios = [1.0]
    checkpoints = [120]
    min_cum_moves = [20]
    taker_imb_options = [None]
    stop_loss_options = [-0.1,-0.25,-0.35,-0.5]  # Add more like [-0.1, -0.25, -0.35]
    vwap_window_options = [15,30,60]

    # Fixed parameters
    fee_pct = 0.03
    gap_widening_required = 3

    # Generate all combinations
    combinations = list(product(
        parkinson_thresholds, volume_ratios, checkpoints, min_cum_moves,
        taker_imb_options, taker_imb_options,  # long and short
        stop_loss_options, vwap_window_options
    ))
    total_runs = len(combinations)
    print(f"\n{'='*80}")
    print(f"YEAR {year} - Total combinations: {total_runs}")
    print("=" * 80)

    results = []
    start_time = time.time()

    for i, (parkinson, vol_ratio, cp, min_cum, taker_long, taker_short, stop_loss, vwap_win) in enumerate(combinations, 1):
        run_start = time.time()

        trades_df = run_strategy(
            df=df,
            checkpoint_bars=[cp],
            parkinson_threshold=parkinson,
            stop_loss_pct=stop_loss,
            min_cumulative_move_bps=min_cum,
            volume_ratio_threshold=vol_ratio,
            volume_ratio_col='volume_ratio_30',
            taker_imb_long_col=taker_long,
            taker_imb_short_col=taker_short,
            vwap_window=vwap_win,
            verbose=False,
            entry_bar=5
        )

        run_time = time.time() - run_start

        # Calculate overall metrics
        n_trades = len(trades_df)
        if n_trades > 0:
            n_wins = (trades_df['pnl_pct'] > 0).sum()
            win_rate = n_wins / n_trades * 100
            avg_pnl = trades_df['pnl_pct'].mean()
            total_pnl = trades_df['pnl_pct'].sum()
        else:
            n_wins = win_rate = avg_pnl = total_pnl = 0

        # Calculate long/short metrics separately
        long_df = trades_df[trades_df['direction'] == 'long'] if n_trades > 0 else pd.DataFrame()
        short_df = trades_df[trades_df['direction'] == 'short'] if n_trades > 0 else pd.DataFrame()

        n_long = len(long_df)
        n_short = len(short_df)

        if n_long > 0:
            long_wins = (long_df['pnl_pct'] > 0).sum()
            long_win_rate = long_wins / n_long * 100
            long_total_pnl = long_df['pnl_pct'].sum()
        else:
            long_wins = long_win_rate = long_total_pnl = 0

        if n_short > 0:
            short_wins = (short_df['pnl_pct'] > 0).sum()
            short_win_rate = short_wins / n_short * 100
            short_total_pnl = short_df['pnl_pct'].sum()
        else:
            short_wins = short_win_rate = short_total_pnl = 0

        # Short labels for taker imb columns
        taker_long_label = taker_long.replace('taker_imb_ema_', 'EMA') if taker_long else 'None'
        taker_short_label = taker_short.replace('taker_imb_ema_', 'EMA') if taker_short else 'None'

        results.append({
            'year': year,
            'parkinson': parkinson,
            'vol_ratio': vol_ratio,
            'checkpoint': cp,
            'min_cum_bps': min_cum,
            'taker_imb_long': taker_long_label,
            'taker_imb_short': taker_short_label,
            'stop_loss': stop_loss,
            'vwap_window': vwap_win,
            # Overall metrics
            'trades': n_trades,
            'wins': n_wins,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl,
            # Long metrics
            'long_trades': n_long,
            'long_wins': long_wins,
            'long_win_rate': long_win_rate,
            'long_pnl': long_total_pnl,
            # Short metrics
            'short_trades': n_short,
            'short_wins': short_wins,
            'short_win_rate': short_win_rate,
            'short_pnl': short_total_pnl,
        })

        # Progress
        elapsed = time.time() - start_time
        eta = elapsed / i * (total_runs - i)
        print(f"[{i:3}/{total_runs}] park={parkinson} vwap={vwap_win:2} sl={stop_loss:+.2f} "
              f"tL={taker_long_label:5} tS={taker_short_label:5} | "
              f"trades={n_trades:4} (L:{n_long:3}/S:{n_short:3}) pnl={total_pnl:+7.2f}% "
              f"(L:{long_total_pnl:+6.2f}/S:{short_total_pnl:+6.2f}) | {run_time:.1f}s")

    print(f"\nYear {year} complete in {time.time() - start_time:.1f}s")
    
    return pd.DataFrame(results)


def print_summary(results_df: pd.DataFrame, year: int):
    """Print summary for a single year."""
    year_df = results_df[results_df['year'] == year].sort_values('total_pnl', ascending=False)
    
    print(f"\n{'='*140}")
    print(f"YEAR {year} - TOP 10 by Total PnL:")
    print("-" * 140)
    print(f"{'vwap':>4} {'sl':>6} {'tL':>5} {'tS':>5} | "
          f"{'trades':>6} {'win%':>5} {'pnl':>8} | "
          f"{'L_tr':>5} {'L_w%':>5} {'L_pnl':>7} | "
          f"{'S_tr':>5} {'S_w%':>5} {'S_pnl':>7}")
    print("-" * 140)
    for _, row in year_df.head(10).iterrows():
        print(f"{row['vwap_window']:>4} {row['stop_loss']:>+6.2f} "
              f"{row['taker_imb_long']:>5} {row['taker_imb_short']:>5} | "
              f"{row['trades']:>6} {row['win_rate']:>4.1f}% {row['total_pnl']:>+7.2f}% | "
              f"{row['long_trades']:>5} {row['long_win_rate']:>4.1f}% {row['long_pnl']:>+6.2f}% | "
              f"{row['short_trades']:>5} {row['short_win_rate']:>4.1f}% {row['short_pnl']:>+6.2f}%")

    # Long-only top 5
    print(f"\n  LONG-ONLY TOP 5:")
    long_sorted = year_df.sort_values('long_pnl', ascending=False)
    for _, row in long_sorted.head(5).iterrows():
        print(f"    vwap={row['vwap_window']} sl={row['stop_loss']:+.2f} "
              f"tL={row['taker_imb_long']:>5} | L: {row['long_trades']} trades, "
              f"{row['long_win_rate']:.1f}% win, {row['long_pnl']:+.2f}% PnL")

    # Short-only top 5
    print(f"\n  SHORT-ONLY TOP 5:")
    short_sorted = year_df.sort_values('short_pnl', ascending=False)
    for _, row in short_sorted.head(5).iterrows():
        print(f"    vwap={row['vwap_window']} sl={row['stop_loss']:+.2f} "
              f"tS={row['taker_imb_short']:>5} | S: {row['short_trades']} trades, "
              f"{row['short_win_rate']:.1f}% win, {row['short_pnl']:+.2f}% PnL")


def run_grid_search(years: list[int] = None):
    """Run grid search across specified years."""
    if years is None:
        years = [2023, 2024, 2025]
    
    data_path = Path("/Volumes/Extreme SSD/trading_data/cex/ohlvc/binance_btcusdt_perp_1m/BTCUSDT-1m-vwap-sessions.csv")
    
    all_results = []
    
    for year in years:
        print(f"\nLoading {year} data...")
        df = load_data(data_path, year=year)
        
        year_results = run_grid_search_for_year(df, year)
        all_results.append(year_results)
        
        # Save individual year results
        output_path = Path(__file__).parent / f"grid_search_{year}_results.csv"
        year_results.to_csv(output_path, index=False)
        print(f"Year {year} results saved to: {output_path}")
    
    # Combine all results
    results_df = pd.concat(all_results, ignore_index=True)
    
    # Save combined results
    output_path = Path(__file__).parent / "grid_search_all_years_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nCombined results saved to: {output_path}")
    
    # Print summaries
    print("\n" + "=" * 140)
    print("GRID SEARCH COMPLETE - SUMMARY")
    print("=" * 140)
    
    for year in years:
        print_summary(results_df, year)
    
    # Cross-year analysis
    print("\n" + "=" * 140)
    print("CROSS-YEAR ANALYSIS - Best Consistent Performers")
    print("=" * 140)
    
    # Aggregate across years
    agg_df = results_df.groupby([
        'parkinson', 'vol_ratio', 'checkpoint', 'min_cum_bps',
        'taker_imb_long', 'taker_imb_short', 'stop_loss', 'vwap_window'
    ]).agg({
        'trades': 'sum',
        'wins': 'sum',
        'total_pnl': 'sum',
        'long_trades': 'sum',
        'long_wins': 'sum',
        'long_pnl': 'sum',
        'short_trades': 'sum',
        'short_wins': 'sum',
        'short_pnl': 'sum',
    }).reset_index()
    agg_df['win_rate'] = agg_df['wins'] / agg_df['trades'] * 100
    agg_df['long_win_rate'] = agg_df['long_wins'] / agg_df['long_trades'].replace(0, 1) * 100
    agg_df['short_win_rate'] = agg_df['short_wins'] / agg_df['short_trades'].replace(0, 1) * 100
    agg_df = agg_df.sort_values('total_pnl', ascending=False)
    
    print("\nTOP 10 by Combined Total PnL (all years):")
    print("-" * 140)
    for _, row in agg_df.head(10).iterrows():
        print(f"vwap={int(row['vwap_window'])} sl={row['stop_loss']:+.2f} "
              f"tL={row['taker_imb_long']:>5} tS={row['taker_imb_short']:>5} | "
              f"Total: {int(row['trades'])} trades, {row['total_pnl']:+.2f}% | "
              f"Long: {int(row['long_trades'])} tr, {row['long_pnl']:+.2f}% | "
              f"Short: {int(row['short_trades'])} tr, {row['short_pnl']:+.2f}%")

    # Best long-only configs
    print("\nBEST LONG-ONLY CONFIGS (aggregated):")
    long_sorted = agg_df.sort_values('long_pnl', ascending=False)
    for _, row in long_sorted.head(5).iterrows():
        print(f"  vwap={int(row['vwap_window'])} sl={row['stop_loss']:+.2f} "
              f"tL={row['taker_imb_long']:>5} | {int(row['long_trades'])} trades, {row['long_pnl']:+.2f}% PnL")

    # Best short-only configs
    print("\nBEST SHORT-ONLY CONFIGS (aggregated):")
    short_sorted = agg_df.sort_values('short_pnl', ascending=False)
    for _, row in short_sorted.head(5).iterrows():
        print(f"  vwap={int(row['vwap_window'])} sl={row['stop_loss']:+.2f} "
              f"tS={row['taker_imb_short']:>5} | {int(row['short_trades'])} trades, {row['short_pnl']:+.2f}% PnL")
    
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run grid search for VWAP momentum strategy")
    parser.add_argument('--years', type=int, nargs='+', default=[2023, 2024, 2025],
                        help='Years to run grid search on (default: 2023 2024 2025)')
    args = parser.parse_args()
    
    run_grid_search(years=args.years)
