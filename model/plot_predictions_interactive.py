import argparse
import re
from pathlib import Path
from typing import Tuple, List

import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def _load_pred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        df['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    else:
        df['timestamp'] = pd.RangeIndex(start=0, stop=len(df))
    return df


def _make_traces(df: pd.DataFrame, name_prefix: str) -> Tuple[List[go.Scatter], bool]:
    x = df['timestamp']
    traces: List[go.Scatter] = []

    # Detect available quantile columns like pred_q05, pred_q50, pred_q95
    q_cols = {}
    for c in df.columns:
        m = re.fullmatch(r"pred_q(\d{2})", str(c))
        if m:
            q = int(m.group(1))
            q_cols[q] = c

    # Prepare symmetric bands (outer to inner)
    band_colors = {
        95: ('#d62728', 'rgba(214,39,40,0.12)'),   # red
        90: ('#9467bd', 'rgba(148,103,189,0.12)'), # purple
        85: ('#17becf', 'rgba(23,190,207,0.12)'),  # teal
        75: ('#ff7f0e', 'rgba(255,127,14,0.15)'),  # orange
    }

    added_any_band = False
    plotted_cols: set[str] = set()
    for upper in [95, 90, 85, 75]:
        lower = 100 - upper
        if upper in q_cols and lower in q_cols:
            ucol = q_cols[upper]
            lcol = q_cols[lower]
            ucolor, fillcolor = band_colors.get(upper, ('#999999', 'rgba(153,153,153,0.12)'))
            # Upper line
            traces.append(
                go.Scatter(
                    x=x, y=df[ucol], name=f'{name_prefix} {ucol}',
                    line=dict(color=ucolor, width=1),
                    mode='lines',
                    hovertemplate='%{x}<br>'+ucol+'=%{y:.6f}<extra></extra>'
                )
            )
            # Lower line with fill to create band
            traces.append(
                go.Scatter(
                    x=x, y=df[lcol], name=f'{name_prefix} {lcol}',
                    line=dict(color=ucolor, width=1),
                    mode='lines',
                    fill='tonexty', fillcolor=fillcolor,
                    hovertemplate='%{x}<br>'+lcol+'=%{y:.6f}<extra></extra>'
                )
            )
            added_any_band = True
            plotted_cols.add(ucol)
            plotted_cols.add(lcol)

    # Median (q50) as center line if available
    if 50 in q_cols:
        traces.append(
            go.Scatter(
                x=x, y=df[q_cols[50]], name=f'{name_prefix} pred_q50',
                line=dict(color='green', width=1),
                mode='lines',
                hovertemplate='%{x}<br>q50=%{y:.6f}<extra></extra>'
            )
        )
        plotted_cols.add(q_cols[50])

    # Plot any single quantiles not part of a symmetric band (e.g., q05, q10, q15 when q95/q90/q85 missing)
    for q in sorted(q_cols.keys()):
        col = q_cols[q]
        if col in plotted_cols:
            continue
        color = '#1f77b4' if q < 50 else '#d62728'
        traces.append(
            go.Scatter(
                x=x, y=df[col], name=f'{name_prefix} {col}',
                line=dict(color=color, width=1, dash='dot'),
                mode='lines',
                hovertemplate='%{x}<br>'+col+'=%{y:.6f}<extra></extra>'
            )
        )

    # Regression point estimate (pred_reg) as green line
    if 'pred_reg' in df.columns:
        reg = df['pred_reg']
        traces.append(
            go.Scatter(
                x=x, y=reg, name=f'{name_prefix} pred_reg',
                line=dict(color='green', width=1),
                mode='lines',
                hovertemplate='%{x}<br>reg=%{y:.6f}<extra></extra>'
            )
        )

    # Target (original values)
    y_true = df['y_true']
    traces.append(
        go.Scatter(
            x=x, y=y_true, name=f'{name_prefix} y_true',
            line=dict(color='black', width=1.2),
            mode='lines',
            hovertemplate='%{x}<br>y=%{y:.6f}<extra></extra>'
        )
    )

    return traces, added_any_band


def _load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Prefer 'timestamp' if present, else 'time'
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    elif 'time' in df.columns:
        ts = pd.to_datetime(df['time'], errors='coerce', utc=True)
    else:
        raise ValueError("OHLCV CSV must have 'timestamp' or 'time' column")
    df['timestamp'] = ts.dt.tz_convert('UTC').dt.tz_localize(None)
    # Normalize column names
    colmap = {}
    for c in df.columns:
        lc = str(c).lower()
        if lc in {'open', 'high', 'low', 'close'}:
            colmap[c] = lc
    df = df.rename(columns=colmap)
    required = {'open', 'high', 'low', 'close', 'timestamp'}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"OHLCV CSV missing columns: {sorted(missing)}")
    return df[['timestamp', 'open', 'high', 'low', 'close']].dropna(subset=['timestamp'])


def _subset_ohlcv_for_window(ohlcv: pd.DataFrame, start_ts, end_ts) -> pd.DataFrame:
    mask = (ohlcv['timestamp'] >= start_ts) & (ohlcv['timestamp'] <= end_ts)
    return ohlcv.loc[mask]


def main() -> None:
    parser = argparse.ArgumentParser(description='Interactive plot of quantile predictions vs target (val/test) with OHLCV candlesticks. Optionally overlay expected return from signals CSVs.')
    parser.add_argument('--run-dir', type=Path, required=True, help='Path to model run directory with pred_val.csv and pred_test.csv')
    parser.add_argument('--prefix', type=str, default='quantile_preds_interactive', help='Output filename prefix (will create <prefix>_val.html and <prefix>_test.html)')
    parser.add_argument('--ohlcv-csv', type=Path, default=Path('data/BINANCE_BTCUSDT.P, 60.csv'), help='Path to original OHLCV CSV (with time/timestamp, open, high, low, close)')
    parser.add_argument('--signals-val', type=Path, required=False, help='Optional path to signals CSV for validation (e.g., pred_val_signals.csv)')
    parser.add_argument('--signals-test', type=Path, required=False, help='Optional path to signals CSV for test (e.g., pred_test_signals.csv)')
    parser.add_argument('--expected-method', choices=['avg', 'conservative'], default='avg', help='Which expected return column to plot from signals (exp_ret_avg or exp_ret_conservative)')
    parser.add_argument('--mark-cross-violations', action='store_true', help='Mark points where cross_violations > 0 on expected return line')
    parser.add_argument('--show-prob-up', action='store_true', help='Overlay probability of positive return (prob_up) on a secondary y-axis')
    parser.add_argument('--prob-thresholds', type=float, nargs=2, metavar=('TAU_LONG', 'TAU_SHORT'), default=None, help='Optional probability thresholds to draw as reference lines when showing prob_up')
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    val_df = _load_pred(run_dir / 'pred_val.csv')
    test_df = _load_pred(run_dir / 'pred_test.csv')
    sig_val_df: pd.DataFrame | None = None
    sig_test_df: pd.DataFrame | None = None
    method_col = 'exp_ret_avg' if args.expected_method == 'avg' else 'exp_ret_conservative'
    if args.signals_val and args.signals_val.exists():
        sig_val_df = _load_pred(args.signals_val)
    if args.signals_test and args.signals_test.exists():
        sig_test_df = _load_pred(args.signals_test)
    ohlcv_df = _load_ohlcv(args.ohlcv_csv)

    # Validation figure with candlestick subplot (secondary y on row 1 for prob_up)
    val_fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.65, 0.35],
        specs=[[{"secondary_y": True}], [{}]],
    )
    val_traces, _ = _make_traces(val_df, 'val')
    for tr in val_traces:
        val_fig.add_trace(tr, row=1, col=1, secondary_y=False)
    # Overlay expected return from signals if available
    if sig_val_df is not None and method_col in sig_val_df.columns:
        val_fig.add_trace(
            go.Scatter(
                x=sig_val_df['timestamp'], y=sig_val_df[method_col], name=f'val {method_col}',
                line=dict(color='#bcbd22', width=2),
                mode='lines',
                hovertemplate='%{x}<br>'+method_col+'=%{y:.6f}<extra></extra>'
            ),
            row=1, col=1, secondary_y=False,
        )
        if args.mark_cross_violations and 'cross_violations' in sig_val_df.columns:
            mask = (sig_val_df['cross_violations'] > 0) & sig_val_df[method_col].notna()
            if mask.any():
                custom = list(zip(sig_val_df.loc[mask, 'cross_violations'], sig_val_df.loc[mask, 'max_cross_gap'] if 'max_cross_gap' in sig_val_df.columns else [None] * mask.sum()))
                val_fig.add_trace(
                    go.Scatter(
                        x=sig_val_df.loc[mask, 'timestamp'],
                        y=sig_val_df.loc[mask, method_col],
                        name='val cross violations',
                        mode='markers',
                        marker=dict(color='#d62728', size=7, symbol='x'),
                        customdata=custom,
                        hovertemplate='%{x}<br>'+method_col+'=%{y:.6f}<br>cross=%{customdata[0]}<br>gap=%{customdata[1]:.6f}<extra></extra>'
                    ),
                    row=1, col=1, secondary_y=False,
                )

    # Optional prob_up on secondary y-axis
    if args.show_prob_up and sig_val_df is not None and 'prob_up' in sig_val_df.columns:
        val_fig.add_trace(
            go.Scatter(
                x=sig_val_df['timestamp'], y=sig_val_df['prob_up'], name='val prob_up',
                line=dict(color='#2ca02c', width=1), mode='lines',
                hovertemplate='%{x}<br>prob_up=%{y:.3f}<extra></extra>'
            ),
            row=1, col=1, secondary_y=True,
        )
        # Threshold reference lines, if provided
        if args.prob_thresholds:
            tau_long, tau_short = float(args.prob_thresholds[0]), float(args.prob_thresholds[1])
            for tau, name in [(tau_long, 'τ_long'), (tau_short, 'τ_short')]:
                val_fig.add_trace(
                    go.Scatter(
                        x=[sig_val_df['timestamp'].min(), sig_val_df['timestamp'].max()],
                        y=[tau, tau], name=f'val {name}',
                        line=dict(color='#2ca02c', width=1, dash='dash'), mode='lines',
                        hovertemplate=f'threshold={tau:.3f}<extra></extra>'
                    ),
                    row=1, col=1, secondary_y=True,
                )
    # Candlestick under predictions
    val_start, val_end = val_df['timestamp'].min(), val_df['timestamp'].max()
    val_ohlcv = _subset_ohlcv_for_window(ohlcv_df, val_start, val_end)
    if not val_ohlcv.empty:
        val_fig.add_trace(
            go.Candlestick(
                x=val_ohlcv['timestamp'],
                open=val_ohlcv['open'], high=val_ohlcv['high'], low=val_ohlcv['low'], close=val_ohlcv['close'],
                showlegend=False,
                increasing_line_color='green', decreasing_line_color='red',
            ),
            row=2, col=1,
        )
    val_fig.update_layout(
        title='Validation: Quantile predictions vs target (interactive)',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1.0),
        hovermode='x unified',
    )
    val_fig.update_yaxes(title_text='y / preds', row=1, col=1)
    val_fig.update_yaxes(title_text='price (OHLC)', row=2, col=1)
    val_fig.update_xaxes(title_text='timestamp', row=2, col=1, rangeslider=dict(visible=True))
    out_val = run_dir / f"{args.prefix}_val.html"
    val_fig.write_html(out_val)

    # Test figure with candlestick subplot (secondary y on row 1 for prob_up)
    test_fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.65, 0.35],
        specs=[[{"secondary_y": True}], [{}]],
    )
    test_traces, _ = _make_traces(test_df, 'test')
    for tr in test_traces:
        test_fig.add_trace(tr, row=1, col=1, secondary_y=False)
    # Overlay expected return from signals if available
    if sig_test_df is not None and method_col in sig_test_df.columns:
        test_fig.add_trace(
            go.Scatter(
                x=sig_test_df['timestamp'], y=sig_test_df[method_col], name=f'test {method_col}',
                line=dict(color='#bcbd22', width=2),
                mode='lines',
                hovertemplate='%{x}<br>'+method_col+'=%{y:.6f}<extra></extra>'
            ),
            row=1, col=1, secondary_y=False,
        )
        if args.mark_cross_violations and 'cross_violations' in sig_test_df.columns:
            mask = (sig_test_df['cross_violations'] > 0) & sig_test_df[method_col].notna()
            if mask.any():
                custom = list(zip(sig_test_df.loc[mask, 'cross_violations'], sig_test_df.loc[mask, 'max_cross_gap'] if 'max_cross_gap' in sig_test_df.columns else [None] * mask.sum()))
                test_fig.add_trace(
                    go.Scatter(
                        x=sig_test_df.loc[mask, 'timestamp'],
                        y=sig_test_df.loc[mask, method_col],
                        name='test cross violations',
                        mode='markers',
                        marker=dict(color='#d62728', size=7, symbol='x'),
                        customdata=custom,
                        hovertemplate='%{x}<br>'+method_col+'=%{y:.6f}<br>cross=%{customdata[0]}<br>gap=%{customdata[1]:.6f}<extra></extra>'
                    ),
                    row=1, col=1, secondary_y=False,
                )

    # Optional prob_up on secondary y-axis (test)
    if args.show_prob_up and sig_test_df is not None and 'prob_up' in sig_test_df.columns:
        test_fig.add_trace(
            go.Scatter(
                x=sig_test_df['timestamp'], y=sig_test_df['prob_up'], name='test prob_up',
                line=dict(color='#2ca02c', width=1), mode='lines',
                hovertemplate='%{x}<br>prob_up=%{y:.3f}<extra></extra>'
            ),
            row=1, col=1, secondary_y=True,
        )
        if args.prob_thresholds:
            tau_long, tau_short = float(args.prob_thresholds[0]), float(args.prob_thresholds[1])
            for tau, name in [(tau_long, 'τ_long'), (tau_short, 'τ_short')]:
                test_fig.add_trace(
                    go.Scatter(
                        x=[sig_test_df['timestamp'].min(), sig_test_df['timestamp'].max()],
                        y=[tau, tau], name=f'test {name}',
                        line=dict(color='#2ca02c', width=1, dash='dash'), mode='lines',
                        hovertemplate=f'threshold={tau:.3f}<extra></extra>'
                    ),
                    row=1, col=1, secondary_y=True,
                )
    test_start, test_end = test_df['timestamp'].min(), test_df['timestamp'].max()
    test_ohlcv = _subset_ohlcv_for_window(ohlcv_df, test_start, test_end)
    if not test_ohlcv.empty:
        test_fig.add_trace(
            go.Candlestick(
                x=test_ohlcv['timestamp'],
                open=test_ohlcv['open'], high=test_ohlcv['high'], low=test_ohlcv['low'], close=test_ohlcv['close'],
                showlegend=False,
                increasing_line_color='green', decreasing_line_color='red',
            ),
            row=2, col=1,
        )
    test_fig.update_layout(
        title='Test: Quantile predictions vs target (interactive)',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1.0),
        hovermode='x unified',
    )
    test_fig.update_yaxes(title_text='y / preds', row=1, col=1)
    test_fig.update_yaxes(title_text='price (OHLC)', row=2, col=1)
    test_fig.update_xaxes(title_text='timestamp', row=2, col=1, rangeslider=dict(visible=True))
    out_test = run_dir / f"{args.prefix}_test.html"
    test_fig.write_html(out_test)

    print(f'Wrote interactive plots: {out_val} and {out_test}')


if __name__ == '__main__':
    main()


