import argparse
from pathlib import Path
from typing import Tuple

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


def _make_traces(df: pd.DataFrame, name_prefix: str) -> Tuple[list[go.Scatter], bool]:
    x = df['timestamp']
    traces: list[go.Scatter] = []
    has_band = ('pred_q05' in df.columns) and ('pred_q95' in df.columns)
    has_inner_band = ('pred_q25' in df.columns) and ('pred_q75' in df.columns)

    # Outer Band (q95 then q05 with fill)
    if has_band:
        q95 = df['pred_q95']
        q05 = df['pred_q05']
        traces.append(
            go.Scatter(
                x=x, y=q95, name=f'{name_prefix} q95',
                line=dict(color='red', width=1),
                mode='lines',
                hovertemplate='%{x}<br>q95=%{y:.6f}<extra></extra>'
            )
        )
        traces.append(
            go.Scatter(
                x=x, y=q05, name=f'{name_prefix} q05',
                line=dict(color='blue', width=1),
                mode='lines',
                fill='tonexty', fillcolor='rgba(100,149,237,0.15)',
                hovertemplate='%{x}<br>q05=%{y:.6f}<extra></extra>'
            )
        )

    # Inner Band (q75 then q25 with fill) if available
    if has_inner_band:
        q75 = df['pred_q75']
        q25 = df['pred_q25']
        traces.append(
            go.Scatter(
                x=x, y=q75, name=f'{name_prefix} q75',
                line=dict(color='orange', width=1),
                mode='lines',
                hovertemplate='%{x}<br>q75=%{y:.6f}<extra></extra>'
            )
        )
        traces.append(
            go.Scatter(
                x=x, y=q25, name=f'{name_prefix} q25',
                line=dict(color='orange', width=1),
                mode='lines',
                fill='tonexty', fillcolor='rgba(255,165,0,0.15)',
                hovertemplate='%{x}<br>q25=%{y:.6f}<extra></extra>'
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

    return traces, has_band


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
    parser = argparse.ArgumentParser(description='Interactive plot of quantile predictions vs target (val/test) with OHLCV candlesticks')
    parser.add_argument('--run-dir', type=Path, required=True, help='Path to model run directory with pred_val.csv and pred_test.csv')
    parser.add_argument('--prefix', type=str, default='quantile_preds_interactive', help='Output filename prefix (will create <prefix>_val.html and <prefix>_test.html)')
    parser.add_argument('--ohlcv-csv', type=Path, default=Path('data/BINANCE_BTCUSDT.P, 60.csv'), help='Path to original OHLCV CSV (with time/timestamp, open, high, low, close)')
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    val_df = _load_pred(run_dir / 'pred_val.csv')
    test_df = _load_pred(run_dir / 'pred_test.csv')
    ohlcv_df = _load_ohlcv(args.ohlcv_csv)

    # Validation figure with candlestick subplot
    val_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.65, 0.35])
    val_traces, _ = _make_traces(val_df, 'val')
    for tr in val_traces:
        val_fig.add_trace(tr, row=1, col=1)
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

    # Test figure with candlestick subplot
    test_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.65, 0.35])
    test_traces, _ = _make_traces(test_df, 'test')
    for tr in test_traces:
        test_fig.add_trace(tr, row=1, col=1)
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


