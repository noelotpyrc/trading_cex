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


def _maybe_roll(s: pd.Series, window: int) -> pd.Series:
    if window and window > 1:
        return s.rolling(window, min_periods=max(1, window // 3)).mean()
    return s


def _make_traces(df: pd.DataFrame, name_prefix: str, rolling: int = 0) -> Tuple[list[go.Scatter], bool]:
    x = df['timestamp']
    traces: list[go.Scatter] = []
    has_band = ('pred_q05' in df.columns) and ('pred_q95' in df.columns)

    # Band (q95 then q05 with fill)
    if has_band:
        q95 = _maybe_roll(df['pred_q95'], rolling)
        q05 = _maybe_roll(df['pred_q05'], rolling)
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

    # Median
    if 'pred_q50' in df.columns:
        q50 = _maybe_roll(df['pred_q50'], rolling)
        traces.append(
            go.Scatter(
                x=x, y=q50, name=f'{name_prefix} q50',
                line=dict(color='green', width=1),
                mode='lines',
                hovertemplate='%{x}<br>q50=%{y:.6f}<extra></extra>'
            )
        )

    # Target
    y_true = _maybe_roll(df['y_true'], rolling)
    traces.append(
        go.Scatter(
            x=x, y=y_true, name=f'{name_prefix} y_true',
            line=dict(color='black', width=1.2),
            mode='lines',
            hovertemplate='%{x}<br>y=%{y:.6f}<extra></extra>'
        )
    )

    return traces, has_band


def main() -> None:
    parser = argparse.ArgumentParser(description='Interactive plot of quantile predictions vs target (val/test)')
    parser.add_argument('--run-dir', type=Path, required=True, help='Path to model run directory with pred_val.csv and pred_test.csv')
    parser.add_argument('--prefix', type=str, default='quantile_preds_interactive', help='Output filename prefix (will create <prefix>_val.html and <prefix>_test.html)')
    parser.add_argument('--rolling', type=int, default=0, help='Optional rolling window for smoothing (0=off)')
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    val_df = _load_pred(run_dir / 'pred_val.csv')
    test_df = _load_pred(run_dir / 'pred_test.csv')

    # Validation figure
    val_fig = go.Figure()
    val_traces, _ = _make_traces(val_df, 'val', rolling=args.rolling)
    for tr in val_traces:
        val_fig.add_trace(tr)
    val_fig.update_layout(
        title='Validation: Quantile predictions vs target (interactive)',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1.0),
        hovermode='x unified',
    )
    val_fig.update_yaxes(title_text='y / preds')
    val_fig.update_xaxes(title_text='timestamp', rangeslider=dict(visible=True))
    out_val = run_dir / f"{args.prefix}_val.html"
    val_fig.write_html(out_val)

    # Test figure
    test_fig = go.Figure()
    test_traces, _ = _make_traces(test_df, 'test', rolling=args.rolling)
    for tr in test_traces:
        test_fig.add_trace(tr)
    test_fig.update_layout(
        title='Test: Quantile predictions vs target (interactive)',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1.0),
        hovermode='x unified',
    )
    test_fig.update_yaxes(title_text='y / preds')
    test_fig.update_xaxes(title_text='timestamp', rangeslider=dict(visible=True))
    out_test = run_dir / f"{args.prefix}_test.html"
    test_fig.write_html(out_test)

    print(f'Wrote interactive plots: {out_val} and {out_test}')


if __name__ == '__main__':
    main()


