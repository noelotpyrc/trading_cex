from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


@dataclass
class PlotConfig:
    figure_size: tuple[float, float] = (14, 6)
    up_color: str = "#2ca02c"
    down_color: str = "#d62728"
    wick_width: float = 1.0
    body_alpha: float = 0.6
    candle_width: float = 0.6
    price_color: str = "#1f77b4"
    marker_edgecolor: str = "black"
    marker_size: float = 50.0


def _to_utc(series: pd.Series) -> pd.Series:
    """Parse to pandas datetime in UTC (tz-aware)."""
    return pd.to_datetime(series, errors="coerce", utc=True)


def load_predictions_split(diagnosis_dir: str | Path, split: str) -> pd.DataFrame:
    """Load predictions CSV for a split: one of {'train','val','test'}.

    Expected columns: timestamp, y_true, y_pred
    """
    diagnosis_dir = Path(diagnosis_dir)
    path = diagnosis_dir / f"pred_{split}.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])  # parse, then normalize to UTC
    df["timestamp"] = _to_utc(df["timestamp"])
    required = {"timestamp", "y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Predictions CSV missing columns: {sorted(missing)} in {path}")
    return df


def load_ohlcv_csv(ohlcv_path: str | Path) -> pd.DataFrame:
    """Load OHLCV CSV and normalize columns to ['timestamp','open','high','low','close'].

    Mirrors the helper used in the notebook: prefers 'timestamp' if present, otherwise 'time'.
    """
    ohlcv_path = Path(ohlcv_path)
    df = pd.read_csv(ohlcv_path)

    # Map timestamp/time column
    ts_col: Optional[str] = None
    for candidate in ("timestamp", "time", "date", "datetime"):
        if candidate in df.columns:
            ts_col = candidate
            break
    if ts_col is None:
        raise ValueError("OHLCV CSV missing a timestamp-like column ('timestamp'/'time').")
    df["timestamp"] = _to_utc(df[ts_col])

    # Normalize price column names to lowercase open,high,low,close
    colmap: dict[str, str] = {}
    for c in df.columns:
        lc = str(c).lower()
        if lc in {"open", "high", "low", "close"}:
            colmap[c] = lc
    df = df.rename(columns=colmap)

    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV CSV missing columns: {sorted(missing)} in {ohlcv_path}")

    df = df[["timestamp", "open", "high", "low", "close"]].dropna(subset=["timestamp"]).copy()
    df.sort_values("timestamp", inplace=True)
    return df


def generate_signal(df: pd.DataFrame, mode: str = "long_only", out_col: str = "ind_pred_pos") -> pd.DataFrame:
    """Generate signal column from y_pred.

    - long_only: 1 if y_pred > 0 else 0
    - long_short: sign(y_pred) in {-1, 0, 1}
    """
    y_pred_num = pd.to_numeric(df["y_pred"], errors="coerce")
    if mode == "long_short":
        df[out_col] = np.sign(y_pred_num).astype(int)
    else:
        df[out_col] = (y_pred_num > 0).astype(int)
    return df


def _prepare_join_on_hour(ohlcv_df: pd.DataFrame, preds_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align OHLCV and predictions by flooring timestamps to the hour in UTC (tz-aware)."""
    o = ohlcv_df.copy()
    p = preds_df.copy()
    o["ts_h"] = _to_utc(o["timestamp"]).dt.floor("h")
    p["ts_h"] = _to_utc(p["timestamp"]).dt.floor("h")
    o.sort_values("ts_h", inplace=True)
    p.sort_values("ts_h", inplace=True)
    return o, p


def plot_candles_with_signal_markers(
    ohlcv_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    signal_col: str = "ind_pred_pos",
    signal_value: int = 1,
    config: PlotConfig | None = None,
    title: Optional[str] = None,
):
    """Draw candlesticks and overlay signal markers.

    - Mark all bars where `preds_df[signal_col] == signal_value`.
    - Marker shape encodes outcome by `y_true`: up-triangle for y_true>0 (win), down-triangle for y_true<=0 (loss).
    - Candles: green for up (close>=open), red for down (close<open).
    """
    if config is None:
        config = PlotConfig()

    # Align on hour, restrict to split period
    o, p = _prepare_join_on_hour(ohlcv_df, preds_df)
    if p.empty:
        raise ValueError("No prediction rows after timestamp parsing.")

    start_ts, end_ts = p["ts_h"].min(), p["ts_h"].max()
    o = o[(o["ts_h"] >= start_ts) & (o["ts_h"] <= end_ts)].copy()
    o.reset_index(drop=True, inplace=True)
    o["x"] = np.arange(len(o), dtype=float)

    # Merge predictions onto OHLCV x-index for plotting markers
    p_mark = p[["ts_h", signal_col, "y_true"]].copy()
    p_mark = p_mark[p_mark[signal_col].astype(int) == int(signal_value)]
    merged = p_mark.merge(o[["ts_h", "x", "open", "high", "low", "close"]], on="ts_h", how="inner")

    fig, ax = plt.subplots(figsize=config.figure_size)

    # Draw candlesticks
    for _, row in o.iterrows():
        x = row["x"]
        o_, h_, l_, c_ = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]) 
        color = config.up_color if c_ >= o_ else config.down_color
        # Wick
        ax.vlines(x, l_, h_, color=color, linewidth=config.wick_width, zorder=1)
        # Body
        body_bottom = min(o_, c_)
        body_height = abs(c_ - o_)
        if body_height == 0:
            # Draw a small line if doji
            ax.hlines(body_bottom, x - config.candle_width / 2.0, x + config.candle_width / 2.0, color=color, linewidth=2, zorder=2)
        else:
            rect = Rectangle(
                (x - config.candle_width / 2.0, body_bottom),
                config.candle_width,
                body_height,
                facecolor=color,
                edgecolor=color,
                alpha=config.body_alpha,
                zorder=2,
            )
            ax.add_patch(rect)

    # Overlay markers for signals
    if not merged.empty:
        win_mask = pd.to_numeric(merged["y_true"], errors="coerce") > 0
        # Position markers just above candle high
        bar_span = (o["high"].max() - o["low"].min())
        offset = 0.004 * float(bar_span) if bar_span > 0 else 0.0

        # Wins
        m_win = merged.loc[win_mask]
        if not m_win.empty:
            ax.scatter(
                m_win["x"],
                m_win["high"] + offset,
                marker="^",
                color=config.up_color,
                edgecolor=config.marker_edgecolor,
                s=config.marker_size,
                linewidths=0.5,
                label="signal & y_true>0",
                zorder=4,
            )
        # Losses
        m_loss = merged.loc[~win_mask]
        if not m_loss.empty:
            ax.scatter(
                m_loss["x"],
                m_loss["high"] + offset,
                marker="v",
                color=config.down_color,
                edgecolor=config.marker_edgecolor,
                s=config.marker_size,
                linewidths=0.5,
                label="signal & y_true<=0",
                zorder=4,
            )

    # Axes formatting
    ax.set_xlim(-0.5, len(o) - 0.5)
    ax.set_ylabel("price")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.2, axis="y")

    # Time labels: show up to ~10 ticks evenly spaced
    num = len(o)
    if num > 0:
        num_ticks = min(10, num)
        idx = (np.linspace(0, num - 1, num_ticks)).astype(int)
        ax.set_xticks(idx)
        ax.set_xticklabels(o.loc[idx, "timestamp"].dt.strftime("%Y-%m-%d %H:%M"), rotation=30, ha="right")
    ax.legend(loc="best")
    plt.tight_layout()
    return fig, ax


def plot_candles_with_signal_markers_interactive(
    ohlcv_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    signal_col: str = "ind_pred_pos",
    signal_value: int = 1,
    title: Optional[str] = None,
):
    """Interactive version using Plotly.

    - Candlesticks for OHLCV
    - Scatter markers at bars where signal==signal_value
      - Up triangle if y_true>0, down triangle otherwise
    """
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except Exception as exc:
        raise RuntimeError(
            "Plotly is required for --interactive. Please `pip install plotly`."
        ) from exc

    # Align on hour and restrict period
    o, p = _prepare_join_on_hour(ohlcv_df, preds_df)
    if p.empty:
        raise ValueError("No prediction rows after timestamp parsing.")

    start_ts, end_ts = p["ts_h"].min(), p["ts_h"].max()
    o = o[(o["ts_h"] >= start_ts) & (o["ts_h"] <= end_ts)].copy()

    # Markers dataset
    p_mark = p[["ts_h", signal_col, "y_true"]].copy()
    p_mark = p_mark[p_mark[signal_col].astype(int) == int(signal_value)]
    merged = p_mark.merge(o[["ts_h", "open", "high", "low", "close"]], on="ts_h", how="inner")

    fig = go.Figure()

    # Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=o["ts_h"],
            open=o["open"],
            high=o["high"],
            low=o["low"],
            close=o["close"],
            name="OHLC",
            increasing_line_color="#2ca02c",
            decreasing_line_color="#d62728",
            increasing_fillcolor="#2ca02c",
            decreasing_fillcolor="#d62728",
            showlegend=False,
        )
    )

    if not merged.empty:
        win_mask = pd.to_numeric(merged["y_true"], errors="coerce") > 0
        # Offset above candle highs
        bar_span = float(o["high"].max() - o["low"].min()) if len(o) else 0.0
        offset = 0.004 * bar_span if bar_span > 0 else 0.0

        win_df = merged.loc[win_mask]
        if not win_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=win_df["ts_h"],
                    y=win_df["high"] + offset,
                    mode="markers",
                    name="signal & y_true>0",
                    marker=dict(symbol="triangle-up", color="#2ca02c", size=10, line=dict(color="black", width=0.5)),
                )
            )
        loss_df = merged.loc[~win_mask]
        if not loss_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=loss_df["ts_h"],
                    y=loss_df["high"] + offset,
                    mode="markers",
                    name="signal & y_true<=0",
                    marker=dict(symbol="triangle-down", color="#d62728", size=10, line=dict(color="black", width=0.5)),
                )
            )

    fig.update_layout(
        title=title or "Candles with signals",
        xaxis_title="timestamp",
        yaxis_title="price",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig

def main():
    parser = argparse.ArgumentParser(description="Plot candlesticks with signal markers from predictions.")
    parser.add_argument("--ohlcv", required=True, help="Path to OHLCV CSV (must contain timestamp, open, high, low, close).")
    parser.add_argument("--diagnosis", required=True, help="Diagnosis directory containing pred_{split}.csv files.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test", help="Which predictions split to plot.")
    parser.add_argument("--signal-mode", choices=["long_only", "long_short"], default="long_only", help="How to derive signal from y_pred.")
    parser.add_argument("--signal-value", type=int, default=1, help="Signal value to mark (1 for long-only).")
    parser.add_argument("--output", default=None, help="Optional path to save the static (PNG) matplotlib figure.")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive Plotly candlestick output.")
    parser.add_argument("--html", default=None, help="When --interactive is set, write Plotly figure to this HTML file.")
    args = parser.parse_args()

    ohlcv_df = load_ohlcv_csv(args.ohlcv)
    preds_df = load_predictions_split(args.diagnosis, args.split)
    preds_df = generate_signal(preds_df, mode=args.signal_mode, out_col="ind_pred_pos")

    title = f"Candles with signals ({args.split})"

    if args.interactive:
        fig = plot_candles_with_signal_markers_interactive(
            ohlcv_df=ohlcv_df,
            preds_df=preds_df,
            signal_col="ind_pred_pos",
            signal_value=int(args.signal_value),
            title=title,
        )
        if args.html:
            from plotly.io import write_html

            out_path = Path(args.html)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            write_html(fig, file=str(out_path), auto_open=False)
            print(f"Saved interactive HTML to {out_path}")
        else:
            fig.show()
    else:
        fig, _ = plot_candles_with_signal_markers(
            ohlcv_df=ohlcv_df,
            preds_df=preds_df,
            signal_col="ind_pred_pos",
            signal_value=int(args.signal_value),
            title=title,
        )
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=150)
            print(f"Saved figure to {out_path}")
        else:
            plt.show()


if __name__ == "__main__":
    main()


