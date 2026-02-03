"""
New VWAP session strategy implemented on top of simple_backtesting.

Timing model for this strategy:
- Entry signal evaluated at bar close (bar N).
- Entry order fills using OHLC/4 of bar N+1 (continuous fill from open to close).
- Entry is considered complete at close of bar N+1.
- Stop is placed at close of the fill bar using that bar's close.
- Stop is active for N bars; if it expires, schedule a time-exit at next bar open
  using the expiration bar close as the exit price.
- Exit fills at stop price (limit-style fill).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

DATETIME_COL = "datetime_utc"

_order_id_counter = 0


def _next_order_id() -> int:
    global _order_id_counter
    _order_id_counter += 1
    return _order_id_counter


def _reset_order_id() -> None:
    global _order_id_counter
    _order_id_counter = 0


def _evaluate(value: Any, op: str, threshold: float | None) -> bool:
    if threshold is None:
        return True
    if pd.isna(value):
        return False
    if op == ">":
        return value > threshold
    if op == ">=":
        return value >= threshold
    if op == "<":
        return value < threshold
    if op == "<=":
        return value <= threshold
    if op == "==":
        return value == threshold
    if op == "!=":
        return value != threshold
    raise ValueError(f"Unsupported operator: {op}")


def check_entry_signal(df: pd.DataFrame, bar_idx: int, config: dict) -> dict | None:
    row = df.iloc[bar_idx]

    for filt in config["filters"]:
        value = row[filt["col"]]
        if not _evaluate(value, filt.get("op", ">="), filt.get("threshold")):
            return None

    direction_cfg = config["direction"]
    direction_filter = direction_cfg.get("filter", "both")
    if direction_filter == "long":
        long_rule = direction_cfg["long_rule"]
        long_value = row[long_rule["col"]]
        if not _evaluate(long_value, long_rule.get("op", ">="), long_rule.get("threshold")):
            return None
        direction = "long"
    elif direction_filter == "short":
        short_rule = direction_cfg["short_rule"]
        short_value = row[short_rule["col"]]
        if not _evaluate(short_value, short_rule.get("op", "<="), short_rule.get("threshold")):
            return None
        direction = "short"
    else:
        long_rule = direction_cfg["long_rule"]
        short_rule = direction_cfg["short_rule"]
        long_value = row[long_rule["col"]]
        short_value = row[short_rule["col"]]

        long_ok = _evaluate(long_value, long_rule.get("op", ">="), long_rule.get("threshold"))
        short_ok = _evaluate(short_value, short_rule.get("op", "<="), short_rule.get("threshold"))

        if long_ok and short_ok:
            tie_breaker = direction_cfg.get("tie_breaker", "skip")
            if tie_breaker == "long":
                direction = "long"
            elif tie_breaker == "short":
                direction = "short"
            else:
                return None
        elif long_ok:
            direction = "long"
        elif short_ok:
            direction = "short"
        else:
            return None

    return {
        "order_type": "market",
        "direction": direction,
        "qty": config["qty"],
    }


def create_entry_order(entry_signal: dict, bar_idx: int, config: dict) -> dict:
    expiry_bars = config["expiry_bars"]
    placed_at = config["placed_at"]
    return {
        "id": _next_order_id(),
        "side": "entry",
        "direction": entry_signal.get("direction", "long"),
        "qty": entry_signal.get("qty", config["qty"]),
        "order_type": entry_signal.get("order_type", "market"),
        "price": entry_signal.get("price"),
        "placed_bar": bar_idx,
        "placed_at": placed_at,
        "expires_at_bar": bar_idx + expiry_bars,
        "status": "working",
    }


def create_exit_order(exit_signal: dict, position: dict, bar_idx: int, config: dict) -> dict:
    expiry_bars = exit_signal.get("expiry_bars", config.get("expiry_bars", 30))
    placed_at = exit_signal.get("placed_at", "close")
    return {
        "id": _next_order_id(),
        "side": "exit",
        "direction": position["direction"],
        "qty": position.get("qty", 1),
        "order_type": exit_signal.get("order_type", "market"),
        "price": exit_signal.get("price"),
        "placed_bar": bar_idx,
        "placed_at": placed_at,
        "expires_at_bar": bar_idx + expiry_bars,
        "status": "working",
        "reason": exit_signal.get("reason"),
    }


def _ohlc4(bar: pd.Series) -> float:
    return (bar["open"] + bar["high"] + bar["low"] + bar["close"]) / 4.0


def _calc_stop_from_close(direction: str, close_price: float, stop_loss_pct: float) -> float:
    if direction == "long":
        return close_price * (1 + stop_loss_pct / 100)
    return close_price * (1 - stop_loss_pct / 100)


def _dynamic_stop_loss_pct(
    df: pd.DataFrame,
    entry_bar_idx: int,
    exit_config: dict,
) -> float | None:
    """
    Compute a volatility-based stop distance using only information available
    prior to the entry bar to avoid lookahead.

    dynamic_stop = min(stop_vol * stop_multiplier, stop_cutoff)

    - stop_vol is read from (entry_bar_idx - 1) row, using stop_vol_col
    - stop_cutoff is in return units (e.g. 0.005 == 50bps)
    - returned value is stop_loss_pct in *percent* units, negative for loss
      (e.g. -0.5 means -50bps).
    """
    vol_col = exit_config.get("stop_vol_col", "parkinson_30")
    stop_multiplier = float(exit_config.get("stop_multiplier", 4.0))

    if "stop_cutoff_bps" in exit_config:
        stop_cutoff = float(exit_config["stop_cutoff_bps"]) / 10000.0
    else:
        stop_cutoff = float(exit_config.get("stop_cutoff", 0.005))
        # Allow users to pass 50 (bps) by mistake; treat >1 as bps.
        if stop_cutoff > 1:
            stop_cutoff = stop_cutoff / 10000.0

    vol_idx = entry_bar_idx - 1
    if vol_idx < 0:
        return None

    stop_vol = df.iloc[vol_idx][vol_col]
    if pd.isna(stop_vol):
        return None

    dynamic_stop = min(float(stop_vol) * stop_multiplier, stop_cutoff)
    return -dynamic_stop * 100.0


def _calc_pnl_pct(entry_price: float, exit_price: float, direction: str, fee_pct: float) -> float:
    if direction == "long":
        pnl = (exit_price / entry_price - 1) * 100
    else:
        pnl = (1 - exit_price / entry_price) * 100
    return pnl - fee_pct


def _exit_trade(
    position: dict,
    trades: list,
    bar: pd.Series,
    bar_idx: int,
    exit_price: float,
    reason: str,
    fee_pct: float,
) -> None:
    pnl_pct = _calc_pnl_pct(position["entry_price"], exit_price, position["direction"], fee_pct)
    trades.append({
        "entry_time": position["entry_time"],
        "exit_time": bar[DATETIME_COL],
        "direction": position["direction"],
        "entry_price": position["entry_price"],
        "exit_price": exit_price,
        "bars_held": bar_idx - position["entry_bar"],
        "exit_reason": reason,
        "stop_loss_pct": position.get("stop_loss_pct"),
        "pnl_pct": pnl_pct,
    })


def process_order(
    pending_order: dict | None,
    position: dict | None,
    trades: list,
    df: pd.DataFrame,
    bar: pd.Series,
    bar_idx: int,
    exit_config: dict,
) -> tuple[dict | None, dict | None, list]:
    if not pending_order:
        if position is not None and bar_idx == len(df) - 1:
            _exit_trade(position, trades, bar, bar_idx, bar["close"], "end_of_data", exit_config["fee_pct"])
            return None, None, trades
        return pending_order, position, trades

    if bar_idx <= pending_order["placed_bar"]:
        return pending_order, position, trades

    if pending_order["side"] == "entry":
        fill_price = _ohlc4(bar)
        position = {
            "entry_bar": bar_idx,
            "entry_price": fill_price,
            "entry_time": bar[DATETIME_COL],
            "direction": pending_order["direction"],
            "qty": pending_order.get("qty", 1),
        }
        return None, position, trades

    if pending_order["side"] == "exit":
        order_type = pending_order["order_type"]
        if order_type == "time_exit":
            _exit_trade(
                position,
                trades,
                bar,
                bar_idx,
                pending_order["price"],
                pending_order["reason"],
                exit_config["fee_pct"],
            )
            return None, None, trades

        if order_type == "stop":
            stop_price = pending_order["price"]
            if position["direction"] == "long" and bar["low"] <= stop_price:
                _exit_trade(
                    position,
                    trades,
                    bar,
                    bar_idx,
                    stop_price,
                    pending_order["reason"],
                    exit_config["fee_pct"],
                )
                return None, None, trades
            if position["direction"] == "short" and bar["high"] >= stop_price:
                _exit_trade(
                    position,
                    trades,
                    bar,
                    bar_idx,
                    stop_price,
                    pending_order["reason"],
                    exit_config["fee_pct"],
                )
                return None, None, trades

            if bar_idx >= pending_order["expires_at_bar"]:
                if bar_idx == len(df) - 1:
                    _exit_trade(
                        position,
                        trades,
                        bar,
                        bar_idx,
                        bar["close"],
                        "end_of_data",
                        exit_config["fee_pct"],
                    )
                    return None, None, trades

                return None, position, trades

        return pending_order, position, trades

    return pending_order, position, trades


def check_exit_signal(df: pd.DataFrame, bar_idx: int, position: dict, config: dict) -> dict | None:
    bar = df.iloc[bar_idx]
    bars_held = bar_idx - position["entry_bar"]

    if bars_held == 0:
        stop_loss_pct = config.get("stop_loss_pct")
        if config.get("use_dynamic_stop") or config.get("dynamic_stop"):
            dyn_stop_loss_pct = _dynamic_stop_loss_pct(df, position["entry_bar"], config)
            if dyn_stop_loss_pct is not None:
                stop_loss_pct = dyn_stop_loss_pct

        if stop_loss_pct is None:
            raise ValueError("stop_loss_pct is required (or provide dynamic stop settings).")

        stop_price = _calc_stop_from_close(position["direction"], bar["close"], float(stop_loss_pct))
        return {
            "order_type": "stop",
            "price": stop_price,
            "reason": "stop_loss",
            "expiry_bars": config["expiry_bars"],
            "placed_at": "close",
            "stop_loss_pct": float(stop_loss_pct),
        }

    if bars_held >= config["expiry_bars"]:
        return {
            "order_type": "time_exit",
            "price": bar["close"],
            "reason": "reached_time_limit",
            "expiry_bars": 1,
            "placed_at": "close",
        }

    return None


def update_pending_order(
    pending_order: dict | None,
    position: dict | None,
    df: pd.DataFrame,
    bar: pd.Series,
    bar_idx: int,
    exit_config: dict,
) -> dict | None:
    return pending_order


def run_backtest(df: pd.DataFrame, entry_config: dict, exit_config: dict) -> pd.DataFrame:
    _reset_order_id()
    pending_order = None
    position = None
    trades = []

    for bar_idx in range(len(df)):
        bar = df.iloc[bar_idx]

        # Update/replace pending order (placeholder for now).
        pending_order = update_pending_order(pending_order, position, df, bar, bar_idx, exit_config)

        # Process fills for orders active during this bar using this bar's OHLC
        # (evaluated at bar close; entry fills use OHLC/4 of this bar).
        pending_order, position, trades = process_order(
            pending_order,
            position,
            trades,
            df,
            bar,
            bar_idx,
            exit_config,
        )

        if position is None and pending_order is None:
            # Entry signal evaluated at bar close; orders placed after close (fill next bar).
            entry_signal = check_entry_signal(df, bar_idx, entry_config)
            if entry_signal:
                # Create entry order at bar close; fills on next bar open/close per process_order.
                pending_order = create_entry_order(entry_signal, bar_idx, entry_config)

        if position is not None and pending_order is None:
            # Exit signal evaluated at bar close; stop/time-exit orders placed after close.
            exit_signal = check_exit_signal(df, bar_idx, position, exit_config)
            if exit_signal:
                if exit_signal.get("order_type") == "stop" and "stop_loss_pct" in exit_signal:
                    position["stop_loss_pct"] = exit_signal["stop_loss_pct"]
                # Create exit order at bar close; fills on subsequent bars via process_order.
                pending_order = create_exit_order(exit_signal, position, bar_idx, exit_config)

    return pd.DataFrame(trades)


def analyze_results(trades_df: pd.DataFrame) -> None:
    """Analyze strategy performance."""
    if len(trades_df) == 0:
        print("No trades to analyze")
        return

    print("=" * 70)
    print("PERFORMANCE ANALYSIS")
    print("=" * 70)

    total_trades = len(trades_df)
    win_trades = len(trades_df[trades_df["pnl_pct"] > 0])
    loss_trades = len(trades_df[trades_df["pnl_pct"] <= 0])
    win_rate = win_trades / total_trades * 100

    print(f"Total Trades: {total_trades:,}")
    print(f"Wins: {win_trades:,} | Losses: {loss_trades:,}")
    print(f"Win Rate: {win_rate:.2f}%")
    print()

    avg_pnl = trades_df["pnl_pct"].mean()
    avg_win = trades_df[trades_df["pnl_pct"] > 0]["pnl_pct"].mean() if win_trades > 0 else 0
    avg_loss = trades_df[trades_df["pnl_pct"] <= 0]["pnl_pct"].mean() if loss_trades > 0 else 0

    print(f"Average PnL: {avg_pnl:.4f}%")
    print(f"Average Win: {avg_win:.4f}%")
    print(f"Average Loss: {avg_loss:.4f}%")
    print()

    print("Exit Reason Breakdown:")
    for reason, count in trades_df["exit_reason"].value_counts().items():
        pct = count / total_trades * 100
        avg_pnl_reason = trades_df[trades_df["exit_reason"] == reason]["pnl_pct"].mean()
        print(f"  {reason}: {count:,} ({pct:.1f}%) - Avg PnL: {avg_pnl_reason:.4f}%")
    print()

    if "direction" in trades_df.columns:
        print("Performance by Direction:")
        for direction in ["long", "short"]:
            subset = trades_df[trades_df["direction"] == direction]
            if len(subset) == 0:
                continue
            wins = len(subset[subset["pnl_pct"] > 0])
            losses = len(subset[subset["pnl_pct"] <= 0])
            win_rate_dir = wins / len(subset) * 100
            avg_pnl_dir = subset["pnl_pct"].mean()
            avg_win_dir = subset[subset["pnl_pct"] > 0]["pnl_pct"].mean() if wins > 0 else 0
            avg_loss_dir = subset[subset["pnl_pct"] <= 0]["pnl_pct"].mean() if losses > 0 else 0
            print(
                f"  {direction}: {len(subset):,} trades, {wins} wins, {losses} losses "
                f"({win_rate_dir:.2f}% win), Avg PnL: {avg_pnl_dir:.4f}%, "
                f"Avg Win: {avg_win_dir:.4f}%, Avg Loss: {avg_loss_dir:.4f}%"
            )
        print()

    if "trigger_bar" in trades_df.columns:
        print("Performance by Trigger Bar:")
        for bar in sorted(trades_df["trigger_bar"].unique()):
            subset = trades_df[trades_df["trigger_bar"] == bar]
            wins = len(subset[subset["pnl_pct"] > 0])
            total_pnl = subset["pnl_pct"].sum()
            print(
                f"  Bar {bar}: {len(subset)} trades, {wins} wins "
                f"({wins/len(subset)*100:.1f}%), PnL: {total_pnl:.2f}%"
            )
        print()

    capital = 100_000
    max_position = 100_000
    for pnl_pct in trades_df.sort_values("entry_time")["pnl_pct"]:
        position_size = min(capital, max_position)
        capital += position_size * (pnl_pct / 100)

    total_roi = (capital - 100_000) / 100_000 * 100
    print(f"Total ROI: {total_roi:.2f}%")
    print(f"Final Capital: ${capital:,.2f}")
    print()
