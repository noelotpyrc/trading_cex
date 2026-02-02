"""
Simple Backtesting Framework

Timing model via attributes:
- Orders have: placed_at ("open" or "close"), placed_bar (bar index)
- Positions have: entry_at ("open" or "close"), entry_bar (bar index)

process_order runs at each bar's OPEN and uses these attributes to determine
which bar's data to use for fill checks:
- Order placed at "open" of bar N-1 → check against bar N-1's HLC
- Order placed at "close" of bar N-1 → market fills at bar N's open,
  stop/limit check against bar N-1's HLC (order was active during N-1 after close)
"""

import pandas as pd

# Order ID generator (reset per backtest)
_order_id_counter = 0

def _next_order_id():
    global _order_id_counter
    _order_id_counter += 1
    return _order_id_counter

def _reset_order_id():
    global _order_id_counter
    _order_id_counter = 0


def run_backtest(df, entry_config, exit_config):
    """
    Main backtest loop. Structure unchanged - timing handled via attributes.
    """
    _reset_order_id()
    pending_order = None
    position = None
    trades = []

    for bar_idx in range(len(df)):
        bar = df.iloc[bar_idx]

        # 0) Optional: update/replace pending order (placeholder)
        pending_order = update_pending_order(pending_order, position, df, bar, bar_idx, exit_config)

        # 1) Process pending order
        pending_order, position, trades = process_order(pending_order, position, trades, df, bar, bar_idx)

        # 2) Entry (flat + no pending)
        if position is None and pending_order is None:
            entry_signal = check_entry_signal(df, bar_idx, entry_config)
            if entry_signal:
                pending_order = create_entry_order(entry_signal, bar_idx, entry_config)

        # 3) Exit (in position + no pending)
        if position is not None and pending_order is None:
            exit_signal = check_exit_signal(df, bar_idx, position, exit_config)
            if exit_signal:
                pending_order = create_exit_order(exit_signal, position, bar_idx, exit_config)
            else:
                # Optional fallback: always have a protective stop if configured
                if exit_config.get("stop_loss_pct"):
                    pending_order = create_stop_exit_order(position, bar_idx, exit_config)

    return pd.DataFrame(trades)


# --------- Exit logic (you implement) ---------

def check_exit_signal(df, bar_idx, position, config):
    """
    Return dict describing desired exit order or None.

    Examples of returned dict:
      {"order_type": "market", "reason": "time_exit"}
      {"order_type": "limit",  "price": 105.2, "reason": "target_hit"}
      {"order_type": "stop",   "price":  98.7, "reason": "indicator_stop"}
    """
    return None


def create_exit_order(exit_signal, position, bar_idx, config):
    """
    Create an exit order from an exit_signal.
    placed_at defaults to "close" (decision made at bar close).
    """
    expiry_bars = config.get("expiry_bars", 10)
    placed_at = config.get("placed_at", "close")
    return {
        "id": _next_order_id(),
        "side": "exit",
        "direction": position["direction"],
        "qty": position.get("qty", 1),
        "order_type": exit_signal.get("order_type", "market"),
        "price": exit_signal.get("price"),  # may be None for market
        "placed_bar": bar_idx,
        "placed_at": placed_at,
        "expires_at_bar": bar_idx + expiry_bars,
        "status": "working",
        "reason": exit_signal.get("reason"),
    }


def create_stop_exit_order(position, bar_idx, config):
    """
    Fallback protective stop-loss exit.
    placed_at="open" because stop is placed immediately after entry fills at open.
    """
    expiry_bars = config.get("expiry_bars", 10)
    stop_price = calc_stop_price(position, config["stop_loss_pct"])

    # If position was just entered this bar (entry_bar == bar_idx),
    # the stop is placed at "open" of this bar
    if position.get("entry_bar") == bar_idx:
        placed_at = "open"
    else:
        placed_at = "close"  # Stop recreated at close of a later bar

    return {
        "id": _next_order_id(),
        "side": "exit",
        "direction": position["direction"],
        "qty": position.get("qty", 1),
        "order_type": "stop",
        "price": stop_price,
        "placed_bar": bar_idx,
        "placed_at": placed_at,
        "expires_at_bar": bar_idx + expiry_bars,
        "status": "working",
        "reason": "stop_loss",
    }


def update_pending_order(pending_order, position, df, bar, bar_idx, exit_config):
    """
    Optional "single-order" replacement behavior:
    If you keep a protective stop working, but a discretionary exit triggers,
    you can replace that stop (since only 1 pending order is allowed).

    Keep as placeholder for now.
    """
    return pending_order


# --------- Entry logic placeholder ---------

def check_entry_signal(df, bar_idx, config):
    return None


def create_entry_order(entry_signal, bar_idx, config):
    """
    Create an entry order from an entry_signal.
    placed_at defaults to "close" (decision made at bar close, fills next bar open).
    """
    expiry_bars = config.get("expiry_bars", 1)
    placed_at = config.get("placed_at", "close")
    return {
        "id": _next_order_id(),
        "side": "entry",
        "direction": entry_signal.get("direction", "long"),
        "qty": entry_signal.get("qty", 1),
        "order_type": entry_signal.get("order_type", "market"),
        "price": entry_signal.get("price"),
        "placed_bar": bar_idx,
        "placed_at": placed_at,
        "expires_at_bar": bar_idx + expiry_bars,
        "status": "working",
    }


# --------- Order processing ---------

def process_order(pending_order, position, trades, df, bar, bar_idx):
    """
    Placeholder - override in strategy.

    Should return: (pending_order, position, trades)
    """
    return pending_order, position, trades


def try_fill(order, prices):
    """
    Placeholder - override in strategy.

    Should return: fill_price or None
    """
    return None


def calc_stop_price(position, stop_loss_pct):
    """Calculate stop price from position and stop loss percentage."""
    if position["direction"] == "long":
        return position["entry_price"] * (1 - stop_loss_pct / 100)
    else:
        return position["entry_price"] * (1 + stop_loss_pct / 100)


def calc_pnl(position, exit_price):
    """Calculate PnL percentage (standard percentage return)."""
    if position["direction"] == "long":
        return (exit_price / position["entry_price"] - 1) * 100
    else:
        return (1 - exit_price / position["entry_price"]) * 100
