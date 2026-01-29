import pandas as pd

# Order ID generator
def _make_order_id():
    i = 0
    while True:
        yield i
        i += 1

_order_id = _make_order_id()


def run_backtest(df, entry_config, exit_config):
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
    Still one position, one pending order.
    """
    expiry_bars = config.get("expiry_bars", 10)
    return {
        "id": next(_order_id),
        "side": "exit",
        "direction": position["direction"],
        "qty": position.get("qty", 1),
        "order_type": exit_signal.get("order_type", "market"),
        "price": exit_signal.get("price"),  # may be None for market
        "created_bar": bar_idx,
        "expires_at_bar": bar_idx + expiry_bars,
        "status": "working",
        "reason": exit_signal.get("reason"),
    }


def create_stop_exit_order(position, bar_idx, config):
    """Fallback protective stop-loss exit."""
    expiry_bars = config.get("expiry_bars", 10)
    stop_price = calc_stop_price(position, config["stop_loss_pct"])
    return {
        "id": next(_order_id),
        "side": "exit",
        "direction": position["direction"],
        "qty": position.get("qty", 1),
        "order_type": "stop",
        "price": stop_price,
        "created_bar": bar_idx,
        "expires_at_bar": bar_idx + expiry_bars,
        "status": "working",
        "reason": "stop_loss",
    }


def update_pending_order(pending_order, position, df, bar, bar_idx, exit_config):
    """
    Optional “single-order” replacement behavior:
    If you keep a protective stop working, but a discretionary exit triggers,
    you can replace that stop (since only 1 pending order is allowed).

    Keep as placeholder for now.
    """
    return pending_order


# --------- Entry logic placeholder ---------

def check_entry_signal(df, bar_idx, config):
    return None


def create_entry_order(entry_signal, bar_idx, config):
    """Create an entry order from an entry_signal."""
    expiry_bars = config.get("expiry_bars", 1)
    return {
        "id": next(_order_id),
        "side": "entry",
        "direction": entry_signal.get("direction", "long"),
        "qty": entry_signal.get("qty", 1),
        "order_type": entry_signal.get("order_type", "market"),
        "price": entry_signal.get("price"),
        "created_bar": bar_idx,
        "expires_at_bar": bar_idx + expiry_bars,
        "status": "working",
    }


# --------- Order processing ---------

def process_order(pending_order, position, trades, df, bar, bar_idx):
    """Process pending order: check fill or expiry."""
    if pending_order is None:
        return None, position, trades

    # Check expiry
    if bar_idx >= pending_order["expires_at_bar"]:
        return None, position, trades

    # Try fill
    filled, fill_price = try_fill(pending_order, bar)
    if not filled:
        return pending_order, position, trades

    # Handle fill
    if pending_order["side"] == "entry":
        position = {
            "entry_price": fill_price,
            "entry_bar": bar_idx,
            "entry_time": bar["datetime_utc"],
            "direction": pending_order["direction"],
            "qty": pending_order.get("qty", 1),
        }
    else:  # exit
        pnl = calc_pnl(position, fill_price)
        trades.append({
            "entry_time": position["entry_time"],
            "entry_price": position["entry_price"],
            "exit_time": bar["datetime_utc"],
            "exit_price": fill_price,
            "exit_bar": bar_idx,
            "direction": position["direction"],
            "reason": pending_order.get("reason"),
            "pnl_pct": pnl,
        })
        position = None

    return None, position, trades


def try_fill(order, bar):
    """Check if order fills this bar. Returns (filled, fill_price)."""
    order_type = order["order_type"]
    direction = order["direction"]
    side = order["side"]

    if order_type == "market":
        return True, bar["open"]

    price = order["price"]
    if price is None:
        return False, None

    if order_type == "limit":
        # Buy limit fills if low <= price
        if (side == "entry" and direction == "long") or (side == "exit" and direction == "short"):
            if bar["low"] <= price:
                return True, price
        # Sell limit fills if high >= price
        if (side == "entry" and direction == "short") or (side == "exit" and direction == "long"):
            if bar["high"] >= price:
                return True, price

    elif order_type == "stop":
        # Stop for long exit (sell stop) fills if low <= price
        if side == "exit" and direction == "long":
            if bar["low"] <= price:
                return True, price
        # Stop for short exit (buy stop) fills if high >= price
        if side == "exit" and direction == "short":
            if bar["high"] >= price:
                return True, price

    return False, None


def calc_stop_price(position, stop_loss_pct):
    """Calculate stop price from position and stop loss percentage."""
    if position["direction"] == "long":
        return position["entry_price"] * (1 - stop_loss_pct / 100)
    else:
        return position["entry_price"] * (1 + stop_loss_pct / 100)


def calc_pnl(position, exit_price):
    """Calculate PnL percentage."""
    if position["direction"] == "long":
        return (exit_price / position["entry_price"] - 1) * 100
    else:
        return (position["entry_price"] / exit_price - 1) * 100
