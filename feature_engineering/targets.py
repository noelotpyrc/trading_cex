from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd


TiePolicy = Literal["conservative", "proximity_to_open"]
BarrierMode = Literal["ternary", "binary", "both"]


def _format_horizon_label(horizon_bars: int, horizon_label: Optional[str]) -> str:
    if horizon_label and isinstance(horizon_label, str):
        return horizon_label
    return f"{horizon_bars}b"


def compute_forward_return(
    entry_price: float,
    forward_close: pd.Series,
    horizon_bars: int,
    *,
    log: bool = True,
    horizon_label: Optional[str] = None,
    column_name: str = "close",
) -> Dict[str, float]:
    """
    Compute fixed-horizon forward return using only the entry price at t and the
    forward window closes (t+1..t+H].

    If there are insufficient forward bars, returns NaN values.
    """
    label = _format_horizon_label(horizon_bars, horizon_label)
    keys = {
        True: f"y_logret_{label}",
        False: f"y_ret_{label}",
    }

    if forward_close is None or len(forward_close) < horizon_bars:
        return {keys[log]: np.nan}

    exit_price = float(forward_close.iloc[horizon_bars - 1])
    if not np.isfinite(entry_price) or not np.isfinite(exit_price) or entry_price <= 0.0 or exit_price <= 0.0:
        return {keys[log]: np.nan}

    if log:
        value = float(np.log(exit_price) - np.log(entry_price))
    else:
        value = float(exit_price / entry_price - 1.0)
    return {keys[log]: value}


def compute_mfe_mae(
    forward_high: pd.Series,
    forward_low: pd.Series,
    entry_price: float,
    horizon_bars: int,
    *,
    horizon_label: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute Max Favorable Excursion (MFE) and Max Adverse Excursion (MAE) for a long entry
    at entry_price over the forward window (t+1..t+H]. Returns simple returns relative to entry.

    If insufficient bars, returns NaNs.
    """
    label = _format_horizon_label(horizon_bars, horizon_label)
    mfe_key = f"y_mfe_{label}"
    mae_key = f"y_mae_{label}"

    if (
        forward_high is None
        or forward_low is None
        or len(forward_high) < horizon_bars
        or len(forward_low) < horizon_bars
        or not np.isfinite(entry_price)
        or entry_price <= 0.0
    ):
        return {mfe_key: np.nan, mae_key: np.nan}

    window_high = forward_high.iloc[:horizon_bars]
    window_low = forward_low.iloc[:horizon_bars]

    max_high = float(np.nanmax(window_high.values))
    min_low = float(np.nanmin(window_low.values))

    if not np.isfinite(max_high) or not np.isfinite(min_low):
        return {mfe_key: np.nan, mae_key: np.nan}

    mfe = max_high / entry_price - 1.0
    mae = min_low / entry_price - 1.0
    return {mfe_key: float(mfe), mae_key: float(mae)}


def compute_barrier_outcomes(
    forward_high: pd.Series,
    forward_low: pd.Series,
    entry_price: float,
    up_pct: float,
    down_pct: float,
    horizon_bars: int,
    *,
    horizon_label: Optional[str] = None,
    mode: BarrierMode = "ternary",
    tie_policy: TiePolicy = "conservative",
    forward_open: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """
    Compute barrier outcomes within the forward window (t+1..t+H].

    - Ternary triple-barrier: +1 upper first, -1 lower first, 0 if neither by H.
      Key: y_tb_label_u{up}_d{down}_{H}
    - Binary TP-before-SL: 1 if upper first else 0 (timeouts and SL-first -> 0).
      Key: y_tp_before_sl_u{up}_d{down}_{H}

    If insufficient bars or invalid inputs, returns NaNs for requested keys.
    """
    label = _format_horizon_label(horizon_bars, horizon_label)
    up_str = f"{float(up_pct):g}"  # compact formatting (e.g., 1.5)
    dn_str = f"{float(down_pct):g}"

    ternary_key = f"y_tb_label_u{up_str}_d{dn_str}_{label}"
    binary_key = f"y_tp_before_sl_u{up_str}_d{dn_str}_{label}"

    def _nan_result() -> Dict[str, float]:
        if mode == "ternary":
            return {ternary_key: np.nan}
        if mode == "binary":
            return {binary_key: np.nan}
        return {ternary_key: np.nan, binary_key: np.nan}

    # Validate inputs
    if (
        forward_high is None
        or forward_low is None
        or len(forward_high) < horizon_bars
        or len(forward_low) < horizon_bars
        or not np.isfinite(entry_price)
        or entry_price <= 0.0
        or up_pct <= 0.0
        or down_pct <= 0.0
    ):
        return _nan_result()

    upper_threshold = entry_price * (1.0 + up_pct)
    lower_threshold = entry_price * (1.0 - down_pct)

    # Iterate forward bars to find the first hit
    first_hit: Optional[int] = None
    first_label: Optional[int] = None  # +1 upper, -1 lower, 0 tie/none

    for i in range(min(horizon_bars, len(forward_high))):
        bar_high = float(forward_high.iloc[i])
        bar_low = float(forward_low.iloc[i])

        hit_upper = np.isfinite(bar_high) and bar_high >= upper_threshold
        hit_lower = np.isfinite(bar_low) and bar_low <= lower_threshold

        if not hit_upper and not hit_lower:
            continue

        first_hit = i
        if hit_upper and not hit_lower:
            first_label = 1
        elif hit_lower and not hit_upper:
            first_label = -1
        else:
            # Both hit within the same bar → tie handling
            if tie_policy == "proximity_to_open" and forward_open is not None and len(forward_open) > i:
                bar_open = float(forward_open.iloc[i])
                if not np.isfinite(bar_open):
                    first_label = 0
                else:
                    dist_upper = abs(upper_threshold - bar_open)
                    dist_lower = abs(bar_open - lower_threshold)
                    first_label = 1 if dist_upper < dist_lower else -1
            else:
                # Conservative: ambiguous intrabar → label 0
                first_label = 0
        break

    # Compose result(s)
    if mode == "ternary":
        return {ternary_key: 0.0 if first_label is None else float(first_label)}
    if mode == "binary":
        # 1 if TP first else 0 (including timeouts and SL-first)
        binary_value = 1.0 if first_label == 1 else 0.0
        return {binary_key: binary_value}

    # both
    ternary_value = 0.0 if first_label is None else float(first_label)
    binary_value = 1.0 if first_label == 1 else 0.0
    return {ternary_key: ternary_value, binary_key: binary_value}


@dataclass
class TargetGenerationConfig:
    """
    Configuration for generating multiple targets for a single row.
    - horizons_bars: list of horizons in bars
    - barrier_pairs: list of (tp_pct, sl_pct) percent thresholds (e.g., 0.015 for 1.5%)
    - tie_policy: tie handling policy for intrabar dual hits
    - horizon_labels: optional mapping from bars to human labels (e.g., 24 -> '24h')
    - include_returns, include_mfe_mae, include_barriers: toggles
    """

    horizons_bars: List[int]
    barrier_pairs: Optional[List[Tuple[float, float]]] = None
    tie_policy: TiePolicy = "conservative"
    horizon_labels: Optional[Dict[int, str]] = None
    include_returns: bool = True
    include_mfe_mae: bool = True
    include_barriers: bool = True
    log_returns: bool = True


def generate_targets_for_row(
    forward_window: pd.DataFrame,
    entry_price: float,
    config: TargetGenerationConfig,
) -> Dict[str, float]:
    """
    Generate selected targets for one row given the forward window DataFrame containing
    columns: 'high', 'low', 'close' (and optionally 'open' for tie policy).

    The forward window must represent (t+1..t+Hmax]. Only OHLCV-forward data is used.
    Missing or insufficient data for a given horizon leads to NaN for those keys.
    """
    required_cols = {"high", "low", "close"}
    if not required_cols.issubset(set(forward_window.columns)):
        raise ValueError(f"forward_window must contain columns {sorted(required_cols)}")

    results: Dict[str, float] = {}
    forward_high = forward_window["high"]
    forward_low = forward_window["low"]
    forward_close = forward_window["close"]
    forward_open = forward_window["open"] if "open" in forward_window.columns else None

    for horizon_bars in config.horizons_bars:
        horizon_label = (
            config.horizon_labels.get(horizon_bars) if config.horizon_labels else None
        )

        if config.include_returns:
            results.update(
                compute_forward_return(
                    entry_price=entry_price,
                    forward_close=forward_close,
                    horizon_bars=horizon_bars,
                    log=config.log_returns,
                    horizon_label=horizon_label,
                )
            )

        if config.include_mfe_mae:
            results.update(
                compute_mfe_mae(
                    forward_high=forward_high,
                    forward_low=forward_low,
                    entry_price=entry_price,
                    horizon_bars=horizon_bars,
                    horizon_label=horizon_label,
                )
            )

        if config.include_barriers and config.barrier_pairs:
            for (tp_pct, sl_pct) in config.barrier_pairs:
                results.update(
                    compute_barrier_outcomes(
                        forward_high=forward_high,
                        forward_low=forward_low,
                        entry_price=entry_price,
                        up_pct=tp_pct,
                        down_pct=sl_pct,
                        horizon_bars=horizon_bars,
                        horizon_label=horizon_label,
                        mode="both",
                        tie_policy=config.tie_policy,
                        forward_open=forward_open,
                    )
                )

    return results


def extract_forward_window(
    data: pd.DataFrame,
    current_idx: int,
    horizon_bars: int,
) -> pd.DataFrame:
    """
    Convenience utility to slice a forward window (t+1..t+H] from an OHLCV DataFrame
    indexed by integer position.
    """
    start = current_idx + 1
    end = current_idx + 1 + horizon_bars
    if start >= len(data):
        return data.iloc[0:0]
    return data.iloc[start:end]



