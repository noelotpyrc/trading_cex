"""Consolidate canonical feature and target tables for BINANCE_BTCUSDT.P,60.

This script reads the source-of-truth feature and target CSV files, aligns
them on their common timestamp index, ensures unique column names, and writes
the consolidated CSV outputs to the feature_store folder.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import re


BASE_DIR = Path("/Volumes/Extreme SSD/trading_data/cex")
LOOKBACK_DIR = BASE_DIR / "lookbacks" / "BINANCE_BTCUSDT.P, 60"
TARGET_DIR = BASE_DIR / "targets" / "BINANCE_BTCUSDT.P, 60"
TRAINING_DIR = BASE_DIR / "training" / "BINANCE_BTCUSDT.P, 60"
OUTPUT_DIR = TRAINING_DIR / "feature_store"

# Consolidated filenames
FEATURES_OUTPUT_CSV = OUTPUT_DIR / "features.csv"
TARGETS_OUTPUT_CSV = OUTPUT_DIR / "targets.csv"
FEATURES_META_JSON = OUTPUT_DIR / "features.json"
TARGETS_META_JSON = OUTPUT_DIR / "targets.json"


@dataclass
class Dataset:
    key: str
    path: Path
    kind: str  # "feature" or "target"
    required: bool = True


def _normalize_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the first column is called timestamp and parsed to UTC."""

    if df.empty:
        return df

    ts_col = df.columns[0]
    if ts_col != "timestamp":
        df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _disambiguate_columns(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Make sure columns are unique within the dataframe before global renaming."""

    new_cols: List[str] = []
    seen = {}
    for col in df.columns:
        if col == "timestamp":
            new_cols.append(col)
            continue
        count = seen.get(col, 0)
        if count:
            new_name = f"{col}__dup{count + 1}"
        else:
            new_name = col
        seen[col] = count + 1
        new_cols.append(new_name)
    df.columns = new_cols
    return df


def _rename_with_registry(
    df: pd.DataFrame,
    key: str,
    registry: dict[str, dict[str, str]],
) -> tuple[pd.DataFrame, List[dict[str, str]]]:
    """Rename columns to be unique across all datasets and record metadata."""

    rename_map = {}
    metadata_entries: List[dict[str, str]] = []

    for col in df.columns:
        if col == "timestamp":
            continue
        # Special handling: prefix HMM regime outputs for clarity
        if key == "regimes":
            base_name = f"hmm_regime_{col}"
        else:
            base_name = col
        candidate = base_name
        suffix = 1
        while candidate in registry:
            candidate = f"{base_name}__{key}" if suffix == 1 else f"{base_name}__{key}_{suffix}"
            suffix += 1
        rename_map[col] = candidate
        registry[candidate] = {"source": key, "original_column": col}
        metadata_entries.append(
            {
                "column_name": candidate,
                "source_file": key,
                "original_column": col,
            }
        )

    df = df.rename(columns=rename_map)
    return df, metadata_entries


def _load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV with encoding fallback and timestamp normalization."""

    for enc in ("utf-8", "latin1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError(
            "utf-8", b"", 0, 1, f"Failed to decode {path} with utf-8 or latin1"
        )

    df = _normalize_timestamp_column(df)
    df = _disambiguate_columns(df, key=path.stem)
    return df


def _intersection(indices: Iterable[pd.Index]) -> pd.Index:
    """Compute the sorted intersection across multiple indices."""

    indices = list(indices)
    if not indices:
        raise ValueError("No indices provided for intersection")

    common = reduce(lambda a, b: a.intersection(b), indices)
    return common.sort_values()


def _split_timeframe_suffix(column_name: str) -> tuple[str, str | None]:
    """Split trailing timeframe suffix like _1H/_4H/_12H/_1D if present.

    Returns (base_name, timeframe_suffix or None)
    """
    m = re.search(r"_(1H|4H|12H|1D)$", column_name)
    if not m:
        return column_name, None
    tf = m.group(1)
    base = column_name[: -(len(tf) + 1)]
    return base, tf


def _parse_feature_family(base_name: str) -> tuple[str, dict, str]:
    """Classify feature family, extract params, and build a concise description (without timeframe).

    Returns (family, params, short_desc)
    """
    params: dict = {}
    b = base_name

    # Interactions
    if "_x_" in b:
        left, right = b.split("_x_", 1)
        return (
            "interaction",
            {"left": left, "right": right},
            f"Interaction: {left} x {right}",
        )

    # Current bar core features
    if b.startswith("close_logret_current"):
        return ("current_bar", {}, "Log return of close (current bar)")
    if b.startswith("high_low_range_pct_current"):
        return ("current_bar", {}, "(High−Low)/Open (current bar)")
    if b.startswith("close_open_pct_current"):
        return ("current_bar", {}, "(Close−Open)/Open (current bar)")
    if b == "log_volume":
        return ("current_bar", {}, "log1p(volume)")
    if b.startswith("log_volume_delta_current"):
        return ("current_bar", {}, "Δ log1p(volume) (current vs prev)")
    if b.startswith("sign_close_logret_current"):
        return ("current_bar", {}, "sign(close log return)")

    # Lag annotations
    lag_matches = re.findall(r"lag_(\d+)", b)
    if lag_matches:
        params["lags"] = [int(x) for x in lag_matches]
        # strip lag tokens for family identification (best-effort)
        b = re.sub(r"_lag_\d+", "", b)
        b = b.replace("_current", "")

    # Trend / averages
    m = re.match(r"(?P<series>\w+)_sma_(?P<window>\d+)$", b)
    if m:
        params.update({"series": m["series"], "window": int(m["window"])})
        return (
            "trend",
            params,
            f"SMA({m['window']}) of {m['series']}",
        )
    m = re.match(r"(?P<series>\w+)_ema_(?P<span>\d+)$", b)
    if m:
        params.update({"series": m["series"], "span": int(m["span"])})
        return ("trend", params, f"EMA({m['span']}) of {m['series']}")
    m = re.match(r"(?P<series>\w+)_wma_(?P<window>\d+)$", b)
    if m:
        params.update({"series": m["series"], "window": int(m["window"])})
        return ("trend", params, f"WMA({m['window']}) of {m['series']}")
    m = re.match(r"(?P<series>\w+)_ma_cross_(?P<kind>diff|ratio|signal)_(?P<fast>\d+)_(?P<slow>\d+)$", b)
    if m:
        params.update({"series": m["series"], "fast": int(m["fast"]), "slow": int(m["slow"]), "kind": m["kind"]})
        return ("trend", params, f"MA crossover {m['kind']} {m['fast']}/{m['slow']} on {m['series']}")

    # Momentum / oscillators
    m = re.match(r"(?P<series>\w+)_rsi_(?P<window>\d+)$", b)
    if m:
        params.update({"series": m["series"], "window": int(m["window"])})
        return ("momentum", params, f"RSI({m['window']}) of {m['series']}")
    m = re.match(r"(?P<series>\w+)_stoch_k_(?P<k>\d+)$", b)
    if m:
        params.update({"series": m["series"], "k": int(m["k"])})
        return ("momentum", params, f"Stochastic %K({m['k']}) of {m['series']}")
    m = re.match(r"(?P<series>\w+)_stoch_d_(?P<k>\d+)_(?P<d>\d+)$", b)
    if m:
        params.update({"series": m["series"], "k": int(m["k"]), "d": int(m["d"])})
        return ("momentum", params, f"Stochastic %D({m['k']},{m['d']}) of {m['series']}")
    m = re.match(r"(?P<series>\w+)_cci_(?P<window>\d+)$", b)
    if m:
        params.update({"series": m["series"], "window": int(m["window"])})
        return ("momentum", params, f"CCI({m['window']}) of {m['series']}")
    m = re.match(r"(?P<series>\w+)_roc_(?P<period>\d+)$", b)
    if m:
        params.update({"series": m["series"], "period": int(m["period"])})
        return ("momentum", params, f"Rate of change({m['period']}) of {m['series']}")
    m = re.match(r"(?P<series>\w+)_williams_r_(?P<window>\d+)$", b)
    if m:
        params.update({"series": m["series"], "window": int(m["window"])})
        return ("momentum", params, f"Williams %R({m['window']}) of {m['series']}")
    m = re.match(r"(?P<series>\w+)_uo_(?P<p1>\d+)_(?P<p2>\d+)_(?P<p3>\d+)$", b)
    if m:
        params.update({"series": m["series"], "periods": [int(m["p1"]), int(m["p2"]), int(m["p3"])]})
        return ("momentum", params, f"Ultimate Oscillator of {m['series']}")
    m = re.match(r"(?P<series>\w+)_macd_line_(?P<fast>\d+)_(?P<slow>\d+)(?:_over_(?P<norm>close|atr\d+))?$", b)
    if m:
        params.update({"series": m["series"], "fast": int(m["fast"]), "slow": int(m["slow"]), "normalized_by": m.group("norm")})
        return ("momentum", params, f"MACD line {m['fast']}/{m['slow']} of {m['series']}")
    m = re.match(r"(?P<series>\w+)_macd_signal_(?P<signal>\d+)$", b)
    if m:
        params.update({"series": m["series"], "signal": int(m["signal"])})
        return ("momentum", params, f"MACD signal {m['signal']} of {m['series']}")
    m = re.match(r"(?P<series>\w+)_macd_histogram_(?P<fast>\d+)_(?P<slow>\d+)_(?P<signal>\d+)(?:_over_(?P<norm>close|atr\d+))?$", b)
    if m:
        params.update({"series": m["series"], "fast": int(m["fast"]), "slow": int(m["slow"]), "signal": int(m["signal"]), "normalized_by": m.group("norm")})
        return ("momentum", params, f"MACD histogram {m['fast']}/{m['slow']}/{m['signal']} of {m['series']}")

    # Volatility
    m = re.match(r"(?P<series>\w+)_atr_(?P<window>\d+)$", b)
    if m:
        params.update({"series": m["series"], "window": int(m["window"])})
        return ("volatility", params, f"ATR({m['window']})")
    m = re.match(r"(?P<series>\w+)_hv_(?P<window>\d+)$", b)
    if m:
        params.update({"series": m["series"], "window": int(m["window"])})
        return ("volatility", params, f"Historical volatility({m['window']}) of {m['series']}")
    m = re.match(r"(?P<series>\w+)_bb_(?P<which>upper|middle|lower|width|percent)_(?P<window>\d+)(?:_(?P<std>\d+|\d+_\d+))?$", b)
    if m:
        params.update({"series": m["series"], "which": m["which"], "window": int(m["window"]), "std": m.group("std")})
        return ("volatility", params, f"Bollinger {m['which']}({m['window']}) of {m['series']}")
    m = re.match(r"(?P<series>\w+)_vol_ratio_(?P<short>\d+)_(?P<long>\d+)$", b)
    if m:
        params.update({"series": m["series"], "short": int(m["short"]), "long": int(m["long"])})
        return ("volatility", params, f"Volatility ratio {m['short']}/{m['long']} of {m['series']}")
    m = re.match(r"(?P<series>\w+)_parkinson_(?P<window>\d+)$", b)
    if m:
        params.update({"series": m["series"], "window": int(m["window"])})
        return ("volatility", params, f"Parkinson volatility({m['window']})")
    m = re.match(r"(?P<series>\w+)_gk_(?P<window>\d+)$", b)
    if m:
        params.update({"series": m["series"], "window": int(m["window"])})
        return ("volatility", params, f"Garman–Klass volatility({m['window']})")
    m = re.match(r"(?P<series>\w+)_bb_width_pct_(?P<window>\d+)_(?P<std>\d+|\d+_\d+)$", b)
    if m:
        params.update({"series": m["series"], "window": int(m["window"]), "std": m["std"]})
        return ("volatility", params, f"Bollinger width %({m['window']}) of {m['series']}")

    # Volume
    if b == "close_obv":
        return ("volume", {}, "On-Balance Volume")
    if b == "close_vwap":
        return ("volume", {}, "VWAP")
    if b == "close_adl":
        return ("volume", {}, "Accumulation/Distribution Line")
    m = re.match(r"(?P<series>\w+)_chaikin_(?P<fast>\d+)_(?P<slow>\d+)$", b)
    if m:
        params.update({"fast": int(m["fast"]), "slow": int(m["slow"])})
        return ("volume", params, f"Chaikin Oscillator {m['fast']}/{m['slow']}")
    m = re.match(r"volume_roc_(?P<period>\d+)$", b)
    if m:
        params.update({"period": int(m["period"])})
        return ("volume", params, f"Volume ROC({m['period']})")
    m = re.match(r"volume_rvol_(?P<window>\d+)$", b)
    if m:
        params.update({"window": int(m["window"])})
        return ("volume", params, f"Relative volume({m['window']})")
    m = re.match(r"close_obv_over_dollar_vol_(?P<window>\d+)$", b)
    if m:
        params.update({"window": int(m["window"])})
        return ("volume", params, f"OBV over dollar volume({m['window']})")
    m = re.match(r"close_adl_over_dollar_vol_(?P<window>\d+)$", b)
    if m:
        params.update({"window": int(m["window"])})
        return ("volume", params, f"ADL over dollar volume({m['window']})")

    # Statistical / distributional
    m = re.match(r"(?P<series>\w+)_percentile_(?P<p>\d+)_(?P<window>\d+)$", b)
    if m:
        params.update({"series": m["series"], "percentile": int(m["p"]), "window": int(m["window"])})
        return ("statistical", params, f"Percentile {m['p']}% over {m['window']} of {m['series']}")
    m = re.match(r"(?P<series>\w+)_skew_(?P<window>\d+)$", b)
    if m:
        params.update({"series": m["series"], "window": int(m["window"])})
        return ("statistical", params, f"Skewness({m['window']}) of {m['series']}")
    m = re.match(r"(?P<series>\w+)_kurt_(?P<window>\d+)$", b)
    if m:
        params.update({"series": m["series"], "window": int(m["window"])})
        return ("statistical", params, f"Kurtosis({m['window']}) of {m['series']}")
    m = re.match(r"(?P<series>\w+)_autocorr_(?P<lag>\d+)_(?P<window>\d+)$", b)
    if m:
        params.update({"series": m["series"], "lag": int(m["lag"]), "window": int(m["window"])})
        return ("statistical", params, f"Autocorr lag {m['lag']} over {m['window']} ({m['series']})")
    m = re.match(r"(?P<series>\w+)_hurst_(?P<window>\d+)$", b)
    if m:
        params.update({"series": m["series"], "window": int(m["window"])})
        return ("statistical", params, f"Hurst exponent({m['window']}) of {m['series']}")
    m = re.match(r"(?P<series>\w+)_entropy_(?P<window>\d+)$", b)
    if m:
        params.update({"series": m["series"], "window": int(m["window"])})
        return ("statistical", params, f"Approximate entropy({m['window']}) of {m['series']}")
    m = re.match(r"(?P<series>\w+)_perm_entropy_(?P<m>\d+)_(?P<window>\d+)$", b)
    if m:
        params.update({"series": m["series"], "m": int(m["m"]), "window": int(m["window"])})
        return ("statistical", params, f"Permutation entropy(m={m['m']}, w={m['window']}) of {m['series']}")

    # Liquidity / microstructure
    m = re.match(r"(?P<series>\w+)_roll_spread_(?P<window>\d+)$", b)
    if m:
        params.update({"series": m["series"], "window": int(m["window"])})
        return ("liquidity", params, f"Roll spread estimate({m['window']}) of {m['series']}")
    m = re.match(r"(?P<series>\w+)_amihud_(?P<window>\d+)$", b)
    if m:
        params.update({"series": m["series"], "window": int(m["window"])})
        return ("liquidity", params, f"Amihud illiquidity({m['window']}) of {m['series']}")

    # Risk
    m = re.match(r"(?P<series>\w+)_var_(?P<pct>\d+)_(?P<window>\d+)$", b)
    if m:
        params.update({"series": m["series"], "alpha_pct": int(m["pct"]), "window": int(m["window"])})
        return ("risk", params, f"VaR({m['pct']}%) over {m['window']} ({m['series']})")
    m = re.match(r"(?P<series>\w+)_cvar_(?P<pct>\d+)_(?P<window>\d+)$", b)
    if m:
        params.update({"series": m["series"], "alpha_pct": int(m["pct"]), "window": int(m["window"])})
        return ("risk", params, f"CVaR({m['pct']}%) over {m['window']} ({m['series']})")

    # Cycle / time / pattern / derived
    m = re.match(r"(?P<series>\w+)_dominant_cycle_length_(?P<window>\d+)$", b)
    if m:
        params.update({"series": m["series"], "window": int(m["window"])})
        return ("cycle", params, f"Dominant cycle length({m['window']}) of {m['series']}")
    m = re.match(r"(?P<series>\w+)_cycle_strength_(?P<window>\d+)$", b)
    if m:
        params.update({"series": m["series"], "window": int(m["window"])})
        return ("cycle", params, f"Cycle strength({m['window']}) of {m['series']}")
    if b.startswith("time_"):
        return ("time", {}, b.replace("_", " "))
    m = re.match(r"(?P<series>\w+)_candle_(?P<which>body_ratio|upper_shadow_ratio|lower_shadow_ratio)$", b)
    if m:
        return ("pattern", {"series": m["series"], "which": m["which"]}, f"Candle {m['which'].replace('_',' ')} of {m['series']}")
    m = re.match(r"(?P<series>\w+)_typical_price$", b)
    if m:
        return ("price_derived", {"series": m["series"]}, f"Typical price of {m['series']}")
    m = re.match(r"(?P<series>\w+)_ohlc_average$", b)
    if m:
        return ("price_derived", {"series": m["series"]}, f"OHLC average of {m['series']}")
    m = re.match(r"(?P<series>\w+)_ret_zscore_(?P<window>\d+)$", b)
    if m:
        return ("statistical", {"series": m["series"], "window": int(m["window"])} , f"Z-score of last return over {m['window']}")
    m = re.match(r"(?P<series>\w+)_over_ema_(?P<span>\d+)$", b)
    if m:
        return ("normalized", {"series": m["series"], "span": int(m["span"])}, f"{m['series']} / EMA({m['span']})")
    m = re.match(r"(?P<series>\w+)_log_ratio_ema_(?P<span>\d+)$", b)
    if m:
        return ("normalized", {"series": m["series"], "span": int(m["span"])}, f"log({m['series']} / EMA({m['span']}))")
    m = re.match(r"(?P<series>\w+)_dist_ema(?P<span>\d+)_atr$", b)
    if m:
        return ("normalized", {"series": m["series"], "span": int(m["span"])}, f"(price−EMA({m['span']})) / ATR")
    m = re.match(r"(?P<left>\w+)_volume_ratio$", b)
    if m:
        return ("normalized", {"series": m["left"]}, f"{m['left']} / volume")
    m = re.match(r"(?P<series>\w+)_donchian_(?P<which>pos|upper_dist|lower_dist)_(?P<window>\d+)$", b)
    if m:
        return ("trend", {"series": m["series"], "which": m["which"], "window": int(m["window"])}, f"Donchian {m['which']}({m['window']}) of {m['series']}")
    m = re.match(r"(?P<series>\w+)_aroon_(?P<which>up|down|osc)_(?P<window>\d+)$", b)
    if m:
        return ("trend", {"series": m["series"], "which": m["which"], "window": int(m["window"])}, f"Aroon {m['which']}({m['window']})")
    m = re.match(r"(?P<series>\w+)_adx_(?P<window>\d+)$", b)
    if m:
        return ("trend", {"series": m["series"], "window": int(m["window"])}, f"ADX({m['window']})")
    m = re.match(r"(?P<series>\w+)_di_(?P<which>plus|minus)_(?P<window>\d+)$", b)
    if m:
        return ("trend", {"series": m["series"], "which": m["which"], "window": int(m["window"])}, f"DI {m['which']}(" + str(int(m["window"])) + ")")

    # Fallback
    return ("other", {}, base_name.replace("_", " "))


def _build_feature_metadata(columns: List[str], registry_entries: List[dict]) -> List[dict]:
    """Create rich metadata for feature columns."""
    registry_map = {e["column_name"]: {"source_file": e.get("source_file"), "original_column": e.get("original_column")} for e in registry_entries}
    out: List[dict] = []
    for col in columns:
        if col == "timestamp":
            continue
        # HMM regime special-case
        if col.startswith("hmm_regime_"):
            base, tf = _split_timeframe_suffix(col)
            name = base[len("hmm_regime_"):]
            if name == "state":
                desc = "HMM predicted regime (integer-coded)"
            elif name.startswith("p_state_"):
                state_id = name.split("p_state_")[-1]
                desc = f"HMM probability of regime {state_id}"
            else:
                desc = f"HMM regime feature {name}"
            out.append({
                "column_name": col,
                "module": "hmm_regime",
                "family": "regime",
                "timeframe": tf,
                "params": {},
                "source_file": "regimes",
                "original_column": name,
                "description": desc,
            })
            continue

        base, tf = _split_timeframe_suffix(col)
        family, params, short = _parse_feature_family(base)
        reg = registry_map.get(col, {})

        # Origin module inference
        source_file = reg.get("source_file")
        if source_file == "regimes":
            module = "hmm_regime"
        elif source_file and source_file.startswith("current_bar_with_lags"):
            module = "current_bar_features"
        else:
            module = "multi_timeframe_features"

        desc = short + (f" on {tf}" if tf else "")
        out.append({
            "column_name": col,
            "module": module,
            "family": family,
            "timeframe": tf,
            "params": params,
            "source_file": source_file,
            "original_column": reg.get("original_column"),
            "description": desc,
        })
    return out


def _parse_target_metadata(column_name: str) -> dict:
    """Parse target column name into structured metadata and description."""
    c = column_name
    # y_logret_24h or y_ret_24h
    m = re.match(r"^y_(?P<kind>logret|ret)_(?P<horizon>.+)$", c)
    if m:
        kind = m["kind"]
        horizon = m["horizon"]
        return {
            "column_name": c,
            "module": "targets",
            "family": "forward_return",
            "horizon": horizon,
            "description": ("Log return" if kind == "logret" else "Simple return") + f" over horizon {horizon}",
        }
    # y_mfe_24h / y_mae_24h
    m = re.match(r"^y_(?P<kind>mfe|mae)_(?P<horizon>.+)$", c)
    if m:
        kind = m["kind"]
        horizon = m["horizon"]
        return {
            "column_name": c,
            "module": "targets",
            "family": "excursion",
            "horizon": horizon,
            "description": ("Max favorable" if kind == "mfe" else "Max adverse") + f" excursion over {horizon}",
        }
    # Triple-barrier label and TP-before-SL
    m = re.match(r"^y_tb_label_u(?P<up>[^_]+)_d(?P<down>[^_]+)_(?P<horizon>.+)$", c)
    if m:
        return {
            "column_name": c,
            "module": "targets",
            "family": "triple_barrier_ternary",
            "horizon": m["horizon"],
            "params": {"up": m["up"], "down": m["down"]},
            "description": f"Ternary triple-barrier label (+1/-1/0) up={m['up']} down={m['down']} over {m['horizon']}",
        }
    m = re.match(r"^y_tp_before_sl_u(?P<up>[^_]+)_d(?P<down>[^_]+)_(?P<horizon>.+)$", c)
    if m:
        return {
            "column_name": c,
            "module": "targets",
            "family": "tp_before_sl_binary",
            "horizon": m["horizon"],
            "params": {"up": m["up"], "down": m["down"]},
            "description": f"Binary: TP before SL (1/0) up={m['up']} down={m['down']} over {m['horizon']}",
        }
    # Fallback
    return {"column_name": c, "module": "targets", "family": "other", "description": c}


def _build_target_metadata(columns: List[str]) -> List[dict]:
    return [_parse_target_metadata(c) for c in columns if c != "timestamp"]


def _load_datasets(
    datasets: Iterable[Dataset],
    base_index: pd.Index | None = None,
) -> tuple[list[pd.DataFrame], pd.Index, list[dict[str, str]], list[str]]:
    """Load datasets, align on shared/base index, and collect metadata."""

    registry: dict[str, dict[str, str]] = {}
    frames: List[pd.DataFrame] = []
    timestamp_indices: List[pd.Index] = []
    metadata: List[dict[str, str]] = []
    missing_reports: List[str] = []
    datasets = list(datasets)

    for ds in datasets:
        df = _load_csv(ds.path)
        if df.empty:
            raise ValueError(f"Dataset {ds.path} is empty")
        index = pd.Index(df["timestamp"])
        timestamp_indices.append(index)

        df, ds_metadata = _rename_with_registry(df, ds.key, registry)
        metadata.extend(ds_metadata)

        df = df.sort_values("timestamp").set_index("timestamp")
        frames.append(df)

    if base_index is None:
        required_indices = [idx for idx, ds in zip(timestamp_indices, datasets) if ds.required]
        indices_for_base = required_indices if required_indices else timestamp_indices
        base_index = _intersection(indices_for_base)
    else:
        base_index = base_index.sort_values()

    aligned_frames: List[pd.DataFrame] = []
    for df, ds, idx in zip(frames, datasets, timestamp_indices):
        missing_ts = base_index.difference(idx)
        if missing_ts.size:
            msg = f"Dataset {ds.key} missing {missing_ts.size} timestamps relative to base index"
            if ds.required:
                raise ValueError(msg)
            missing_reports.append(msg)
        aligned = df.reindex(base_index)
        aligned_frames.append(aligned)

    return aligned_frames, base_index, metadata, missing_reports


def _consolidate_feature_store() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    feature_datasets = [
        Dataset(key=path.stem, path=path, kind="feature", required=True)
        for path in sorted(LOOKBACK_DIR.glob("*.csv"))
        if not path.name.startswith("._")
    ]

    feature_datasets.extend(
        [
            Dataset(
                key="regimes",
                path=TRAINING_DIR / "regimes.csv",
                kind="feature",
                required=False,
            ),
        ]
    )

    target_datasets = [
        Dataset(key=path.stem, path=path, kind="target", required=True)
        for path in sorted(TARGET_DIR.glob("*.csv"))
        if not path.name.startswith("._")
    ]

    feature_frames, base_index, feature_metadata, feature_missing = _load_datasets(
        feature_datasets
    )

    target_frames, _, target_metadata, target_missing = _load_datasets(
        target_datasets, base_index=base_index
    )

    final_index = base_index

    features_df = pd.concat([df.reindex(final_index) for df in feature_frames], axis=1)
    targets_df = pd.concat([df.reindex(final_index) for df in target_frames], axis=1)

    features_df = features_df.reset_index().rename(columns={"index": "timestamp"})
    targets_df = targets_df.reset_index().rename(columns={"index": "timestamp"})

    features_df.to_csv(FEATURES_OUTPUT_CSV, index=False)
    targets_df.to_csv(TARGETS_OUTPUT_CSV, index=False)

    # Metadata JSONs are generated by a separate script.

    print("Feature rows:", len(features_df))
    print("Target rows:", len(targets_df))
    print("Feature columns:", len(features_df.columns) - 1)
    print("Target columns:", len(targets_df.columns) - 1)
    if feature_missing:
        print("Feature alignment warnings:")
        for msg in feature_missing:
            print(" -", msg)
    if target_missing:
        print("Target alignment warnings:")
        for msg in target_missing:
            print(" -", msg)


if __name__ == "__main__":
    _consolidate_feature_store()

