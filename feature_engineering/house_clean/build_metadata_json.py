"""Generate rich metadata JSONs for consolidated feature and target tables.

Reads features.csv and targets.csv from the feature_store folder, infers origins
and families from column names, and writes features.json and targets.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

import pandas as pd

BASE_DIR = Path("/Volumes/Extreme SSD/trading_data/cex")
TRAINING_DIR = BASE_DIR / "training" / "BINANCE_BTCUSDT.P, 60"
OUTPUT_DIR = TRAINING_DIR / "feature_store"

FEATURES_CSV = OUTPUT_DIR / "features.csv"
TARGETS_CSV = OUTPUT_DIR / "targets.csv"
FEATURES_JSON = OUTPUT_DIR / "features.json"
TARGETS_JSON = OUTPUT_DIR / "targets.json"


def _split_timeframe_suffix(column_name: str) -> tuple[str, str | None]:
    m = re.search(r"_(1H|4H|12H|1D)$", column_name)
    if not m:
        return column_name, None
    tf = m.group(1)
    base = column_name[: -(len(tf) + 1)]
    return base, tf


def _parse_feature_family(base_name: str) -> tuple[str, dict, str]:
    params: dict = {}
    b = base_name

    if "_x_" in b:
        left, right = b.split("_x_", 1)
        # Capture lag annotations appearing in either side of the interaction
        lags = sorted({int(x) for x in re.findall(r"lag_(\d+)", b)})
        params_inter = {"left": left, "right": right}
        if lags:
            params_inter["lags"] = lags
        return ("interaction", params_inter, f"Interaction: {left} x {right}")

    if b.startswith("close_logret_current") or b == "close_logret":
        return ("current_bar", {}, "Log return of close (current bar)")
    if b.startswith("high_low_range_pct_current") or b == "high_low_range_pct":
        return ("current_bar", {}, "(High−Low)/Open (current bar)")
    if b.startswith("close_open_pct_current") or b == "close_open_pct":
        return ("current_bar", {}, "(Close−Open)/Open (current bar)")
    if b == "log_volume":
        return ("current_bar", {}, "log1p(volume)")
    if b.startswith("log_volume_delta_current") or b == "log_volume_delta":
        return ("current_bar", {}, "Δ log1p(volume) (current vs prev)")
    if b.startswith("sign_close_logret_current") or b == "sign_close_logret":
        return ("current_bar", {}, "sign(close log return)")

    lag_matches = re.findall(r"lag_(\d+)", b)
    if lag_matches:
        params["lags"] = [int(x) for x in lag_matches]
        # If the base reflects a lagged current-bar feature (or its interaction), classify as current_bar
        cb_roots = (
            "close_logret_current", "high_low_range_pct_current", "close_open_pct_current",
            "log_volume", "log_volume_delta_current", "sign_close_logret_current",
        )
        normalized = re.sub(r"_lag_\d+", "_current", b)
        if any(normalized.startswith(root) or any(part.startswith(root) for part in normalized.split('_x_')) for root in cb_roots):
            pretty = re.sub(r"_lag_(\d+)", r" lag \1", b).replace("_", " ")
            return ("current_bar", params, pretty)
        # otherwise strip lag tokens and continue parsing
        b = re.sub(r"_lag_\d+", "", b)
        b = b.replace("_current", "")

    m = re.match(r"(?P<series>\w+)_sma_(?P<window>\d+)$", b)
    if m:
        params.update({"series": m["series"], "window": int(m["window"])})
        return ("trend", params, f"SMA({m['window']}) of {m['series']}")
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
        params.update({"series": m["series"], "periods": [int(m["p1"]), int(m["p2"]), int(m["p3"]) ]})
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

    m = re.match(r"(?P<series>\w+)_roll_spread_(?P<window>\d+)$", b)
    if m:
        params.update({"series": m["series"], "window": int(m["window"])})
        return ("liquidity", params, f"Roll spread estimate({m['window']}) of {m['series']}")
    m = re.match(r"(?P<series>\w+)_amihud_(?P<window>\d+)$", b)
    if m:
        params.update({"series": m["series"], "window": int(m["window"])})
        return ("liquidity", params, f"Amihud illiquidity({m['window']}) of {m['series']}")

    m = re.match(r"(?P<series>\w+)_var_(?P<pct>\d+)_(?P<window>\d+)$", b)
    if m:
        params.update({"series": m["series"], "alpha_pct": int(m["pct"]), "window": int(m["window"])})
        return ("risk", params, f"VaR({m['pct']}%) over {m['window']} ({m['series']})")
    m = re.match(r"(?P<series>\w+)_cvar_(?P<pct>\d+)_(?P<window>\d+)$", b)
    if m:
        params.update({"series": m["series"], "alpha_pct": int(m["pct"]), "window": int(m["window"])})
        return ("risk", params, f"CVaR({m['pct']}%) over {m['window']} ({m['series']})")

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
        return ("statistical", {"series": m["series"], "window": int(m["window"])}, f"Z-score of last return over {m['window']}")
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

    return ("other", {}, base_name.replace("_", " "))


def _build_feature_metadata_from_headers(columns: List[str]) -> List[dict]:
    out: List[dict] = []
    for col in columns:
        if col == "timestamp":
            continue
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
        # Determine module based on family or recognizable roots
        if family == "current_bar":
            module = "current_bar_features"
        else:
            cb_roots = (
                "close_logret", "high_low_range_pct", "close_open_pct",
                "log_volume", "log_volume_delta", "sign_close_logret",
            )
            module = "current_bar_features" if any(tok in base for tok in cb_roots) else "multi_timeframe_features"
        desc = short + (f" on {tf}" if tf else "")
        out.append({
            "column_name": col,
            "module": module,
            "family": family,
            "timeframe": tf,
            "params": params,
            "description": desc,
        })
    return out


def _parse_target_metadata(column_name: str) -> dict:
    c = column_name
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
    return {"column_name": c, "module": "targets", "family": "other", "description": c}


def _build_target_metadata_from_headers(columns: List[str]) -> List[dict]:
    return [_parse_target_metadata(c) for c in columns if c != "timestamp"]


def main() -> None:
    if not FEATURES_CSV.exists() or not TARGETS_CSV.exists():
        raise FileNotFoundError("features.csv or targets.csv not found. Run build_feature_store.py first.")

    feat_cols = pd.read_csv(FEATURES_CSV, nrows=0).columns.tolist()
    tgt_cols = pd.read_csv(TARGETS_CSV, nrows=0).columns.tolist()

    features_meta = _build_feature_metadata_from_headers(feat_cols)
    targets_meta = _build_target_metadata_from_headers(tgt_cols)

    FEATURES_JSON.write_text(json.dumps(features_meta, indent=2))
    TARGETS_JSON.write_text(json.dumps(targets_meta, indent=2))
    print(f"Wrote {len(features_meta)} feature metadata entries and {len(targets_meta)} target metadata entries.")


if __name__ == "__main__":
    main()
