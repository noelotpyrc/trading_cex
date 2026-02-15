"""
Strategy Exploration App (TTL-first)

Analyze per-TTL behavior before strategy simulation:
1) edge distribution (empirical prob - market prob)
2) accuracy comparison (empirical prob vs market prob)

Usage:
    streamlit run apps/btc1m_trade_analysis/strategy_exploration_app.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =============================================================================
# CONFIG
# =============================================================================

JOINED_PATH = Path(
    "/Users/noel/projects/trading_cex/apps/btc1m_trade_analysis/btc_15m_features_vol_inner_join.csv"
)
DEFAULT_START = date(2025, 9, 15)
DEFAULT_END = date(2025, 11, 30)

OUR_PROB_COL = "prob_yes_emp"
ALL_PROB_COLS = ["prob_yes_emp", "prob_yes", "pm_p"]
MARKET_COMPARE_COL = "pm_p_next"


# =============================================================================
# DATA
# =============================================================================

@st.cache_data
def load_joined_data() -> pd.DataFrame:
    df = pd.read_csv(JOINED_PATH, parse_dates=["datetime_utc"])

    for col in ["close", "strike_K", "ttl", *ALL_PROB_COLS]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("datetime_utc").reset_index(drop=True)


@st.cache_data
def add_resolved_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve YES/NO outcome using timestamp + ttl lookup:
    settle_dt = datetime_utc + ttl minutes
    outcome_yes = 1[settle_close > strike_K]
    """
    out = df.copy()
    required = {"datetime_utc", "close", "strike_K", "ttl"}
    if not required.issubset(out.columns):
        out["settle_close"] = np.nan
        out["outcome_yes"] = np.nan
        return out

    ttl = pd.to_numeric(out["ttl"], errors="coerce")
    settle_dt = out["datetime_utc"] + pd.to_timedelta(ttl, unit="m")

    close_map = out.set_index("datetime_utc")["close"]
    out["settle_close"] = settle_dt.map(close_map)

    valid = out["strike_K"].notna() & out["settle_close"].notna()
    out["outcome_yes"] = np.nan
    out.loc[valid, "outcome_yes"] = (
        out.loc[valid, "settle_close"] > out.loc[valid, "strike_K"]
    ).astype(float)
    return out


@st.cache_data
def add_next_market_prob(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach next-bar market probability:
    pm_p_next[t] = pm_p[t+1 minute]
    """
    out = df.copy()
    required = {"datetime_utc", "pm_p"}
    if not required.issubset(out.columns):
        out[MARKET_COMPARE_COL] = np.nan
        return out

    market_map = (
        out[["datetime_utc", "pm_p"]]
        .dropna()
        .sort_values("datetime_utc")
        .drop_duplicates(subset=["datetime_utc"], keep="last")
        .set_index("datetime_utc")["pm_p"]
    )
    out[MARKET_COMPARE_COL] = (out["datetime_utc"] + pd.Timedelta(minutes=1)).map(market_map)
    return out


# =============================================================================
# METRICS
# =============================================================================

def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0:
        return np.nan
    return float(np.mean((p - y) ** 2))


def log_loss_binary(y: np.ndarray, p: np.ndarray, eps: float = 1e-6) -> float:
    if len(y) == 0:
        return np.nan
    p_clip = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p_clip) + (1.0 - y) * np.log(1.0 - p_clip)))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    if len(y) == 0:
        return np.nan
    p_clip = np.clip(p, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(y)
    for idx in range(bins):
        lo = edges[idx]
        hi = edges[idx + 1]
        if idx == bins - 1:
            mask = (p_clip >= lo) & (p_clip <= hi)
        else:
            mask = (p_clip >= lo) & (p_clip < hi)
        if not np.any(mask):
            continue
        p_bin = p_clip[mask]
        y_bin = y[mask]
        ece += (len(p_bin) / n) * abs(float(np.mean(y_bin)) - float(np.mean(p_bin)))
    return float(ece)


def build_edge_stats_by_ttl(df: pd.DataFrame, our_col: str, market_col: str) -> pd.DataFrame:
    d = df[["ttl", our_col, market_col]].dropna().copy()
    if d.empty:
        return pd.DataFrame()

    d["ttl"] = d["ttl"].astype(int)
    d["edge"] = d[our_col] - d[market_col]

    out = (
        d.groupby("ttl")
        .agg(
            count=("edge", "size"),
            edge_mean=("edge", "mean"),
            edge_std=("edge", "std"),
            edge_q05=("edge", lambda s: float(s.quantile(0.05))),
            edge_q25=("edge", lambda s: float(s.quantile(0.25))),
            edge_q50=("edge", "median"),
            edge_q75=("edge", lambda s: float(s.quantile(0.75))),
            edge_q95=("edge", lambda s: float(s.quantile(0.95))),
            pos_edge_rate=("edge", lambda s: float((s > 0).mean())),
        )
        .reset_index()
        .sort_values("ttl")
    )
    return out


def probability_detail_tables(prob: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s = pd.to_numeric(prob, errors="coerce").dropna()
    if s.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    q_levels = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    q_vals = s.quantile(q_levels)
    abs_dev = (s - 0.5).abs()

    summary = pd.DataFrame(
        [
            {
                "count": int(len(s)),
                "min": float(s.min()),
                "p01": float(q_vals.loc[0.01]),
                "p05": float(q_vals.loc[0.05]),
                "p10": float(q_vals.loc[0.10]),
                "p25": float(q_vals.loc[0.25]),
                "p50": float(q_vals.loc[0.50]),
                "p75": float(q_vals.loc[0.75]),
                "p90": float(q_vals.loc[0.90]),
                "p95": float(q_vals.loc[0.95]),
                "p99": float(q_vals.loc[0.99]),
                "max": float(s.max()),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=0)),
                "skew": float(s.skew()),
                "excess_kurtosis": float(s.kurt()),
                "absdev_mean": float(abs_dev.mean()),
                "absdev_p50": float(abs_dev.quantile(0.50)),
                "absdev_p90": float(abs_dev.quantile(0.90)),
                "absdev_p95": float(abs_dev.quantile(0.95)),
            }
        ]
    )

    band_rows = []
    band_thresholds = [0.005, 0.010, 0.020, 0.030, 0.050, 0.100, 0.150, 0.200]
    n = len(s)
    for d in band_thresholds:
        within = int((abs_dev <= d).sum())
        outside = int((abs_dev >= d).sum())
        band_rows.append(
            {
                "absdev_threshold": d,
                "within_count": within,
                "within_pct": float(within / n),
                "outside_count": outside,
                "outside_pct": float(outside / n),
            }
        )
    bands = pd.DataFrame(band_rows)

    edges = np.linspace(0.0, 1.0, 11)
    counts, _ = np.histogram(s.to_numpy(dtype=float), bins=edges)
    bin_rows = []
    for i, c in enumerate(counts):
        lo = edges[i]
        hi = edges[i + 1]
        label = f"[{lo:.1f},{hi:.1f})" if i < len(counts) - 1 else f"[{lo:.1f},{hi:.1f}]"
        bin_rows.append(
            {
                "prob_bin": label,
                "count": int(c),
                "pct": float(c / n),
            }
        )
    coarse_bins = pd.DataFrame(bin_rows)
    return summary, bands, coarse_bins


def build_accuracy_by_ttl(df: pd.DataFrame, our_col: str, market_col: str) -> pd.DataFrame:
    d = df[["ttl", "outcome_yes", our_col, market_col]].dropna().copy()
    if d.empty:
        return pd.DataFrame()

    d["ttl"] = d["ttl"].astype(int)
    rows = []
    for ttl_val, g in d.groupby("ttl", sort=True):
        y = g["outcome_yes"].to_numpy(dtype=float)
        p_emp = g[our_col].to_numpy(dtype=float)
        p_market = g[market_col].to_numpy(dtype=float)
        brier_emp = brier_score(y, p_emp)
        brier_market = brier_score(y, p_market)
        ll_emp = log_loss_binary(y, p_emp)
        ll_market = log_loss_binary(y, p_market)
        ece_emp = expected_calibration_error(y, p_emp, bins=10)
        ece_market = expected_calibration_error(y, p_market, bins=10)
        rows.append(
            {
                "ttl": int(ttl_val),
                "count": int(len(g)),
                "yes_rate": float(np.mean(y)),
                "brier_emp": brier_emp,
                "brier_market": brier_market,
                "brier_delta_market_minus_emp": float(brier_market - brier_emp),
                "logloss_emp": ll_emp,
                "logloss_market": ll_market,
                "logloss_delta_market_minus_emp": float(ll_market - ll_emp),
                "ece_emp": ece_emp,
                "ece_market": ece_market,
                "ece_delta_market_minus_emp": float(ece_market - ece_emp),
                "acc_emp_0p5": float(np.mean((p_emp >= 0.5) == (y == 1.0))),
                "acc_market_0p5": float(np.mean((p_market >= 0.5) == (y == 1.0))),
            }
        )

    out = pd.DataFrame(rows).sort_values("ttl").reset_index(drop=True)
    out["winner_brier"] = np.where(
        out["brier_delta_market_minus_emp"] > 0.0,
        "empirical",
        np.where(out["brier_delta_market_minus_emp"] < 0.0, "market", "tie"),
    )
    return out


def run_simple_threshold_strategy(
    df: pd.DataFrame,
    market_prices_df: pd.DataFrame,
    signal_col: str,
    threshold: float,
    mode: str = "symmetric",
) -> pd.DataFrame:
    """
    Decision rule (same as vol feature analysis):
    - YES entry if signal_p >= 0.5 + threshold
    - NO entry  if signal_p <= 0.5 - threshold
    - else skip

    Pricing rule (market-priced, next bar execution):
    - YES entry price = pm_p[t+1]
    - NO entry price  = 1 - pm_p[t+1]
    Realized PnL:
    - YES: y - pm_p[t+1]
    - NO : pm_p[t+1] - y
    """
    required = {"datetime_utc", "ttl", "outcome_yes", signal_col}
    required_market = {"datetime_utc", "pm_p"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    if not required_market.issubset(market_prices_df.columns):
        return pd.DataFrame()

    d = df[["datetime_utc", "ttl", "outcome_yes", signal_col]].dropna().copy()
    market_next = (
        market_prices_df[["datetime_utc", "pm_p"]]
        .dropna()
        .sort_values("datetime_utc")
        .drop_duplicates(subset=["datetime_utc"], keep="last")
        .rename(columns={"datetime_utc": "entry_dt", "pm_p": "entry_market_p"})
    )
    d["entry_dt"] = d["datetime_utc"] + pd.Timedelta(minutes=1)
    d = d.merge(market_next, on="entry_dt", how="left")
    d = d.dropna(subset=["entry_market_p"])
    if d.empty:
        return pd.DataFrame()

    d["ttl"] = d["ttl"].astype(int)
    p_signal = d[signal_col].to_numpy(dtype=float)
    p_market = d["entry_market_p"].to_numpy(dtype=float)
    y = d["outcome_yes"].to_numpy(dtype=float)

    yes_entry = p_signal >= (0.5 + threshold)
    no_entry = p_signal <= (0.5 - threshold)
    if mode == "yes_only":
        no_entry = np.zeros_like(no_entry, dtype=bool)
    elif mode == "no_only":
        yes_entry = np.zeros_like(yes_entry, dtype=bool)

    entered = yes_entry | no_entry
    pnl_yes = y - p_market
    pnl_no = p_market - y
    pnl = np.where(yes_entry, pnl_yes, np.where(no_entry, pnl_no, 0.0))

    d["signal_p"] = p_signal
    d["entry_market_p"] = p_market
    d["yes_entry"] = yes_entry
    d["no_entry"] = no_entry
    d["entered"] = entered
    d["pnl"] = pnl
    d["win"] = np.where(yes_entry, y == 1.0, np.where(no_entry, y == 0.0, np.nan))
    return d


def summarize_simple_strategy_by_ttl(row_df: pd.DataFrame) -> pd.DataFrame:
    if row_df.empty:
        return pd.DataFrame()

    rows = []
    for ttl_val, grp in row_df.groupby("ttl", sort=True):
        entered = grp["entered"].to_numpy(dtype=bool)
        n_total = int(len(grp))
        n_enter = int(entered.sum())
        n_yes = int(grp["yes_entry"].sum())
        n_no = int(grp["no_entry"].sum())

        win_rate = float(grp.loc[entered, "win"].mean()) if n_enter > 0 else np.nan
        abs_conf = np.abs(grp["signal_p"].to_numpy(dtype=float) - 0.5)
        avg_abs_conf = float(abs_conf[entered].mean()) if n_enter > 0 else np.nan
        pnl_values = grp["pnl"].to_numpy(dtype=float)
        pnl_per_entered = float(pnl_values[entered].mean()) if n_enter > 0 else np.nan

        rows.append(
            {
                "ttl": int(ttl_val),
                "count_total": n_total,
                "count_entered": n_enter,
                "count_yes_entries": n_yes,
                "count_no_entries": n_no,
                "participation_rate": float(n_enter / n_total) if n_total else np.nan,
                "win_rate_entered": win_rate,
                "avg_abs_confidence": avg_abs_conf,
                "pnl_total": float(pnl_values.sum()),
                "pnl_per_entered_trade": pnl_per_entered,
                "pnl_per_opportunity": float(pnl_values.mean()) if n_total else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("ttl").reset_index(drop=True)


# =============================================================================
# UI
# =============================================================================

def main() -> None:
    st.set_page_config(page_title="Strategy Exploration", page_icon="🧪", layout="wide")
    st.title("🧪 Strategy Exploration (TTL-first)")
    st.caption(
        "Dataset: inner join of volatility features + 15-minute market prices. "
        "This app analyzes each TTL separately before any strategy simulation."
    )

    df_all = add_next_market_prob(add_resolved_outcome(load_joined_data()))
    required_cols = {"datetime_utc", "ttl", "pm_p", MARKET_COMPARE_COL, OUR_PROB_COL, "outcome_yes"}
    missing = sorted(required_cols - set(df_all.columns))
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    # Sidebar filters
    st.sidebar.header("Filters")
    data_min = df_all["datetime_utc"].dt.date.min()
    data_max = df_all["datetime_utc"].dt.date.max()

    default_start = max(DEFAULT_START, data_min)
    default_end = min(DEFAULT_END, data_max)
    if default_start > default_end:
        default_start = data_min
        default_end = data_max

    date_start = st.sidebar.date_input(
        "Start date", value=default_start, min_value=data_min, max_value=data_max
    )
    date_end = st.sidebar.date_input(
        "End date", value=default_end, min_value=data_min, max_value=data_max
    )

    mask = (
        (df_all["datetime_utc"].dt.date >= date_start)
        & (df_all["datetime_utc"].dt.date <= date_end)
    )
    df = df_all.loc[mask].reset_index(drop=True)

    st.sidebar.caption(f"Our probability column: `{OUR_PROB_COL}`")
    st.sidebar.metric("Rows (filtered)", f"{len(df):,}")
    st.sidebar.metric("Rows w/ outcome", f"{df['outcome_yes'].notna().sum():,}")

    tab_plan, tab_data, tab_ttl, tab_strategy = st.tabs(
        ["🚦 Run First", "📦 Data Health", "📊 TTL Analysis", "⚙️ Simple Strategy"]
    )

    with tab_plan:
        st.subheader("Recommended First Analyses")
        st.markdown(
            """
1. **Data health first**: confirm outcome and probability coverage for each TTL.
2. **Edge distribution by TTL**: inspect `prob_yes_emp - pm_p[t+1]` spread per TTL (not pooled).
3. **Accuracy by TTL**: compare empirical probability vs next-bar market probability against realized outcome.
4. **Simple strategy check**: thresholded entries by TTL using market-priced YES/NO fills.
            """
        )
        st.info(
            "Accuracy is reported with Brier score, log loss, and ECE (lower is better). "
            "Positive delta `market(t+1) - empirical` means empirical is better."
        )

    with tab_data:
        st.subheader("Coverage & Integrity")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Filtered rows", f"{len(df):,}")
        c2.metric("Outcome rows", f"{df['outcome_yes'].notna().sum():,}")
        c3.metric("Empirical prob rows", f"{df[OUR_PROB_COL].notna().sum():,}")
        c4.metric("Market prob rows (t+1)", f"{df[MARKET_COMPARE_COL].notna().sum():,}")

        ttl_cov = (
            df.groupby("ttl")
            .agg(
                rows=("ttl", "size"),
                outcome_rows=("outcome_yes", lambda s: s.notna().sum()),
                empirical_rows=(OUR_PROB_COL, lambda s: s.notna().sum()),
                market_rows_t1=(MARKET_COMPARE_COL, lambda s: s.notna().sum()),
            )
            .reset_index()
            .sort_values("ttl")
        )
        st.dataframe(
            ttl_cov.style.format(
                {
                    "rows": "{:,.0f}",
                    "outcome_rows": "{:,.0f}",
                    "empirical_rows": "{:,.0f}",
                    "market_rows_t1": "{:,.0f}",
                }
            ),
            use_container_width=True,
        )

        by_day = (
            df.assign(day=df["datetime_utc"].dt.date)
            .groupby("day")
            .agg(rows=("day", "size"))
            .reset_index()
        )
        fig_day = px.bar(by_day, x="day", y="rows", title="Rows per Day (Filtered)")
        fig_day.update_layout(template="plotly_dark", height=320, margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(fig_day, use_container_width=True)

    with tab_ttl:
        st.subheader("Per-TTL Edge Distribution")
        edge_stats = build_edge_stats_by_ttl(df, OUR_PROB_COL, MARKET_COMPARE_COL)
        if edge_stats.empty:
            st.warning("No valid rows for edge analysis.")
        else:
            st.dataframe(
                edge_stats.style.format(
                    {
                        "count": "{:,.0f}",
                        "edge_mean": "{:.6f}",
                        "edge_std": "{:.6f}",
                        "edge_q05": "{:.6f}",
                        "edge_q25": "{:.6f}",
                        "edge_q50": "{:.6f}",
                        "edge_q75": "{:.6f}",
                        "edge_q95": "{:.6f}",
                        "pos_edge_rate": "{:.2%}",
                    }
                ),
                use_container_width=True,
            )

            edge_plot_df = df[["ttl", OUR_PROB_COL, MARKET_COMPARE_COL]].dropna().copy()
            edge_plot_df["ttl"] = edge_plot_df["ttl"].astype(int)
            edge_plot_df["edge"] = edge_plot_df[OUR_PROB_COL] - edge_plot_df[MARKET_COMPARE_COL]

            fig_edge = px.violin(
                edge_plot_df,
                x="ttl",
                y="edge",
                box=True,
                points=False,
                title="Edge Distribution by TTL (empirical - market[t+1])",
            )
            fig_edge.add_hline(y=0.0, line_dash="dot")
            fig_edge.update_layout(
                template="plotly_dark",
                height=380,
                margin=dict(l=50, r=20, t=50, b=40),
                xaxis_title="TTL",
                yaxis_title="empirical - market[t+1]",
            )
            st.plotly_chart(fig_edge, use_container_width=True)

            ttl_detail = st.selectbox(
                "Inspect one TTL distribution",
                options=sorted(edge_plot_df["ttl"].unique().tolist()),
                index=0,
            )
            bins = st.slider("Histogram bins", min_value=20, max_value=120, value=60, step=5)
            g = edge_plot_df[edge_plot_df["ttl"] == ttl_detail]
            fig_hist = px.histogram(
                g,
                x="edge",
                nbins=bins,
                title=f"Edge Histogram at TTL={ttl_detail}",
            )
            fig_hist.add_vline(x=0.0, line_dash="dot")
            fig_hist.update_layout(
                template="plotly_dark",
                height=320,
                margin=dict(l=50, r=20, t=50, b=40),
                xaxis_title="empirical - market[t+1]",
                yaxis_title="count",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            st.subheader(f"TTL={ttl_detail} Probability Detail ({OUR_PROB_COL})")
            prob_summary, prob_bands, prob_coarse_bins = probability_detail_tables(
                edge_plot_df.loc[edge_plot_df["ttl"] == ttl_detail, OUR_PROB_COL]
            )
            if prob_summary.empty:
                st.warning("No valid probabilities for selected TTL.")
            else:
                st.dataframe(
                    prob_summary.style.format(
                        {
                            "count": "{:,.0f}",
                            "min": "{:.6f}",
                            "p01": "{:.6f}",
                            "p05": "{:.6f}",
                            "p10": "{:.6f}",
                            "p25": "{:.6f}",
                            "p50": "{:.6f}",
                            "p75": "{:.6f}",
                            "p90": "{:.6f}",
                            "p95": "{:.6f}",
                            "p99": "{:.6f}",
                            "max": "{:.6f}",
                            "mean": "{:.6f}",
                            "std": "{:.6f}",
                            "skew": "{:.6f}",
                            "excess_kurtosis": "{:.6f}",
                            "absdev_mean": "{:.6f}",
                            "absdev_p50": "{:.6f}",
                            "absdev_p90": "{:.6f}",
                            "absdev_p95": "{:.6f}",
                        }
                    ),
                    use_container_width=True,
                )

                c1, c2 = st.columns(2)
                with c1:
                    st.caption("Concentration around 0.5")
                    st.dataframe(
                        prob_bands.style.format(
                            {
                                "absdev_threshold": "{:.3f}",
                                "within_count": "{:,.0f}",
                                "within_pct": "{:.2%}",
                                "outside_count": "{:,.0f}",
                                "outside_pct": "{:.2%}",
                            }
                        ),
                        use_container_width=True,
                    )
                with c2:
                    st.caption("Coarse probability bins (width=0.1)")
                    st.dataframe(
                        prob_coarse_bins.style.format(
                            {
                                "count": "{:,.0f}",
                                "pct": "{:.2%}",
                            }
                        ),
                        use_container_width=True,
                    )

        st.subheader("Per-TTL Accuracy: Empirical vs Market (Next Bar)")
        st.caption("Market probability in this section uses `pm_p[t+1]`.")
        acc_tbl = build_accuracy_by_ttl(df, OUR_PROB_COL, MARKET_COMPARE_COL)
        if acc_tbl.empty:
            st.warning("No valid rows for accuracy comparison.")
        else:
            n_ttl = len(acc_tbl)
            emp_better = int((acc_tbl["winner_brier"] == "empirical").sum())
            market_better = int((acc_tbl["winner_brier"] == "market").sum())
            ties = int((acc_tbl["winner_brier"] == "tie").sum())
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("TTL buckets compared", f"{n_ttl:,}")
            c2.metric("Empirical better (Brier)", f"{emp_better:,}")
            c3.metric("Market better (Brier)", f"{market_better:,}")
            c4.metric("Ties", f"{ties:,}")

            st.dataframe(
                acc_tbl.style.format(
                    {
                        "count": "{:,.0f}",
                        "yes_rate": "{:.4f}",
                        "brier_emp": "{:.6f}",
                        "brier_market": "{:.6f}",
                        "brier_delta_market_minus_emp": "{:.6f}",
                        "logloss_emp": "{:.6f}",
                        "logloss_market": "{:.6f}",
                        "logloss_delta_market_minus_emp": "{:.6f}",
                        "ece_emp": "{:.6f}",
                        "ece_market": "{:.6f}",
                        "ece_delta_market_minus_emp": "{:.6f}",
                        "acc_emp_0p5": "{:.2%}",
                        "acc_market_0p5": "{:.2%}",
                    }
                ),
                use_container_width=True,
            )

            fig_brier_delta = px.bar(
                acc_tbl,
                x="ttl",
                y="brier_delta_market_minus_emp",
                color="winner_brier",
                title="Brier Delta by TTL (market[t+1] - empirical)",
            )
            fig_brier_delta.add_hline(y=0.0, line_dash="dot")
            fig_brier_delta.update_layout(
                template="plotly_dark",
                height=340,
                margin=dict(l=50, r=20, t=50, b=40),
                xaxis_title="TTL",
                yaxis_title="Brier delta (market[t+1] - empirical)",
            )
            st.plotly_chart(fig_brier_delta, use_container_width=True)

    with tab_strategy:
        st.subheader("Simple Threshold Strategy (No Edge Filter)")
        st.caption(
            "Decision uses model probability only (0.5 ± δ). "
            "Execution uses next bar market price: YES at `pm_p[t+1]`, NO at `1-pm_p[t+1]`."
        )
        st.caption(f"Signal column fixed to `{OUR_PROB_COL}`.")

        c1, c2 = st.columns(2)
        with c1:
            decision_mode_label = st.selectbox(
                "Decision mode",
                options=["Symmetric (YES and NO)", "YES only", "NO only"],
                index=0,
                key="simple_decision_mode",
            )
        with c2:
            threshold = st.slider(
                "Entry threshold δ around 0.5",
                min_value=0.00,
                max_value=0.20,
                value=0.02,
                step=0.0025,
                key="simple_threshold",
            )

        mode_map = {
            "Symmetric (YES and NO)": "symmetric",
            "YES only": "yes_only",
            "NO only": "no_only",
        }
        sim_rows = run_simple_threshold_strategy(
            df=df,
            market_prices_df=df_all,
            signal_col=OUR_PROB_COL,
            threshold=float(threshold),
            mode=mode_map[decision_mode_label],
        )

        if sim_rows.empty:
            st.warning("No valid rows for strategy simulation.")
        else:
            sim_ttl = summarize_simple_strategy_by_ttl(sim_rows)
            total_opps = int(sim_ttl["count_total"].sum())
            total_entered = int(sim_ttl["count_entered"].sum())
            total_pnl = float(sim_ttl["pnl_total"].sum())
            ev_per_entered = (total_pnl / total_entered) if total_entered > 0 else np.nan
            ev_per_opp = (total_pnl / total_opps) if total_opps > 0 else np.nan
            participation = (total_entered / total_opps) if total_opps > 0 else np.nan

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Entered trades", f"{total_entered:,}")
            m2.metric("Participation", f"{participation:.2%}" if np.isfinite(participation) else "nan")
            m3.metric("Total Sim EV", f"{total_pnl:.4f}")
            m4.metric("Sim EV / Entered", f"{ev_per_entered:.6f}" if np.isfinite(ev_per_entered) else "nan")
            m5.metric("Sim EV / Opportunity", f"{ev_per_opp:.6f}" if np.isfinite(ev_per_opp) else "nan")

            st.dataframe(
                sim_ttl.rename(
                    columns={
                        "ttl": "TTL",
                        "count_total": "Total",
                        "count_entered": "Entered",
                        "count_yes_entries": "YES Entries",
                        "count_no_entries": "NO Entries",
                        "participation_rate": "Participation",
                        "win_rate_entered": "Win Rate (Entered)",
                        "avg_abs_confidence": "Avg |p-0.5| (Entered)",
                        "pnl_total": "Sim EV Total",
                        "pnl_per_entered_trade": "Sim EV / Entered",
                        "pnl_per_opportunity": "Sim EV / Opportunity",
                    }
                ).style.format(
                    {
                        "Participation": "{:.2%}",
                        "Win Rate (Entered)": "{:.2%}",
                        "Avg |p-0.5| (Entered)": "{:.6f}",
                        "Sim EV Total": "{:.4f}",
                        "Sim EV / Entered": "{:.6f}",
                        "Sim EV / Opportunity": "{:.6f}",
                    }
                ),
                use_container_width=True,
            )

            fig_ev = px.line(
                sim_ttl,
                x="ttl",
                y="pnl_per_entered_trade",
                markers=True,
                title=f"Sim EV / Entered by TTL ({OUR_PROB_COL}, δ={threshold:.4f})",
            )
            fig_ev.add_hline(y=0.0, line_dash="dot")
            fig_ev.update_layout(
                template="plotly_dark",
                height=340,
                margin=dict(l=50, r=20, t=50, b=40),
                xaxis_title="TTL",
                yaxis_title="Sim EV / Entered",
            )
            st.plotly_chart(fig_ev, use_container_width=True)


if __name__ == "__main__":
    main()
