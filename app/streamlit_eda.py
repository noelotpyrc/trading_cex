from __future__ import annotations

import io
import os
from datetime import datetime, timedelta, date
from pathlib import Path
import sys
from typing import List, Optional
import re
import random

import pandas as pd
import streamlit as st

# Ensure local imports work when run via `streamlit run app/streamlit_eda.py`
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
for p in (THIS_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eda.loader import (
    EDAConfig,
    available_time_range,
    load_config,
    fs_min_max,
    fs_load_ohlcv,
    fs_load_features,
    fs_load_targets,
    join_on_timestamp,
    fs_detect_binary_targets,
)
from eda.plots import corr_heatmap, histograms, missingness, scatter_xy, time_series
from eda.recipes import direction_label, forward_return, triple_barrier


st.set_page_config(page_title="EDA: Features + OHLCV", layout="wide")
st.title("Streamlit EDA: Features + OHLCV")


@st.cache_data(show_spinner=False)
def _cached_load_config(path: str) -> EDAConfig:
    return load_config(Path(path))


# (DuckDB loaders not used in CSV-only app)


def _coerce_date_to_pd(d: date | datetime | str | None) -> Optional[pd.Timestamp]:
    if d is None:
        return None
    return pd.Timestamp(d)


def _load_profile_report():
    try:
        from ydata_profiling import ProfileReport  # type: ignore
    except Exception as e:  # pragma: no cover
        st.error("ydata-profiling not installed. Install with: pip install ydata-profiling")
        return None
    return ProfileReport


def _infer_feature_columns(df: pd.DataFrame) -> List[str]:
    base_cols = {"timestamp", "open", "high", "low", "close", "volume"}
    return [c for c in df.columns if c not in base_cols]


def app_body() -> None:
    st.sidebar.header("Configuration")
    default_cfg_path = str(Path("configs/eda_config.yaml").resolve())
    cfg_path = st.sidebar.text_input("Config path", value=default_cfg_path)

    try:
        cfg = _cached_load_config(cfg_path)
        st.sidebar.success(f"Loaded config: {cfg.dataset_id}")
    except Exception as e:
        st.sidebar.warning(f"Config load failed, using defaults. Details: {e}")
        cfg = EDAConfig(
            dataset_id="dataset",
            ohlcv_duckdb_path=Path("/path/to/ohlcv.duckdb"),
            ohlcv_table="ohlcv_btcusdt_1h",
            features_duckdb_path=Path("/path/to/features.duckdb"),
            features_table="features",
            default_feature_key=None,
        )

    # Paths & selectors
    st.sidebar.subheader("Data Sources")
    fs_folder = st.sidebar.text_input("Feature store folder", value=str(cfg.feature_store_folder or ""))
    fs_ohlcv_csv = st.sidebar.text_input("OHLCV CSV filename", value=cfg.fs_ohlcv_csv)
    fs_features_csv = st.sidebar.text_input("Features CSV filename", value=cfg.fs_features_csv)
    fs_targets_csv = st.sidebar.text_input("Targets CSV filename", value=cfg.fs_targets_csv)

    st.sidebar.subheader("Date Range")
    trailing_days = st.sidebar.number_input("Default trailing days", min_value=1, max_value=3650, value=int(cfg.date_range_days))

    @st.cache_data(show_spinner=False)
    def _cached_fs_range(folder: str, ohlcv_csv: str):
        return fs_min_max(folder, ohlcv_csv)

    t_min, t_max = _cached_fs_range(fs_folder, fs_ohlcv_csv)
    if t_min is None or t_max is None:
        st.error("Failed to query OHLCV time range from feature_store CSV. Check folder and filename.")
        st.stop()

    default_start = (t_max - pd.Timedelta(days=int(trailing_days))).to_pydatetime().date()
    start_date, end_date = st.sidebar.date_input(
        "Window",
        value=(default_start, t_max.to_pydatetime().date()),
        min_value=t_min.to_pydatetime().date(),
        max_value=t_max.to_pydatetime().date(),
    )

    start_ts = _coerce_date_to_pd(start_date)
    end_ts = _coerce_date_to_pd(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    # General options
    drop_na = st.sidebar.checkbox("Drop NaNs in joined view", value=True)

    # Load data for the window
    ohlcv = fs_load_ohlcv(fs_folder, fs_ohlcv_csv, start_ts, end_ts)
    features_csv_df = fs_load_features(fs_folder, fs_features_csv, start_ts, end_ts)
    targets_df = fs_load_targets(fs_folder, fs_targets_csv, start_ts, end_ts)
    if ohlcv.empty:
        st.error("OHLCV CSV returned no rows for the selected window.")
        st.stop()

    # Keep only OHLCV columns for OHLCV view
    ohlcv_cols = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in ohlcv.columns]
    ohlcv_view = ohlcv[ohlcv_cols].copy()

    # Features selection (group by timeframe) with autocomplete/filter support
    # Exclude any feature columns containing the token "_all_tf_normalized"
    _exclude_token = "_all_tf_normalized"
    feature_cols_all = [
        c for c in features_csv_df.columns if c != "timestamp" and _exclude_token not in str(c)
    ]
    st.sidebar.subheader("Features")
    tf_options = ["1H", "4H", "12H", "1D"]
    tf_choice = st.sidebar.radio("Timeframe", tf_options, index=0, horizontal=True)
    # Reset selection when timeframe changes
    if st.session_state.get("_tf_choice") != tf_choice:
        st.session_state["_tf_choice"] = tf_choice
        if "selected_feature_cols" in st.session_state:
            del st.session_state["selected_feature_cols"]

    # Pick only columns that end with _<TF>, e.g., *_1H
    tf_suffix = f"_{tf_choice}"
    suffix_re = re.compile(r"_(1H|4H|12H|1D)$")
    tf_feature_pool = [c for c in feature_cols_all if c.endswith(tf_suffix)]
    # Fallback for 1H: include features without explicit timeframe suffix
    if not tf_feature_pool and tf_choice == "1H":
        tf_feature_pool = [c for c in feature_cols_all if not suffix_re.search(c)]

    feat_search = st.sidebar.text_input(
        "Search features (type to filter)", value="", placeholder="substring or regex (toggle)", help="Type to filter the list; also supports regex when enabled"
    )
    feat_search_mode = st.sidebar.radio("Search mode", ["contains", "regex"], index=0, horizontal=True)
    feat_case_sensitive = st.sidebar.checkbox("Case sensitive", value=False)

    filtered_feature_opts = tf_feature_pool
    if feat_search.strip():
        q = feat_search.strip()
        if feat_search_mode == "regex":
            try:
                flags = 0 if feat_case_sensitive else re.IGNORECASE
                pat = re.compile(q, flags)
                filtered_feature_opts = [c for c in tf_feature_pool if pat.search(c)]
            except re.error:
                st.sidebar.warning("Invalid regex; showing all features")
        else:
            # contains: match all tokens (space-separated)
            tokens = [t for t in q.split() if t]
            if tokens:
                def norm(s: str) -> str:
                    return s if feat_case_sensitive else s.lower()
                toks = [norm(t) for t in tokens]
                filtered_feature_opts = [c for c in tf_feature_pool if all(t in norm(c) for t in toks)]

    # Default to 5 random features from the selected timeframe (once per timeframe)
    default_feats: List[str] = []
    if "selected_feature_cols" not in st.session_state:
        k = min(5, len(filtered_feature_opts))
        if k > 0:
            default_feats = random.sample(filtered_feature_opts, k=k)

    selected_features = st.sidebar.multiselect(
        "Feature columns", options=filtered_feature_opts, default=default_feats, key="selected_feature_cols"
    )

    # Targets selection (no join; use targets.csv)
    st.sidebar.subheader("Targets")
    target_cols_all = [c for c in targets_df.columns if c != "timestamp"]
    if not target_cols_all:
        st.info("No columns found in targets.csv")
    preferred = [c for c in target_cols_all if c.lower().startswith(("y", "target"))]
    default_targets = preferred[: min(3, len(preferred))] if preferred else target_cols_all[:1]
    selected_target_cols: List[str] = st.sidebar.multiselect(
        "Target columns (from targets.csv)", options=target_cols_all, default=default_targets
    )

    # Placeholder for derived target on the fly
    with st.sidebar.expander("Add derived target (coming soon)"):
        st.selectbox("Recipe", ["Forward return", "Direction label", "Triple barrier"], disabled=True)
        st.number_input("Horizon (hours)", min_value=1, max_value=1000, value=24, disabled=True)
        st.number_input("Threshold (for direction)", min_value=0.0, max_value=1.0, value=0.0, step=0.001, disabled=True)
        st.number_input("Take profit (frac)", min_value=0.0, max_value=5.0, value=0.02, step=0.001, disabled=True)
        st.number_input("Stop loss (frac)", min_value=0.0, max_value=5.0, value=0.02, step=0.001, disabled=True)
        st.button("Compute derived target", disabled=True)

    # ================= OHLCV Section =================
    st.subheader("OHLCV")
    n_rows, n_cols = ohlcv_view.shape
    t0, t1 = available_time_range(ohlcv_view)
    st.caption(f"Rows: {n_rows:,} | Cols: {n_cols:,} | Window: {t0} → {t1}")
    st.dataframe(ohlcv_view.head(10))
    st.plotly_chart(time_series(ohlcv_view, ["close"], title="Close"), use_container_width=True)
    st.dataframe(missingness(ohlcv_view, [c for c in ohlcv_view.columns if c != "timestamp"]).head(50))

    # ================= Features Section =================
    st.subheader("Features")
    feat_view = features_csv_df[["timestamp"] + [c for c in selected_features if c in features_csv_df.columns]].copy()
    if not feat_view.empty and selected_features:
        st.caption(f"Rows: {feat_view.shape[0]:,} | Selected features: {len(selected_features)}")
        st.dataframe(feat_view.head(10))
        st.plotly_chart(time_series(feat_view, selected_features, title="Selected Features"), use_container_width=True)
        st.plotly_chart(histograms(feat_view, selected_features, title="Feature Distributions"), use_container_width=True)
        if len(selected_features) >= 2:
            st.plotly_chart(corr_heatmap(feat_view, selected_features, title="Feature Correlation"), use_container_width=True)
        st.dataframe(missingness(feat_view, selected_features).head(50))
    else:
        st.info("Select one or more feature columns to explore.")

    # ================= Targets Section =================
    st.subheader("Targets")
    if selected_target_cols:
        targ_view = targets_df[["timestamp"] + selected_target_cols].copy()
        st.caption(f"Rows: {targ_view.shape[0]:,} | Selected targets: {len(selected_target_cols)}")
        st.dataframe(targ_view.head(10))
        st.plotly_chart(time_series(targ_view, selected_target_cols, title="Targets"), use_container_width=True)
        st.plotly_chart(histograms(targ_view, selected_target_cols, title="Target Distributions"), use_container_width=True)
        if len(selected_target_cols) >= 2:
            st.plotly_chart(corr_heatmap(targ_view, selected_target_cols, title="Target Correlation"), use_container_width=True)
        st.dataframe(missingness(targ_view, selected_target_cols).head(50))
    else:
        st.info("Select target columns from targets.csv.")

    # ================= Feature ↔ Target Correlation (Discovery) =================
    st.subheader("Feature ↔ Target Correlation (Discovery)")
    if not selected_target_cols:
        st.info("Pick at least one target in the sidebar to compute correlations.")
    else:
        col_cfg1, col_cfg2, col_cfg3 = st.columns([1,1,1])
        with col_cfg1:
            corr_target = st.selectbox("Target for correlation", options=selected_target_cols)
        with col_cfg2:
            corr_method = st.radio("Method", ["pearson", "spearman"], index=0, horizontal=True)
        with col_cfg3:
            corr_abs_thr = st.slider("Abs correlation threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.01)

        # Feature set: all features vs current timeframe
        feature_set_mode = st.radio("Feature set", ["All features", f"{tf_choice} features"], index=0, horizontal=True)
        if feature_set_mode == "All features":
            corr_feature_cols = [
                c for c in features_csv_df.columns if c != "timestamp" and _exclude_token not in str(c)
            ]
        else:
            corr_feature_cols = tf_feature_pool

        # Build joined frame: target + candidate features
        join_parts = [targets_df[["timestamp", corr_target]]]
        if corr_feature_cols:
            join_parts.append(features_csv_df[["timestamp"] + corr_feature_cols])
        df_corr = join_on_timestamp(*join_parts)

        if df_corr.empty:
            st.warning("No overlapping rows between features and target in the selected window.")
        else:
            # Compute pairwise correlations target↔each feature with pairwise NA handling
            import numpy as np
            res = []
            s_t = pd.to_numeric(df_corr[corr_target], errors="coerce")
            for c in corr_feature_cols:
                s_f = pd.to_numeric(df_corr[c], errors="coerce")
                mask = s_t.notna() & s_f.notna()
                n = int(mask.sum())
                if n < 10:
                    continue
                try:
                    r = s_t[mask].corr(s_f[mask], method=corr_method)
                except Exception:
                    r = np.nan
                if pd.notna(r):
                    res.append({"feature": c, "corr": float(r), "n": n})

            if not res:
                st.info("No valid correlations could be computed (insufficient overlap or all-NaN pairs).")
            else:
                corr_df = pd.DataFrame(res).assign(abs_corr=lambda d: d["corr"].abs()).sort_values("abs_corr", ascending=False)
                st.dataframe(corr_df.head(50))

                # Filter by threshold
                passed = corr_df[corr_df["abs_corr"] >= corr_abs_thr]
                if passed.empty:
                    st.info(f"No features reached |corr| ≥ {corr_abs_thr:.2f} for target '{corr_target}'.")
                else:
                    top_cols = passed["feature"].tolist()
                    st.caption(f"{len(top_cols)} features passed threshold. Showing correlation heatmap with target.")
                    cols_for_heat = [corr_target] + top_cols
                    st.plotly_chart(corr_heatmap(df_corr, cols_for_heat, method=corr_method, title="Filtered Corr Heatmap (features + target)"), use_container_width=True)

    # ================= Binary Target Insights =================
    st.subheader("Binary Target Insights")
    # Detect binary targets by scanning the entire targets.csv (not just the current window)
    # A column is binary if its non-NaN values across the full file are a subset of {0,1}
    try:
        all_binary_targets = set(fs_detect_binary_targets(fs_folder, fs_targets_csv))
    except Exception as e:
        all_binary_targets = set()
        st.warning(f"Binary detection failed: {e}")
    # Only keep those also selected in the sidebar; if none, fall back to all detected
    binary_targets = [c for c in selected_target_cols if c in all_binary_targets]
    if not binary_targets and all_binary_targets:
        st.caption(f"No selected targets are binary; showing all detected binary targets ({len(all_binary_targets)}).")
        binary_targets = sorted(all_binary_targets)

    if not binary_targets:
        st.info("No binary targets (0/1) detected.")
    else:
        col_bt1, col_bt2, col_bt3 = st.columns([1,1,1])
        st.caption(f"Detected binary targets: {len(binary_targets)}")
        with col_bt1:
            tbin = st.selectbox("Binary target", options=binary_targets)
        with col_bt2:
            auc_thr = st.slider("Min separability (AUC or 1-AUC)", min_value=0.5, max_value=1.0, value=0.55, step=0.01)
        with col_bt3:
            fset = st.radio("Feature set", ["All features", f"{tf_choice} features"], index=1, horizontal=True)

        # Build joined frame with target and chosen feature set
        if fset == "All features":
            feat_cols_for_bin = [c for c in features_csv_df.columns if c != "timestamp" and _exclude_token not in str(c)]
        else:
            feat_cols_for_bin = tf_feature_pool

        join_parts = [targets_df[["timestamp", tbin]]]
        if feat_cols_for_bin:
            join_parts.append(features_csv_df[["timestamp"] + feat_cols_for_bin])
        df_bt = join_on_timestamp(*join_parts)

        if df_bt.empty:
            st.warning("No overlapping rows between features and the binary target in the selected window.")
        else:
            import numpy as np
            import plotly.express as px

            y_raw = pd.to_numeric(df_bt[tbin], errors="coerce")
            y = y_raw.where(y_raw.isin([0, 1]), np.nan)

            results = []
            for c in feat_cols_for_bin:
                s = pd.to_numeric(df_bt[c], errors="coerce")
                mask = y.notna() & s.notna()
                n = int(mask.sum())
                if n < 20:
                    continue
                yy = y[mask].astype(int)
                xx = s[mask]
                n_pos = int((yy == 1).sum())
                n_neg = int((yy == 0).sum())
                if n_pos < 5 or n_neg < 5:
                    continue
                # Point-biserial (Pearson with binary labels)
                try:
                    r_pb = float(yy.corr(xx, method="pearson"))
                except Exception:
                    r_pb = np.nan
                # Spearman rank
                try:
                    rho_s = float(yy.corr(xx, method="spearman"))
                except Exception:
                    rho_s = np.nan
                # AUC via rank-based formula (ties handled by average ranks)
                ranks = xx.rank(method="average")
                sum_ranks_pos = float(ranks[yy == 1].sum())
                auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
                auc_eff = float(max(auc, 1.0 - auc))  # separability regardless of direction
                results.append({
                    "feature": c,
                    "auc": float(auc),
                    "auc_eff": auc_eff,
                    "r_pb": r_pb,
                    "rho_s": rho_s,
                    "n": n,
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                })

            if not results:
                st.info("No features with sufficient data to evaluate against the binary target.")
            else:
                tbl = pd.DataFrame(results).sort_values(["auc_eff", "n"], ascending=[False, False])
                st.dataframe(tbl.head(100))

                # Threshold filter by AUC separability
                filt = tbl[tbl["auc_eff"] >= auc_thr]
                if filt.empty:
                    st.info(f"No features reached separability ≥ {auc_thr:.2f} (AUC or 1-AUC).")
                else:
                    topN = st.slider("Top N features for plots", min_value=3, max_value=50, value=min(10, len(filt)))
                    top_feats = filt.head(topN)["feature"].tolist()
                    # Bar chart of AUC separability
                    st.plotly_chart(
                        px.bar(filt.head(topN), x="feature", y="auc_eff", title="Top features by AUC separability", range_y=[0.5, 1.0]),
                        use_container_width=True,
                    )
                    # Heatmap of point-biserial correlations for top features + target
                    df_join_hm = join_on_timestamp(df_bt[["timestamp", tbin]], features_csv_df[["timestamp"] + top_feats])
                    st.plotly_chart(
                        corr_heatmap(df_join_hm, [tbin] + top_feats, method="pearson", title="Point-biserial Corr Heatmap (target + top features)"),
                        use_container_width=True,
                    )

    # ================= Optional Join for Cross Analysis =================
    st.subheader("Cross Analysis (optional join)")
    enable_join = st.checkbox("Enable join on timestamp for interactions/correlation", value=False)
    if enable_join:
        # Decide target for join
        join_target_col: Optional[str] = None
        join_target_df = pd.DataFrame(columns=["timestamp"])  # placeholder
        if selected_target_cols:
            join_target_col = st.selectbox("Target column for join", options=selected_target_cols)
            join_target_df = targets_df[["timestamp", join_target_col]].copy()
        else:
            st.info("Select a target to enable cross-source join.")

        join_feature_cols = st.multiselect(
            "Features to join", options=selected_features, default=selected_features[: min(5, len(selected_features))]
        )
        include_close = st.checkbox("Include close in joined view", value=True)

        if join_target_col and join_feature_cols:
            join_parts = []
            if include_close and "close" in ohlcv_view.columns:
                join_parts.append(ohlcv_view[["timestamp", "close"]])
            if join_feature_cols:
                join_parts.append(features_csv_df[["timestamp"] + join_feature_cols])
            join_parts.append(join_target_df)
            df_join = join_on_timestamp(*join_parts)
            if drop_na:
                df_join = df_join.dropna(subset=[c for c in [join_target_col] + join_feature_cols if c in df_join.columns])

            st.caption(f"Joined rows: {df_join.shape[0]:,} | Cols: {df_join.shape[1]:,}")
            st.dataframe(df_join.head(10))
            # Correlation with target
            cols_corr = join_feature_cols + [join_target_col]
            if len(cols_corr) >= 2:
                st.plotly_chart(corr_heatmap(df_join, cols_corr, title="Feature ↔ Target Correlation"), use_container_width=True)
            # Interaction scatter
            if join_feature_cols:
                x_col = st.selectbox("Scatter X (feature)", options=join_feature_cols)
                st.plotly_chart(scatter_xy(df_join, x_col, join_target_col, title=f"{x_col} vs {join_target_col}"), use_container_width=True)
        else:
            st.info("Pick a target and at least one feature to run cross analysis.")

    # ================= Report Export =================
    st.subheader("Report Export")
    prof_cls = _load_profile_report()
    if prof_cls:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Profile OHLCV"):
                with st.spinner("Profiling OHLCV..."):
                    profile = prof_cls(ohlcv_view, minimal=cfg.profile_minimal, title=f"OHLCV: {cfg.dataset_id}")
                    out_dir = Path(cfg.report_dir); out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"profile_ohlcv_{cfg.dataset_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.html"
                    profile.to_file(out_path)
                    st.success(f"Saved {out_path}")
                    with open(out_path, "rb") as f:
                        st.download_button("Download OHLCV report", data=f.read(), file_name=out_path.name, mime="text/html")
        with col2:
            if st.button("Profile Features") and selected_features:
                with st.spinner("Profiling features..."):
                    feat_view = features_csv_df[["timestamp"] + selected_features]
                    profile = prof_cls(feat_view, minimal=cfg.profile_minimal, title=f"Features: {cfg.dataset_id}")
                    out_dir = Path(cfg.report_dir); out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"profile_features_{cfg.dataset_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.html"
                    profile.to_file(out_path)
                    st.success(f"Saved {out_path}")
                    with open(out_path, "rb") as f:
                        st.download_button("Download features report", data=f.read(), file_name=out_path.name, mime="text/html")
        with col3:
            if st.button("Profile Targets"):
                with st.spinner("Profiling targets..."):
                    targ_view = targets_df[["timestamp"] + selected_target_cols] if selected_target_cols else pd.DataFrame(columns=["timestamp"])  # empty
                    profile = prof_cls(targ_view, minimal=cfg.profile_minimal, title=f"Targets: {cfg.dataset_id}")
                    out_dir = Path(cfg.report_dir); out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"profile_targets_{cfg.dataset_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.html"
                    profile.to_file(out_path)
                    st.success(f"Saved {out_path}")
                    with open(out_path, "rb") as f:
                        st.download_button("Download targets report", data=f.read(), file_name=out_path.name, mime="text/html")
        with col4:
            if enable_join:
                if st.button("Profile Joined"):
                    with st.spinner("Profiling joined view..."):
                        # Build joined view similar to above
                        join_parts = [ohlcv_view[["timestamp", "close"]]] if "close" in ohlcv_view.columns else []
                        if selected_features:
                            join_parts.append(features_csv_df[["timestamp"] + selected_features])
                        if selected_target_cols:
                            join_parts.append(targets_df[["timestamp"] + selected_target_cols])
                        df_join = join_on_timestamp(*join_parts) if join_parts else pd.DataFrame(columns=["timestamp"]) 
                        profile = prof_cls(df_join, minimal=cfg.profile_minimal, title=f"Joined: {cfg.dataset_id}")
                        out_dir = Path(cfg.report_dir); out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = out_dir / f"profile_joined_{cfg.dataset_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.html"
                        profile.to_file(out_path)
                        st.success(f"Saved {out_path}")
                        with open(out_path, "rb") as f:
                            st.download_button("Download joined report", data=f.read(), file_name=out_path.name, mime="text/html")

    # ================= Downloads =================
    st.subheader("Download Subsets")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("OHLCV CSV", data=ohlcv_view.to_csv(index=False).encode("utf-8"), file_name=f"ohlcv_{cfg.dataset_id}.csv", mime="text/csv")
    with col2:
        fv = features_csv_df[["timestamp"] + selected_features] if selected_features else features_csv_df
        st.download_button("Features CSV", data=fv.to_csv(index=False).encode("utf-8"), file_name=f"features_{cfg.dataset_id}.csv", mime="text/csv")
    with col3:
        tv = targets_df[["timestamp"] + selected_target_cols] if selected_target_cols else targets_df
        st.download_button("Targets CSV", data=tv.to_csv(index=False).encode("utf-8"), file_name=f"targets_{cfg.dataset_id}.csv", mime="text/csv")


if __name__ == "__main__":
    app_body()
