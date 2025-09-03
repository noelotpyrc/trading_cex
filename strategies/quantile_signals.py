#!/usr/bin/env python3
"""
Utilities to compute monotone-quantile fixes, expected returns from quantile
predictions, probability-of-positive, tail spreads, and basic trading signals.

Designed to operate on merged prediction CSVs produced by
`model/merge_quantile_predictions.py` that contain columns:
  - timestamp (optional), y_true, pred_q05, pred_q10, ..., pred_q95

This module focuses on pure computations. Threshold selection and data I/O are
handled by the CLI wrapper in `model/compute_signals_from_quantiles.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_QCOLS: List[str] = [
    "pred_q05",
    "pred_q10",
    "pred_q15",
    "pred_q25",
    "pred_q50",
    "pred_q75",
    "pred_q85",
    "pred_q90",
    "pred_q95",
]


@dataclass(frozen=True)
class MonotonicDiagnostics:
    cross_violations: pd.Series
    max_cross_gap: pd.Series


def detect_monotonic_violations(df: pd.DataFrame, qcols: Sequence[str] = DEFAULT_QCOLS) -> MonotonicDiagnostics:
    """Compute row-wise monotonicity diagnostics before fixing.

    - cross_violations: number of pairs (q_i, q_{i+1}) where q_{i+1} < q_i
    - max_cross_gap: max over i of (q_i - q_{i+1}) where q_{i+1} < q_i, else 0
    """
    qdf = df[list(qcols)].copy()
    # Differences between successive quantiles
    diffs = qdf.diff(axis=1).iloc[:, 1:]
    # Violations are negative diffs
    violations = (diffs < 0)
    cross_violations = violations.sum(axis=1)
    # Gap magnitude where violations occur
    gap = (-(diffs.where(violations))).max(axis=1).fillna(0.0)
    gap = gap.astype(float)
    return MonotonicDiagnostics(cross_violations=cross_violations, max_cross_gap=gap)


def apply_monotonic_fix(df: pd.DataFrame, qcols: Sequence[str] = DEFAULT_QCOLS) -> Tuple[pd.DataFrame, MonotonicDiagnostics]:
    """Return a copy with non-decreasing quantiles per row and diagnostics from the original values.
    """
    diagnostics = detect_monotonic_violations(df, qcols)
    fixed = df.copy()
    fixed[list(qcols)] = fixed[list(qcols)].cummax(axis=1)
    return fixed, diagnostics


def _build_p_q_nodes(row: pd.Series, qcols: Sequence[str] = DEFAULT_QCOLS) -> Tuple[np.ndarray, np.ndarray]:
    """Construct p- and q-nodes for integration with duplicated endpoints.

    p: [0.00, 0.05, 0.10, 0.15, 0.25, 0.50, 0.75, 0.85, 0.90, 0.95, 1.00]
    q: [q05, q05, q10, q15, q25, q50, q75, q85, q90, q95, q95]
    """
    p_nodes = np.array([0.00, 0.05, 0.10, 0.15, 0.25, 0.50, 0.75, 0.85, 0.90, 0.95, 1.00], dtype=float)
    qvals = row[list(qcols)].to_numpy(dtype=float)
    q_nodes = np.concatenate(([qvals[0]], qvals, [qvals[-1]]))
    return p_nodes, q_nodes


def expected_return_trapezoid(row: pd.Series, qcols: Sequence[str] = DEFAULT_QCOLS) -> float:
    p, q = _build_p_q_nodes(row, qcols)
    # Trapezoidal integration over successive segments
    dq = (q[:-1] + q[1:]) * 0.5
    dp = (p[1:] - p[:-1])
    return float(np.sum(dp * dq))


def expected_return_left_riemann(row: pd.Series, qcols: Sequence[str] = DEFAULT_QCOLS) -> float:
    p, q = _build_p_q_nodes(row, qcols)
    dp = (p[1:] - p[:-1])
    q_left = q[:-1]
    return float(np.sum(dp * q_left))


def estimate_prob_up(row: pd.Series, qcols: Sequence[str] = DEFAULT_QCOLS) -> float:
    """Estimate P(y > 0) via piecewise-linear interpolation on (p, q).

    Assumes q is non-decreasing.
    """
    p, q = _build_p_q_nodes(row, qcols)
    if q[1] >= 0:  # q05 >= 0
        return 1.0
    if q[-2] <= 0:  # q95 <= 0
        return 0.0
    # Find segment where q crosses 0
    idx = np.searchsorted(q, 0.0, side="left") - 1
    idx = int(np.clip(idx, 0, len(q) - 2))
    q0, q1 = q[idx], q[idx + 1]
    p0, p1 = p[idx], p[idx + 1]
    if q1 == q0:
        # Flat segment at zero; choose upper bound for a conservative estimate
        p_star = p1
    else:
        slope = (q1 - q0) / (p1 - p0)
        p_star = p0 + (0.0 - q0) / slope
    p_star = float(np.clip(p_star, 0.0, 1.0))
    return float(1.0 - p_star)


def compute_tail_spread(row: pd.Series, qcols: Sequence[str] = DEFAULT_QCOLS) -> float:
    q05 = float(row[qcols[0]])
    q95 = float(row[qcols[-1]])
    return q95 - q05


def add_core_metrics(df: pd.DataFrame, qcols: Sequence[str] = DEFAULT_QCOLS) -> pd.DataFrame:
    """Compute diagnostics, expected returns, probability, and tail spread.

    Returns a new DataFrame copy with added columns:
      - cross_violations, max_cross_gap (diagnostics from original values)
      - exp_ret_avg, exp_ret_conservative, prob_up, tail_spread
    """
    # Diagnostics from original values
    diags = detect_monotonic_violations(df, qcols)

    # Apply monotonic fix for downstream computations
    df_fixed = df.copy()
    df_fixed[list(qcols)] = df_fixed[list(qcols)].cummax(axis=1)

    df_fixed["cross_violations"] = diags.cross_violations.values
    df_fixed["max_cross_gap"] = diags.max_cross_gap.values

    df_fixed["exp_ret_avg"] = df_fixed.apply(lambda r: expected_return_trapezoid(r, qcols), axis=1)
    df_fixed["exp_ret_conservative"] = df_fixed.apply(lambda r: expected_return_left_riemann(r, qcols), axis=1)
    df_fixed["prob_up"] = df_fixed.apply(lambda r: estimate_prob_up(r, qcols), axis=1)
    df_fixed["tail_spread"] = df_fixed.apply(lambda r: compute_tail_spread(r, qcols), axis=1)

    return df_fixed


def compute_static_thresholds(
    frames: Iterable[pd.DataFrame],
    columns: Sequence[str],
    percentiles: Sequence[float],
) -> pd.Series:
    """Compute percentile-based thresholds on the union of provided frames.

    Returns a Series indexed by "{col}@{pct}" → threshold.
    """
    concat = pd.concat(frames, ignore_index=True)
    results = {}
    for col in columns:
        series = pd.to_numeric(concat[col], errors="coerce").dropna()
        for pct in percentiles:
            key = f"{col}@{pct:.4f}"
            results[key] = float(np.percentile(series.to_numpy(), pct * 100.0)) if not series.empty else np.nan
    return pd.Series(results)


def add_basic_signals(
    df: pd.DataFrame,
    selected_method: str = "avg",
    exp_zero_band: float = 1e-6,
    prob_thresholds: Tuple[float, float] = (0.55, 0.45),
    quantile_thresholds: Tuple[float, float] | None = None,
    qcols: Sequence[str] = DEFAULT_QCOLS,
) -> pd.DataFrame:
    """Add basic signals derived from expected returns, probability, and quantile cutoffs.

    - selected_method: "avg" or "conservative" for picking exp_ret_* when a single signal is desired.
    - exp_zero_band: deadband around zero for expected-return-based signal.
    - prob_thresholds: (tau_long, tau_short) for probability-based signal.
    - quantile_thresholds: optional (thr_q90_long, thr_q10_short) to create a quantile-based signal.
    """
    out = df.copy()
    # Expected return signal
    exp_col = "exp_ret_avg" if selected_method == "avg" else "exp_ret_conservative"
    exp_vals = out[exp_col].to_numpy()
    signal_exp = np.where(exp_vals > exp_zero_band, 1, np.where(exp_vals < -exp_zero_band, -1, 0))
    out["signal_exp"] = signal_exp.astype(int)

    # Probability signal
    tau_long, tau_short = prob_thresholds
    prob = out["prob_up"].to_numpy()
    signal_prob = np.where(prob >= tau_long, 1, np.where(prob <= tau_short, -1, 0))
    out["signal_prob"] = signal_prob.astype(int)

    # Quantile thresholds signal (optional)
    if quantile_thresholds is not None:
        thr_q90_long, thr_q10_short = quantile_thresholds
        q90 = out.get("pred_q90")
        q10 = out.get("pred_q10")
        if q90 is not None and q10 is not None:
            cond_long = q90.to_numpy() >= thr_q90_long
            cond_short = q10.to_numpy() <= thr_q10_short
            signal_q = np.where(cond_long, 1, np.where(cond_short, -1, 0))
            out["signal_quantile"] = signal_q.astype(int)
    return out


