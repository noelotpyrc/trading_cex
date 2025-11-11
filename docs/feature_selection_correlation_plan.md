# Correlation‑Driven Feature Selection — Plan & Logic (no code)

This document defines a target‑free, correlation‑aware feature selection approach that reduces redundancy across multi‑timeframe features while preserving features deemed important by trained LightGBM runs. It focuses on logic and process only (no implementation).

## Objectives
- Reduce redundant features (especially many windows per indicator/timeframe) without using the target.
- Keep a curated whitelist of “important” features obtained from prior trained models.
- Prefer stable, simple, lower‑latency features when choices are equivalent.
- Produce a concise, reproducible feature set with transparent diagnostics.

Non‑goals:
- Do not optimize for a specific target metric here.
- Do not rely on SHAP/permutation with the target in the selection logic (can be used later for validation only).

## Data & Assumptions
- Feature names carry timeframe suffixes like `_1H`, `_4H`, `_12H`, `_1D` (consistent with existing tables).
- Availability of trained model run folders containing `feature_importance.csv` near `model_path` (from predictions table or run directories).
- Time‑series data; leakage is possible if selection peeks ahead. All diagnostics/splits must be time‑aware.

## Redundancy Metric
- Use absolute Spearman correlation |ρ| between features as the redundancy signal (robust to monotonic transforms and outliers).
- Define distance for clustering as `1 − |ρ|`.
- Primary threshold τ ≈ 0.90 (start); explore 0.85–0.95 if needed.

## Unsupervised Priority (No Target)
When two features are highly correlated, pick the one with higher unsupervised “quality”. The score combines:
- Robust variability: standard deviation of rank‑transformed values (uniform [0,1]); favors informative spread.
- Missingness penalty: multiply by `(1 − NaN_rate)^penalty` (default penalty = 1); favors reliable features.
- Optional timeframe bias: slightly up‑weight shorter TFs (e.g., `_1H` > `_1D`) for responsiveness/latency when ties occur.

Rationale: preserves features that carry information consistently, penalizes noisy/sparse ones, and codifies a mild real‑time preference.

## Selection Strategies
Pick one (greedy is the baseline); both ignore the target.

1) Greedy Correlation Filter (unsupervised)
   - Order features by the quality score (whitelist first, see below).
   - Iterate in order; keep a feature if its max |ρ| to the current kept set is < τ.
   - Complexity is manageable; no clustering dependencies.

2) Correlation Clustering + Representative
   - Compute |Spearman| correlation matrix; distance = `1 − |ρ|`.
   - Hierarchical clustering (average/complete); cut at `1 − τ`.
   - Choose the best representative per cluster via the quality score.
   - Produces clear “families”; easy to visualize.

3) Optional: Pivoted QR (linear de‑dup)
   - Column‑pivoted QR on standardized features; keep columns until R‑diag < tol.
   - Fast for near‑linear redundancy; complement to correlation methods.
   - Can be used as a pre‑filter before correlation selection.

## Whitelist from Trained Models (Pinned Features)
Source: LightGBM `feature_importance.csv` near each `model_path` (file or parent folders).

Aggregation logic:
- Per run, normalize importances so they sum to 1 (comparability).
- Restrict to top‑K per run (e.g., 50–150) to avoid flooding the whitelist.
- Aggregate across runs by mean importance; record how many runs each feature appears in.
- Whitelist = features appearing in ≥ min_runs (start with 1; increase to 2+ with more runs).

Pinning policy:
- Pinned features are included a priori in the final set.
- If pinned features are mutually redundant, keep them anyway (they represent proven usage) but consider per‑family caps (see below) to maintain diversity.
- If pinning prevents diversity, either reduce top‑K per run or raise τ slightly.

## Family/Timeframe Constraints
- Group features by “family” (base name without timeframe suffix) to detect variants of the same indicator across TFs/windows.
- Cap representatives per family, e.g., 2 (fast/slow) or 3 (fast/medium/slow). Pinned features may exceed caps lightly, but aim to keep the cap tight.
- Prefer derived contrasts over many raw windows when possible (e.g., spreads, z‑scores, slopes often carry the gist of multi‑window families with fewer columns).

## Stability Across Time (Target‑Free)
- Split history into chronological blocks/folds (e.g., 5–10 non‑overlapping segments).
- Run the chosen selector in each block independently (no target leakage).
- Keep features that appear in ≥ presence% of blocks (e.g., 60%).
- Outcome: stable features that persist across regimes; transient correlations are filtered out.

## Tuning Knobs (Start Points)
- Correlation threshold τ: 0.90 (test 0.85–0.95).
- Top‑K per run for whitelist: 100.
- min_runs to join whitelist: 1 (raise to 2+ as runs accumulate).
- Stability presence across blocks: 0.6.
- Per‑family cap: 2 (or 3 if needed).
- Missingness penalty: 1.0.

## Diagnostics & Reporting
Produce a small report for each selection run:
- Pairwise |ρ| distribution before vs after selection.
- Cluster sizes and chosen representatives (if clustering).
- Retained vs dropped counts per family and timeframe.
- Whitelist coverage: pinned features included; those missing (not found in current X) with counts.
- Stability: presence histogram across folds for the final set.
- Sensitivity: selected count vs τ curve; changes in representatives when τ changes.

## Recommended Metrics (Concise Set)
Use these few, robust, target‑free metrics to quantify overall correlation/diversification. Compute on train‑only (and per time block for stability).

- Mean |ρ| and 95th |ρ|
  - Definition: mean and 95th percentile of absolute Spearman correlations over all i≠j pairs.
  - Interpretation: lower is better; 95th captures tail redundancy (near duplicates).
  - Targets: mean |ρ| < 0.25 (good), 0.25–0.40 (watch); 95th |ρ| < 0.85 (good), 0.85–0.95 (watch).

- Effective Rank (eRank)
  - Definition: eigenvalues λ of the correlation matrix C; p_i = λ_i/∑λ; H = −∑ p_i log p_i; eRank = exp(H).
  - Normalized: eRank/p (0–1). Higher means more effective dimensions.
  - Targets: eRank/p > 0.6 (good), 0.4–0.6 (watch).

- Largest Correlated Component @ τ
  - Definition: build a graph with edges where |ρ| ≥ τ (τ≈0.90). Report fraction of features in the largest connected component.
  - Interpretation: smaller is better; large component indicates big redundant cluster(s).
  - Targets: < 0.20 (good), 0.20–0.40 (watch).

- Diversification Ratio (equal‑weight)
  - Definition: DR = 1 / sqrt(wᵀ C w) with w = (1/p,…,1/p) and C the correlation matrix.
  - Range: [1, √p]. Normalize by √p for comparability: DR_norm = DR/√p (0–1). Higher is better.
  - Targets: DR/√p > 0.70 (good), 0.50–0.70 (watch).

Computation notes
- Prefer Spearman correlations; drop all‑NaN/constant columns before metrics.
- For very wide p, compute metrics on a stratified feature sample (e.g., by family/timeframe) and full metrics on the shortlisted set.
- Report medians over chronological blocks to avoid regime bias; track deltas before/after selection.

## Integration Plan (Phases)
Phase 1 — Analysis (Notebook)
- Load candidate features X and selected run `model_path`s.
- Build whitelist from `feature_importance.csv` around each run.
- Run greedy or clustering selection with the whitelist and produce diagnostics.
- Persist the final list as an artifact (e.g., `selected_features.txt` or `selected_features.yaml`).

Phase 2 — Pipeline Integration
- Add a small selection step that:
  - Accepts X, τ, family cap, presence threshold, and whitelist paths.
  - Writes the final selected feature list next to the model outputs and into MLflow artifacts.
  - Logs selection metadata: τ, caps, number pinned, number per family, stability stats.
- Ensure training uses only the selected list; keep a config flag to bypass selection for experiments.

Phase 3 — Hardening & Validation
- Stability validation across alternative fold partitions.
- Sensitivity analysis over τ and family caps; choose settings that keep performance stable.
- Out‑of‑fold validation using permutation/drop‑column importance (read‑only; not used for selection) to confirm the reduced set doesn’t degrade lift.

## Risks & Mitigations
- Over‑pinning from whitelist may freeze redundant features → cap per family and reduce top‑K per run.
- τ too low prunes diversity → increase τ or relax family caps.
- Transient correlations lead to unstable sets → apply stability selection across time blocks.
- Data gaps drive the quality score → ensure missingness penalty is present; impute only for diagnostics, not selection logic.

## Success Criteria
- Final feature count reduced substantially (e.g., 5–10×) with minimal or no CV performance loss.
- Selected features are stable across time blocks and robust to moderate τ perturbations.
- Report artifacts make the selection transparent and reproducible (list, params, diagnostics).

## Quick Checklist (for execution later)
- [ ] Choose τ, presence, per‑family cap defaults.
- [ ] Aggregate whitelist from trained runs (top‑K, min_runs).
- [ ] Run greedy/clustering selection; generate diagnostics.
- [ ] Apply stability selection; update final list.
- [ ] Persist selected list + metadata; wire into training pipeline.

## MLflow‑First Greedy (Simplified)
Always start from features proven by prior models and only de‑duplicate via correlation. This keeps the process simple and deterministic while leveraging historical model signal.

- Candidate collection
  - Query MLflow runs by experiment + metric filter (e.g., AUC ≥ x or RMSE ≤ y).
  - For each run, take top‑N features from `feature_importance.csv`.
  - Build `counts: feature → occurrences` across runs.

- Candidate filtering
  - Keep features with `count ≥ min_occurrence` (e.g., 2).
  - Optionally cap to `topK_overall` by count (e.g., 200–400).
  - Intersect with available columns in X (current dataset).

- Ordering (no target)
  - Primary: `count` descending.
  - Tie‑breakers: lower NaN rate, higher rank‑std (robust variability), shorter timeframe suffix (e.g., `_1H` over `_1D`), then name for determinism.

- Greedy correlation prune
  - Iterate ordered candidates; keep if max |Spearman ρ| to already kept < τ (default 0.90).
  - Enforce per‑family cap (e.g., ≤2 per base indicator across TFs). Families defined by stripping the trailing timeframe suffix; extend to window families if needed.
  - Correlation computed with pairwise complete observations; require a minimum overlap (e.g., ≥200 rows) to accept ρ; otherwise treat as 0 for redundancy.

- Optional stability across time
  - Split history into chronological blocks (e.g., 5–10).
  - Run the greedy step per block on the same candidate list.
  - Final set = features present in ≥ `presence` fraction of blocks (e.g., 60%).
  - Optionally pin “very frequent” candidates (≥ 90th percentile of count) so they’re always included.

- Defaults (starting points)
  - MLflow: `top_n_per_run=100`, `min_occurrence=2`, `topK_overall=300`.
  - Correlation: `tau=0.90`, `min_overlap=200` samples, Spearman.
  - Family cap: `cap_per_family=2` (consider 3 if needed).
  - Stability: `presence=0.6`, blocks=5–10.

- Minimal diagnostics
  - On the candidate set, report before→after deltas for: mean |ρ|, 95th |ρ|, eRank/p, largest component @ τ=0.90, DR/√p.
  - Selection summary: kept vs dropped counts; per‑family kept; top reasons (redundant vs over‑cap).
  - Counts table: feature → occurrences (from MLflow) for auditability.

- Config stub (for later wiring)
  - feature_selection.method: `mlflow_greedy`
  - feature_selection.mlflow: `{ experiment, metric, comparator, threshold, top_n_per_run, min_occurrence, topK_overall, tracking_uri? }`
  - feature_selection.corr: `{ tau, min_overlap }`
  - feature_selection.family: `{ cap_per_family, known_tfs: ['1H','4H','12H','1D'] }`
  - feature_selection.stability (optional): `{ presence, n_blocks }`
