#!/usr/bin/env python3
"""
Greedy Forward Feature Selection for LGBM Binary Classification.

This script performs supervised greedy forward selection:
1. Starts with a base feature (close_parkinson_20_1H, AUC ~0.62)
2. Tests adding each candidate feature individually
3. Selects the feature that gives the best test AUC improvement
4. Repeats until test AUC stops improving
5. Logs all results and saves the optimal feature list
6. Logs everything to MLflow for tracking and comparison

Algorithm:
- Round 1: base_feature + each candidate → pick best combination
- Round 2: best_from_round1 + each remaining candidate → pick best
- Continue until test AUC decreases
- Stop criteria: best_auc_test(round_N) < best_auc_test(round_N-1)

Usage:
    python analysis/greedy_forward_selection.py \
        --base-config configs/model_configs/binance_btcusdt_perp_1h_since_2020_lgbm_y_binary4u2d_24h_tuning_selected_feature_greedy.json \
        --candidates-json configs/feature_lists/binance_btcusdt_p60_from_corr_since2020_1.json \
        --output-dir /Volumes/Extreme\ SSD/trading_data/cex/greedy_selection \
        --baseline-auc 0.62 \
        --mlflow-tracking-uri http://127.0.0.1:5000 \
        --mlflow-experiment greedy-feature-selection \
        --exclude-same-family
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd

# Add project root to Python path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import pipeline functions
try:
    from model.run_lgbm_pipeline import (
        load_config,
        prepare_training_data,
        tune_hyperparameters,
        train_model,
        persist_results,
    )
except ImportError as e:
    raise SystemExit(f"Failed to import pipeline functions: {e}")


def setup_logging(output_dir: Path) -> None:
    """Setup logging for greedy selection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"greedy_selection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(stream=sys.stdout),
            logging.FileHandler(str(log_file))
        ]
    )
    logging.info(f"Logging to: {log_file}")


def get_feature_family(feature_name: str) -> str:
    """Extract feature family by removing timeframe suffix."""
    # Known timeframe suffixes
    timeframes = ['_1H', '_4H', '_12H', '_1D']
    for tf in timeframes:
        if feature_name.endswith(tf):
            return feature_name[:-len(tf)]
    return feature_name


def load_candidates(
    candidates_json: Path,
    base_features: List[str],
    exclude_same_family: bool = False
) -> List[str]:
    """
    Load candidate features from JSON, excluding the base features.

    Args:
        candidates_json: Path to JSON file with candidate features
        base_features: List of base features to exclude
        exclude_same_family: If True, exclude all features from the same family
                            (e.g., if base is close_parkinson_20_1H, exclude
                             close_parkinson_20_4H, close_parkinson_20_12H, etc.)
    """
    with open(candidates_json, 'r') as f:
        all_features = json.load(f)

    if not isinstance(all_features, list):
        raise ValueError("Candidates JSON must contain a list of feature names")

    # Remove base features if present
    candidates = [f for f in all_features if f not in base_features]

    # Optionally exclude same family
    if exclude_same_family:
        base_families = [get_feature_family(bf) for bf in base_features]
        excluded_count = 0
        filtered_candidates = []

        for f in candidates:
            if get_feature_family(f) in base_families:
                excluded_count += 1
                logging.info(f"  Excluding same-family feature: {f}")
            else:
                filtered_candidates.append(f)

        candidates = filtered_candidates
        logging.info(f"Excluded {excluded_count} features from the same families as base features")

    logging.info(f"Loaded {len(candidates)} candidate features (excluding {len(base_features)} base features)")
    return candidates


def train_with_features(
    base_config: Dict[str, Any],
    features: List[str],
    round_num: int,
    combo_num: int,
    output_dir: Path,
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_experiment: Optional[str] = None
) -> Tuple[Dict[str, float], Path]:
    """
    Train a model with the specified feature list and register to MLflow.

    Returns:
        (metrics, run_dir) where metrics contains 'auc_test', 'auc_val', etc.
    """
    # Create a modified config with the feature list
    config = deepcopy(base_config)
    config['feature_selection'] = {'include': features}

    # Create a unique run name
    run_name = f"round{round_num:02d}_combo{combo_num:03d}_{len(features)}features"

    logging.info(f"\n{'='*80}")
    logging.info(f"Training: {run_name}")
    logging.info(f"Features ({len(features)}): {features}")
    logging.info(f"{'='*80}\n")

    try:
        # Step 1: Prepare data
        data_dir = prepare_training_data(config)

        # Step 2: Tune hyperparameters
        tuned_params = tune_hyperparameters(config, data_dir)

        # Step 3: Choose final params
        use_best = bool(config['model'].get('use_best_params_for_final', True))
        final_params = tuned_params if use_best else config['model'].get('params', {})

        # Step 4: Train and evaluate
        _, metrics, run_dir = train_model(config, data_dir, final_params)

        # Step 5: Persist all results
        run_dir = persist_results(config, run_dir, metrics, final_params, data_dir)

        logging.info(f"✓ {run_name} completed")
        logging.info(f"  Test AUC:  {metrics.get('auc_test', 0.0):.6f}")
        logging.info(f"  Val AUC:   {metrics.get('auc_val', 0.0):.6f}")
        logging.info(f"  Train AUC: {metrics.get('auc_train', 0.0):.6f}")
        logging.info(f"  Run dir: {run_dir}")

        # Register to MLflow using the existing registrar
        if mlflow_tracking_uri and mlflow_experiment:
            try:
                import subprocess
                repo_root = Path(__file__).resolve().parent.parent
                registrar = repo_root / "mlflow" / "mlflow_register.py"

                cmd = [
                    sys.executable,
                    str(registrar),
                    "--tracking-uri", mlflow_tracking_uri,
                    "--experiment", mlflow_experiment,
                    "--run-dir", str(run_dir),
                    "--no-model-register",  # Just log the run, don't register as model
                    "--artifact-mode", "all"
                ]

                logging.info(f"  Registering to MLflow...")
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)

                if result.returncode == 0:
                    logging.info(f"  ✓ MLflow registration successful")
                else:
                    logging.warning(f"  ✗ MLflow registration failed: {result.stderr}")

            except Exception as e:
                logging.warning(f"Failed to register to MLflow: {e}")

        return metrics, run_dir

    except Exception as e:
        logging.error(f"✗ {run_name} failed: {e}")
        raise


def greedy_forward_selection(
    base_config: Dict[str, Any],
    base_features: List[str],
    candidates: List[str],
    baseline_auc: float,
    output_dir: Path,
    min_improvement: float = 0.0,
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_experiment: Optional[str] = None
) -> Tuple[List[str], pd.DataFrame]:
    """
    Perform greedy forward feature selection.

    Args:
        base_config: Base pipeline configuration
        base_features: Starting features (list)
        candidates: List of candidate features to test
        baseline_auc: Baseline test AUC for the base features
        output_dir: Directory to save results
        min_improvement: Minimum AUC improvement to accept a feature (default 0.0)
        mlflow_tracking_uri: MLflow tracking URI (optional)
        mlflow_experiment: MLflow experiment name (optional)

    Returns:
        (best_features, results_df) where results_df contains all trials
    """
    results = []
    current_features = base_features.copy()  # Start with all base features
    current_best_auc = baseline_auc
    remaining_candidates = candidates.copy()
    round_num = 0

    logging.info(f"\n{'#'*80}")
    logging.info(f"STARTING GREEDY FORWARD SELECTION")
    logging.info(f"{'#'*80}")
    logging.info(f"Base features ({len(base_features)}): {base_features}")
    logging.info(f"Baseline test AUC: {baseline_auc:.6f}")
    logging.info(f"Candidates: {len(remaining_candidates)}")
    logging.info(f"Min improvement threshold: {min_improvement:.6f}")
    if mlflow_tracking_uri:
        logging.info(f"MLflow tracking: {mlflow_tracking_uri}")
        logging.info(f"MLflow experiment: {mlflow_experiment}")
    logging.info(f"{'#'*80}\n")

    # Run the greedy selection loop
    try:
        while remaining_candidates:
            round_num += 1
            round_results = []

            logging.info(f"\n{'='*80}")
            logging.info(f"ROUND {round_num}: Testing {len(remaining_candidates)} candidates")
            logging.info(f"Current features ({len(current_features)}): {current_features}")
            logging.info(f"Current best test AUC: {current_best_auc:.6f}")
            logging.info(f"{'='*80}\n")

            # Test each remaining candidate
            for idx, candidate in enumerate(remaining_candidates, 1):
                test_features = current_features + [candidate]

                try:
                    metrics, run_dir = train_with_features(
                        base_config=base_config,
                        features=test_features,
                        round_num=round_num,
                        combo_num=idx,
                        output_dir=output_dir,
                        mlflow_tracking_uri=mlflow_tracking_uri,
                        mlflow_experiment=mlflow_experiment
                    )

                    result = {
                        'round': round_num,
                        'combo_num': idx,
                        'candidate_feature': candidate,
                        'num_features': len(test_features),
                        'features': test_features.copy(),
                        'auc_test': metrics.get('auc_test', 0.0),
                        'auc_val': metrics.get('auc_val', 0.0),
                        'auc_train': metrics.get('auc_train', 0.0),
                        'improvement': metrics.get('auc_test', 0.0) - current_best_auc,
                        'run_dir': str(run_dir)
                    }

                    round_results.append(result)
                    results.append(result)

                except Exception as e:
                    logging.error(f"Failed to train with candidate {candidate}: {e}")
                    continue

            if not round_results:
                logging.warning(f"Round {round_num} produced no valid results. Stopping.")
                break

            # Find best candidate from this round
            round_results_sorted = sorted(round_results, key=lambda x: x['auc_test'], reverse=True)
            best_result = round_results_sorted[0]

            logging.info(f"\n{'='*80}")
            logging.info(f"ROUND {round_num} RESULTS:")
            logging.info(f"{'='*80}")
            for i, res in enumerate(round_results_sorted[:5], 1):
                logging.info(f"{i}. {res['candidate_feature']}: "
                            f"AUC={res['auc_test']:.6f} (Δ={res['improvement']:+.6f})")
            if len(round_results_sorted) > 5:
                logging.info(f"... and {len(round_results_sorted)-5} more")

            logging.info(f"\nBest candidate: {best_result['candidate_feature']}")
            logging.info(f"  Test AUC: {best_result['auc_test']:.6f}")
            logging.info(f"  Improvement: {best_result['improvement']:+.6f}")

            # Check stopping criteria
            if best_result['auc_test'] < current_best_auc - min_improvement:
                logging.info(f"\n{'!'*80}")
                logging.info(f"STOPPING: Best AUC from round {round_num} "
                            f"({best_result['auc_test']:.6f}) is not better than "
                            f"previous best ({current_best_auc:.6f})")
                logging.info(f"{'!'*80}\n")
                break

            # Accept best candidate
            current_features = best_result['features'].copy()
            current_best_auc = best_result['auc_test']
            remaining_candidates.remove(best_result['candidate_feature'])

            logging.info(f"\n✓ Accepted: {best_result['candidate_feature']}")
            logging.info(f"  Current feature set ({len(current_features)}): {current_features}")
            logging.info(f"  Current best test AUC: {current_best_auc:.6f}")

            # Save intermediate results
            save_results(results, output_dir, round_num)

    finally:
        # Create results DataFrame
        results_df = pd.DataFrame(results)

        logging.info(f"\n{'#'*80}")
        logging.info(f"GREEDY SELECTION COMPLETED")
        logging.info(f"{'#'*80}")
        logging.info(f"Final features ({len(current_features)}): {current_features}")
        logging.info(f"Final test AUC: {current_best_auc:.6f}")
        logging.info(f"Improvement over baseline: {current_best_auc - baseline_auc:+.6f}")
        logging.info(f"Total rounds: {round_num}")
        logging.info(f"Total models trained: {len(results)}")
        logging.info(f"{'#'*80}\n")

    return current_features, results_df


def save_results(results: List[Dict], output_dir: Path, round_num: int = None) -> None:
    """Save intermediate and final results."""
    suffix = f"_round{round_num}" if round_num else "_final"

    # Save as JSON
    json_path = output_dir / f"greedy_selection_results{suffix}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logging.info(f"Results saved to: {json_path}")

    # Save as CSV
    if results:
        df = pd.DataFrame(results)
        # Drop the 'features' column for CSV (it's a list)
        df_csv = df.drop(columns=['features'], errors='ignore')
        csv_path = output_dir / f"greedy_selection_results{suffix}.csv"
        df_csv.to_csv(csv_path, index=False)
        logging.info(f"Results CSV saved to: {csv_path}")


def save_feature_list(features: List[str], output_dir: Path, baseline_auc: float, final_auc: float) -> None:
    """Save the final selected feature list as JSON."""
    feature_list_path = output_dir / "selected_features_greedy.json"
    with open(feature_list_path, 'w') as f:
        json.dump(features, f, indent=2)

    logging.info(f"\nFinal feature list saved to: {feature_list_path}")

    # Also save metadata
    metadata = {
        'num_features': len(features),
        'features': features,
        'baseline_auc_test': baseline_auc,
        'final_auc_test': final_auc,
        'improvement': final_auc - baseline_auc,
        'timestamp': datetime.now().isoformat()
    }

    metadata_path = output_dir / "selection_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"Metadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Greedy forward feature selection for LGBM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--base-config',
        type=Path,
        required=True,
        help='Base LGBM pipeline config JSON'
    )
    parser.add_argument(
        '--candidates-json',
        type=Path,
        required=True,
        help='JSON file with candidate feature list'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('/Volumes/Extreme SSD/trading_data/cex/greedy_selection'),
        help='Output directory for results'
    )
    parser.add_argument(
        '--baseline-auc',
        type=float,
        default=0.62,
        help='Baseline test AUC for the base feature'
    )
    parser.add_argument(
        '--min-improvement',
        type=float,
        default=0.0,
        help='Minimum AUC improvement to accept a feature'
    )
    parser.add_argument(
        '--mlflow-tracking-uri',
        type=str,
        default=None,
        help='MLflow tracking URI (e.g., http://127.0.0.1:5000)'
    )
    parser.add_argument(
        '--mlflow-experiment',
        type=str,
        default='greedy-feature-selection',
        help='MLflow experiment name'
    )
    parser.add_argument(
        '--exclude-same-family',
        action='store_true',
        help='Exclude features from the same indicator family as the base feature'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.output_dir)

    logging.info("="*80)
    logging.info("GREEDY FORWARD FEATURE SELECTION")
    logging.info("="*80)
    logging.info(f"Base config: {args.base_config}")
    logging.info(f"Candidates: {args.candidates_json}")
    logging.info(f"Output dir: {args.output_dir}")
    logging.info(f"Baseline AUC: {args.baseline_auc}")
    logging.info(f"Min improvement: {args.min_improvement}")
    logging.info(f"Exclude same family: {args.exclude_same_family}")
    if args.mlflow_tracking_uri:
        logging.info(f"MLflow tracking URI: {args.mlflow_tracking_uri}")
        logging.info(f"MLflow experiment: {args.mlflow_experiment}")
    else:
        logging.info("MLflow: Disabled (no tracking URI provided)")
    logging.info("="*80 + "\n")

    try:
        # Load base config
        base_config = load_config(args.base_config)
        logging.info(f"✓ Loaded base config from: {args.base_config}")

        # Extract base features (can be one or more)
        base_features = base_config.get('feature_selection', {}).get('include', [])
        if not base_features:
            raise ValueError("Base config must have 'feature_selection.include' with at least one feature")

        logging.info(f"✓ Base features ({len(base_features)}): {base_features}")

        # Load candidates (exclude all base features)
        candidates = load_candidates(
            args.candidates_json,
            base_features,
            exclude_same_family=args.exclude_same_family
        )
        logging.info(f"✓ Loaded {len(candidates)} candidate features\n")

        # Run greedy selection
        best_features, results_df = greedy_forward_selection(
            base_config=base_config,
            base_features=base_features,
            candidates=candidates,
            baseline_auc=args.baseline_auc,
            output_dir=args.output_dir,
            min_improvement=args.min_improvement,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            mlflow_experiment=args.mlflow_experiment
        )

        # Save final results
        save_results(results_df.to_dict('records'), args.output_dir)

        # Get final AUC
        final_auc = args.baseline_auc
        if not results_df.empty:
            # Find the best AUC achieved (from accepted features, not all trials)
            accepted_features_count = len(best_features)
            final_auc = results_df[results_df['num_features'] == accepted_features_count]['auc_test'].max() if accepted_features_count > len(base_features) else args.baseline_auc

        # Save feature list
        save_feature_list(best_features, args.output_dir, args.baseline_auc, final_auc)

        logging.info("\n" + "="*80)
        logging.info("SUCCESS: Greedy selection completed!")
        logging.info("="*80)

    except Exception as e:
        logging.error(f"\nFATAL ERROR: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
