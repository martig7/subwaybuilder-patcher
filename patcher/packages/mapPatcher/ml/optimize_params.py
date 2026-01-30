"""
Parameter Optimization using Bayesian Optimization

Uses Optuna's Tree-structured Parzen Estimator (TPE) to find optimal parameters
that minimize the error between generated and ground truth data.

Usage: python optimize_params.py [--n-trials N] [--n-jobs N] [--city CITY_CODE]
  --n-trials N   Number of optimization trials (default: 100)
  --n-jobs N     Number of parallel jobs (default: 4)
  --city CODE    Only optimize for specific city (default: all matching cities)
"""

import json
import optuna
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import argparse
import sys
import threading
import time
import gc
import multiprocessing as mp

# Try to import psutil for memory monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Import the Python processing module directly
from process_data import process_all_places, ProcessingConfig, ensure_preclassified_exists, MAP_PATCHER_DIR


# Thread-local storage for worker IDs
_thread_local = threading.local()
_worker_id_lock = threading.Lock()
_next_worker_id = 0
_n_workers = 4  # Will be updated based on --n-jobs


def get_worker_id() -> int:
    """Get a stable worker ID for the current thread"""
    global _next_worker_id

    if not hasattr(_thread_local, 'worker_id'):
        with _worker_id_lock:
            _thread_local.worker_id = _next_worker_id % _n_workers
            _next_worker_id += 1

    return _thread_local.worker_id


# Parameter search space
PARAM_SPACE = {
    # Stage 1: Building → Population
    'residential-sqft-multiplier': (0.5, 2.0),
    'commercial-sqft-multiplier': (0.5, 2.0),
    'mixed-use-residential-ratio': (0.0, 1.0),
    'yes-building-sqft-per-resident': (100, 800),
    'yes-building-sqft-per-job': (50, 500),
    'levels-default': (1.0, 4.0),
    'levels-multiplier': (0.5, 2.0),
    'min-building-area-sqft': (0, 1000),
    'max-building-area-sqft': (0, 500000),
    'residential-occupancy-rate': (0.7, 1.0),
    
    # Stage 2: Cluster management
    'max-place-size-for-splitting': (500, 10000),
    'merge-places-distance-meters': (0, 200),
    'splitting-num-clusters-divisor': (625, 5000),
    'splitting-cluster-min-separation': (0.001, 0.05),
    
    # Stage 3: Connection generation
    'gravity-distance-floor-meters': (100, 5000),
    'gravity-exponent': (0.3, 1.5),
    'min-connections-per-cluster': (1, 5),
    'max-connections-per-cluster': (5, 25),
    'connection-scaling-divisor': (20, 160),
    'population-scale-factor': (0.25, 2.0),
    'max-total-connections': (0, 5000),
    'max-connection-distance-meters': (0, 50000),  # 0 = unlimited
}

# Parameters to hold constant during optimization
FIXED_PARAMS = {
    'tile-zoom-level': 16,
    'connection-size-cap': 200,
    'skip-buildings-index': True,  # Skip for ML (faster) - will be created on final run
}

# Weights for aggregate error calculation (must sum to 1.0)
# Emphasis on distribution shape: std, percentiles, min/max over mean/median
ERROR_WEIGHTS = {
    # Stage 1: Total Population/Jobs - CRITICAL for map scale
    'total_population': 0.04,
    'total_jobs': 0.03,

    # Stage 2a: Residential size distribution
    'res_size_std': 0.03,
    'res_size_p10': 0.02,
    'res_size_p25': 0.02,
    'res_size_p75': 0.02,
    'res_size_p90': 0.02,
    'res_size_max': 0.03,  # Prevents mega-clusters

    # Stage 2b: Commercial/job size distribution
    'job_size_std': 0.03,
    'job_size_p10': 0.02,
    'job_size_p25': 0.02,
    'job_size_p75': 0.02,
    'job_size_p90': 0.02,
    'job_size_max': 0.03,  # Prevents mega-clusters

    # Stage 2c: Residential spacing distribution
    'res_nn_dist_std': 0.02,
    'res_nn_dist_p10': 0.06,
    'res_nn_dist_p25': 0.01,
    'res_nn_dist_p75': 0.01,
    'res_nn_dist_p90': 0.02,
    'res_nn_dist_max': 0.01,

    # Stage 2d: Commercial spacing distribution
    'job_nn_dist_std': 0.02,
    'job_nn_dist_p10': 0.06,
    'job_nn_dist_p25': 0.01,
    'job_nn_dist_p75': 0.01,
    'job_nn_dist_p90': 0.02,
    'job_nn_dist_max': 0.01,

    # Stage 2e: Overall cluster structure
    'cluster_count': 0.05,

    # Stage 3a: Connection distance distribution
    'conn_dist_std': 0.02,
    'conn_dist_p10': 0.02,
    'conn_dist_p25': 0.02,
    'conn_dist_p75': 0.02,
    'conn_dist_p90': 0.02,
    'conn_dist_max': 0.02,

    # Stage 3b: Connection structure
    'connections_per_cluster': 0.05,
    'conn_size_mean': 0.04,
    'graph_density': 0.06,
}

# City code mapping
CITY_CODE_MAPPING = {
    'HNL_GEN': 'HNL',
    'CHI_GEN': 'CHI',
    'SF_GEN': 'SF',
    'BOS_GEN': 'BOS',
    'DC_GEN': 'DC',
    'ATL_GEN': 'ATL',
    'SEA_GEN': 'SEA',
    'DEN_GEN': 'DEN',
    'LON_GEN': 'LON',
    'PHL_GEN': 'PHL',
}


class OptimizationContext:
    """Shared context for optimization trials"""

    def __init__(self, target_places: list, gt_by_city: dict, worker_id: int = 0):
        self.target_places = target_places
        self.gt_by_city = gt_by_city  # Now shared across workers
        self.script_dir = Path(__file__).parent
        self.map_patcher_dir = self.script_dir.parent
        self.worker_id = worker_id
        self.last_result = None  # Store last processing result for feature extraction
        self._temp_files = []  # Track temp files to clean up

    def run_pipeline(self, params: Dict) -> bool:
        """Run Python processing pipeline directly (no subprocess)"""
        try:
            # Build config dict
            config_dict = {
                **FIXED_PARAMS,
                **params,
                'places': self.target_places,
                'skip-buildings-index': True,  # Skip during optimization (faster)
            }

            # Run processing directly
            results = process_all_places(config_dict)

            # Check for failures
            for result in results:
                if not result['success']:
                    print(f"W{self.worker_id}: Pipeline failed for {result['code']}: {result.get('error', 'unknown')}", file=sys.stderr)
                    return False

            self.last_result = results
            return True

        except Exception as e:
            print(f"W{self.worker_id}: Pipeline error: {e}", file=sys.stderr)
            return False
    
    def cleanup(self):
        """Clean up temporary data to free memory"""
        self.last_result = None
        for temp_file in self._temp_files:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
        self._temp_files.clear()
        gc.collect()

    def extract_features(self) -> Optional[List[Dict]]:
        """Extract features from last processing result (no subprocess)"""
        if not self.last_result:
            print(f"W{self.worker_id}: No processing result to extract features from", file=sys.stderr)
            return None

        try:
            from extract_generated import extract_features, CityFeatures
            from dataclasses import asdict

            features_list = []
            for result in self.last_result:
                if not result['success']:
                    continue

                city_code = result['code']
                data = result['data']

                if not data.get('points'):
                    print(f"W{self.worker_id}: No points for {city_code}", file=sys.stderr)
                    continue

                features = extract_features(data, city_code)
                features_list.append(asdict(features))

            return features_list if features_list else None

        except Exception as e:
            print(f"W{self.worker_id}: Feature extraction error: {e}", file=sys.stderr)
            return None

    def calculate_aggregate_error(self, gen: dict, gt: dict) -> float:
        """Calculate weighted aggregate error"""
        total_error = 0.0
        for metric, weight in ERROR_WEIGHTS.items():
            gen_val = gen.get(metric, 0)
            gt_val = gt.get(metric, 0)
            
            if gt_val == 0:
                error = 0.0 if gen_val == 0 else 1.0
            else:
                error = abs(gen_val - gt_val) / abs(gt_val)
            
            total_error += error * weight
        
        return total_error


def objective(trial: optuna.Trial, context: OptimizationContext) -> float:
    """Objective function for Optuna optimization"""

    # Get stable worker ID for this thread (not trial-number based)
    worker_id = get_worker_id()
    context.worker_id = worker_id

    # Memory monitoring (if available)
    if HAS_PSUTIL:
        import os
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

    # Suggest parameters based on their types
    params = {}

    for param_name, (low, high) in PARAM_SPACE.items():
        if param_name in ['max-place-size-for-splitting', 'merge-places-distance-meters',
                          'splitting-num-clusters-divisor', 'min-connections-per-cluster',
                          'max-connections-per-cluster', 'connection-scaling-divisor',
                          'max-total-connections', 'max-connection-distance-meters',
                          'gravity-distance-floor-meters',
                          'yes-building-sqft-per-resident', 'yes-building-sqft-per-job',
                          'min-building-area-sqft', 'max-building-area-sqft']:
            # Integer parameters
            params[param_name] = trial.suggest_int(param_name, int(low), int(high))
        else:
            # Float parameters
            params[param_name] = trial.suggest_float(param_name, low, high)

    # Run pipeline directly (no subprocess) — timed for reward signal
    t0 = time.perf_counter()
    if not context.run_pipeline(params):
        context.cleanup()  # Clean up on failure
        gc.collect()
        return float('inf')
    elapsed = time.perf_counter() - t0

    # Extract features directly from results (no file I/O)
    generated = context.extract_features()
    if not generated:
        context.cleanup()
        gc.collect()
        return float('inf')

    total_error = 0.0
    match_count = 0

    for gen in generated:
        gt_code = CITY_CODE_MAPPING.get(gen['city'], gen['city'])
        gt = context.gt_by_city.get(gt_code)

        if not gt:
            print(f"W{worker_id}: No ground truth for {gen['city']} -> {gt_code}", file=sys.stderr)
            continue

        error = context.calculate_aggregate_error(gen, gt)
        total_error += error
        match_count += 1

    if match_count == 0:
        print(f"W{worker_id}: No matching cities found", file=sys.stderr)
        return float('inf')
    
    avg_error = total_error / match_count

    # Time-based reward: penalize slow configs
    TIME_WEIGHT = 0.02   # 2% of objective from processing time
    TIME_BASELINE = 240.0 # seconds — expected normal duration
    time_penalty = max(0.0, (elapsed - TIME_BASELINE) / TIME_BASELINE) * TIME_WEIGHT
    score = avg_error + time_penalty

    # Report intermediate value for pruning
    trial.report(score, match_count)

    # Check if trial should be pruned
    if trial.should_prune():
        raise optuna.TrialPruned()

    return score


def load_ground_truth_shared(gt_path: Path) -> dict:
    """Load ground truth into shared memory for worker reuse"""
    print('Loading ground truth into shared memory...')
    
    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)
    
    # Create shared dict accessible by all workers
    manager = mp.Manager()
    shared_gt = manager.dict()
    
    for gt in ground_truth:
        shared_gt[gt['city']] = gt
    
    print(f'  Loaded {len(shared_gt)} cities into shared memory')
    return shared_gt


def main():
    """Main optimization loop"""
    parser = argparse.ArgumentParser(description='Optimize parameters using Bayesian optimization')
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of optimization trials (default: 100)')
    parser.add_argument('--n-jobs', type=int, default=4,
                       help='Number of parallel jobs (default: 4)')
    parser.add_argument('--city', type=str, default=None,
                       help='Only optimize for specific city')
    args = parser.parse_args()

    # Set global worker count for thread-local worker ID assignment
    global _n_workers
    _n_workers = args.n_jobs

    script_dir = Path(__file__).parent
    map_patcher_dir = script_dir.parent
    
    # Load ground truth into shared memory
    gt_path = script_dir / 'ground_truth_features.json'
    if not gt_path.exists():
        print('Error: Ground truth not found. Run extract_ground_truth.py first.')
        return
    
    gt_by_city = load_ground_truth_shared(gt_path)
    
    # Load current config to get places
    config_path = map_patcher_dir / 'config.js'
    config_json_path = map_patcher_dir / 'config.json'
    
    # Try to load from config.json first, fallback to parsing config.js
    if config_json_path.exists():
        with open(config_json_path, 'r') as f:
            current_config = json.load(f)
    else:
        print('Warning: config.json not found, trying to parse config.js...')
        # Simple parsing - this is fragile, user should convert to JSON
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Extract places array (very basic parsing)
        import re
        places_match = re.search(r'places:\s*(\[.*?\])', content, re.DOTALL)
        if not places_match:
            print('Error: Could not parse places from config.js')
            print('Please create config.json with your configuration.')
            return
        
        # Evaluate the places array (UNSAFE but works for this case)
        places_str = places_match.group(1)
        # Replace JavaScript syntax with Python
        places_str = places_str.replace('true', 'True').replace('false', 'False').replace('null', 'None')
        current_config = {'places': eval(places_str)}
    
    places = current_config['places']
    
    print(f"Cities to optimize: {', '.join(p['code'] for p in places)}")
    
    # Filter to target city if specified
    if args.city:
        target_places = [p for p in places 
                        if p['code'] == args.city or 
                        CITY_CODE_MAPPING.get(p['code']) == args.city]
        
        if not target_places:
            print(f"Error: No matching places found for city: {args.city}")
            return
        
        print(f"Filtering to city: {args.city}")
    else:
        target_places = places
    
    # Create optimization context with shared ground truth
    context = OptimizationContext(target_places, gt_by_city)
    
    # Preclassify buildings for all target cities (one-time cost, huge speedup per trial)
    print(f"\nEnsuring preclassified buildings exist...")
    preclassify_count = 0
    for place_dict in target_places:
        code = place_dict['code']
        raw_data_dir = MAP_PATCHER_DIR / 'raw_data' / code
        preclassified_path = raw_data_dir / 'buildings_preclassified.msgpack'
        
        if preclassified_path.exists():
            size_mb = preclassified_path.stat().st_size / 1_000_000
            print(f"  {code}: Using cached preclassified buildings ({size_mb:.1f}MB)")
        else:
            print(f"  {code}: Creating preclassified buildings cache...")
            result = ensure_preclassified_exists(code, raw_data_dir)
            if result:
                preclassify_count += 1
                print(f"  {code}: ✓ Preclassified successfully")
            else:
                print(f"  {code}: ✗ Failed - will use slower path")
    
    if preclassify_count > 0:
        print(f"  Created {preclassify_count} new preclassified caches")
    
    # Create Optuna study
    print(f"\nStarting Bayesian Optimization:")
    print(f"  Trials: {args.n_trials}")
    print(f"  Parallel jobs: {args.n_jobs}")
    print(f"  Algorithm: Tree-structured Parzen Estimator (TPE)")
    print(f"  Pruner: Median pruner (early stopping)")
    
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=0,
            interval_steps=1
        ),
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=20,  # Increase for better initial coverage
            multivariate=True,
            group=True,  # Group correlated params automatically
            seed=42,
            constant_liar=True,  # Better parallel performance
        )
    )
    
    # Seed study with known-good parameters from prior runs
    seed_count = 0

    # 1. Enqueue current config.json params (only params that exist in search space)
    seed_params = {k: current_config[k] for k in PARAM_SPACE if k in current_config}
    if seed_params:
        study.enqueue_trial(seed_params)
        seed_count += 1

    # 2. Enqueue top trials from previous optimization results
    results_path = script_dir / 'optimization_results.json'
    if results_path.exists():
        try:
            with open(results_path, 'r') as f:
                prev_results = json.load(f)
            for trial_data in prev_results.get('trials', [])[:5]:
                if trial_data.get('value') is None:
                    continue
                # Only include params that exist in current search space;
                # new params will be sampled randomly by Optuna
                trial_params = {k: v for k, v in trial_data['params'].items()
                                if k in PARAM_SPACE}
                if trial_params and trial_params != seed_params:
                    study.enqueue_trial(trial_params)
                    seed_count += 1
        except Exception as e:
            print(f"  Warning: Could not load previous results: {e}")

    if seed_count > 0:
        print(f"  Seeded {seed_count} trials from prior runs")
        missing = set(PARAM_SPACE) - set(seed_params)
        if missing:
            print(f"  New params (randomly initialized): {', '.join(sorted(missing))}")

    # Run optimization
    try:
        study.optimize(
            lambda trial: objective(trial, context),
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
    
    # Print results
    print('\n' + '=' * 80)
    print('OPTIMIZATION COMPLETE')
    print('=' * 80)
    
    if study.best_trial:
        best_params = study.best_params
        best_error = study.best_value
        
        print(f"\nBest aggregate error: {best_error * 100:.1f}%")
        print(f"Found in trial: {study.best_trial.number}")
        print("\nBest parameters:")
        for key, value in sorted(best_params.items()):
            if isinstance(value, float):
                print(f'  "{key}": {value:.4f},')
            else:
                print(f'  "{key}": {value},')
        
        # Write best config to main config.json
        config_json_path = context.map_patcher_dir / 'config.json'
        best_config = {
            **FIXED_PARAMS,
            **best_params,
            'places': places,
        }
        with open(config_json_path, 'w') as f:
            json.dump(best_config, f, indent=2)
        
        print(f"\n✓ Best config written to {config_json_path}")
        
        # Save optimization results
        results = {
            'best_params': best_params,
            'best_error': float(best_error),
            'n_trials': len(study.trials),
            'n_completed': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'trials': [
                {
                    'number': t.number,
                    'params': t.params,
                    'value': float(t.value) if t.value is not None else None,
                    'state': str(t.state)
                }
                for t in study.trials[:20]  # Save top 20
            ]
        }
        
        results_path = script_dir / 'optimization_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to {results_path}")
        
        # Print statistics
        print(f"\nTrials completed: {results['n_completed']}")
        print(f"Trials pruned: {results['n_pruned']}")
        
        if results['n_completed'] > 0:
            completed_values = [t.value for t in study.trials 
                              if t.state == optuna.trial.TrialState.COMPLETE]
            print(f"\nError statistics:")
            print(f"  Best: {min(completed_values) * 100:.1f}%")
            print(f"  Mean: {np.mean(completed_values) * 100:.1f}%")
            print(f"  Median: {np.median(completed_values) * 100:.1f}%")
            print(f"  Worst: {max(completed_values) * 100:.1f}%")
    else:
        print("\nNo valid trials found.")


if __name__ == '__main__':
    main()
