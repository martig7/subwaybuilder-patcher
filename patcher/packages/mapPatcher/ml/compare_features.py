"""
Compare Features

Compares generated features against ground truth features
and computes error metrics for each city and overall.

Usage: python compare_features.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict


# Map generated city codes to ground truth city codes
CITY_CODE_MAPPING = {
    'HNL_GEN': 'HNL',
    'CHI_GEN': 'CHI',
    'SF_GEN': 'SF',
    'BOS_GEN': 'BOS',
    'DC_GEN': 'DC',
    'ATL_GEN': 'ATL',
    'SEA_GEN': 'SEA',
    'DEN_GEN': 'DEN',
}

# Metrics to compare and their weights in the aggregate error
METRIC_CONFIG = {
    # Stage 1: Total Population/Jobs (weight: 0.25 total) - CRITICAL for map scale
    'total_population': {'weight': 0.15, 'display_name': 'Total Population'},
    'total_jobs': {'weight': 0.10, 'display_name': 'Total Jobs'},
    
    # Stage 2: Cluster Distribution (weight: 0.35 total)
    'cluster_count': {'weight': 0.08, 'display_name': 'Cluster Count'},
    'size_mean': {'weight': 0.05, 'display_name': 'Size Mean'},
    'size_median': {'weight': 0.05, 'display_name': 'Size Median'},
    'size_max': {'weight': 0.10, 'display_name': 'Size Max'},  # Critical: prevents mega-clusters
    'size_std': {'weight': 0.02, 'display_name': 'Size Std Dev'},
    'nn_dist_median': {'weight': 0.05, 'display_name': 'NN Distance Median'},
    
    # Stage 3: Connection Distribution (weight: 0.40 total)
    'connections_per_cluster': {'weight': 0.12, 'display_name': 'Connections/Cluster'},
    'conn_dist_median': {'weight': 0.10, 'display_name': 'Conn Distance Median'},
    'conn_size_mean': {'weight': 0.08, 'display_name': 'Conn Size Mean'},
    'graph_density': {'weight': 0.10, 'display_name': 'Graph Density'},
}


def relative_error(generated: float, ground_truth: float) -> float:
    """Calculate relative error between generated and ground truth"""
    if ground_truth == 0:
        return 0.0 if generated == 0 else 1.0
    return abs(generated - ground_truth) / abs(ground_truth)


def calculate_aggregate_error(gen: Dict, gt: Dict) -> float:
    """Calculate weighted aggregate error for a city comparison"""
    total_error = 0.0
    for metric, config in METRIC_CONFIG.items():
        error = relative_error(gen[metric], gt[metric])
        total_error += error * config['weight']
    return total_error


def main():
    """Compare generated and ground truth features"""
    script_dir = Path(__file__).parent
    
    ground_truth_path = script_dir / 'ground_truth_features.json'
    generated_path = script_dir / 'generated_features.json'
    
    print('Loading feature files...')
    
    # Load files
    if not ground_truth_path.exists():
        print(f"Error: Ground truth features not found at {ground_truth_path}")
        print("Run: python extract_ground_truth.py")
        return
    
    if not generated_path.exists():
        print(f"Error: Generated features not found at {generated_path}")
        print("Run: python extract_generated.py")
        return
    
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    with open(generated_path, 'r') as f:
        generated = json.load(f)
    
    print(f"Ground truth: {len(ground_truth)} cities")
    print(f"Generated: {len(generated)} cities")
    
    # Index ground truth by city code
    gt_by_city = {gt['city']: gt for gt in ground_truth}
    
    # Find matching cities (using code mapping)
    matching_cities = [g for g in generated 
                      if CITY_CODE_MAPPING.get(g['city'], g['city']) in gt_by_city]
    
    print(f"Matching cities: {len(matching_cities)}")
    
    if not matching_cities:
        print("\nNo matching cities found between generated and ground truth data.")
        print("Generated cities:", [g['city'] for g in generated])
        print("Ground truth cities:", list(gt_by_city.keys()))
        return
    
    # Compare each matching city
    comparisons = []
    
    for gen in matching_cities:
        gt_code = CITY_CODE_MAPPING.get(gen['city'], gen['city'])
        gt = gt_by_city[gt_code]
        
        errors = {}
        aggregate_error = 0.0
        
        for metric, config in METRIC_CONFIG.items():
            gen_value = gen[metric]
            gt_value = gt[metric]
            error = relative_error(gen_value, gt_value)
            
            errors[metric] = {
                'generated': gen_value,
                'ground_truth': gt_value,
                'relative_error': error,
                'percent_error': f"{error * 100:.1f}%",
            }
            
            aggregate_error += error * config['weight']
        
        comparisons.append({
            'city': gen['city'],
            'gt_city': gt_code,
            'errors': errors,
            'aggregate_error': aggregate_error,
        })
    
    # Sort by aggregate error
    comparisons.sort(key=lambda x: x['aggregate_error'])
    
    # Print per-city results
    print('\n' + '=' * 80)
    print('PER-CITY COMPARISON')
    print('=' * 80)
    
    for comp in comparisons:
        gt_label = f" → {comp['gt_city']}" if comp['city'] != comp['gt_city'] else ''
        print(f"\n{comp['city']}{gt_label} (Aggregate Error: {comp['aggregate_error'] * 100:.1f}%)")
        print('-' * 60)
        
        for metric, config in METRIC_CONFIG.items():
            e = comp['errors'][metric]
            status = 'OK' if e['relative_error'] < 0.25 else '~~' if e['relative_error'] < 0.50 else 'XX'
            
            gen_val = f"{e['generated']:.2f}".rjust(10)
            gt_val = f"{e['ground_truth']:.2f}".rjust(10)
            err_val = e['percent_error'].rjust(7)
            
            print(f"  {status} {config['display_name']:25} Gen: {gen_val} | GT: {gt_val} | Err: {err_val}")
    
    # Calculate overall metrics
    print('\n' + '=' * 80)
    print('OVERALL METRICS')
    print('=' * 80)
    
    overall_errors = {}
    for metric in METRIC_CONFIG.keys():
        errors = [c['errors'][metric]['relative_error'] for c in comparisons]
        overall_errors[metric] = {
            'mean': np.mean(errors),
            'median': np.median(errors),
            'max': np.max(errors),
            'min': np.min(errors),
        }
    
    print('\nMetric                       Mean Err    Median Err    Max Err')
    print('-' * 70)
    
    for metric, config in METRIC_CONFIG.items():
        e = overall_errors[metric]
        status = 'OK' if e['mean'] < 0.25 else '~~' if e['mean'] < 0.50 else 'XX'
        print(f"{status} {config['display_name']:25} {e['mean'] * 100:8.1f}%    "
              f"{e['median'] * 100:8.1f}%    {e['max'] * 100:8.1f}%")
    
    overall_aggregate_error = np.mean([c['aggregate_error'] for c in comparisons])
    print('-' * 70)
    print(f"AGGREGATE ERROR:             {overall_aggregate_error * 100:8.1f}%")
    
    # Identify worst metrics
    print('\n' + '=' * 80)
    print('RECOMMENDATIONS')
    print('=' * 80)
    
    sorted_metrics = sorted(
        [(metric, e['mean'], METRIC_CONFIG[metric]) 
         for metric, e in overall_errors.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    print('\nMetrics to focus on (highest error first):')
    for i, (metric, error, config) in enumerate(sorted_metrics[:3], 1):
        print(f"  {i}. {config['display_name']} ({error * 100:.1f}% avg error)")
    
    # Save comparison results
    results_path = script_dir / 'comparison_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'comparisons': comparisons,
            'overall_errors': {k: {kk: float(vv) for kk, vv in v.items()} 
                             for k, v in overall_errors.items()},
            'overall_aggregate_error': float(overall_aggregate_error),
            'timestamp': pd.Timestamp.now().isoformat(),
        }, f, indent=2)
    
    print(f"\nSaved detailed results to: {results_path}")
    
    # Return success/failure based on aggregate error
    if overall_aggregate_error < 0.25:
        print('\nOK SUCCESS: Aggregate error is below 25% target')
    else:
        print('\nXX NEEDS IMPROVEMENT: Aggregate error exceeds 25% target')


if __name__ == '__main__':
    main()
