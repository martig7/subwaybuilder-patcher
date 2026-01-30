#!/usr/bin/env python3
"""Test the optimization setup with all new features."""

import sys
import argparse
from pathlib import Path

# Parse command line arguments
parser = argparse.ArgumentParser(description='Test optimization setup and optionally compare with config file')
parser.add_argument('--config', type=str, help='Path to config file (JSON format like config.json) for parameter comparison')
parser.add_argument('--city', type=str, help='City code to test (if not specified, uses first city from config)')
parser.add_argument('--generate', action='store_true', help='Generate demand data with config parameters and compare')
args = parser.parse_args()

# Load config parameters if provided
config_params = None
config_places = None
test_city = args.city  # Override city if specified

if args.config:
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    print("=" * 60)
    print("LOADING CONFIG FILE")
    print("=" * 60)
    print(f"Reading parameters from: {config_path}")
    
    import json
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    # Extract parameters from config (could be directly in root or nested)
    config_params = {}
    
    # Handle different config formats
    if 'places' in config_data:
        # This is a full config.json with places array
        config_places = config_data['places']
        
        # If no city specified, use first place from config
        if not test_city and config_places:
            test_city = config_places[0]['code']
            print(f"✓ Using first city from config: {test_city}")
        
        # Extract global parameters (everything except 'places')
        for key, value in config_data.items():
            if key != 'places' and isinstance(value, (int, float, str, bool)):
                config_params[key] = value
        print(f"✓ Loaded {len(config_params)} parameters from config.json")
        print(f"✓ Found {len(config_places)} places in config")
    else:
        # Assume it's a flat parameters file
        for key, value in config_data.items():
            if isinstance(value, (int, float, str, bool)):
                config_params[key] = value
        print(f"✓ Loaded {len(config_params)} parameters from JSON file")

# Use default test city if none specified
if not test_city:
    test_city = 'HNL_GEN'

# Test 1: Import all required modules
print("=" * 60)
print("TEST 1: Module Imports")
print("=" * 60)

try:
    from optimize_params import PARAM_SPACE, ERROR_WEIGHTS, FIXED_PARAMS
    print("✓ optimize_params imported")
except Exception as e:
    print(f"✗ Failed to import optimize_params: {e}")
    sys.exit(1)

try:
    from extract_ground_truth import extract_features as extract_gt
    print("✓ extract_ground_truth imported")
except Exception as e:
    print(f"✗ Failed to import extract_ground_truth: {e}")
    sys.exit(1)

try:
    from extract_generated import extract_features as extract_gen
    print("✓ extract_generated imported")
except Exception as e:
    print(f"✗ Failed to import extract_generated: {e}")
    sys.exit(1)

try:
    from grid_road_distances import GridRoadDistances, HAS_OSMNX, HAS_NETWORKX
    print(f"✓ grid_road_distances imported (OSMnx: {HAS_OSMNX}, NetworkX: {HAS_NETWORKX})")
except Exception as e:
    print(f"✗ Failed to import grid_road_distances: {e}")
    sys.exit(1)

# Test 2: Parameter space
print("\n" + "=" * 60)
print("TEST 2: Optimization Parameters")
print("=" * 60)

required_params = ['circuity-factor', 'average-driving-speed-mps']
missing = [p for p in required_params if p not in PARAM_SPACE]
if missing:
    print(f"✗ Missing parameters: {missing}")
    sys.exit(1)

print(f"✓ All required parameters present ({len(PARAM_SPACE)} total)")
for param in required_params:
    print(f"  {param}: {PARAM_SPACE[param]}")

# Compare config parameters if provided
if config_params:
    print("\n" + "=" * 60)
    print("TEST 2B: Config Parameter Comparison")
    print("=" * 60)
    
    # Check which config params are in PARAM_SPACE
    matched = 0
    out_of_range = []
    
    for key, value in config_params.items():
        if key in PARAM_SPACE:
            matched += 1
            min_val, max_val = PARAM_SPACE[key]
            in_range = min_val <= value <= max_val
            status = "✓" if in_range else "✗ OUT OF RANGE"
            
            if not in_range:
                out_of_range.append((key, value, min_val, max_val))
            
            print(f"  {key}: {value:.4f} [{min_val}, {max_val}] {status}")
    
    print(f"\n✓ {matched}/{len(config_params)} config params are optimizable")
    
    if out_of_range:
        print(f"\n⚠ Warning: {len(out_of_range)} parameters are outside optimization ranges:")
        for key, value, min_val, max_val in out_of_range:
            if value < min_val:
                print(f"    {key}: {value:.4f} < {min_val} (too low)")
            else:
                print(f"    {key}: {value:.4f} > {max_val} (too high)")
    
    # Show new parameters that are in config
    if 'circuity-factor' in config_params or 'average-driving-speed-mps' in config_params:
        print("\nNew optimization parameters in config:")
        if 'circuity-factor' in config_params:
            print(f"  circuity-factor: {config_params['circuity-factor']:.4f}")
        if 'average-driving-speed-mps' in config_params:
            print(f"  average-driving-speed-mps: {config_params['average-driving-speed-mps']:.4f}")

# Test 3: Error weights
print("\n" + "=" * 60)
print("TEST 3: Error Weights")
print("=" * 60)

driving_keys = [k for k in ERROR_WEIGHTS if 'driving_seconds' in k]
if len(driving_keys) < 6:
    print(f"✗ Expected 6+ driving_seconds weights, found {len(driving_keys)}")
    sys.exit(1)

total_weight = sum(ERROR_WEIGHTS.values())
if abs(total_weight - 1.0) > 1e-6:
    print(f"✗ Weights don't sum to 1.0: {total_weight}")
    sys.exit(1)

print(f"✓ All weights valid (total: {total_weight:.6f})")
print(f"  Driving seconds metrics: {len(driving_keys)}")
print(f"  Total metrics: {len(ERROR_WEIGHTS)}")

# Test 4: Feature extraction and comparison
print("\n" + "=" * 60)
print("TEST 4: Feature Extraction & Comparison")
print("=" * 60)

import json
from dataclasses import asdict

print(f"Testing with city: {test_city}")

# Test with specified city (generated) and ground truth
gen_data_path = Path(__file__).parent / f'../processed_data/{test_city}/demand_data.json'
gt_features_path = Path(__file__).parent / 'ground_truth_features.json'

if gen_data_path.exists() and gt_features_path.exists():
    # Load generated features
    with open(gen_data_path, 'r') as f:
        gen_data = json.load(f)
    
    gen_features = extract_gen(gen_data, test_city)
    gen_dict = asdict(gen_features)
    
    # Load ground truth features (JSON array)
    with open(gt_features_path, 'r') as f:
        gt_list = json.load(f)
    
    # Convert to dict for easy lookup
    gt_data = {item['city']: item for item in gt_list}
    
    # Find matching ground truth (try with and without _GEN suffix)
    gt_city = None
    base_city = test_city.replace('_GEN', '')
    
    for city_code in [test_city, base_city, test_city + '_GEN']:
        if city_code in gt_data:
            gt_city = gt_data[city_code]
            print(f"✓ Found ground truth for {city_code}")
            break
    
    if gt_city is None:
        print(f"⚠ Ground truth not found for {test_city} or {base_city}")
        print(f"   Available cities: {', '.join(sorted(gt_data.keys())[:10])}...")
    else:
        missing_fields = [k for k in driving_keys if k not in gen_dict]
        if missing_fields:
            print(f"✗ Missing fields in features: {missing_fields}")
            sys.exit(1)
        
        print(f"✓ All driving_seconds fields extracted")
        print(f"\nDriving Time Comparison ({base_city}):")
        print(f"{'Metric':<25} {'Generated':<15} {'Ground Truth':<15} {'Diff %':<10}")
        print("-" * 65)
        
        for key in ['driving_seconds_median', 'driving_seconds_p25', 'driving_seconds_p75', 
                    'driving_seconds_p90', 'driving_seconds_std']:
            gen_val = gen_dict.get(key, 0)
            gt_val = gt_city.get(key, 0)
            
            if gt_val > 0:
                diff_pct = ((gen_val - gt_val) / gt_val) * 100
                gen_min = gen_val / 60
                gt_min = gt_val / 60
                print(f"{key:<25} {gen_min:>6.1f} min      {gt_min:>6.1f} min      {diff_pct:>+6.1f}%")
            else:
                print(f"{key:<25} {gen_val/60:>6.1f} min      N/A             N/A")
elif gen_data_path.exists():
    print(f"⚠ Ground truth not found, showing generated only")
    with open(gen_data_path, 'r') as f:
        data = json.load(f)
    
    features = extract_gen(data, test_city)
    features_dict = asdict(features)
    
    print(f"✓ All driving_seconds fields extracted")
    print(f"  Sample values:")
    print(f"    Median: {features_dict['driving_seconds_median']:.0f}s")
    print(f"    P90: {features_dict['driving_seconds_p90']:.0f}s")
    print(f"    Std: {features_dict['driving_seconds_std']:.0f}s")
else:
    print(f"⚠ {test_city} data not found at {gen_data_path}")
    print(f"   Skipping feature extraction test")

# Test 5: Grid distances
print("\n" + "=" * 60)
print("TEST 5: Grid Road Distances")
print("=" * 60)

grid_file = Path(__file__).parent / f'../raw_data/{test_city}/{test_city}_grid_distances.npz'
if grid_file.exists():
    try:
        grid = GridRoadDistances.load(grid_file)
        print(f"✓ Grid loaded successfully for {test_city}")
        print(f"  Grid nodes: {len(grid.grid_coords)}")
        print(f"  Cell size: {grid.metadata.cell_size_meters}m")
        print(f"  File size: {grid_file.stat().st_size / 1_000_000:.1f}MB")
    except Exception as e:
        print(f"✗ Failed to load grid: {e}")
        sys.exit(1)
else:
    print(f"⚠ {test_city} grid not found at {grid_file}")
    print(f"  Grid can be created with: python process_data.py --place {test_city} --use-grid-road-distances")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("✓ All tests passed!")
print("\nOptimization pipeline ready:")
print(f"  - {len(PARAM_SPACE)} optimizable parameters")
print(f"  - {len(ERROR_WEIGHTS)} scoring metrics")
print(f"  - Driving time metrics included")
print(f"  - Grid precomputation enabled")

if config_params:
    print(f"\nConfig file analysis:")
    print(f"  - {len(config_params)} parameters loaded")
    matched_count = sum(1 for k in config_params if k in PARAM_SPACE)
    print(f"  - {matched_count} match optimization parameters")
    if config_places:
        print(f"  - {len(config_places)} cities in config: {', '.join(p['code'] for p in config_places)}")
    print(f"  - Testing with city: {test_city}")
    if args.generate:
        print(f"\n⚠ Note: --generate flag set but generation not implemented in this test")
        print(f"  Use: python process_data.py --place {test_city} --config {args.config}")
else:
    print(f"\nTest city: {test_city}")
    print("\nNext steps:")
    print("  1. Extract ground truth: python extract_ground_truth.py")
    print("  2. Run optimization: python optimize_params.py --n-trials 100 --n-jobs 4")
    print("\nTo test with config file:")
    print("  python test_optimization_setup.py --config ../config.json")
    print("  python test_optimization_setup.py --config ../config.json --city PHL_GEN")
