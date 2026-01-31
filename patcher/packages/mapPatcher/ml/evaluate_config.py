"""
Evaluate a config file against ground truth and report per-feature errors.

Usage: python evaluate_config.py [--config <file>] [--city CITY_CODE]
  --config <file>  Config JSON file (default: ../config.json)
  --city CODE      Only evaluate a specific city
"""

import json
import argparse
from pathlib import Path
from dataclasses import asdict

from process_data import process_place, ProcessingConfig, Place, MAP_PATCHER_DIR
from extract_generated import extract_features
from optimize_params import ERROR_WEIGHTS, CITY_CODE_MAPPING, FIXED_PARAMS


def evaluate(config_path: Path, gt_path: Path, city_filter: str = None):
    with open(config_path, 'r') as f:
        config = json.load(f)

    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)

    gt_by_city = {gt['city']: gt for gt in ground_truth}

    places = config['places']
    if city_filter:
        places = [p for p in places if p['code'] == city_filter or
                  CITY_CODE_MAPPING.get(p['code']) == city_filter]
        if not places:
            print(f"No matching places for: {city_filter}")
            return

    # Build processing config from the config file params + fixed params
    processing_params = {k: config[k] for k in config if k != 'places'}
    processing_params.update(FIXED_PARAMS)
    processing_params['places'] = [p for p in config['places']]
    processing_params['skip-buildings-index'] = True

    # Enable grid road distances if any city has a precomputed grid
    for p in places:
        grid_file = MAP_PATCHER_DIR / 'raw_data' / p['code'] / f"{p['code']}_grid_distances.npz"
        if grid_file.exists():
            processing_params['use-grid-road-distances'] = True
            print(f"Grid road distances enabled (found {grid_file.name})")
            break

    proc_config = ProcessingConfig.from_dict(processing_params)

    total_error = 0.0
    match_count = 0

    for place_dict in places:
        code = place_dict['code']
        gt_code = CITY_CODE_MAPPING.get(code, code)
        gt = gt_by_city.get(gt_code)

        if not gt:
            print(f"Skipping {code}: no ground truth for {gt_code}")
            continue

        print(f"\n{'=' * 70}")
        print(f"  {place_dict.get('name', code)} ({code})  ->  ground truth: {gt_code}")
        print(f"{'=' * 70}")

        place = Place(
            code=code,
            name=place_dict.get('name', code),
            description=place_dict.get('description', ''),
            bbox=place_dict['bbox'],
            population=place_dict.get('population', 0),
        )

        try:
            demand_data = process_place(place, proc_config)
        except Exception as e:
            print(f"  Processing failed: {e}")
            continue

        if not demand_data.get('points'):
            print(f"  No points generated")
            continue

        features = extract_features(demand_data, code)
        gen = asdict(features)

        # Print per-feature breakdown
        city_error = 0.0
        print(f"\n  {'Feature':<30} {'Weight':>6} {'Generated':>12} {'Truth':>12} {'Error':>8} {'Weighted':>8}")
        print(f"  {'-'*30} {'-'*6} {'-'*12} {'-'*12} {'-'*8} {'-'*8}")

        for metric, weight in sorted(ERROR_WEIGHTS.items(), key=lambda x: -x[1]):
            gen_val = gen.get(metric, 0)
            gt_val = gt.get(metric, 0)

            if gt_val == 0:
                error = 0.0 if gen_val == 0 else 1.0
            else:
                error = abs(gen_val - gt_val) / abs(gt_val)

            weighted = error * weight
            city_error += weighted

            # Format values compactly
            def fmt(v):
                if isinstance(v, int) or (isinstance(v, float) and v == int(v) and abs(v) > 10):
                    return f"{int(v):,}"
                return f"{v:.4f}" if abs(v) < 100 else f"{v:,.1f}"

            print(f"  {metric:<30} {weight:>6.2f} {fmt(gen_val):>12} {fmt(gt_val):>12} {error:>7.1%} {weighted:>7.3f}")

        print(f"\n  City aggregate error: {city_error:.4f} ({city_error*100:.1f}%)")
        total_error += city_error
        match_count += 1

    if match_count > 0:
        avg = total_error / match_count
        print(f"\n{'=' * 70}")
        print(f"  Overall average error: {avg:.4f} ({avg*100:.1f}%) across {match_count} cities")
        print(f"{'=' * 70}")
    else:
        print("\nNo cities matched ground truth.")


def main():
    parser = argparse.ArgumentParser(description='Evaluate config against ground truth')
    parser.add_argument('--config', type=str, default=None,
                        help='Config JSON file (default: ../config.json)')
    parser.add_argument('--city', type=str, default=None,
                        help='Only evaluate a specific city')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    map_patcher_dir = script_dir.parent

    config_path = Path(args.config) if args.config else map_patcher_dir / 'config.json'
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return

    gt_path = script_dir / 'ground_truth_features.json'
    if not gt_path.exists():
        print(f"Ground truth not found: {gt_path}")
        print("Run extract_ground_truth.py first.")
        return

    print(f"Config: {config_path}")
    print(f"Ground truth: {gt_path}")

    evaluate(config_path, gt_path, args.city)


if __name__ == '__main__':
    main()
