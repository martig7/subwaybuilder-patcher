"""
Fit a piecewise driving speed function from ground truth data.

Learns effective_speed(road_distance) from ground truth connections.
The model: flat speed below a breakpoint, power curve above.

  if distance <= breakpoint:
      speed = floor_speed
  else:
      speed = a * distance^b

Usage: python fit_speed_curve.py [--samples N] [--plot]
  --samples N   Max connections to sample per city (default: 5000)
  --plot        Show a matplotlib scatter + fit plot
"""

import json
import gzip
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
import argparse
import sys
import os


def get_appdata_path() -> Path:
    if sys.platform == 'win32':
        return Path(os.environ['APPDATA'])
    elif sys.platform == 'darwin':
        return Path.home() / 'Library' / 'Application Support'
    else:
        return Path.home() / '.config'


def piecewise_speed(distance, floor_speed, breakpoint, a, b):
    """Piecewise model: flat floor below breakpoint, power curve above."""
    distance = np.asarray(distance, dtype=float)
    result = np.full_like(distance, floor_speed)
    above = distance > breakpoint
    result[above] = a * np.power(distance[above], b)
    return result


def fit_piecewise(distances, speeds):
    """Fit the piecewise model by minimizing RMSE on binned medians.

    Uses log-spaced bins to reduce 200k samples to ~30 representative
    points, weighted by bin sample count. Much faster and less noisy.
    """
    # Bin the data into log-spaced bins for fast fitting
    bins = np.logspace(np.log10(max(distances.min(), 1)), np.log10(distances.max()), 40)
    bin_centers = []
    bin_medians = []
    bin_weights = []
    for i in range(len(bins) - 1):
        mask = (distances >= bins[i]) & (distances < bins[i+1])
        count = mask.sum()
        if count > 10:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            bin_medians.append(np.median(speeds[mask]))
            bin_weights.append(np.sqrt(count))  # Weight by sqrt(n)

    bin_centers = np.array(bin_centers)
    bin_medians = np.array(bin_medians)
    bin_weights = np.array(bin_weights)
    bin_weights /= bin_weights.sum()  # Normalize

    def objective(params):
        floor_speed, breakpoint, a, b = params
        if floor_speed <= 0 or breakpoint <= 0 or a <= 0 or b <= 0 or b > 1:
            return 1e10
        predicted = piecewise_speed(bin_centers, floor_speed, breakpoint, a, b)
        # Ensure continuity: at breakpoint, power curve should equal floor_speed
        continuity_error = (a * breakpoint**b - floor_speed)**2 * 100
        return np.sqrt(np.sum(bin_weights * (bin_medians - predicted)**2)) + continuity_error

    best_result = None
    best_cost = float('inf')

    for bp_init in [1500, 3000, 5000]:
        for floor_init in [7.5, 8.5]:
            try:
                result = minimize(
                    objective,
                    x0=[floor_init, bp_init, 0.5, 0.3],
                    method='Nelder-Mead',
                    options={'maxiter': 10000, 'xatol': 1e-4, 'fatol': 1e-4}
                )
                if result.fun < best_cost:
                    best_cost = result.fun
                    best_result = result
            except Exception:
                continue

    return best_result


def main():
    parser = argparse.ArgumentParser(description='Fit driving speed curve from ground truth')
    parser.add_argument('--samples', type=int, default=5000,
                        help='Max connections per city (default: 5000)')
    parser.add_argument('--plot', action='store_true',
                        help='Show scatter + fit plot')
    args = parser.parse_args()

    appdata_path = get_appdata_path()
    game_data_path = appdata_path / 'metro-maker4' / 'cities' / 'data'

    if not game_data_path.exists():
        print(f"Game data not found at {game_data_path}")
        return

    # Collect (distance, seconds) pairs from all cities
    all_distances = []
    all_seconds = []

    cities_dir = game_data_path
    city_codes = sorted([d.name for d in cities_dir.iterdir() if d.is_dir()])

    for city_code in city_codes:
        demand_file = game_data_path / city_code / 'demand_data.json.gz'
        if not demand_file.exists():
            continue

        with gzip.open(demand_file, 'rt', encoding='utf-8') as f:
            data = json.load(f)

        pops = data.get('pops', [])
        if not pops:
            continue

        # Sample connections
        rng = np.random.default_rng(42)
        if len(pops) > args.samples:
            indices = rng.choice(len(pops), args.samples, replace=False)
            sampled = [pops[i] for i in indices]
        else:
            sampled = pops

        distances = np.array([p['drivingDistance'] for p in sampled], dtype=float)
        seconds = np.array([p['drivingSeconds'] for p in sampled], dtype=float)

        # Filter out zero/invalid entries
        valid = (distances > 0) & (seconds > 0)
        distances = distances[valid]
        seconds = seconds[valid]

        all_distances.append(distances)
        all_seconds.append(seconds)

        print(f"  {city_code}: {len(distances)} connections, "
              f"median distance {np.median(distances):.0f}m, "
              f"median speed {np.median(distances/seconds):.1f} m/s")

    all_distances = np.concatenate(all_distances)
    all_seconds = np.concatenate(all_seconds)
    all_speeds = all_distances / all_seconds

    print(f"\nTotal samples: {len(all_distances)}")
    print(f"Distance range: {all_distances.min():.0f} - {all_distances.max():.0f} m")
    print(f"Speed range: {all_speeds.min():.2f} - {all_speeds.max():.2f} m/s "
          f"({all_speeds.min()*3.6:.1f} - {all_speeds.max()*3.6:.1f} km/h)")
    print(f"Median speed: {np.median(all_speeds):.2f} m/s ({np.median(all_speeds)*3.6:.1f} km/h)")

    # Fit piecewise model
    print(f"\nFitting piecewise model...")
    result = fit_piecewise(all_distances, all_speeds)

    if result is None:
        print("Fitting failed!")
        return

    floor_speed, breakpoint, a, b = result.x
    predicted = piecewise_speed(all_distances, floor_speed, breakpoint, a, b)
    residuals = all_speeds - predicted
    rmse = np.sqrt(np.mean(residuals**2))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((all_speeds - all_speeds.mean())**2)
    r2 = 1 - ss_res / ss_tot

    print(f"\nPiecewise model:")
    print(f"  if distance <= {breakpoint:.0f}m:")
    print(f"      speed = {floor_speed:.4f} m/s ({floor_speed*3.6:.1f} km/h)")
    print(f"  else:")
    print(f"      speed = {a:.6f} * distance^{b:.4f}")
    print(f"  RMSE: {rmse:.4f} m/s, R²: {r2:.4f}")
    print(f"  Continuity check at breakpoint: "
          f"floor={floor_speed:.2f}, curve={a * breakpoint**b:.2f} m/s")

    # Comparison table
    print(f"\n{'Distance':>10}  {'Flat 8.33':>10}  {'Piecewise':>10}  {'GT median':>10}")
    print(f"{'':>10}  {'(m/s)':>10}  {'(m/s)':>10}  {'(m/s)':>10}")
    print("-" * 50)
    for dist in [500, 1000, 2000, 3000, 5000, 10000, 20000, 50000]:
        mask = (all_distances > dist * 0.8) & (all_distances < dist * 1.2)
        gt_median = np.median(all_speeds[mask]) if mask.sum() > 10 else float('nan')
        pw_val = float(piecewise_speed(np.array([dist]), floor_speed, breakpoint, a, b)[0])
        print(f"{dist:>8}m  {8.33:>10.2f}  {pw_val:>10.2f}  {gt_median:>10.2f}")

    # Compute error stats for flat vs piecewise
    flat_seconds = all_distances / 8.33
    pw_seconds = all_distances / predicted
    flat_error = np.abs(flat_seconds - all_seconds) / all_seconds
    pw_error = np.abs(pw_seconds - all_seconds) / all_seconds
    print(f"\nMedian absolute error (driving seconds):")
    print(f"  Flat 8.33 m/s: {np.median(flat_error)*100:.1f}%")
    print(f"  Piecewise:     {np.median(pw_error)*100:.1f}%")

    # Save coefficients
    script_dir = Path(__file__).parent
    output = {
        'model': 'piecewise',
        'description': 'if distance <= breakpoint: speed = floor_speed; else: speed = a * distance^b',
        'floor_speed_mps': float(floor_speed),
        'breakpoint_m': float(breakpoint),
        'a': float(a),
        'b': float(b),
        'rmse': float(rmse),
        'r2': float(r2),
        'total_samples': int(len(all_distances)),
    }

    output_path = script_dir / 'speed_curve.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved coefficients to {output_path}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Left: speed vs distance scatter
            ax = axes[0]
            ax.scatter(all_distances / 1000, all_speeds, alpha=0.02, s=1, color='gray')

            # Bin medians
            bins = np.logspace(np.log10(all_distances.min()), np.log10(all_distances.max()), 30)
            bin_centers = []
            bin_medians = []
            for i in range(len(bins) - 1):
                mask = (all_distances >= bins[i]) & (all_distances < bins[i+1])
                if mask.sum() > 20:
                    bin_centers.append((bins[i] + bins[i+1]) / 2 / 1000)
                    bin_medians.append(np.median(all_speeds[mask]))
            ax.plot(bin_centers, bin_medians, 'ko-', label='GT bin medians', markersize=4)

            x_fit = np.linspace(all_distances.min(), all_distances.max(), 500)
            pw_fit = piecewise_speed(x_fit, floor_speed, breakpoint, a, b)
            ax.plot(x_fit / 1000, pw_fit, 'r-', label='Piecewise model', linewidth=2)
            ax.axhline(y=8.33, color='green', linestyle=':', label='Flat 8.33 m/s')
            ax.axvline(x=breakpoint / 1000, color='orange', linestyle='--', alpha=0.5,
                       label=f'Breakpoint ({breakpoint:.0f}m)')

            ax.set_xlabel('Road Distance (km)')
            ax.set_ylabel('Effective Speed (m/s)')
            ax.set_title('Effective Driving Speed vs Road Distance')
            ax.legend()
            ax.set_xscale('log')

            # Right: driving seconds error comparison
            ax = axes[1]
            pw_seconds_all = all_distances / piecewise_speed(all_distances, floor_speed, breakpoint, a, b)
            flat_seconds_all = all_distances / 8.33
            ax.scatter(all_distances / 1000,
                       (pw_seconds_all - all_seconds) / all_seconds * 100,
                       alpha=0.02, s=1, color='red', label='Piecewise error')
            ax.scatter(all_distances / 1000,
                       (flat_seconds_all - all_seconds) / all_seconds * 100,
                       alpha=0.02, s=1, color='green', label='Flat speed error')
            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.set_xlabel('Road Distance (km)')
            ax.set_ylabel('Driving Seconds Error (%)')
            ax.set_title('Prediction Error: Flat vs Piecewise')
            ax.legend()
            ax.set_xscale('log')
            ax.set_ylim(-100, 100)

            plt.tight_layout()
            plt.show()
        except ImportError:
            print("matplotlib not installed, skipping plot")


if __name__ == '__main__':
    main()
