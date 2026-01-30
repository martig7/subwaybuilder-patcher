#!/usr/bin/env python3
"""Test optimization configuration"""

from optimize_params import PARAM_SPACE, ERROR_WEIGHTS, FIXED_PARAMS

print('New optimization parameters:')
for param in ['circuity-factor', 'average-driving-speed-mps']:
    if param in PARAM_SPACE:
        print(f'  {param}: {PARAM_SPACE[param]}')
    else:
        print(f'  {param}: NOT FOUND')

print('\nDriving seconds weights:')
total_driving_weight = 0
for key in sorted(ERROR_WEIGHTS.keys()):
    if 'driving_seconds' in key:
        weight = ERROR_WEIGHTS[key]
        total_driving_weight += weight
        print(f'  {key}: {weight:.4f}')

print(f'\nTotal driving seconds weight: {total_driving_weight:.4f}')
print(f'Total ERROR_WEIGHTS: {sum(ERROR_WEIGHTS.values()):.6f}')

print('\nFixed params for grid usage:')
use_grid = FIXED_PARAMS.get('use-grid-road-distances', 'NOT SET')
print(f'  use-grid-road-distances: {use_grid}')
