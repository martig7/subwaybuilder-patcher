"""
Audit building types in raw OSM data.

Lists all building types found, split into included/excluded
based on process_data.py classification logic.

Usage:
    python audit_building_types.py LON_GEN
"""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from process_data import SQFT_PER_POPULATION, SQFT_PER_JOB, load_data_file, MAP_PATCHER_DIR


def get_place_config(code: str, config_file: str = 'config.json') -> dict | None:
    """Look up a place by code in a config file."""
    config_path = MAP_PATCHER_DIR / config_file
    if not config_path.exists():
        return None
    with open(config_path, 'r') as f:
        config = json.load(f)
    for place in config.get('places', []):
        if place['code'] == code:
            return place
    return None


def ensure_raw_data(code: str, config_file: str = 'config.json') -> Path:
    """Ensure raw data exists, downloading if needed. Returns raw_data dir."""
    raw_dir = MAP_PATCHER_DIR / 'raw_data' / code
    buildings_json = raw_dir / 'buildings.json'
    buildings_msgpack = raw_dir / 'buildings.msgpack'

    if buildings_json.exists() or buildings_msgpack.exists():
        return raw_dir

    print(f"Raw data not found for {code}, downloading...")
    subprocess.run(
        ['node', '--max-old-space-size=8096', 'download_data.js', '--config', config_file],
        cwd=MAP_PATCHER_DIR,
        check=True,
    )

    if not buildings_json.exists() and not buildings_msgpack.exists():
        print(f"Error: download_data.js did not produce buildings data for {code}")
        sys.exit(1)

    return raw_dir


def classify_type(building_type: str) -> tuple[str, str]:
    """Return (status, category) for a building type."""
    if building_type == 'yes':
        return 'included', 'mixed'
    if building_type in SQFT_PER_POPULATION:
        return 'included', 'residential'
    if building_type in SQFT_PER_JOB:
        return 'included', 'commercial'
    return 'excluded', 'excluded'


def main():
    if len(sys.argv) < 2:
        print("Usage: python audit_building_types.py <CITY_CODE> [config_file]")
        print("Example: python audit_building_types.py LON_GEN")
        print("         python audit_building_types.py LON_GEN my_config.json")
        sys.exit(1)

    code = sys.argv[1]
    config_file = sys.argv[2] if len(sys.argv) > 2 else 'config.json'
    place = get_place_config(code, config_file)
    if not place:
        print(f"Warning: {code} not found in config.json, proceeding with raw data lookup only")

    raw_dir = ensure_raw_data(code, config_file)

    print(f"Loading buildings for {code}...")
    buildings = load_data_file(raw_dir / 'buildings.json')
    print(f"Loaded {len(buildings)} building elements")

    # Tally building types
    type_counts = Counter()
    for b in buildings:
        bt = b.get('tags', {}).get('building')
        if bt:
            type_counts[bt] += 1
        else:
            type_counts['<no building tag>'] += 1

    # Classify and sort
    rows = []
    for bt, count in type_counts.items():
        status, category = classify_type(bt)
        rows.append((bt, count, status, category))

    included = sorted([r for r in rows if r[2] == 'included'], key=lambda r: -r[1])
    excluded = sorted([r for r in rows if r[2] == 'excluded'], key=lambda r: -r[1])

    # Print table
    total = sum(r[1] for r in rows)
    inc_count = sum(r[1] for r in included)
    exc_count = sum(r[1] for r in excluded)

    col_w = max(len(r[0]) for r in rows) + 2
    header = f"{'Building Type':<{col_w}} {'Count':>8}  {'Status':<10} {'Category':<12}"
    sep = '-' * len(header)

    print(f"\n=== Building Type Audit: {code} ===\n")
    print(header)
    print(sep)

    for r in included:
        print(f"{r[0]:<{col_w}} {r[1]:>8}  {r[2]:<10} {r[3]:<12}")

    if excluded:
        print(sep)
        for r in excluded:
            print(f"{r[0]:<{col_w}} {r[1]:>8}  {r[2]:<10} {r[3]:<12}")

    print(sep)
    print(f"\nTotal buildings:  {total:>8}")
    print(f"Included:         {inc_count:>8} ({inc_count/total*100:.1f}%)")
    print(f"Excluded:         {exc_count:>8} ({exc_count/total*100:.1f}%)")


if __name__ == '__main__':
    main()
