"""
Extract Ground Truth Features

Reads demand_data.json.gz from the game's AppData folder for each city
and computes metrics for comparison with generated data.

Usage: python extract_ground_truth.py
"""

import json
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict
import os
import sys


# Available cities in ground truth data
CITIES = [
    'ATL', 'AUS', 'BAL', 'BIR', 'BOS', 'CHI', 'CIN', 'CLE', 'CLT', 'COL',
    'DAL', 'DC', 'DEN', 'DET', 'HNL', 'HOU', 'IND', 'LIV', 'LON', 'MAN',
    'MIA', 'MKE', 'MSP', 'NEW', 'NYC', 'PDX', 'PHL', 'PHX', 'PIT', 'SAN',
    'SEA', 'SF', 'SLC', 'STL'
]


@dataclass
class CityFeatures:
    """Statistical features for a city's ground truth data"""
    city: str
    
    # Stage 2: Cluster Distribution
    cluster_count: int
    size_mean: float
    size_median: float
    size_std: float
    size_p90: float
    size_min: float
    size_max: float
    total_population: int
    total_jobs: int
    jobs_ratio: float
    nn_dist_mean: float
    nn_dist_median: float
    nn_dist_p90: float
    area_km2: float
    density: float
    
    # Stage 3: Connection Distribution
    connection_count: int
    connections_per_cluster: float
    conn_dist_mean: float
    conn_dist_median: float
    conn_dist_p90: float
    conn_size_mean: float
    conn_size_median: float
    conn_size_p90: float
    graph_density: float


def get_appdata_path() -> Path:
    """Get AppData path based on OS"""
    if sys.platform == 'win32':
        return Path(os.environ['APPDATA'])
    elif sys.platform == 'darwin':
        return Path.home() / 'Library' / 'Application Support'
    else:
        return Path.home() / '.config'


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate distance between two points in meters using Haversine formula"""
    R = 6371000  # Earth radius in meters
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat / 2) ** 2 + \
        np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c


def calculate_area_km2(points: List[Dict]) -> float:
    """Calculate bounding box area in km²"""
    if not points:
        return 0.0
    
    lons = np.array([p['location'][0] for p in points])
    lats = np.array([p['location'][1] for p in points])
    
    min_lon, max_lon = lons.min(), lons.max()
    min_lat, max_lat = lats.min(), lats.max()
    
    width = haversine_distance(min_lon, (min_lat + max_lat) / 2, 
                               max_lon, (min_lat + max_lat) / 2)
    height = haversine_distance((min_lon + max_lon) / 2, min_lat,
                                (min_lon + max_lon) / 2, max_lat)
    
    return (width * height) / 1_000_000  # Convert to km²


def calculate_nearest_neighbor_distances(points: List[Dict]) -> np.ndarray:
    """Calculate nearest neighbor distances for all points using vectorization"""
    if len(points) <= 1:
        return np.array([])
    
    # Extract coordinates
    coords = np.array([[p['location'][0], p['location'][1]] for p in points])
    
    # Vectorized haversine distance matrix calculation
    lons = coords[:, 0]
    lats = coords[:, 1]
    
    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)
    
    # Broadcasting for pairwise distances
    dlat = lat_rad[:, None] - lat_rad[None, :]
    dlon = lon_rad[:, None] - lon_rad[None, :]
    
    a = np.sin(dlat / 2) ** 2 + \
        np.cos(lat_rad[:, None]) * np.cos(lat_rad[None, :]) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distances = 6371000 * c  # Earth radius in meters
    
    # Set diagonal to infinity to exclude self-distances
    np.fill_diagonal(distances, np.inf)
    
    # Get minimum distance for each point
    min_distances = distances.min(axis=1)
    
    return min_distances[min_distances != np.inf]


def extract_features(data: Dict, city_code: str) -> CityFeatures:
    """Extract statistical features from demand data"""
    points = data['points']
    pops = data['pops']
    
    if not points:
        raise ValueError(f"No points found for {city_code}")
    
    # Stage 2: Cluster Distribution Metrics
    sizes = np.array([p.get('jobs', 0) + p.get('residents', 0) for p in points])
    total_jobs = sum(p.get('jobs', 0) for p in points)
    total_residents = sum(p.get('residents', 0) for p in points)
    total_size = total_jobs + total_residents
    
    area_km2 = calculate_area_km2(points)
    nn_distances = calculate_nearest_neighbor_distances(points)
    
    # Stage 3: Connection Distribution Metrics
    conn_distances = np.array([p.get('drivingDistance', 0) for p in pops])
    conn_sizes = np.array([p.get('size', 0) for p in pops])
    
    # Count unique edges (residence-job pairs)
    unique_edges = set((p['residenceId'], p['jobId']) for p in pops)
    
    # Possible edges = points with residents * points with jobs
    residential_points = sum(1 for p in points if p.get('residents', 0) > 0)
    job_points = sum(1 for p in points if p.get('jobs', 0) > 0)
    possible_edges = residential_points * job_points
    
    return CityFeatures(
        city=city_code,
        
        # Stage 2: Cluster Distribution
        cluster_count=len(points),
        size_mean=float(sizes.mean()) if len(sizes) > 0 else 0.0,
        size_median=float(np.median(sizes)) if len(sizes) > 0 else 0.0,
        size_std=float(sizes.std()) if len(sizes) > 0 else 0.0,
        size_p90=float(np.percentile(sizes, 90)) if len(sizes) > 0 else 0.0,
        size_min=float(sizes.min()) if len(sizes) > 0 else 0.0,
        size_max=float(sizes.max()) if len(sizes) > 0 else 0.0,
        total_population=int(total_residents),
        total_jobs=int(total_jobs),
        jobs_ratio=total_jobs / total_size if total_size > 0 else 0.0,
        nn_dist_mean=float(nn_distances.mean()) if len(nn_distances) > 0 else 0.0,
        nn_dist_median=float(np.median(nn_distances)) if len(nn_distances) > 0 else 0.0,
        nn_dist_p90=float(np.percentile(nn_distances, 90)) if len(nn_distances) > 0 else 0.0,
        area_km2=area_km2,
        density=len(points) / area_km2 if area_km2 > 0 else 0.0,
        
        # Stage 3: Connection Distribution
        connection_count=len(pops),
        connections_per_cluster=len(pops) / len(points) if len(points) > 0 else 0.0,
        conn_dist_mean=float(conn_distances.mean()) if len(conn_distances) > 0 else 0.0,
        conn_dist_median=float(np.median(conn_distances)) if len(conn_distances) > 0 else 0.0,
        conn_dist_p90=float(np.percentile(conn_distances, 90)) if len(conn_distances) > 0 else 0.0,
        conn_size_mean=float(conn_sizes.mean()) if len(conn_sizes) > 0 else 0.0,
        conn_size_median=float(np.median(conn_sizes)) if len(conn_sizes) > 0 else 0.0,
        conn_size_p90=float(np.percentile(conn_sizes, 90)) if len(conn_sizes) > 0 else 0.0,
        graph_density=len(unique_edges) / possible_edges if possible_edges > 0 else 0.0,
    )


def main():
    """Extract features from all ground truth cities"""
    appdata_path = get_appdata_path()
    game_data_path = appdata_path / 'subway_builder' / 'resources' / 'cities'
    
    if not game_data_path.exists():
        print(f"Error: Game data not found at {game_data_path}")
        print("Please ensure Subway Builder is installed and has been run at least once.")
        return
    
    features_list = []
    
    for city_code in CITIES:
        demand_file = game_data_path / city_code / 'demand_data.json.gz'
        
        if not demand_file.exists():
            print(f"Skipping {city_code}: demand_data.json.gz not found")
            continue
        
        print(f"Extracting features for {city_code}...")
        
        try:
            with gzip.open(demand_file, 'rt', encoding='utf-8') as f:
                data = json.load(f)
            
            features = extract_features(data, city_code)
            features_list.append(asdict(features))
            
            print(f"  OK {features.cluster_count} clusters, "
                  f"{features.total_population:,} population, "
                  f"{features.connection_count} connections")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    if not features_list:
        print("\nNo features extracted. Check game installation.")
        return
    
    # Save as JSON
    script_dir = Path(__file__).parent
    output_json = script_dir / 'ground_truth_features.json'
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(features_list, f, indent=2)
    print(f"\nOK Saved {len(features_list)} city features to {output_json}")
    
    # Save as CSV for easy viewing
    output_csv = script_dir / 'ground_truth_features.csv'
    df = pd.DataFrame(features_list)
    df.to_csv(output_csv, index=False)
    print(f"OK Saved CSV to {output_csv}")


if __name__ == '__main__':
    main()
