"""
Process OSM data into game format.

Python port of process_data.js for integration with the optimizer.
Eliminates subprocess overhead and allows direct parameter injection.

Usage:
    from process_data import process_all_places
    process_all_places(config)
"""

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial import KDTree

# Try to import msgpack for binary format (much faster than JSON)
try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False

# Try to import ijson for streaming JSON parsing (optional)
try:
    import ijson
    HAS_IJSON = True
except ImportError:
    HAS_IJSON = False

# Base directory
SCRIPT_DIR = Path(__file__).parent
MAP_PATCHER_DIR = SCRIPT_DIR.parent


# === Building type mappings ===
# Square feet per resident for residential building types
SQFT_PER_POPULATION = {
    'apartments': 240,
    'barracks': 100,
    'bungalow': 600,
    'cabin': 600,
    'detached': 600,
    'annexe': 240,
    'dormitory': 125,
    'farm': 600,
    'ger': 240,
    'hotel': 240,
    'house': 600,
    'houseboat': 600,
    'residential': 600,
    'semidetached_house': 400,
    'static_caravan': 500,
    'stilt_house': 600,
    'terrace': 500,
    'tree_house': 240,
    'trullo': 240,
}

# Square feet per job for commercial building types
SQFT_PER_JOB = {
    'commercial': 150,
    'industrial': 500,
    'kiosk': 50,
    'office': 150,
    'retail': 300,
    'supermarket': 300,
    'warehouse': 500,
    'religious': 100,
    'cathedral': 100,
    'chapel': 100,
    'church': 100,
    'kingdom_hall': 100,
    'monastery': 100,
    'mosque': 100,
    'presbytery': 100,
    'shrine': 100,
    'synagogue': 100,
    'temple': 100,
    'bakehouse': 300,
    'college': 250,
    'fire_station': 500,
    'government': 150,
    'gatehouse': 150,
    'hospital': 150,
    'kindergarten': 100,
    'museum': 300,
    'public': 300,
    'school': 100,
    'train_station': 1000,
    'transportation': 1000,
    'university': 250,
    'grandstand': 150,
    'pavilion': 150,
    'riding_hall': 150,
    'sports_hall': 150,
    'sports_centre': 150,
    'stadium': 150,
}


@dataclass
class ProcessingConfig:
    """Configuration for data processing"""
    # Stage 1: Building -> Population
    residential_sqft_multiplier: float = 1.0
    commercial_sqft_multiplier: float = 1.0
    mixed_use_residential_ratio: float = 0.5
    mixed_use_threshold_small: int = 5000
    mixed_use_threshold_large: int = 20000
    max_connection_distance_meters: float = 0  # 0 = unlimited
    
    # Stage 2: Cluster management
    max_place_size_for_splitting: int = 0
    merge_places_distance_meters: int = 0
    min_place_size_for_protection: int = 5000
    splitting_num_clusters_divisor: int = 1250
    splitting_cluster_min_separation: float = 0.002
    grid_cell_size: float = 0.0009

    # Stage 3: Connection generation
    gravity_exponent: float = 0.5
    min_connections_per_cluster: int = 5
    max_connections_per_cluster: int = 25
    connection_size_cap: int = 200
    connection_scaling_divisor: int = 40
    population_scale_factor: float = 1.0
    max_total_connections: int = 0  # 0 = unlimited, else probabilistically limit total

    # Other
    tile_zoom_level: int = 16
    disable_3d_buildings: bool = False
    skip_buildings_index: bool = False  # Skip for ML optimization (faster)

    @classmethod
    def from_dict(cls, d: Dict) -> 'ProcessingConfig':
        """Create config from dictionary (handles hyphenated keys)"""
        return cls(
            residential_sqft_multiplier=d.get('residential-sqft-multiplier', 1.0),
            commercial_sqft_multiplier=d.get('commercial-sqft-multiplier', 1.0),
            mixed_use_residential_ratio=d.get('mixed-use-residential-ratio', 0.5),
            mixed_use_threshold_small=d.get('mixed-use-threshold-small', 5000),
            mixed_use_threshold_large=d.get('mixed-use-threshold-large', 20000),
            max_place_size_for_splitting=d.get('max-place-size-for-splitting', 0),
            merge_places_distance_meters=d.get('merge-places-distance-meters', 0),
            min_place_size_for_protection=d.get('min-place-size-for-protection', 5000),
            splitting_num_clusters_divisor=d.get('splitting-num-clusters-divisor', 1250),
            splitting_cluster_min_separation=d.get('splitting-cluster-min-separation', 0.002),
            grid_cell_size=d.get('grid-cell-size', 0.0009),
            gravity_exponent=d.get('gravity-exponent', 0.5),
            min_connections_per_cluster=d.get('min-connections-per-cluster', 5),
            max_connections_per_cluster=d.get('max-connections-per-cluster', 25),
            connection_size_cap=d.get('connection-size-cap', 200),
            connection_scaling_divisor=d.get('connection-scaling-divisor', 40),
            population_scale_factor=d.get('population-scale-factor', 1.0),
            tile_zoom_level=d.get('tile-zoom-level', 16),
            disable_3d_buildings=d.get('disable-3d-buildings', False),
            skip_buildings_index=d.get('skip-buildings-index', False),
        )


@dataclass
class Place:
    """A city/place to process"""
    code: str
    name: str
    description: str
    bbox: List[float]  # [minLon, minLat, maxLon, maxLat]
    population: int = 0


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate distance between two points in meters"""
    R = 6371000  # Earth radius in meters
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat / 2) ** 2 + \
        math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def polygon_area(coords: List[List[float]]) -> float:
    """Calculate polygon area in square meters using shoelace formula + geodesic correction"""
    if len(coords) < 3:
        return 0.0

    # Use shoelace formula for area in degrees^2
    n = len(coords)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += coords[i][0] * coords[j][1]
        area -= coords[j][0] * coords[i][1]
    area = abs(area) / 2.0

    # Convert to square meters (approximate)
    # 1 degree latitude ≈ 111,320 meters
    # 1 degree longitude ≈ 111,320 * cos(lat) meters
    center_lat = sum(c[1] for c in coords) / len(coords)
    lat_meters = 111320
    lon_meters = 111320 * math.cos(math.radians(center_lat))

    return area * lat_meters * lon_meters


def polygon_centroid(coords: List[List[float]]) -> Tuple[float, float]:
    """Calculate polygon centroid"""
    if len(coords) < 3:
        return (coords[0][0], coords[0][1]) if coords else (0, 0)

    cx = sum(c[0] for c in coords) / len(coords)
    cy = sum(c[1] for c in coords) / len(coords)
    return (cx, cy)


def extract_neighborhoods(raw_places: List[Dict]) -> Dict[str, Dict]:
    """Extract neighborhoods from OSM places data"""
    neighborhoods = {}
    centers = {}

    for place in raw_places:
        tags = place.get('tags', {})

        # Check if it's a neighborhood-type place
        is_neighborhood = (
            tags.get('place') in ('quarter', 'neighbourhood') or
            tags.get('aeroway') == 'terminal' or
            tags.get('amenity') == 'university'
        )

        if not is_neighborhood:
            continue

        place_id = str(place.get('id'))
        neighborhoods[place_id] = place

        # Calculate center
        if place.get('type') == 'node':
            centers[place_id] = [place.get('lon'), place.get('lat')]
        elif place.get('type') in ('way', 'relation'):
            bounds = place.get('bounds', {})
            center = [
                (bounds.get('minlon', 0) + bounds.get('maxlon', 0)) / 2,
                (bounds.get('minlat', 0) + bounds.get('maxlat', 0)) / 2
            ]
            centers[place_id] = center

    return neighborhoods, centers


def classify_building(building: Dict, config: ProcessingConfig) -> Optional[Dict]:
    """Classify a building and calculate population/jobs"""
    tags = building.get('tags', {})
    building_type = tags.get('building')

    if not building_type:
        return None

    geometry = building.get('geometry', [])
    if len(geometry) < 3:
        return None

    # Convert geometry to coords
    coords = [[p.get('lon'), p.get('lat')] for p in geometry]

    # Close polygon if needed
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    if len(coords) < 4:
        return None

    # Calculate area
    base_area = polygon_area(coords)
    try:
        levels = max(1, int(float(tags.get('building:levels', 1) or 1)))
    except (ValueError, TypeError):
        levels = 1
    total_area_sqft = base_area * levels * 10.7639  # Convert m² to sqft

    # Calculate centroid
    center = polygon_centroid(coords)

    result = {
        'id': building.get('id'),
        'center': center,
        'coords': coords,
        'area_sqft': total_area_sqft,
        'tags': tags,
        'approx_pop': 0,
        'approx_jobs': 0,
    }

    # Mixed-use buildings (building=yes)
    if building_type == 'yes':
        # Size-based residential/commercial split
        if total_area_sqft < config.mixed_use_threshold_small:
            res_ratio = 0.8
        elif total_area_sqft < config.mixed_use_threshold_large:
            t = (total_area_sqft - config.mixed_use_threshold_small) / \
                (config.mixed_use_threshold_large - config.mixed_use_threshold_small)
            res_ratio = 0.8 - (0.8 - config.mixed_use_residential_ratio) * t
        else:
            t = min(1, (total_area_sqft - config.mixed_use_threshold_large) / 80000)
            res_ratio = config.mixed_use_residential_ratio - \
                       (config.mixed_use_residential_ratio - 0.3) * t

        res_area = total_area_sqft * res_ratio
        com_area = total_area_sqft * (1 - res_ratio)

        result['approx_pop'] = int(res_area / (400 * config.residential_sqft_multiplier))
        result['approx_jobs'] = int(com_area / (200 * config.commercial_sqft_multiplier))
        result['is_mixed_use'] = True

    # Residential buildings
    elif building_type in SQFT_PER_POPULATION:
        sqft_per_person = SQFT_PER_POPULATION[building_type] * config.residential_sqft_multiplier
        result['approx_pop'] = int(total_area_sqft / sqft_per_person)

    # Commercial buildings
    elif building_type in SQFT_PER_JOB:
        sqft_per_job = SQFT_PER_JOB[building_type] * config.commercial_sqft_multiplier
        result['approx_jobs'] = int(total_area_sqft / sqft_per_job)

        # Airport terminals get multiplier
        if tags.get('aeroway') == 'terminal':
            result['approx_jobs'] *= 20

    else:
        # Unclassified building
        return None

    return result


def assign_buildings_to_neighborhoods(
    buildings: Dict[str, Dict],
    neighborhood_centers: Dict[str, List[float]]
) -> Dict[str, List[str]]:
    """Assign each building to its nearest neighborhood using KDTree"""
    if not buildings or not neighborhood_centers:
        return {}

    # Build arrays for KDTree
    place_ids = list(neighborhood_centers.keys())
    center_coords = np.array([neighborhood_centers[pid] for pid in place_ids])

    # Create KDTree for fast nearest-neighbor lookup
    tree = KDTree(center_coords)

    # Assign buildings
    assignments = {pid: [] for pid in place_ids}

    for building_id, building in buildings.items():
        center = building['center']
        _, idx = tree.query(center)
        nearest_place = place_ids[idx]
        assignments[nearest_place].append(building_id)

    return assignments


def merge_small_neighborhoods(
    neighborhoods: Dict[str, Dict],
    centers: Dict[str, List[float]],
    metadata: Dict[str, Dict],
    config: ProcessingConfig
) -> Tuple[Dict, Dict, Dict]:
    """Merge small nearby neighborhoods"""
    if config.merge_places_distance_meters <= 0:
        return neighborhoods, centers, metadata

    merge_dist = config.merge_places_distance_meters
    min_size = config.min_place_size_for_protection

    place_ids = list(metadata.keys())
    merged = set()
    merged_count = 0

    for i, id1 in enumerate(place_ids):
        if id1 in merged:
            continue

        place1 = metadata[id1]
        size1 = place1['total_population'] + place1['total_jobs']

        # Skip if large (protected)
        if size1 >= min_size:
            continue

        cluster = [id1]
        cluster_pop = place1['total_population']
        cluster_jobs = place1['total_jobs']

        for j in range(i + 1, len(place_ids)):
            id2 = place_ids[j]
            if id2 in merged:
                continue

            place2 = metadata[id2]
            size2 = place2['total_population'] + place2['total_jobs']

            if size2 >= min_size:
                continue

            # Check distance
            dist = haversine_distance(
                centers[id1][0], centers[id1][1],
                centers[id2][0], centers[id2][1]
            )

            if dist <= merge_dist:
                cluster.append(id2)
                merged.add(id2)
                cluster_pop += place2['total_population']
                cluster_jobs += place2['total_jobs']

        if len(cluster) > 1:
            # Merge into first ID
            main_id = cluster[0]
            avg_lon = sum(centers[cid][0] for cid in cluster) / len(cluster)
            avg_lat = sum(centers[cid][1] for cid in cluster) / len(cluster)
            centers[main_id] = [avg_lon, avg_lat]

            metadata[main_id]['total_population'] = cluster_pop
            metadata[main_id]['total_jobs'] = cluster_jobs

            # Remove merged places
            for cid in cluster[1:]:
                neighborhoods.pop(cid, None)
                centers.pop(cid, None)
                metadata.pop(cid, None)
                merged_count += 1

    print(f"  Merged {merged_count} small neighborhoods. Remaining: {len(metadata)}")
    return neighborhoods, centers, metadata


def split_large_neighborhoods(
    neighborhoods: Dict[str, Dict],
    centers: Dict[str, List[float]],
    metadata: Dict[str, Dict],
    buildings: Dict[str, Dict],
    assignments: Dict[str, List[str]],
    config: ProcessingConfig,
    total_population: int,
    total_jobs: int
) -> Tuple[Dict, Dict, Dict, Dict]:
    """Split large neighborhoods into smaller sub-neighborhoods"""
    if config.max_place_size_for_splitting <= 0:
        return neighborhoods, centers, metadata, assignments

    max_size = config.max_place_size_for_splitting
    divisor = config.splitting_num_clusters_divisor
    min_sep = config.splitting_cluster_min_separation

    place_ids = list(metadata.keys())
    split_count = 0
    new_id_counter = 0

    for place_id in place_ids:
        place = metadata.get(place_id)
        if not place:
            continue

        total_size = place['total_population'] + place['total_jobs']
        if total_size <= max_size:
            continue

        assigned_buildings = assignments.get(place_id, [])
        if not assigned_buildings:
            continue

        # Determine number of sub-clusters
        num_clusters = max(4, min(16, int(total_size / divisor)))

        # Get building locations and weights
        building_data = []
        for bid in assigned_buildings:
            b = buildings.get(bid)
            if b:
                weight = b['approx_pop'] + b['approx_jobs'] + 1
                building_data.append((bid, b['center'], weight))

        if not building_data:
            continue

        total_weight = sum(w for _, _, w in building_data)

        # Select diverse centers using weighted random sampling
        sub_centers = []
        max_attempts = num_clusters * 50

        for _ in range(max_attempts):
            if len(sub_centers) >= num_clusters:
                break

            # Weighted random selection
            rand = random.random() * total_weight
            selected_idx = 0
            cumulative = 0
            for idx, (_, _, weight) in enumerate(building_data):
                cumulative += weight
                if cumulative >= rand:
                    selected_idx = idx
                    break

            candidate_loc = building_data[selected_idx][1]

            # Check distance from existing centers
            too_close = False
            for sub in sub_centers:
                dist = math.sqrt(
                    (candidate_loc[0] - sub['location'][0]) ** 2 +
                    (candidate_loc[1] - sub['location'][1]) ** 2
                )
                if dist < min_sep:
                    too_close = True
                    break

            if not too_close:
                sub_id = f"{place_id}_split_{new_id_counter}"
                new_id_counter += 1
                sub_centers.append({
                    'id': sub_id,
                    'location': candidate_loc,
                    'buildings': [],
                    'total_population': 0,
                    'total_jobs': 0
                })

        if not sub_centers:
            continue

        # Build KDTree for sub-centers
        sub_coords = np.array([s['location'] for s in sub_centers])
        sub_tree = KDTree(sub_coords)

        # Redistribute buildings
        for bid, center, _ in building_data:
            _, idx = sub_tree.query(center)
            b = buildings[bid]
            sub_centers[idx]['buildings'].append(bid)
            sub_centers[idx]['total_population'] += b['approx_pop']
            sub_centers[idx]['total_jobs'] += b['approx_jobs']

        # Remove original
        neighborhoods.pop(place_id, None)
        centers.pop(place_id, None)
        metadata.pop(place_id, None)
        assignments.pop(place_id, None)

        # Add sub-neighborhoods
        for sub in sub_centers:
            if sub['total_population'] > 0 or sub['total_jobs'] > 0:
                sub_id = sub['id']
                neighborhoods[sub_id] = neighborhoods.get(place_id, {'tags': {}})
                centers[sub_id] = sub['location']
                assignments[sub_id] = sub['buildings']
                metadata[sub_id] = {
                    'place_id': sub_id,
                    'name': f"{place.get('name', '')} ({sub_id.split('_')[-1]})",
                    'total_population': sub['total_population'],
                    'total_jobs': sub['total_jobs'],
                    'pct_population': sub['total_population'] / total_population if total_population > 0 else 0,
                    'pct_jobs': sub['total_jobs'] / total_jobs if total_jobs > 0 else 0
                }
                split_count += 1

    print(f"  Split into {split_count} smaller neighborhoods. Total: {len(metadata)}")
    return neighborhoods, centers, metadata, assignments


def generate_connections(
    metadata: Dict[str, Dict],
    centers: Dict[str, List[float]],
    config: ProcessingConfig
) -> List[Dict]:
    """Generate connections using gravity model"""
    connections = []
    total_connections_generated = 0

    # Pre-compute all distances using vectorized operations
    place_ids = list(metadata.keys())
    n = len(place_ids)

    if n == 0:
        return []

    # Build coordinate array
    coords = np.array([centers[pid] for pid in place_ids])

    # Compute pairwise distances (vectorized haversine approximation)
    # For small distances, Euclidean with lat correction is faster
    lat_mid = np.mean(coords[:, 1])
    lon_scale = math.cos(math.radians(lat_mid)) * 111320
    lat_scale = 111320

    scaled_coords = coords.copy()
    scaled_coords[:, 0] *= lon_scale
    scaled_coords[:, 1] *= lat_scale

    for i, outer_id in enumerate(place_ids):
        outer = metadata[outer_id]
        total_demand = outer['total_population']

        if total_demand <= 5:
            continue

        # Probabilistically skip if approaching connection limit
        if config.max_total_connections > 0:
            if total_connections_generated >= config.max_total_connections:
                break  # Hard stop at limit
            elif total_connections_generated > config.max_total_connections * 0.8:
                # Soft limit: reduce probability as we approach cap
                progress = (total_connections_generated - config.max_total_connections * 0.8) / (config.max_total_connections * 0.2)
                skip_prob = progress * 0.9  # Up to 90% chance to skip
                if random.random() < skip_prob:
                    continue

        # Calculate gravity scores
        gravity_scores = []
        total_gravity = 0

        for j, inner_id in enumerate(place_ids):
            if i == j:
                continue
            
            # Skip distant neighborhoods
            if config.max_connection_distance_meters > 0:
                dist = np.linalg.norm(scaled_coords[i] - scaled_coords[j])
                if dist > config.max_connection_distance_meters:
                    continue
            
            inner = metadata[inner_id]
            if inner['total_jobs'] <= 0:
                continue

            # Distance in meters
            dx = scaled_coords[j, 0] - scaled_coords[i, 0]
            dy = scaled_coords[j, 1] - scaled_coords[i, 1]
            distance = math.sqrt(dx * dx + dy * dy)

            # Gravity model
            effective_dist = max(distance, 100)
            gravity = (inner['total_jobs'] * outer['total_population']) / \
                      (effective_dist ** config.gravity_exponent)

            if gravity > 0:
                gravity_scores.append({
                    'inner_id': inner_id,
                    'gravity': gravity,
                    'distance': distance
                })
                total_gravity += gravity

        if not gravity_scores:
            continue

        # Normalize probabilities
        for item in gravity_scores:
            item['probability'] = item['gravity'] / total_gravity

        # Sort by gravity
        gravity_scores.sort(key=lambda x: -x['gravity'])

        # Determine number of connections
        num_connections = min(
            config.max_connections_per_cluster,
            max(config.min_connections_per_cluster,
                int(math.sqrt(total_demand / config.connection_scaling_divisor)))
        )

        # Weighted random selection
        selected = set()
        attempts = 0
        max_attempts = len(gravity_scores) * 5

        while len(selected) < num_connections and len(selected) < len(gravity_scores) and attempts < max_attempts:
            attempts += 1
            rand = random.random()
            cumulative = 0

            for item in gravity_scores:
                if item['inner_id'] in selected:
                    continue
                cumulative += item['probability']
                if rand <= cumulative:
                    selected.add(item['inner_id'])
                    break

        # Distribute demand
        selected_items = [item for item in gravity_scores if item['inner_id'] in selected]
        selected_gravity = sum(item['gravity'] for item in selected_items)

        total_connections_generated += len(selected_items)

        for item in selected_items:
            connection_size = round((item['gravity'] / selected_gravity) * total_demand)
            if connection_size <= 0:
                continue

            distance = item['distance']
            seconds = distance * 0.12

            # Split large connections
            remaining = connection_size
            while remaining > 0:
                chunk = min(config.connection_size_cap, remaining)
                connections.append({
                    'residenceId': outer_id,
                    'jobId': item['inner_id'],
                    'size': chunk,
                    'drivingDistance': round(distance),
                    'drivingSeconds': round(seconds)
                })
                remaining -= chunk

    return connections


def process_place_connections(
    place: Place,
    raw_buildings: List[Dict],
    raw_places: List[Dict],
    config: ProcessingConfig
) -> Dict:
    """Main processing function for demand data"""
    print(f"  Extracting neighborhoods...")
    neighborhoods, centers = extract_neighborhoods(raw_places)
    print(f"  Found {len(neighborhoods)} OSM neighborhoods")

    if not neighborhoods:
        print(f"  WARNING: No neighborhoods found!")
        return {'points': [], 'pops': []}

    # Classify buildings
    print(f"  Classifying {len(raw_buildings)} buildings...")
    buildings = {}
    for building in raw_buildings:
        classified = classify_building(building, config)
        if classified:
            buildings[str(classified['id'])] = classified

    print(f"  Classified {len(buildings)} buildings")

    # Assign buildings to neighborhoods
    print(f"  Assigning buildings to neighborhoods...")
    assignments = assign_buildings_to_neighborhoods(buildings, centers)

    # Calculate population/jobs per neighborhood
    metadata = {}
    total_population = 0
    total_jobs = 0

    for place_id in neighborhoods:
        assigned = assignments.get(place_id, [])
        pop = sum(buildings[bid]['approx_pop'] for bid in assigned if bid in buildings)
        jobs = sum(buildings[bid]['approx_jobs'] for bid in assigned if bid in buildings)
        total_population += pop
        total_jobs += jobs

        metadata[place_id] = {
            'place_id': place_id,
            'name': neighborhoods[place_id].get('tags', {}).get('name', place_id),
            'total_population': pop,
            'total_jobs': jobs,
            'pct_population': 0,
            'pct_jobs': 0
        }

    print(f"  Total population: {total_population}, jobs: {total_jobs}")

    # Apply population scale factor
    if config.population_scale_factor != 1.0:
        scale = config.population_scale_factor
        total_population = round(total_population * scale)
        total_jobs = round(total_jobs * scale)

        for pid in metadata:
            metadata[pid]['total_population'] = round(metadata[pid]['total_population'] * scale)
            metadata[pid]['total_jobs'] = round(metadata[pid]['total_jobs'] * scale)

        print(f"  Scaled to population: {total_population}, jobs: {total_jobs}")

    # Merge small neighborhoods
    print(f"  Merging small neighborhoods...")
    neighborhoods, centers, metadata = merge_small_neighborhoods(
        neighborhoods, centers, metadata, config
    )

    # Split large neighborhoods
    print(f"  Splitting large neighborhoods...")
    neighborhoods, centers, metadata, assignments = split_large_neighborhoods(
        neighborhoods, centers, metadata, buildings, assignments, config,
        total_population, total_jobs
    )

    # Update percentages
    for pid in metadata:
        if total_population > 0:
            metadata[pid]['pct_population'] = metadata[pid]['total_population'] / total_population
        if total_jobs > 0:
            metadata[pid]['pct_jobs'] = metadata[pid]['total_jobs'] / total_jobs

    # Generate connections
    print(f"  Generating connections...")
    connections = generate_connections(metadata, centers, config)
    print(f"  Generated {len(connections)} connections")

    # Build final output
    terminal_ticker = 0
    uni_ticker = 0
    final_neighborhoods = {}

    for place_id, place_meta in metadata.items():
        nh = neighborhoods.get(place_id, {})
        tags = nh.get('tags', {})

        # Determine ID
        if tags.get('aeroway') == 'terminal':
            final_id = f"AIR_Terminal_{terminal_ticker}"
            terminal_ticker += 1
        elif tags.get('amenity') == 'university':
            final_id = f"UNI_{uni_ticker}"
            uni_ticker += 1
        else:
            final_id = place_id

        final_neighborhoods[place_id] = {
            'id': final_id,
            'location': centers[place_id],
            'jobs': place_meta['total_jobs'],
            'residents': place_meta['total_population'],
            'popIds': []
        }

    # Assign connection IDs and update popIds
    for i, conn in enumerate(connections):
        conn_id = str(i)
        conn['id'] = conn_id

        res_id = conn['residenceId']
        job_id = conn['jobId']

        if res_id in final_neighborhoods:
            final_neighborhoods[res_id]['popIds'].append(conn_id)
        if job_id in final_neighborhoods:
            final_neighborhoods[job_id]['popIds'].append(conn_id)

        # Update IDs to final format
        conn['residenceId'] = final_neighborhoods[res_id]['id']
        conn['jobId'] = final_neighborhoods[job_id]['id']

    # Filter out points with no connections
    points = [p for p in final_neighborhoods.values() if p['popIds']]
    print(f"  Final: {len(points)} demand points, {len(connections)} connections")

    return {
        'points': points,
        'pops': [c for c in connections if c['size'] > 0]
    }


def load_data_file(filepath: Path) -> List[Dict]:
    """Load data file (MessagePack preferred, JSON fallback)"""
    # Check for MessagePack version first (much faster)
    msgpack_path = filepath.with_suffix('.msgpack')
    if msgpack_path.exists() and HAS_MSGPACK:
        file_size = msgpack_path.stat().st_size
        print(f"    Loading {file_size / 1_000_000:.1f}MB MessagePack file...")
        with open(msgpack_path, 'rb') as f:
            return msgpack.unpack(f, raw=False)

    # Fall back to JSON
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath} (or {msgpack_path})")

    file_size = filepath.stat().st_size
    print(f"    Loading {file_size / 1_000_000:.1f}MB JSON file...")

    # For large JSON files, try streaming if available
    if file_size > 50_000_000 and HAS_IJSON:
        print(f"    Using streaming parser...")
        items = []
        try:
            with open(filepath, 'rb') as f:
                for item in ijson.items(f, 'item'):
                    items.append(item)
            return items
        except Exception as e:
            print(f"    Streaming failed ({e}), falling back to standard parser...")

    # Standard JSON loading
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        if 'Expecting' in str(e):
            raise ValueError(
                f"JSON file appears to be corrupted or truncated: {filepath.name}\n"
                f"  Error at position {e.pos}: {e.msg}\n"
                f"  Try re-downloading the raw data with: node download_data.js"
            )
        raise


def process_place(place: Place, config: ProcessingConfig) -> Dict:
    """Process a single place"""
    print(f"Processing {place.code}...")

    raw_data_dir = MAP_PATCHER_DIR / 'raw_data' / place.code
    output_dir = MAP_PATCHER_DIR / 'processed_data' / place.code

    # Check for raw data (MessagePack preferred, JSON fallback)
    buildings_msgpack = raw_data_dir / 'buildings.msgpack'
    buildings_json = raw_data_dir / 'buildings.json'
    places_msgpack = raw_data_dir / 'places.msgpack'
    places_json = raw_data_dir / 'places.json'

    if not buildings_msgpack.exists() and not buildings_json.exists():
        raise FileNotFoundError(f"Buildings file not found: {buildings_json} or {buildings_msgpack}")
    if not places_msgpack.exists() and not places_json.exists():
        raise FileNotFoundError(f"Places file not found: {places_json} or {places_msgpack}")

    print(f"  Loading raw data...")
    raw_buildings = load_data_file(buildings_json)  # Will auto-detect msgpack
    raw_places = load_data_file(places_json)  # Will auto-detect msgpack

    # Process connections/demand
    demand_data = process_place_connections(
        place, raw_buildings, raw_places, config
    )

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write demand data
    demand_file = output_dir / 'demand_data.json'
    with open(demand_file, 'w', encoding='utf-8') as f:
        json.dump(demand_data, f)
    print(f"  Wrote demand_data.json")

    # Copy supporting files from raw_data (needed for game patching)
    import shutil
    for filename in ['roads.geojson', 'runways_taxiways.geojson']:
        src = raw_data_dir / filename
        dst = output_dir / filename
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Copied {filename}")

    # Create buildings_index.json (spatial index for 3D buildings)
    # Skip during ML optimization for speed - only needed for final game patch
    if not config.skip_buildings_index:
        print(f"  Creating buildings_index.json...")
        buildings_index = create_buildings_index(raw_buildings, place.bbox)
        buildings_index_file = output_dir / 'buildings_index.json'
        with open(buildings_index_file, 'w', encoding='utf-8') as f:
            json.dump(buildings_index, f)
        print(f"  Wrote buildings_index.json ({buildings_index['stats']['count']} buildings)")

    return demand_data


def create_buildings_index(raw_buildings: List[Dict], bbox: List[float]) -> Dict:
    """Create spatial index for 3D building rendering"""
    CS = 0.0009  # Cell size in degrees (matches game)

    min_lon, min_lat, max_lon, max_lat = float('inf'), float('inf'), float('-inf'), float('-inf')
    processed_buildings = []
    max_depth = 1

    for i, building in enumerate(raw_buildings):
        geometry = building.get('geometry', [])
        if len(geometry) < 3:
            continue

        tags = building.get('tags', {})

        # Convert geometry to coords
        coords = [[p.get('lon'), p.get('lat')] for p in geometry]
        if coords[0] != coords[-1]:
            coords.append(coords[0])

        if len(coords) < 4:
            continue

        # Calculate bounding box
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        b_min_lon, b_max_lon = min(lons), max(lons)
        b_min_lat, b_max_lat = min(lats), max(lats)

        # Update global bounds
        min_lon = min(min_lon, b_min_lon)
        min_lat = min(min_lat, b_min_lat)
        max_lon = max(max_lon, b_max_lon)
        max_lat = max(max_lat, b_max_lat)

        # Get foundation depth
        try:
            depth = int(float(tags.get('building:levels:underground', 1) or 1))
        except (ValueError, TypeError):
            depth = 1
        max_depth = max(max_depth, depth)

        processed_buildings.append({
            'b': [b_min_lon, b_min_lat, b_max_lon, b_max_lat],
            'f': depth,
            'p': [coords],  # Polygon as array of rings
            'center': [(b_min_lon + b_max_lon) / 2, (b_min_lat + b_max_lat) / 2]
        })

    if not processed_buildings:
        return {
            'cs': CS,
            'bbox': bbox,
            'grid': [0, 0],
            'cells': [],
            'buildings': [],
            'stats': {'count': 0, 'maxDepth': 1}
        }

    # Create grid
    lat_mid = (min_lat + max_lat) / 2
    distortion = 1 / math.cos(math.radians(lat_mid))
    cs_x = CS * distortion

    cols = max(1, math.ceil((max_lon - min_lon) / cs_x))
    rows = max(1, math.ceil((max_lat - min_lat) / CS))

    # Assign buildings to cells
    cells_dict = {}
    for idx, b in enumerate(processed_buildings):
        cx, cy = b['center']
        cell_x = min(cols - 1, max(0, int((cx - min_lon) / cs_x)))
        cell_y = min(rows - 1, max(0, int((cy - min_lat) / CS)))
        key = f"{cell_x},{cell_y}"
        if key not in cells_dict:
            cells_dict[key] = []
        cells_dict[key].append(idx)

    # Convert cells to array format
    cells = []
    for key, building_ids in cells_dict.items():
        x, y = map(int, key.split(','))
        cells.append([x, y] + building_ids)

    # Remove center from buildings (not needed in output)
    buildings_output = [{'b': b['b'], 'f': b['f'], 'p': b['p']} for b in processed_buildings]

    return {
        'cs': CS,
        'bbox': [min_lon, min_lat, max_lon, max_lat],
        'grid': [cols, rows],
        'cells': cells,
        'buildings': buildings_output,
        'stats': {'count': len(buildings_output), 'maxDepth': max_depth}
    }


def process_all_places(config_dict: Dict) -> List[Dict]:
    """Process all places in config"""
    config = ProcessingConfig.from_dict(config_dict)
    places = config_dict.get('places', [])

    results = []
    for place_dict in places:
        place = Place(
            code=place_dict['code'],
            name=place_dict.get('name', place_dict['code']),
            description=place_dict.get('description', ''),
            bbox=place_dict['bbox'],
            population=place_dict.get('population', 0)
        )

        try:
            result = process_place(place, config)
            results.append({'code': place.code, 'success': True, 'data': result})
        except Exception as e:
            print(f"ERROR processing {place.code}: {e}")
            results.append({'code': place.code, 'success': False, 'error': str(e)})

    return results


def main():
    """CLI entry point"""
    import sys

    # Load config
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'config.json'
    config_path = MAP_PATCHER_DIR / config_file

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    print(f"Loaded config from {config_file}")
    print(f"Processing {len(config_dict.get('places', []))} places...")

    results = process_all_places(config_dict)

    success_count = sum(1 for r in results if r['success'])
    print(f"\nCompleted: {success_count}/{len(results)} places processed successfully")


if __name__ == '__main__':
    main()
