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
import os
import random
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool, Queue, Process
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

# Try to import numba for JIT compilation (optional but recommended)
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

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
    yes_building_sqft_per_resident: float = 400
    yes_building_sqft_per_job: float = 200
    levels_default: float = 1.0  # Default building:levels when tag is missing
    levels_multiplier: float = 1.0  # Scale reported building:levels
    min_building_area_sqft: float = 0  # Filter out tiny buildings
    max_building_area_sqft: float = 0  # Cap effective area (0 = unlimited)
    residential_occupancy_rate: float = 1.0  # Fraction of housing occupied
    max_connection_distance_meters: float = 0  # 0 = unlimited
    
    # Stage 2: Cluster management
    max_place_size_for_splitting: int = 0
    merge_places_distance_meters: int = 0
    min_place_size_for_protection: int = 5000
    splitting_num_clusters_divisor: int = 1250
    splitting_cluster_min_separation: float = 0.002
    grid_cell_size: float = 0.0009

    # Stage 3: Connection generation
    gravity_distance_floor_meters: float = 100  # Min effective distance for gravity calc
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
            yes_building_sqft_per_resident=d.get('yes-building-sqft-per-resident', 400),
            yes_building_sqft_per_job=d.get('yes-building-sqft-per-job', 200),
            levels_default=d.get('levels-default', 1.0),
            levels_multiplier=d.get('levels-multiplier', 1.0),
            min_building_area_sqft=d.get('min-building-area-sqft', 0),
            max_building_area_sqft=d.get('max-building-area-sqft', 0),
            residential_occupancy_rate=d.get('residential-occupancy-rate', 1.0),
            max_place_size_for_splitting=d.get('max-place-size-for-splitting', 0),
            merge_places_distance_meters=d.get('merge-places-distance-meters', 0),
            min_place_size_for_protection=d.get('min-place-size-for-protection', 5000),
            splitting_num_clusters_divisor=d.get('splitting-num-clusters-divisor', 1250),
            splitting_cluster_min_separation=d.get('splitting-cluster-min-separation', 0.002),
            grid_cell_size=d.get('grid-cell-size', 0.0009),
            gravity_distance_floor_meters=d.get('gravity-distance-floor-meters', 100),
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


if HAS_NUMBA:
    @jit(nopython=True)
    def polygon_area_fast_jit(coords_arr: np.ndarray) -> float:
        """JIT-compiled polygon area calculation (much faster)"""
        x, y = coords_arr[:, 0], coords_arr[:, 1]
        n = len(x)
        
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += x[i] * y[j]
            area -= x[j] * y[i]
        area = abs(area) / 2.0
        
        # Convert to meters
        center_lat = 0.0
        for i in range(n):
            center_lat += y[i]
        center_lat /= n
        
        lat_meters = 111320.0
        lon_meters = 111320.0 * math.cos(math.radians(center_lat))
        
        return area * lat_meters * lon_meters
    
    def polygon_area(coords: List[List[float]]) -> float:
        """Calculate polygon area using JIT-compiled function"""
        if len(coords) < 3:
            return 0.0
        c = np.array(coords, dtype=np.float64)
        return polygon_area_fast_jit(c)
else:
    def polygon_area(coords: List[List[float]]) -> float:
        """Calculate polygon area in square meters using shoelace formula + geodesic correction"""
        if len(coords) < 3:
            return 0.0

        c = np.array(coords)
        # Shoelace formula (vectorized)
        x, y = c[:, 0], c[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))

        # Convert to square meters (approximate)
        center_lat = y.mean()
        lat_meters = 111320
        lon_meters = 111320 * math.cos(math.radians(center_lat))

        return float(area * lat_meters * lon_meters)


def polygon_centroid(coords: List[List[float]]) -> Tuple[float, float]:
    """Calculate polygon centroid"""
    if len(coords) < 3:
        return (coords[0][0], coords[0][1]) if coords else (0, 0)

    c = np.array(coords)
    return (float(c[:, 0].mean()), float(c[:, 1].mean()))


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


def preclassify_building(building: Dict) -> Optional[Dict]:
    """
    Phase 1: Pre-classify building geometry (config-independent).
    
    This is the expensive part - polygon area, centroid calculation.
    Run once per city download, results cached to disk.
    """
    tags = building.get('tags', {})
    building_type = tags.get('building')

    # Early rejection by type
    if not building_type:
        return None
    
    # Check if it's a known type
    is_residential = building_type in SQFT_PER_POPULATION
    is_commercial = building_type in SQFT_PER_JOB
    is_mixed = building_type == 'yes'
    
    if not (is_residential or is_commercial or is_mixed):
        return None

    geometry = building.get('geometry', [])
    if len(geometry) < 3 or len(geometry) > 1000:
        return None

    # Convert geometry to coords
    coords = [[p.get('lon'), p.get('lat')] for p in geometry]

    # Close polygon if needed
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    if len(coords) < 4:
        return None

    # Calculate area (expensive - uses JIT if available)
    base_area_m2 = polygon_area(coords)
    
    # Skip tiny buildings (< 10 sqft base area)
    if base_area_m2 < 1:  # ~10 sqft
        return None

    # Calculate centroid (expensive)
    center = polygon_centroid(coords)

    # Parse levels once (raw value, multiplier applied later)
    raw_levels = tags.get('building:levels')
    if raw_levels is not None and raw_levels != '':
        try:
            parsed_levels = max(1, int(float(raw_levels)))
        except (ValueError, TypeError):
            parsed_levels = None  # Will use config default
    else:
        parsed_levels = None

    # Determine building category
    if is_residential:
        category = 'residential'
        base_sqft_per_unit = SQFT_PER_POPULATION[building_type]
    elif is_commercial:
        category = 'commercial'
        base_sqft_per_unit = SQFT_PER_JOB[building_type]
    else:
        category = 'mixed'
        base_sqft_per_unit = 0  # Calculated dynamically

    return {
        'id': building.get('id'),
        'center': center,
        'base_area_m2': base_area_m2,
        'parsed_levels': parsed_levels,
        'building_type': building_type,
        'category': category,
        'base_sqft_per_unit': base_sqft_per_unit,
        'is_airport_terminal': tags.get('aeroway') == 'terminal',
    }


def apply_building_config(preclassified: Dict, config: ProcessingConfig) -> Optional[Dict]:
    """
    Phase 2: Apply config-dependent calculations to pre-classified building.
    
    This is the cheap part - just math on pre-computed values.
    Run for each optimization trial.
    """
    # Apply levels
    if preclassified['parsed_levels'] is not None:
        levels = preclassified['parsed_levels'] * config.levels_multiplier
    else:
        levels = config.levels_default
    
    # Calculate total area in sqft
    total_area_sqft = preclassified['base_area_m2'] * levels * 10.7639

    # Filter by area bounds
    if config.min_building_area_sqft > 0 and total_area_sqft < config.min_building_area_sqft:
        return None
    if config.max_building_area_sqft > 0:
        total_area_sqft = min(total_area_sqft, config.max_building_area_sqft)

    result = {
        'id': preclassified['id'],
        'center': preclassified['center'],
        'area_sqft': total_area_sqft,
        'approx_pop': 0,
        'approx_jobs': 0,
    }

    category = preclassified['category']
    building_type = preclassified['building_type']

    if category == 'mixed':
        # Mixed-use buildings (building=yes)
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

        result['approx_pop'] = int(res_area / (config.yes_building_sqft_per_resident * config.residential_sqft_multiplier) * config.residential_occupancy_rate)
        result['approx_jobs'] = int(com_area / (config.yes_building_sqft_per_job * config.commercial_sqft_multiplier))

    elif category == 'residential':
        sqft_per_person = preclassified['base_sqft_per_unit'] * config.residential_sqft_multiplier
        result['approx_pop'] = int(total_area_sqft / sqft_per_person * config.residential_occupancy_rate)

    elif category == 'commercial':
        sqft_per_job = preclassified['base_sqft_per_unit'] * config.commercial_sqft_multiplier
        result['approx_jobs'] = int(total_area_sqft / sqft_per_job)

        # Airport terminals get multiplier
        if preclassified['is_airport_terminal']:
            result['approx_jobs'] *= 20

    return result


def classify_building(building: Dict, config: ProcessingConfig) -> Optional[Dict]:
    """Classify a building and calculate population/jobs - optimized version"""
    tags = building.get('tags', {})
    building_type = tags.get('building')

    # Early rejection by type (cheapest check first)
    if not building_type:
        return None
    
    # Check if it's a known type before expensive geometry processing
    is_residential = building_type in SQFT_PER_POPULATION
    is_commercial = building_type in SQFT_PER_JOB
    is_mixed = building_type == 'yes'
    
    if not (is_residential or is_commercial or is_mixed):
        return None  # Skip before geometry calculations

    geometry = building.get('geometry', [])
    
    # Quick filtering in optimization mode to reduce memory
    if config.optimization_mode:
        if len(geometry) < 3 or len(geometry) > 1000:  # Skip tiny and massive buildings
            return None
    elif len(geometry) < 3:
        return None

    # Convert geometry to coords
    coords = [[p.get('lon'), p.get('lat')] for p in geometry]

    # Close polygon if needed
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    if len(coords) < 4:
        return None

    # Quick bounding box area check (much faster than polygon area)
    if config.min_building_area_sqft > 0:
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        bbox_width = max(lons) - min(lons)
        bbox_height = max(lats) - min(lats)
        
        # Very rough bbox area in m² (1 deg ≈ 111km)
        # This is an overestimate, so if bbox fails, polygon definitely fails
        center_lat = sum(lats) / len(lats)
        bbox_area_m2 = (bbox_width * 111320 * math.cos(math.radians(center_lat))) * (bbox_height * 111320)
        bbox_area_sqft = bbox_area_m2 * 10.7639
        
        # If even the bounding box is too small, skip expensive polygon calc
        if bbox_area_sqft < config.min_building_area_sqft * 0.5:  # 50% margin for safety
            return None

    # Calculate area (now using JIT-compiled version if available)
    base_area = polygon_area(coords)
    raw_levels = tags.get('building:levels')
    if raw_levels is not None and raw_levels != '':
        try:
            levels = max(1, int(float(raw_levels))) * config.levels_multiplier
        except (ValueError, TypeError):
            levels = config.levels_default
    else:
        levels = config.levels_default
    total_area_sqft = base_area * levels * 10.7639  # Convert m² to sqft

    # Filter tiny buildings
    if config.min_building_area_sqft > 0 and total_area_sqft < config.min_building_area_sqft:
        return None

    # Cap effective area
    if config.max_building_area_sqft > 0:
        total_area_sqft = min(total_area_sqft, config.max_building_area_sqft)

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

        result['approx_pop'] = int(res_area / (config.yes_building_sqft_per_resident * config.residential_sqft_multiplier) * config.residential_occupancy_rate)
        result['approx_jobs'] = int(com_area / (config.yes_building_sqft_per_job * config.commercial_sqft_multiplier))
        result['is_mixed_use'] = True

    # Residential buildings
    elif building_type in SQFT_PER_POPULATION:
        sqft_per_person = SQFT_PER_POPULATION[building_type] * config.residential_sqft_multiplier
        result['approx_pop'] = int(total_area_sqft / sqft_per_person * config.residential_occupancy_rate)

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


# Valid building types (precomputed set for O(1) lookup)
_VALID_BUILDING_TYPES = set(SQFT_PER_POPULATION.keys()) | set(SQFT_PER_JOB.keys()) | {'yes'}


def _prefilter_buildings(buildings: List[Dict]) -> List[Dict]:
    """Fast pre-filter: remove buildings without valid types before expensive processing"""
    return [
        b for b in buildings
        if b.get('tags', {}).get('building') in _VALID_BUILDING_TYPES
        and len(b.get('geometry', [])) >= 3
    ]


def _classify_buildings_chunk(args: Tuple[List[Dict], 'ProcessingConfig']) -> List[Dict]:
    """Classify a chunk of buildings (for parallel execution)"""
    chunk, config = args
    results = []
    for building in chunk:
        classified = classify_building(building, config)
        if classified:
            results.append(classified)
    return results


def _streaming_producer(msgpack_path: Path, chunk_queue: Queue, chunk_size: int, prefilter: bool = True):
    """Stream chunks from disk into queue (runs in separate process)"""
    try:
        import msgpack
        
        # Local copy of valid types for fast filtering in this process
        valid_types = {'apartments', 'barracks', 'bungalow', 'cabin', 'detached', 'annexe',
                       'dormitory', 'farm', 'ger', 'hotel', 'house', 'houseboat', 'residential',
                       'semidetached_house', 'static_caravan', 'stilt_house', 'terrace',
                       'tree_house', 'trullo', 'commercial', 'industrial', 'kiosk', 'office',
                       'retail', 'supermarket', 'warehouse', 'religious', 'cathedral', 'chapel',
                       'church', 'kingdom_hall', 'monastery', 'mosque', 'presbytery', 'shrine',
                       'synagogue', 'temple', 'bakehouse', 'college', 'fire_station', 'government',
                       'gatehouse', 'hospital', 'kindergarten', 'museum', 'public', 'school',
                       'train_station', 'transportation', 'university', 'grandstand', 'pavilion',
                       'riding_hall', 'sports_hall', 'sports_centre', 'stadium', 'yes'}
        
        filtered_count = 0
        total_count = 0
        
        with open(msgpack_path, 'rb') as f:
            unpacker = msgpack.Unpacker(f, raw=False)
            chunk = []
            
            for item in unpacker:
                total_count += 1
                
                # Pre-filter in producer to reduce queue/serialization overhead
                if prefilter:
                    tags = item.get('tags', {})
                    btype = tags.get('building')
                    geom = item.get('geometry', [])
                    
                    if btype not in valid_types or len(geom) < 3:
                        filtered_count += 1
                        continue
                
                chunk.append(item)
                if len(chunk) >= chunk_size:
                    chunk_queue.put(chunk)
                    chunk = []
            
            if chunk:
                chunk_queue.put(chunk)
        
        # Signal completion
        chunk_queue.put(None)
    except Exception as e:
        print(f"  Streaming producer error: {e}")
        chunk_queue.put(None)


def classify_buildings_stream_parallel(msgpack_path: Path, config: 'ProcessingConfig') -> Optional[Dict[str, Dict]]:
    """Hybrid streaming + parallel: best of both worlds"""
    try:
        import multiprocessing as mp
        
        n_workers = min(mp.cpu_count() or 4, 6)
        chunk_size = 10000
        chunk_queue = mp.Queue(maxsize=n_workers * 2)  # Buffer 2 chunks per worker
        
        # Start producer process (with pre-filtering enabled)
        producer = Process(target=_streaming_producer, args=(msgpack_path, chunk_queue, chunk_size, True))
        producer.start()
        
        buildings = {}
        pending_results = []  # Track async results for true parallelism
        
        # Process chunks in parallel as they arrive
        with Pool(processes=n_workers) as pool:
            while True:
                chunk = chunk_queue.get()
                if chunk is None:  # End signal
                    break
                
                # Submit chunk for async processing (don't wait immediately)
                async_result = pool.apply_async(_classify_buildings_chunk, [(chunk, config)])
                pending_results.append(async_result)
                
                # Collect completed results periodically (every n_workers chunks)
                if len(pending_results) >= n_workers:
                    for res in pending_results:
                        result_list = res.get()
                        for b in result_list:
                            buildings[str(b['id'])] = b
                    pending_results = []
                    
                    # Free memory periodically
                    if len(buildings) % 100000 == 0:
                        gc.collect()
            
            # Collect any remaining results
            for res in pending_results:
                result_list = res.get()
                for b in result_list:
                    buildings[str(b['id'])] = b
        
        producer.join()
        return buildings
        
    except Exception as e:
        print(f"  Warning: Hybrid streaming+parallel failed ({e}), falling back to simple streaming")
        return None


def classify_buildings_streaming(msgpack_path: Path, config: 'ProcessingConfig') -> Dict[str, Dict]:
    """Stream and classify buildings from MessagePack to reduce memory"""
    buildings = {}
    CHUNK_SIZE = 50000  # Process 50K at a time
    filtered_early = 0
    
    try:
        import msgpack
        with open(msgpack_path, 'rb') as f:
            unpacker = msgpack.Unpacker(f, raw=False)
            chunk = []
            
            for item in unpacker:
                chunk.append(item)
                
                if len(chunk) >= CHUNK_SIZE:
                    # Process chunk
                    for building in chunk:
                        classified = classify_building(building, config)
                        if classified:
                            buildings[str(classified['id'])] = classified
                    chunk = []
                    gc.collect()  # Free memory between chunks
                    
            # Process remaining
            for building in chunk:
                classified = classify_building(building, config)
                if classified:
                    buildings[str(classified['id'])] = classified
                    
        return buildings
    except Exception as e:
        print(f"  Warning: Streaming failed ({e}), falling back to full load")
        return None


def classify_buildings_parallel(raw_buildings: List[Dict], config: 'ProcessingConfig') -> Dict[str, Dict]:
    """Classify all buildings using multiple processes"""
    n_workers = min(os.cpu_count() or 4, 8)
    
    # Pre-filter to remove obviously invalid buildings before expensive processing
    filtered_buildings = _prefilter_buildings(raw_buildings)
    n_original = len(raw_buildings)
    n_filtered = len(filtered_buildings)
    
    if n_original > n_filtered:
        print(f"    Pre-filtered: {n_original} → {n_filtered} buildings ({100*(n_original-n_filtered)/n_original:.1f}% rejected early)")

    # Don't bother with multiprocessing for small datasets
    if n_filtered < 10000 or n_workers < 2:
        buildings = {}
        for building in filtered_buildings:
            classified = classify_building(building, config)
            if classified:
                buildings[str(classified['id'])] = classified
        return buildings

    # Split into chunks
    chunk_size = math.ceil(n_filtered / n_workers)
    chunks = [filtered_buildings[i:i + chunk_size] for i in range(0, n_filtered, chunk_size)]

    buildings = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for result_list in pool.map(_classify_buildings_chunk, [(c, config) for c in chunks]):
            for classified in result_list:
                buildings[str(classified['id'])] = classified

    return buildings


# =============================================================================
# PRECLASSIFICATION SYSTEM - Run geometry processing once, apply config later
# =============================================================================

def _preclassify_chunk(buildings: List[Dict]) -> List[Dict]:
    """Preclassify a chunk of buildings (for parallel execution)"""
    results = []
    for building in buildings:
        preclassified = preclassify_building(building)
        if preclassified:
            results.append(preclassified)
    return results


def preclassify_buildings_parallel(raw_buildings: List[Dict]) -> Dict[str, Dict]:
    """
    Pre-classify all buildings (geometry only, no config).
    This is the expensive step - run once per city download.
    """
    n_workers = min(os.cpu_count() or 4, 8)
    
    # Pre-filter first
    filtered_buildings = _prefilter_buildings(raw_buildings)
    n_original = len(raw_buildings)
    n_filtered = len(filtered_buildings)
    
    print(f"    Pre-filtering: {n_original} → {n_filtered} buildings")

    if n_filtered < 10000 or n_workers < 2:
        buildings = {}
        for building in filtered_buildings:
            preclassified = preclassify_building(building)
            if preclassified:
                buildings[str(preclassified['id'])] = preclassified
        return buildings

    # Split into chunks
    chunk_size = math.ceil(n_filtered / n_workers)
    chunks = [filtered_buildings[i:i + chunk_size] for i in range(0, n_filtered, chunk_size)]

    buildings = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for result_list in pool.map(_preclassify_chunk, chunks):
            for preclassified in result_list:
                buildings[str(preclassified['id'])] = preclassified

    return buildings


def save_preclassified_buildings(buildings: Dict[str, Dict], output_path: Path) -> bool:
    """Save preclassified buildings to MessagePack file"""
    if not HAS_MSGPACK:
        print("  Warning: msgpack not available, cannot save preclassified buildings")
        return False
    
    try:
        # Convert to list for more compact storage
        buildings_list = list(buildings.values())
        
        with open(output_path, 'wb') as f:
            msgpack.pack(buildings_list, f)
        
        size_mb = output_path.stat().st_size / 1_000_000
        print(f"  Saved {len(buildings_list)} preclassified buildings ({size_mb:.1f}MB)")
        return True
    except Exception as e:
        print(f"  Error saving preclassified buildings: {e}")
        return False


def load_preclassified_buildings(input_path: Path) -> Optional[Dict[str, Dict]]:
    """Load preclassified buildings from MessagePack file"""
    if not HAS_MSGPACK or not input_path.exists():
        return None
    
    try:
        with open(input_path, 'rb') as f:
            buildings_list = msgpack.unpack(f, raw=False)
        
        # Convert back to dict
        buildings = {str(b['id']): b for b in buildings_list}
        print(f"  Loaded {len(buildings)} preclassified buildings")
        return buildings
    except Exception as e:
        print(f"  Error loading preclassified buildings: {e}")
        return None


def apply_config_to_preclassified(
    preclassified: Dict[str, Dict], 
    config: ProcessingConfig
) -> Dict[str, Dict]:
    """
    Apply config-dependent calculations to pre-classified buildings.
    This is the cheap step - run for each optimization trial.
    
    ~10-50x faster than full classification since geometry is already done.
    """
    buildings = {}
    
    for bid, pre in preclassified.items():
        result = apply_building_config(pre, config)
        if result:
            buildings[bid] = result
    
    return buildings


def ensure_preclassified_exists(city_code: str, raw_data_dir: Path) -> Optional[Path]:
    """
    Ensure preclassified buildings exist for a city.
    If not, create them from raw buildings data.
    
    Returns path to preclassified file, or None if failed.
    """
    preclassified_path = raw_data_dir / 'buildings_preclassified.msgpack'
    
    if preclassified_path.exists():
        return preclassified_path
    
    # Check for raw buildings
    buildings_msgpack = raw_data_dir / 'buildings.msgpack'
    buildings_json = raw_data_dir / 'buildings.json'
    
    if not buildings_msgpack.exists() and not buildings_json.exists():
        print(f"  No raw buildings data found for {city_code}")
        return None
    
    print(f"  Creating preclassified buildings for {city_code}...")
    
    # Load raw buildings
    if buildings_msgpack.exists() and HAS_MSGPACK:
        with open(buildings_msgpack, 'rb') as f:
            raw_buildings = msgpack.unpack(f, raw=False)
    else:
        with open(buildings_json, 'r', encoding='utf-8') as f:
            raw_buildings = json.load(f)
    
    # Preclassify
    preclassified = preclassify_buildings_parallel(raw_buildings)
    
    # Save
    if save_preclassified_buildings(preclassified, preclassified_path):
        return preclassified_path
    
    return None


def classify_buildings_from_preclassified(
    preclassified_path: Path,
    config: ProcessingConfig
) -> Optional[Dict[str, Dict]]:
    """
    Load preclassified buildings and apply config.
    This is the fast path for optimization trials.
    """
    preclassified = load_preclassified_buildings(preclassified_path)
    if preclassified is None:
        return None
    
    return apply_config_to_preclassified(preclassified, config)


def assign_buildings_to_neighborhoods(
    buildings: Dict[str, Dict],
    neighborhood_centers: Dict[str, List[float]]
) -> Dict[str, List[str]]:
    """Assign each building to its nearest neighborhood using KDTree (vectorized batch query)"""
    if not buildings or not neighborhood_centers:
        return {}

    # Build arrays for KDTree
    place_ids = list(neighborhood_centers.keys())
    center_coords = np.array([neighborhood_centers[pid] for pid in place_ids])

    # Create KDTree for fast nearest-neighbor lookup
    tree = KDTree(center_coords)

    # Vectorized: extract all building IDs and centers at once
    building_ids = np.array(list(buildings.keys()))
    building_centers = np.array([buildings[bid]['center'] for bid in building_ids])
    
    # Batch query: find nearest neighborhood for ALL buildings at once (much faster)
    _, nearest_indices = tree.query(building_centers)
    
    # Initialize assignments
    assignments = {pid: [] for pid in place_ids}
    
    # Group buildings by neighborhood using numpy masking (vectorized)
    for i, place_id in enumerate(place_ids):
        mask = nearest_indices == i
        assigned_building_ids = building_ids[mask].tolist()
        assignments[place_id] = assigned_building_ids

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
    n = len(place_ids)

    # Precompute pairwise haversine distance matrix
    center_arr = np.array([centers[pid] for pid in place_ids])
    lons, lats = center_arr[:, 0], center_arr[:, 1]
    lat_rad = np.radians(lats)
    dlat = np.radians(lats[:, np.newaxis] - lats[np.newaxis, :])
    dlon = np.radians(lons[:, np.newaxis] - lons[np.newaxis, :])
    a = np.sin(dlat / 2) ** 2 + \
        np.cos(lat_rad[:, np.newaxis]) * np.cos(lat_rad[np.newaxis, :]) * np.sin(dlon / 2) ** 2
    dist_matrix = 2 * 6371000 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Precompute sizes
    sizes = np.array([metadata[pid]['total_population'] + metadata[pid]['total_jobs']
                       for pid in place_ids])

    merged = set()
    merged_count = 0

    for i, id1 in enumerate(place_ids):
        if id1 in merged:
            continue

        if sizes[i] >= min_size:
            continue

        # Vectorized: find ALL small, unmerged, nearby places in one operation
        nearby_mask = (dist_matrix[i] <= merge_dist) & \
                      (sizes < min_size) & \
                      (~np.isin(place_ids, list(merged)))
        
        # Exclude self
        nearby_mask[i] = False
        
        nearby_indices = np.where(nearby_mask)[0]
        
        if len(nearby_indices) == 0:
            continue
        
        # Create cluster from all nearby small places
        cluster = [id1] + [place_ids[j] for j in nearby_indices]
        cluster_pop = sum(metadata[cid]['total_population'] for cid in cluster)
        cluster_jobs = sum(metadata[cid]['total_jobs'] for cid in cluster)
        
        # Mark all as merged
        for cid in cluster[1:]:
            merged.add(cid)

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
    """Split large neighborhoods into smaller sub-neighborhoods (optimized)"""
    if config.max_place_size_for_splitting <= 0:
        return neighborhoods, centers, metadata, assignments

    max_size = config.max_place_size_for_splitting
    divisor = config.splitting_num_clusters_divisor
    min_sep = config.splitting_cluster_min_separation
    min_sep_sq = min_sep * min_sep  # Precompute squared for faster distance checks

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

        # Pre-extract building data into arrays (avoid repeated dict lookups)
        valid_bids = []
        valid_centers = []
        valid_pops = []
        valid_jobs = []
        
        for bid in assigned_buildings:
            b = buildings.get(bid)
            if b:
                valid_bids.append(bid)
                valid_centers.append(b['center'])
                valid_pops.append(b['approx_pop'])
                valid_jobs.append(b['approx_jobs'])

        if not valid_bids:
            continue

        # Convert to numpy arrays once
        building_ids = np.array(valid_bids)
        building_centers = np.array(valid_centers)
        building_pops = np.array(valid_pops)
        building_jobs = np.array(valid_jobs)
        weights = building_pops + building_jobs + 1
        
        total_weight = weights.sum()
        cum_weights = np.cumsum(weights)

        # Select diverse centers using weighted random sampling
        # Pre-allocate array for sub-center locations (max size)
        sub_center_locs = np.empty((num_clusters, 2), dtype=np.float64)
        sub_center_ids = []
        n_sub = 0
        max_attempts = num_clusters * 10

        for _ in range(max_attempts):
            if n_sub >= num_clusters:
                break

            # Weighted random selection (vectorized lookup)
            rand = random.random() * total_weight
            selected_idx = min(int(np.searchsorted(cum_weights, rand)), len(building_ids) - 1)
            candidate_loc = building_centers[selected_idx]

            # Check distance from existing centers using squared distance (avoid sqrt)
            if n_sub > 0:
                diffs = sub_center_locs[:n_sub] - candidate_loc
                dist_sq = (diffs * diffs).sum(axis=1)
                too_close = bool(np.any(dist_sq < min_sep_sq))
            else:
                too_close = False

            if not too_close:
                sub_id = f"{place_id}_split_{new_id_counter}"
                new_id_counter += 1
                sub_center_locs[n_sub] = candidate_loc
                sub_center_ids.append(sub_id)
                n_sub += 1

        if n_sub == 0:
            continue

        # Trim to actual size
        sub_center_locs = sub_center_locs[:n_sub]

        # Build KDTree for sub-centers and batch query ALL buildings at once
        sub_tree = KDTree(sub_center_locs)
        _, nearest_indices = sub_tree.query(building_centers)

        # Vectorized accumulation using bincount
        sub_pops = np.bincount(nearest_indices, weights=building_pops, minlength=n_sub)
        sub_jobs = np.bincount(nearest_indices, weights=building_jobs, minlength=n_sub)

        # Group building IDs by sub-neighborhood (vectorized masking)
        sub_building_lists = []
        for i in range(n_sub):
            mask = nearest_indices == i
            sub_building_lists.append(building_ids[mask].tolist())

        # Remove original
        neighborhoods.pop(place_id, None)
        centers.pop(place_id, None)
        metadata.pop(place_id, None)
        assignments.pop(place_id, None)

        # Add sub-neighborhoods
        for i in range(n_sub):
            pop_i = int(sub_pops[i])
            jobs_i = int(sub_jobs[i])
            
            if pop_i > 0 or jobs_i > 0:
                sub_id = sub_center_ids[i]
                neighborhoods[sub_id] = {'tags': {}}
                centers[sub_id] = sub_center_locs[i].tolist()
                assignments[sub_id] = sub_building_lists[i]
                metadata[sub_id] = {
                    'place_id': sub_id,
                    'name': f"{place.get('name', '')} ({sub_id.split('_')[-1]})",
                    'total_population': pop_i,
                    'total_jobs': jobs_i,
                    'pct_population': pop_i / total_population if total_population > 0 else 0,
                    'pct_jobs': jobs_i / total_jobs if total_jobs > 0 else 0
                }
                split_count += 1

    print(f"  Split into {split_count} smaller neighborhoods. Total: {len(metadata)}")
    return neighborhoods, centers, metadata, assignments


def generate_connections(
    metadata: Dict[str, Dict],
    centers: Dict[str, List[float]],
    config: ProcessingConfig
) -> List[Dict]:
    """Generate connections using gravity model (vectorized inner loop)"""
    connections = []
    total_connections_generated = 0

    place_ids = list(metadata.keys())
    n = len(place_ids)

    if n == 0:
        return []

    # Build coordinate array and scale to meters
    coords = np.array([centers[pid] for pid in place_ids])
    lat_mid = np.mean(coords[:, 1])
    lon_scale = math.cos(math.radians(lat_mid)) * 111320
    lat_scale = 111320

    scaled_coords = coords.copy()
    scaled_coords[:, 0] *= lon_scale
    scaled_coords[:, 1] *= lat_scale

    # Precompute pairwise distance matrix
    diff = scaled_coords[:, np.newaxis, :] - scaled_coords[np.newaxis, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=2))

    # Precompute arrays from metadata
    pop_array = np.array([metadata[pid]['total_population'] for pid in place_ids])
    jobs_array = np.array([metadata[pid]['total_jobs'] for pid in place_ids])

    # Precompute gravity matrix: gravity[i,j] = jobs[j] * pop[i] / effective_dist^exponent
    effective_dist = np.maximum(dist_matrix, config.gravity_distance_floor_meters)
    gravity_matrix = (jobs_array[np.newaxis, :] * pop_array[:, np.newaxis]) / \
                     (effective_dist ** config.gravity_exponent)

    # Zero out self-connections and targets with no jobs
    np.fill_diagonal(gravity_matrix, 0.0)
    gravity_matrix[:, jobs_array <= 0] = 0.0

    # Apply max distance cutoff
    if config.max_connection_distance_meters > 0:
        gravity_matrix[dist_matrix > config.max_connection_distance_meters] = 0.0

    for i, outer_id in enumerate(place_ids):
        total_demand = pop_array[i]

        if total_demand <= 5:
            continue

        # Probabilistically skip if approaching connection limit
        if config.max_total_connections > 0:
            if total_connections_generated >= config.max_total_connections:
                break
            elif total_connections_generated > config.max_total_connections * 0.8:
                progress = (total_connections_generated - config.max_total_connections * 0.8) / (config.max_total_connections * 0.2)
                skip_prob = progress * 0.9
                if random.random() < skip_prob:
                    continue

        # Get gravity scores from precomputed matrix
        grav_row = gravity_matrix[i]
        valid_mask = grav_row > 0
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            continue

        valid_gravities = grav_row[valid_indices]
        valid_distances = dist_matrix[i, valid_indices]
        total_gravity = valid_gravities.sum()
        probabilities = valid_gravities / total_gravity

        # Sort by gravity descending
        sort_order = np.argsort(-valid_gravities)
        valid_indices = valid_indices[sort_order]
        valid_gravities = valid_gravities[sort_order]
        valid_distances = valid_distances[sort_order]
        probabilities = probabilities[sort_order]

        # Determine number of connections
        num_connections = min(
            config.max_connections_per_cluster,
            max(config.min_connections_per_cluster,
                int(math.sqrt(total_demand / config.connection_scaling_divisor)))
        )
        num_connections = min(num_connections, len(valid_indices))

        # Weighted random selection
        selected_mask = np.zeros(len(valid_indices), dtype=bool)
        n_selected = 0
        max_attempts = len(valid_indices) * 5
        attempts = 0

        while n_selected < num_connections and attempts < max_attempts:
            attempts += 1
            # Renormalize over unselected items
            remaining_probs = probabilities.copy()
            remaining_probs[selected_mask] = 0.0
            prob_sum = remaining_probs.sum()
            if prob_sum <= 0:
                break
            remaining_probs /= prob_sum

            cumulative = np.cumsum(remaining_probs)
            rand = random.random()
            idx = np.searchsorted(cumulative, rand)
            if idx < len(selected_mask) and not selected_mask[idx]:
                selected_mask[idx] = True
                n_selected += 1

        if n_selected == 0:
            continue

        sel_indices = np.where(selected_mask)[0]
        sel_gravities = valid_gravities[sel_indices]
        sel_distances = valid_distances[sel_indices]
        sel_place_indices = valid_indices[sel_indices]
        sel_gravity_total = sel_gravities.sum()

        total_connections_generated += len(sel_indices)

        for k in range(len(sel_indices)):
            connection_size = round((sel_gravities[k] / sel_gravity_total) * total_demand)
            if connection_size <= 0:
                continue

            distance = float(sel_distances[k])
            seconds = distance * 0.12
            inner_id = place_ids[int(sel_place_indices[k])]

            # Split large connections
            remaining = connection_size
            while remaining > 0:
                chunk = min(config.connection_size_cap, remaining)
                connections.append({
                    'residenceId': outer_id,
                    'jobId': inner_id,
                    'size': chunk,
                    'drivingDistance': round(distance),
                    'drivingSeconds': round(seconds)
                })
                remaining -= chunk

    return connections


def process_place_connections(
    place: Place,
    raw_places: List[Dict],
    config: ProcessingConfig,
    preclassified_path: Path,
    buildings_msgpack_path: Optional[Path] = None,
    raw_buildings: Optional[List[Dict]] = None
) -> Dict:
    """Main processing function for demand data"""
    print(f"  Extracting neighborhoods...")
    neighborhoods, centers = extract_neighborhoods(raw_places)
    print(f"  Found {len(neighborhoods)} OSM neighborhoods")

    if not neighborhoods:
        print(f"  WARNING: No neighborhoods found!")
        return {'points': [], 'pops': []}

    # Classify buildings - always use fastest available path
    buildings = None
    
    # FAST PATH: Use preclassified buildings (geometry already done)
    if preclassified_path.exists():
        print(f"  Applying config to preclassified buildings...")
        buildings = classify_buildings_from_preclassified(preclassified_path, config)
        if buildings:
            print(f"  Applied config to {len(buildings)} buildings")
    
    # MEDIUM PATH: Hybrid streaming+parallel from msgpack
    if buildings is None and buildings_msgpack_path and buildings_msgpack_path.exists():
        print(f"  Classifying buildings (streaming+parallel)...")
        buildings = classify_buildings_stream_parallel(buildings_msgpack_path, config)
        if buildings is None:
            buildings = classify_buildings_streaming(buildings_msgpack_path, config)
    
    # FALLBACK: Full classification from raw data
    if buildings is None and raw_buildings:
        print(f"  Classifying {len(raw_buildings)} buildings (parallel)...")
        buildings = classify_buildings_parallel(raw_buildings, config)
    
    if buildings is None:
        buildings = {}
    
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
    preclassified_path = raw_data_dir / 'buildings_preclassified.msgpack'

    if not buildings_msgpack.exists() and not buildings_json.exists():
        raise FileNotFoundError(f"Buildings file not found: {buildings_json} or {buildings_msgpack}")
    if not places_msgpack.exists() and not places_json.exists():
        raise FileNotFoundError(f"Places file not found: {places_json} or {places_msgpack}")

    print(f"  Loading raw data...")
    
    # Ensure preclassified buildings exist (much faster for repeated runs)
    if not preclassified_path.exists():
        print(f"  Creating preclassified buildings cache...")
        ensure_preclassified_exists(place.code, raw_data_dir)
    
    # Only load raw buildings if preclassified and msgpack both unavailable
    raw_buildings = None
    if not preclassified_path.exists() and not buildings_msgpack.exists():
        raw_buildings = load_data_file(buildings_json)
    
    raw_places = load_data_file(places_json)  # Will auto-detect msgpack

    # Process connections/demand
    demand_data = process_place_connections(
        place, raw_places, config,
        preclassified_path=preclassified_path,
        buildings_msgpack_path=buildings_msgpack,
        raw_buildings=raw_buildings
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

    # Check for preclassify command
    if len(sys.argv) > 1 and sys.argv[1] == '--preclassify':
        # Preclassify all cities in config
        config_file = sys.argv[2] if len(sys.argv) > 2 else 'config.json'
        config_path = MAP_PATCHER_DIR / config_file
        
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        places = config_dict.get('places', [])
        print(f"Preclassifying buildings for {len(places)} cities...")
        print("This is a one-time operation that speeds up optimization trials.\n")
        
        for place_dict in places:
            code = place_dict['code']
            raw_data_dir = MAP_PATCHER_DIR / 'raw_data' / code
            preclassified_path = raw_data_dir / 'buildings_preclassified.msgpack'
            
            if preclassified_path.exists():
                size_mb = preclassified_path.stat().st_size / 1_000_000
                print(f"  {code}: Already preclassified ({size_mb:.1f}MB)")
                continue
            
            result = ensure_preclassified_exists(code, raw_data_dir)
            if result:
                print(f"  {code}: ✓ Preclassified successfully")
            else:
                print(f"  {code}: ✗ Failed to preclassify")
        
        print("\nPreclassification complete!")
        print("Run optimization with: python optimize_params.py --n-trials 100 --n-jobs 2")
        return

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
