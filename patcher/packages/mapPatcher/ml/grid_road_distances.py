"""
Grid-based road distance precomputation using OSMnx and NetworkX.

This module downloads the road network for a bounding box, creates a grid of
waypoints snapped to road intersections, and precomputes shortest path distances
between all grid cells. During demand generation, road distances are approximated as:

    straight_line(origin → nearest_grid_node) +
    precomputed_road_distance(grid_node_A → grid_node_B) +
    straight_line(nearest_grid_node → destination)

Usage:
    from grid_road_distances import GridRoadDistances
    
    # Precompute (slow, do once per city)
    grid = GridRoadDistances.from_bbox(bbox, cell_size_meters=500)
    grid.save("city_grid.npz")
    
    # Load and use (fast)
    grid = GridRoadDistances.load("city_grid.npz")
    distances = grid.get_road_distances(origins, destinations)  # vectorized
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from typing import Tuple, List, Optional, Union
from pathlib import Path
import math
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
import json

# Optional imports with graceful fallback
try:
    import osmnx as ox
    HAS_OSMNX = True
except ImportError:
    HAS_OSMNX = False
    warnings.warn("osmnx not installed. Install with: pip install osmnx")

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    warnings.warn("networkx not installed. Install with: pip install networkx")


# Earth radius in meters
EARTH_RADIUS_M = 6371000


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate haversine distance between two points in meters."""
    lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = math.sin(dlat / 2) ** 2 + \
        math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def haversine_distance_vectorized(
    lons1: np.ndarray, lats1: np.ndarray,
    lons2: np.ndarray, lats2: np.ndarray
) -> np.ndarray:
    """Vectorized haversine distance calculation."""
    lat1_rad = np.radians(lats1)
    lat2_rad = np.radians(lats2)
    dlat = np.radians(lats2 - lats1)
    dlon = np.radians(lons2 - lons1)
    
    a = np.sin(dlat / 2) ** 2 + \
        np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


@dataclass
class GridMetadata:
    """Metadata about the road distance grid."""
    bbox: Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
    cell_size_meters: float
    num_nodes: int
    num_edges: int
    grid_shape: Tuple[int, int]  # (rows, cols)
    created_at: str
    osmnx_version: Optional[str] = None


class GridRoadDistances:
    """
    Precomputed grid-based road distance lookup.
    
    Attributes:
        grid_coords: (N, 2) array of [lon, lat] for each grid node
        distance_matrix: (N, N) array of road distances in meters
        kdtree: KDTree for fast nearest-node lookup
        metadata: GridMetadata with bbox, cell size, etc.
    """
    
    def __init__(
        self,
        grid_coords: np.ndarray,
        distance_matrix: np.ndarray,
        metadata: GridMetadata
    ):
        self.grid_coords = grid_coords
        self.distance_matrix = distance_matrix
        self.metadata = metadata
        
        # Build KDTree for fast spatial lookup
        # Convert to approximate Cartesian for KDTree (good enough for local areas)
        lat_mid = np.mean(grid_coords[:, 1])
        self._lon_scale = math.cos(math.radians(lat_mid)) * 111320
        self._lat_scale = 111320
        
        scaled_coords = grid_coords.copy()
        scaled_coords[:, 0] *= self._lon_scale
        scaled_coords[:, 1] *= self._lat_scale
        self.kdtree = KDTree(scaled_coords)
        self._scaled_coords = scaled_coords
    
    @classmethod
    def from_bbox(
        cls,
        bbox: Tuple[float, float, float, float],
        cell_size_meters: float = 500,
        network_type: str = "drive",
        simplify: bool = True,
        verbose: bool = True
    ) -> "GridRoadDistances":
        """
        Create a GridRoadDistances object from a bounding box.
        
        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat)
            cell_size_meters: Grid cell size in meters (default 500m)
            network_type: OSMnx network type ("drive", "walk", "bike", "all")
            simplify: Whether to simplify the graph (removes degree-2 nodes)
            verbose: Print progress messages
        
        Returns:
            GridRoadDistances object with precomputed distances
        """
        if not HAS_OSMNX or not HAS_NETWORKX:
            raise ImportError("osmnx and networkx are required. Install with: pip install osmnx networkx")
        
        min_lon, min_lat, max_lon, max_lat = bbox
        
        if verbose:
            print(f"Downloading road network for bbox: {bbox}")
        
        # Download road network from OSM
        # OSMnx 2.0+ expects bbox as (left, bottom, right, top) = (min_lon, min_lat, max_lon, max_lat)
        G = ox.graph_from_bbox(
            bbox=(min_lon, min_lat, max_lon, max_lat),
            network_type=network_type,
            simplify=simplify
        )
        
        if verbose:
            print(f"  Downloaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Add edge lengths if not present
        G = ox.distance.add_edge_lengths(G)
        
        # Create grid of waypoints
        grid_coords, grid_node_ids = cls._create_grid_waypoints(
            G, bbox, cell_size_meters, verbose
        )
        
        if len(grid_node_ids) == 0:
            raise ValueError("No grid waypoints could be created. Check if bbox contains roads.")
        
        if verbose:
            print(f"  Created {len(grid_node_ids)} grid waypoints")
        
        # Compute all-pairs shortest paths
        distance_matrix = cls._compute_distance_matrix(
            G, grid_node_ids, verbose
        )
        
        # Calculate grid shape
        lon_span = max_lon - min_lon
        lat_span = max_lat - min_lat
        lon_cells = max(1, int(lon_span * 111320 * math.cos(math.radians((min_lat + max_lat) / 2)) / cell_size_meters))
        lat_cells = max(1, int(lat_span * 111320 / cell_size_meters))
        
        metadata = GridMetadata(
            bbox=bbox,
            cell_size_meters=cell_size_meters,
            num_nodes=G.number_of_nodes(),
            num_edges=G.number_of_edges(),
            grid_shape=(lat_cells, lon_cells),
            created_at=str(np.datetime64('now')),
            osmnx_version=ox.__version__ if HAS_OSMNX else None
        )
        
        return cls(grid_coords, distance_matrix, metadata)
    
    @classmethod
    def from_geojson(
        cls,
        geojson_path: Union[str, Path],
        bbox: Tuple[float, float, float, float],
        cell_size_meters: float = 500,
        verbose: bool = True
    ) -> "GridRoadDistances":
        """
        Create from an existing roads.geojson file (avoids re-downloading).
        
        Args:
            geojson_path: Path to roads.geojson file
            bbox: Bounding box for grid creation
            cell_size_meters: Grid cell size in meters
            verbose: Print progress
        
        Returns:
            GridRoadDistances object
        """
        if not HAS_NETWORKX:
            raise ImportError("networkx is required. Install with: pip install networkx")
        
        if verbose:
            print(f"Loading road network from {geojson_path}")
        
        with open(geojson_path, 'r', encoding='utf-8') as f:
            geojson = json.load(f)
        
        # Build NetworkX graph from GeoJSON
        G = nx.Graph()
        
        node_id = 0
        node_coords = {}  # (lon, lat) -> node_id
        
        for feature in geojson.get('features', []):
            geom = feature.get('geometry', {})
            if geom.get('type') != 'LineString':
                continue
            
            coords = geom.get('coordinates', [])
            if len(coords) < 2:
                continue
            
            # Add nodes and edges for this road segment
            prev_node = None
            segment_length = 0
            
            for i, coord in enumerate(coords):
                lon, lat = coord[0], coord[1]
                coord_key = (round(lon, 7), round(lat, 7))
                
                if coord_key in node_coords:
                    curr_node = node_coords[coord_key]
                else:
                    curr_node = node_id
                    node_coords[coord_key] = node_id
                    G.add_node(node_id, x=lon, y=lat)
                    node_id += 1
                
                if prev_node is not None:
                    prev_lon, prev_lat = G.nodes[prev_node]['x'], G.nodes[prev_node]['y']
                    edge_length = haversine_distance(prev_lon, prev_lat, lon, lat)
                    
                    if G.has_edge(prev_node, curr_node):
                        # Keep shorter edge
                        if edge_length < G[prev_node][curr_node]['length']:
                            G[prev_node][curr_node]['length'] = edge_length
                    else:
                        G.add_edge(prev_node, curr_node, length=edge_length)
                
                prev_node = curr_node
        
        if verbose:
            print(f"  Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Create grid waypoints
        grid_coords, grid_node_ids = cls._create_grid_waypoints_nx(
            G, bbox, cell_size_meters, verbose
        )
        
        if len(grid_node_ids) == 0:
            raise ValueError("No grid waypoints could be created.")
        
        # Compute distance matrix
        distance_matrix = cls._compute_distance_matrix_nx(
            G, grid_node_ids, verbose
        )
        
        # Calculate grid shape
        min_lon, min_lat, max_lon, max_lat = bbox
        lon_span = max_lon - min_lon
        lat_span = max_lat - min_lat
        lon_cells = max(1, int(lon_span * 111320 * math.cos(math.radians((min_lat + max_lat) / 2)) / cell_size_meters))
        lat_cells = max(1, int(lat_span * 111320 / cell_size_meters))
        
        metadata = GridMetadata(
            bbox=bbox,
            cell_size_meters=cell_size_meters,
            num_nodes=G.number_of_nodes(),
            num_edges=G.number_of_edges(),
            grid_shape=(lat_cells, lon_cells),
            created_at=str(np.datetime64('now')),
            osmnx_version=None
        )
        
        return cls(grid_coords, distance_matrix, metadata)
    
    @staticmethod
    def _create_grid_waypoints(
        G,  # OSMnx graph
        bbox: Tuple[float, float, float, float],
        cell_size_meters: float,
        verbose: bool
    ) -> Tuple[np.ndarray, List]:
        """Create grid of waypoints snapped to nearest road nodes."""
        min_lon, min_lat, max_lon, max_lat = bbox
        
        # Calculate grid dimensions
        lat_mid = (min_lat + max_lat) / 2
        lon_scale = math.cos(math.radians(lat_mid)) * 111320  # meters per degree lon
        lat_scale = 111320  # meters per degree lat
        
        lon_step = cell_size_meters / lon_scale
        lat_step = cell_size_meters / lat_scale
        
        # Generate grid points
        lons = np.arange(min_lon, max_lon + lon_step, lon_step)
        lats = np.arange(min_lat, max_lat + lat_step, lat_step)
        
        if verbose:
            print(f"  Grid dimensions: {len(lats)} x {len(lons)} = {len(lats) * len(lons)} cells")
        
        # Get all node coordinates from graph
        node_ids = list(G.nodes())
        node_coords = np.array([[G.nodes[n]['x'], G.nodes[n]['y']] for n in node_ids])
        
        # Build KDTree for node lookup
        scaled_node_coords = node_coords.copy()
        scaled_node_coords[:, 0] *= lon_scale
        scaled_node_coords[:, 1] *= lat_scale
        node_tree = KDTree(scaled_node_coords)
        
        # Snap each grid point to nearest road node
        grid_coords = []
        grid_node_ids = []
        seen_nodes = set()
        
        for lat in lats:
            for lon in lons:
                # Find nearest road node
                query_point = np.array([lon * lon_scale, lat * lat_scale])
                dist, idx = node_tree.query(query_point)
                
                # Skip if too far from any road (> 2x cell size)
                if dist > cell_size_meters * 2:
                    continue
                
                node_id = node_ids[idx]
                
                # Avoid duplicate nodes
                if node_id in seen_nodes:
                    continue
                seen_nodes.add(node_id)
                
                grid_coords.append([node_coords[idx, 0], node_coords[idx, 1]])
                grid_node_ids.append(node_id)
        
        return np.array(grid_coords), grid_node_ids
    
    @staticmethod
    def _create_grid_waypoints_nx(
        G: "nx.Graph",
        bbox: Tuple[float, float, float, float],
        cell_size_meters: float,
        verbose: bool
    ) -> Tuple[np.ndarray, List]:
        """Create grid waypoints for a plain NetworkX graph."""
        min_lon, min_lat, max_lon, max_lat = bbox
        
        lat_mid = (min_lat + max_lat) / 2
        lon_scale = math.cos(math.radians(lat_mid)) * 111320
        lat_scale = 111320
        
        lon_step = cell_size_meters / lon_scale
        lat_step = cell_size_meters / lat_scale
        
        lons = np.arange(min_lon, max_lon + lon_step, lon_step)
        lats = np.arange(min_lat, max_lat + lat_step, lat_step)
        
        if verbose:
            print(f"  Grid dimensions: {len(lats)} x {len(lons)} = {len(lats) * len(lons)} cells")
        
        # Get node coordinates
        node_ids = list(G.nodes())
        node_coords = np.array([[G.nodes[n]['x'], G.nodes[n]['y']] for n in node_ids])
        
        scaled_node_coords = node_coords.copy()
        scaled_node_coords[:, 0] *= lon_scale
        scaled_node_coords[:, 1] *= lat_scale
        node_tree = KDTree(scaled_node_coords)
        
        grid_coords = []
        grid_node_ids = []
        seen_nodes = set()
        
        for lat in lats:
            for lon in lons:
                query_point = np.array([lon * lon_scale, lat * lat_scale])
                dist, idx = node_tree.query(query_point)
                
                if dist > cell_size_meters * 2:
                    continue
                
                node_id = node_ids[idx]
                if node_id in seen_nodes:
                    continue
                seen_nodes.add(node_id)
                
                grid_coords.append([node_coords[idx, 0], node_coords[idx, 1]])
                grid_node_ids.append(node_id)
        
        return np.array(grid_coords) if grid_coords else np.array([]).reshape(0, 2), grid_node_ids
    
    @staticmethod
    def _compute_distance_matrix(
        G,  # OSMnx graph
        grid_node_ids: List,
        verbose: bool
    ) -> np.ndarray:
        """Compute all-pairs shortest path distances using scipy sparse Dijkstra."""
        n = len(grid_node_ids)
        
        if verbose:
            print(f"  Computing {n}x{n} = {n*n:,} shortest paths...")
        
        # Create node ID to index mapping for the full graph
        all_nodes = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(all_nodes)}
        
        # Build sparse adjacency matrix
        n_total = len(all_nodes)
        row, col, data = [], [], []
        
        for u, v, edge_data in G.edges(data=True):
            length = edge_data.get('length', 1.0)
            u_idx, v_idx = node_to_idx[u], node_to_idx[v]
            # Add both directions (undirected)
            row.extend([u_idx, v_idx])
            col.extend([v_idx, u_idx])
            data.extend([length, length])
        
        adj_matrix = csr_matrix((data, (row, col)), shape=(n_total, n_total))
        
        # Get indices of grid nodes in the full graph
        grid_indices = [node_to_idx[node_id] for node_id in grid_node_ids]
        
        # Compute shortest paths from each grid node
        # Use dijkstra with indices parameter to only compute to relevant nodes
        dist_matrix = np.full((n, n), np.inf, dtype=np.float32)
        
        # Process in batches for memory efficiency
        batch_size = max(1, min(100, n))
        
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch_indices = grid_indices[batch_start:batch_end]
            
            # Compute distances from this batch of sources to all nodes
            distances = dijkstra(
                adj_matrix,
                directed=False,
                indices=batch_indices,
                return_predecessors=False
            )
            
            # Extract only the distances to other grid nodes
            for i, src_idx in enumerate(range(batch_start, batch_end)):
                for j, dest_graph_idx in enumerate(grid_indices):
                    dist_matrix[src_idx, j] = distances[i, dest_graph_idx]
            
            if verbose and (batch_end % 500 == 0 or batch_end == n):
                print(f"    Processed {batch_end}/{n} source nodes...")
        
        # Replace inf with a large number (for unreachable pairs, use circuity estimate)
        inf_mask = np.isinf(dist_matrix)
        if inf_mask.any():
            if verbose:
                print(f"  Warning: {inf_mask.sum()} unreachable pairs, using fallback distances")
            # For unreachable pairs, estimate as 1.5x straight-line distance
            # This handles disconnected graph components
        
        return dist_matrix
    
    @staticmethod
    def _compute_distance_matrix_nx(
        G: "nx.Graph",
        grid_node_ids: List,
        verbose: bool
    ) -> np.ndarray:
        """Compute distance matrix for plain NetworkX graph."""
        n = len(grid_node_ids)
        
        if verbose:
            print(f"  Computing {n}x{n} = {n*n:,} shortest paths...")
        
        # Build sparse matrix
        all_nodes = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(all_nodes)}
        n_total = len(all_nodes)
        
        row, col, data = [], [], []
        for u, v, edge_data in G.edges(data=True):
            length = edge_data.get('length', 1.0)
            u_idx, v_idx = node_to_idx[u], node_to_idx[v]
            row.extend([u_idx, v_idx])
            col.extend([v_idx, u_idx])
            data.extend([length, length])
        
        adj_matrix = csr_matrix((data, (row, col)), shape=(n_total, n_total))
        
        grid_indices = [node_to_idx[node_id] for node_id in grid_node_ids]
        
        dist_matrix = np.full((n, n), np.inf, dtype=np.float32)
        
        batch_size = max(1, min(100, n))
        
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch_indices = grid_indices[batch_start:batch_end]
            
            distances = dijkstra(
                adj_matrix,
                directed=False,
                indices=batch_indices,
                return_predecessors=False
            )
            
            for i, src_idx in enumerate(range(batch_start, batch_end)):
                for j, dest_graph_idx in enumerate(grid_indices):
                    dist_matrix[src_idx, j] = distances[i, dest_graph_idx]
            
            if verbose and (batch_end % 500 == 0 or batch_end == n):
                print(f"    Processed {batch_end}/{n} source nodes...")
        
        return dist_matrix
    
    def save(self, path: Union[str, Path]) -> None:
        """Save to .npz file."""
        path = Path(path)
        
        # Serialize metadata
        metadata_dict = {
            'bbox': self.metadata.bbox,
            'cell_size_meters': self.metadata.cell_size_meters,
            'num_nodes': self.metadata.num_nodes,
            'num_edges': self.metadata.num_edges,
            'grid_shape': self.metadata.grid_shape,
            'created_at': self.metadata.created_at,
            'osmnx_version': self.metadata.osmnx_version or ''
        }
        
        np.savez_compressed(
            path,
            grid_coords=self.grid_coords,
            distance_matrix=self.distance_matrix,
            metadata=json.dumps(metadata_dict)
        )
        
        print(f"Saved grid road distances to {path}")
        print(f"  Grid nodes: {len(self.grid_coords)}")
        print(f"  File size: {path.stat().st_size / 1024 / 1024:.2f} MB")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "GridRoadDistances":
        """Load from .npz file."""
        path = Path(path)
        
        data = np.load(path, allow_pickle=True)
        
        grid_coords = data['grid_coords']
        distance_matrix = data['distance_matrix']
        metadata_dict = json.loads(str(data['metadata']))
        
        metadata = GridMetadata(
            bbox=tuple(metadata_dict['bbox']),
            cell_size_meters=metadata_dict['cell_size_meters'],
            num_nodes=metadata_dict['num_nodes'],
            num_edges=metadata_dict['num_edges'],
            grid_shape=tuple(metadata_dict['grid_shape']),
            created_at=metadata_dict['created_at'],
            osmnx_version=metadata_dict.get('osmnx_version') or None
        )
        
        return cls(grid_coords, distance_matrix, metadata)
    
    def get_nearest_node_indices(self, coords: np.ndarray) -> np.ndarray:
        """
        Get the indices of the nearest grid nodes for an array of coordinates.
        
        Args:
            coords: (N, 2) array of [lon, lat] coordinates
        
        Returns:
            (N,) array of grid node indices
        """
        # Scale to match KDTree
        scaled = coords.copy()
        scaled[:, 0] *= self._lon_scale
        scaled[:, 1] *= self._lat_scale
        
        _, indices = self.kdtree.query(scaled)
        return indices
    
    def get_road_distances(
        self,
        origins: np.ndarray,
        destinations: np.ndarray,
        fallback_circuity: float = 1.4
    ) -> np.ndarray:
        """
        Get road distances between origin-destination pairs.
        
        This snaps each origin/destination to the nearest grid node,
        looks up the precomputed road distance, and adds straight-line
        distances from the actual points to their grid nodes.
        
        Args:
            origins: (N, 2) array of [lon, lat] origin coordinates
            destinations: (N, 2) array of [lon, lat] destination coordinates
            fallback_circuity: Circuity factor for unreachable pairs
        
        Returns:
            (N,) array of road distances in meters
        """
        n = len(origins)
        
        if n == 0:
            return np.array([])
        
        # Get nearest grid nodes
        origin_indices = self.get_nearest_node_indices(origins)
        dest_indices = self.get_nearest_node_indices(destinations)
        
        # Get grid node coordinates
        origin_grid_coords = self.grid_coords[origin_indices]
        dest_grid_coords = self.grid_coords[dest_indices]
        
        # Calculate straight-line distances to/from grid nodes
        dist_to_grid = haversine_distance_vectorized(
            origins[:, 0], origins[:, 1],
            origin_grid_coords[:, 0], origin_grid_coords[:, 1]
        )
        
        dist_from_grid = haversine_distance_vectorized(
            dest_grid_coords[:, 0], dest_grid_coords[:, 1],
            destinations[:, 0], destinations[:, 1]
        )
        
        # Look up precomputed road distances
        road_distances = self.distance_matrix[origin_indices, dest_indices]
        
        # Total distance = to_grid + road + from_grid
        total_distances = dist_to_grid + road_distances + dist_from_grid
        
        # Handle unreachable pairs (inf in distance matrix)
        inf_mask = np.isinf(road_distances)
        if inf_mask.any():
            # Fall back to circuity-adjusted straight-line distance
            straight_line = haversine_distance_vectorized(
                origins[inf_mask, 0], origins[inf_mask, 1],
                destinations[inf_mask, 0], destinations[inf_mask, 1]
            )
            total_distances[inf_mask] = straight_line * fallback_circuity
        
        return total_distances
    
    def get_road_distance_matrix(
        self,
        points: np.ndarray,
        fallback_circuity: float = 1.4
    ) -> np.ndarray:
        """
        Get pairwise road distances between all points.
        
        Args:
            points: (N, 2) array of [lon, lat] coordinates
            fallback_circuity: Circuity factor for unreachable pairs
        
        Returns:
            (N, N) distance matrix in meters
        """
        n = len(points)
        
        if n == 0:
            return np.array([]).reshape(0, 0)
        
        # Get nearest grid nodes for all points
        point_indices = self.get_nearest_node_indices(points)
        point_grid_coords = self.grid_coords[point_indices]
        
        # Calculate distances from points to their grid nodes
        dist_to_grid = haversine_distance_vectorized(
            points[:, 0], points[:, 1],
            point_grid_coords[:, 0], point_grid_coords[:, 1]
        )
        
        # Get the submatrix of road distances between grid nodes
        road_dist_submatrix = self.distance_matrix[point_indices][:, point_indices]
        
        # Total distance matrix = dist_to_grid[i] + road_dist[i,j] + dist_to_grid[j]
        # Broadcasting: (N, 1) + (N, N) + (1, N)
        total_matrix = (
            dist_to_grid[:, np.newaxis] +
            road_dist_submatrix +
            dist_to_grid[np.newaxis, :]
        )
        
        # Handle unreachable pairs
        inf_mask = np.isinf(road_dist_submatrix)
        if inf_mask.any():
            # Compute straight-line distances for all pairs
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            lat_mid = np.mean(points[:, 1])
            lon_scale = math.cos(math.radians(lat_mid)) * 111320
            lat_scale = 111320
            
            scaled_diff = diff.copy()
            scaled_diff[:, :, 0] *= lon_scale
            scaled_diff[:, :, 1] *= lat_scale
            
            straight_line = np.sqrt((scaled_diff ** 2).sum(axis=2))
            total_matrix[inf_mask] = straight_line[inf_mask] * fallback_circuity
        
        return total_matrix


def precompute_grid_for_city(
    city_code: str,
    bbox: Tuple[float, float, float, float],
    output_dir: Union[str, Path],
    cell_size_meters: float = 500,
    use_existing_roads: bool = True,
    verbose: bool = True
) -> Path:
    """
    Convenience function to precompute grid distances for a city.
    
    Args:
        city_code: City code (e.g., "TYO_GEN")
        bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
        output_dir: Directory to save the grid file
        cell_size_meters: Grid cell size in meters
        use_existing_roads: Try to use roads.geojson if available
        verbose: Print progress
    
    Returns:
        Path to the saved grid file
    """
    output_dir = Path(output_dir)
    output_path = output_dir / f"{city_code}_grid_distances.npz"
    
    # Try to use existing roads.geojson first
    roads_path = output_dir / "roads.geojson"
    
    if use_existing_roads and roads_path.exists():
        if verbose:
            print(f"Using existing roads.geojson for {city_code}")
        grid = GridRoadDistances.from_geojson(roads_path, bbox, cell_size_meters, verbose)
    else:
        if verbose:
            print(f"Downloading road network from OSM for {city_code}")
        grid = GridRoadDistances.from_bbox(bbox, cell_size_meters, verbose=verbose)
    
    grid.save(output_path)
    return output_path


# CLI for standalone usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Precompute grid road distances for a city")
    parser.add_argument("--bbox", type=float, nargs=4, required=True,
                        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
                        help="Bounding box coordinates")
    parser.add_argument("--city-code", type=str, required=True,
                        help="City code (e.g., TYO_GEN)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for grid file")
    parser.add_argument("--cell-size", type=float, default=500,
                        help="Grid cell size in meters (default: 500)")
    parser.add_argument("--use-geojson", type=str, default=None,
                        help="Path to roads.geojson to use instead of downloading")
    
    args = parser.parse_args()
    
    bbox = tuple(args.bbox)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.use_geojson:
        grid = GridRoadDistances.from_geojson(
            args.use_geojson, bbox, args.cell_size, verbose=True
        )
    else:
        grid = GridRoadDistances.from_bbox(
            bbox, args.cell_size, verbose=True
        )
    
    output_path = output_dir / f"{args.city_code}_grid_distances.npz"
    grid.save(output_path)
