/**
 * Extract Ground Truth Features
 *
 * Reads demand_data.json.gz from the game's AppData folder for each city
 * and computes metrics for comparison with generated data.
 *
 * Usage: node extract_ground_truth.js
 */

import fs from 'fs';
import path from 'path';
import zlib from 'zlib';
import os from 'os';

// Available cities in ground truth data
const CITIES = [
  'ATL', 'AUS', 'BAL', 'BIR', 'BOS', 'CHI', 'CIN', 'CLE', 'CLT', 'COL',
  'DAL', 'DC', 'DEN', 'DET', 'HNL', 'HOU', 'IND', 'LIV', 'LON', 'MAN',
  'MIA', 'MKE', 'MSP', 'NEW', 'NYC', 'PDX', 'PHL', 'PHX', 'PIT', 'SAN',
  'SEA', 'SF', 'SLC', 'STL'
];

// Get AppData path based on OS
function getAppDataPath() {
  if (process.platform === 'win32') {
    return process.env.APPDATA;
  } else if (process.platform === 'darwin') {
    return path.join(os.homedir(), 'Library', 'Application Support');
  } else {
    return path.join(os.homedir(), '.config');
  }
}

// Load gzipped JSON file
function loadGzippedJson(filePath) {
  const compressed = fs.readFileSync(filePath);
  const decompressed = zlib.gunzipSync(compressed);
  return JSON.parse(decompressed.toString('utf8'));
}

// Statistical helpers
function mean(arr) {
  if (arr.length === 0) return 0;
  return arr.reduce((sum, v) => sum + v, 0) / arr.length;
}

function median(arr) {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function std(arr) {
  if (arr.length === 0) return 0;
  const avg = mean(arr);
  const squaredDiffs = arr.map(v => (v - avg) ** 2);
  return Math.sqrt(mean(squaredDiffs));
}

function percentile(arr, p) {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[Math.max(0, idx)];
}

// Calculate distance between two points in meters (Haversine formula)
function haversineDistance(lon1, lat1, lon2, lat2) {
  const R = 6371000; // Earth radius in meters
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = Math.sin(dLat / 2) ** 2 +
            Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
            Math.sin(dLon / 2) ** 2;
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

// Calculate bounding box area in km²
function calculateAreaKm2(points) {
  if (points.length === 0) return 0;

  const lons = points.map(p => p.location[0]);
  const lats = points.map(p => p.location[1]);

  const minLon = Math.min(...lons);
  const maxLon = Math.max(...lons);
  const minLat = Math.min(...lats);
  const maxLat = Math.max(...lats);

  const width = haversineDistance(minLon, (minLat + maxLat) / 2, maxLon, (minLat + maxLat) / 2);
  const height = haversineDistance((minLon + maxLon) / 2, minLat, (minLon + maxLon) / 2, maxLat);

  return (width * height) / 1_000_000; // Convert to km²
}

// Calculate nearest neighbor distances for all points
function calculateNearestNeighborDistances(points) {
  const distances = [];

  for (let i = 0; i < points.length; i++) {
    let minDist = Infinity;
    const p1 = points[i];

    for (let j = 0; j < points.length; j++) {
      if (i === j) continue;
      const p2 = points[j];
      const dist = haversineDistance(
        p1.location[0], p1.location[1],
        p2.location[0], p2.location[1]
      );
      if (dist < minDist) minDist = dist;
    }

    if (minDist !== Infinity) {
      distances.push(minDist);
    }
  }

  return distances;
}

// Detect if ground truth is census-based or OSM-based
// Census IDs are typically 15-digit numbers (FIPS codes) or UK census codes (E000xxxxx)
// OSM-based IDs start with AIR_, UNI_, synthetic_, or are OSM node IDs
function detectDataSource(points) {
  if (points.length === 0) return 'unknown';

  // Sample first 10 non-special IDs
  const sampleIds = points
    .map(p => p.id)
    .filter(id => !id.startsWith('AIR_') && !id.startsWith('UNI_'))
    .slice(0, 10);

  if (sampleIds.length === 0) return 'osm'; // Only special places = OSM-based

  // Check if IDs look like census tracts
  const censusPatternsUS = /^\d{15}$/;  // US FIPS: 15 digits
  const censusPatternsUK = /^E\d{8}$/;  // UK: E + 8 digits
  const censusPatternsCA = /^\d{10}$/;  // Canada: 10 digits

  const censusCount = sampleIds.filter(id =>
    censusPatternsUS.test(id) || censusPatternsUK.test(id) || censusPatternsCA.test(id)
  ).length;

  return censusCount > sampleIds.length / 2 ? 'census' : 'osm';
}

// Extract features from demand data
function extractFeatures(data, cityCode) {
  const { points, pops } = data;
  const dataSource = detectDataSource(points);

  // Stage 2: Cluster Distribution Metrics
  const sizes = points.map(p => (p.jobs || 0) + (p.residents || 0));
  const totalJobs = points.reduce((sum, p) => sum + (p.jobs || 0), 0);
  const totalResidents = points.reduce((sum, p) => sum + (p.residents || 0), 0);
  const totalSize = totalJobs + totalResidents;

  const areaKm2 = calculateAreaKm2(points);
  const nnDistances = calculateNearestNeighborDistances(points);

  // Stage 3: Connection Distribution Metrics
  const connDistances = pops.map(p => p.drivingDistance || 0);
  const connSizes = pops.map(p => p.size || 0);

  // Count unique edges (residence-job pairs)
  const uniqueEdges = new Set();
  pops.forEach(p => {
    const edge = `${p.residenceId}->${p.jobId}`;
    uniqueEdges.add(edge);
  });

  // Possible edges = points with residents * points with jobs
  const residentialPoints = points.filter(p => (p.residents || 0) > 0).length;
  const jobPoints = points.filter(p => (p.jobs || 0) > 0).length;
  const possibleEdges = residentialPoints * jobPoints;

  return {
    city: cityCode,
    data_source: dataSource,  // 'census' or 'osm' - only census-based is true ground truth

    // Stage 2: Cluster Distribution
    cluster_count: points.length,
    size_mean: mean(sizes),
    size_median: median(sizes),
    size_std: std(sizes),
    size_p90: percentile(sizes, 90),
    size_min: Math.min(...sizes),
    size_max: Math.max(...sizes),
    total_population: totalResidents,
    total_jobs: totalJobs,
    jobs_ratio: totalSize > 0 ? totalJobs / totalSize : 0,
    nn_dist_mean: mean(nnDistances),
    nn_dist_median: median(nnDistances),
    nn_dist_p90: percentile(nnDistances, 90),
    area_km2: areaKm2,
    density: areaKm2 > 0 ? points.length / areaKm2 : 0,

    // Stage 3: Connection Distribution
    connection_count: pops.length,
    connections_per_cluster: points.length > 0 ? pops.length / points.length : 0,
    conn_dist_mean: mean(connDistances),
    conn_dist_median: median(connDistances),
    conn_dist_p90: percentile(connDistances, 90),
    conn_dist_min: connDistances.length > 0 ? Math.min(...connDistances) : 0,
    conn_dist_max: connDistances.length > 0 ? Math.max(...connDistances) : 0,
    conn_size_mean: mean(connSizes),
    conn_size_median: median(connSizes),
    conn_size_std: std(connSizes),
    unique_edges: uniqueEdges.size,
    graph_density: possibleEdges > 0 ? uniqueEdges.size / possibleEdges : 0,
  };
}

// Main execution
async function main() {
  const appDataPath = getAppDataPath();
  const gameDataPath = path.join(appDataPath, 'metro-maker4', 'cities', 'data');

  console.log(`Looking for ground truth data in: ${gameDataPath}`);

  if (!fs.existsSync(gameDataPath)) {
    console.error(`Game data folder not found: ${gameDataPath}`);
    console.error('Make sure Subway Builder is installed and has been run at least once.');
    process.exit(1);
  }

  const results = [];
  const errors = [];

  for (const city of CITIES) {
    const demandDataPath = path.join(gameDataPath, city, 'demand_data.json.gz');

    if (!fs.existsSync(demandDataPath)) {
      console.log(`  [SKIP] ${city}: No demand_data.json.gz found`);
      errors.push({ city, error: 'File not found' });
      continue;
    }

    try {
      console.log(`  [LOAD] ${city}...`);
      const data = loadGzippedJson(demandDataPath);
      const features = extractFeatures(data, city);
      results.push(features);
      console.log(`  [OK]   ${city}: ${features.cluster_count} clusters, ${features.connection_count} connections`);
    } catch (err) {
      console.log(`  [ERR]  ${city}: ${err.message}`);
      errors.push({ city, error: err.message });
    }
  }

  if (results.length === 0) {
    console.error('\nNo ground truth data could be loaded.');
    process.exit(1);
  }

  // Output to CSV
  const csvPath = path.join(import.meta.dirname, 'ground_truth_features.csv');
  const headers = Object.keys(results[0]);
  const csvContent = [
    headers.join(','),
    ...results.map(r => headers.map(h => {
      const val = r[h];
      return typeof val === 'number' ? val.toFixed(4) : val;
    }).join(','))
  ].join('\n');

  fs.writeFileSync(csvPath, csvContent);
  console.log(`\nSaved ${results.length} cities to: ${csvPath}`);

  // Output to JSON for easier programmatic access
  const jsonPath = path.join(import.meta.dirname, 'ground_truth_features.json');
  fs.writeFileSync(jsonPath, JSON.stringify(results, null, 2));
  console.log(`Saved JSON to: ${jsonPath}`);

  // Print summary statistics
  console.log('\n=== Ground Truth Summary ===');
  console.log(`Cities loaded: ${results.length}/${CITIES.length}`);

  const censusCities = results.filter(r => r.data_source === 'census');
  const osmCities = results.filter(r => r.data_source === 'osm');

  console.log(`\nData sources:`);
  console.log(`  Census-based (true ground truth): ${censusCities.length} cities`);
  console.log(`    ${censusCities.map(r => r.city).join(', ')}`);
  console.log(`  OSM-based (not comparable): ${osmCities.length} cities`);
  console.log(`    ${osmCities.map(r => r.city).join(', ')}`);

  console.log(`\nCensus-based cities stats:`);
  if (censusCities.length > 0) {
    console.log(`  Cluster counts: ${Math.min(...censusCities.map(r => r.cluster_count))} - ${Math.max(...censusCities.map(r => r.cluster_count))}`);
    console.log(`  Avg size/cluster: ${mean(censusCities.map(r => r.size_mean)).toFixed(0)}`);
    console.log(`  Avg connections/cluster: ${mean(censusCities.map(r => r.connections_per_cluster)).toFixed(2)}`);
    console.log(`  Avg connection distance: ${mean(censusCities.map(r => r.conn_dist_median)).toFixed(0)}m`);
  }

  if (errors.length > 0) {
    console.log(`\nCities with errors: ${errors.map(e => e.city).join(', ')}`);
  }
}

main().catch(console.error);
