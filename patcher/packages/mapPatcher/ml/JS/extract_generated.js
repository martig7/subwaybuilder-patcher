/**
 * Extract Generated Features
 *
 * Reads demand_data.json from the processed_data folder for each city
 * and computes the same metrics as extract_ground_truth.js for comparison.
 *
 * Usage: node extract_generated.js
 */

import fs from 'fs';
import path from 'path';

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

// Extract features from demand data
function extractFeatures(data, cityCode) {
  const { points, pops } = data;

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
  const processedDataPath = path.join(import.meta.dirname, '..', 'processed_data');

  console.log(`Looking for generated data in: ${processedDataPath}`);

  if (!fs.existsSync(processedDataPath)) {
    console.error(`Processed data folder not found: ${processedDataPath}`);
    console.error('Run process_data.js first to generate demand data.');
    process.exit(1);
  }

  // Find all city folders
  const cityFolders = fs.readdirSync(processedDataPath).filter(f => {
    const fullPath = path.join(processedDataPath, f);
    return fs.statSync(fullPath).isDirectory();
  });

  if (cityFolders.length === 0) {
    console.error('No city folders found in processed_data.');
    process.exit(1);
  }

  console.log(`Found ${cityFolders.length} city folder(s): ${cityFolders.join(', ')}`);

  const results = [];
  const errors = [];

  for (const city of cityFolders) {
    const demandDataPath = path.join(processedDataPath, city, 'demand_data.json');

    if (!fs.existsSync(demandDataPath)) {
      console.log(`  [SKIP] ${city}: No demand_data.json found`);
      errors.push({ city, error: 'File not found' });
      continue;
    }

    try {
      console.log(`  [LOAD] ${city}...`);
      const data = JSON.parse(fs.readFileSync(demandDataPath, 'utf8'));
      const features = extractFeatures(data, city);
      results.push(features);
      console.log(`  [OK]   ${city}: ${features.cluster_count} clusters, ${features.connection_count} connections`);
    } catch (err) {
      console.log(`  [ERR]  ${city}: ${err.message}`);
      errors.push({ city, error: err.message });
    }
  }

  if (results.length === 0) {
    console.error('\nNo generated data could be loaded.');
    process.exit(1);
  }

  // Output to CSV
  const csvPath = path.join(import.meta.dirname, 'generated_features.csv');
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
  const jsonPath = path.join(import.meta.dirname, 'generated_features.json');
  fs.writeFileSync(jsonPath, JSON.stringify(results, null, 2));
  console.log(`Saved JSON to: ${jsonPath}`);

  // Print summary statistics
  console.log('\n=== Generated Data Summary ===');
  console.log(`Cities loaded: ${results.length}`);
  console.log(`\nCluster counts: ${Math.min(...results.map(r => r.cluster_count))} - ${Math.max(...results.map(r => r.cluster_count))}`);
  console.log(`Avg size/cluster: ${mean(results.map(r => r.size_mean)).toFixed(0)}`);
  console.log(`Avg connections/cluster: ${mean(results.map(r => r.connections_per_cluster)).toFixed(2)}`);
  console.log(`Avg connection distance: ${mean(results.map(r => r.conn_dist_median)).toFixed(0)}m`);

  if (errors.length > 0) {
    console.log(`\nCities with errors: ${errors.map(e => e.city).join(', ')}`);
  }
}

main().catch(console.error);
