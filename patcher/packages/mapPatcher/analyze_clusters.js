/**
 * Clustering Quality Analysis Tool
 *
 * Analyzes the clustering output from process_data.js and provides metrics on:
 * - Cluster sizes (population/jobs)
 * - Inter-cluster distances
 * - Cluster density and distribution
 * - Recommendations for improvement
 *
 * Usage: node analyze_clusters.js [city_code]
 * Example: node analyze_clusters.js IST
 */

import fs from 'fs';
import config from './config.js';

// Haversine distance in meters
function haversineDistance(coord1, coord2) {
  const R = 6371000; // Earth's radius in meters
  const lat1 = coord1[1] * Math.PI / 180;
  const lat2 = coord2[1] * Math.PI / 180;
  const dLat = (coord2[1] - coord1[1]) * Math.PI / 180;
  const dLon = (coord2[0] - coord1[0]) * Math.PI / 180;

  const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(lat1) * Math.cos(lat2) *
            Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

// Calculate statistics for an array
function stats(arr) {
  if (arr.length === 0) return { min: 0, max: 0, mean: 0, median: 0, std: 0, p10: 0, p90: 0 };

  const sorted = [...arr].sort((a, b) => a - b);
  const sum = arr.reduce((a, b) => a + b, 0);
  const mean = sum / arr.length;
  const variance = arr.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / arr.length;

  return {
    min: sorted[0],
    max: sorted[sorted.length - 1],
    mean: mean,
    median: sorted[Math.floor(sorted.length / 2)],
    std: Math.sqrt(variance),
    p10: sorted[Math.floor(sorted.length * 0.1)],
    p90: sorted[Math.floor(sorted.length * 0.9)],
  };
}

function analyzeClusterQuality(placeCode) {
  const demandPath = `${import.meta.dirname}/processed_data/${placeCode}/demand_data.json`;

  if (!fs.existsSync(demandPath)) {
    console.error(`No demand_data.json found for ${placeCode}`);
    console.error(`Run process_data.js first, or check that ${placeCode} is configured.`);
    return null;
  }

  const demandData = JSON.parse(fs.readFileSync(demandPath, 'utf8'));
  const points = demandData.points;
  const connections = demandData.pops;

  console.log(`\n${'='.repeat(60)}`);
  console.log(`CLUSTERING ANALYSIS FOR: ${placeCode}`);
  console.log(`${'='.repeat(60)}\n`);

  // === BASIC COUNTS ===
  console.log(`BASIC COUNTS:`);
  console.log(`  Total clusters (demand points): ${points.length}`);
  console.log(`  Total connections: ${connections.length}`);
  console.log();

  // === CLUSTER SIZE ANALYSIS ===
  const populations = points.map(p => p.residents);
  const jobs = points.map(p => p.jobs);
  const totalSizes = points.map(p => p.residents + p.jobs);

  const popStats = stats(populations);
  const jobStats = stats(jobs);
  const sizeStats = stats(totalSizes);

  console.log(`CLUSTER SIZE ANALYSIS:`);
  console.log(`  Population per cluster:`);
  console.log(`    Min: ${popStats.min.toLocaleString()}, Max: ${popStats.max.toLocaleString()}`);
  console.log(`    Mean: ${Math.round(popStats.mean).toLocaleString()}, Median: ${popStats.median.toLocaleString()}`);
  console.log(`    Std Dev: ${Math.round(popStats.std).toLocaleString()}`);
  console.log(`    10th percentile: ${popStats.p10.toLocaleString()}, 90th: ${popStats.p90.toLocaleString()}`);
  console.log();
  console.log(`  Jobs per cluster:`);
  console.log(`    Min: ${jobStats.min.toLocaleString()}, Max: ${jobStats.max.toLocaleString()}`);
  console.log(`    Mean: ${Math.round(jobStats.mean).toLocaleString()}, Median: ${jobStats.median.toLocaleString()}`);
  console.log();
  console.log(`  Total (pop + jobs) per cluster:`);
  console.log(`    Min: ${sizeStats.min.toLocaleString()}, Max: ${sizeStats.max.toLocaleString()}`);
  console.log(`    Mean: ${Math.round(sizeStats.mean).toLocaleString()}, Median: ${sizeStats.median.toLocaleString()}`);
  console.log();

  // === SIZE DISTRIBUTION ===
  const sizeBuckets = {
    'tiny (0-100)': 0,
    'small (100-500)': 0,
    'medium (500-2000)': 0,
    'large (2000-10000)': 0,
    'very large (10000-50000)': 0,
    'mega (50000+)': 0,
  };

  totalSizes.forEach(size => {
    if (size < 100) sizeBuckets['tiny (0-100)']++;
    else if (size < 500) sizeBuckets['small (100-500)']++;
    else if (size < 2000) sizeBuckets['medium (500-2000)']++;
    else if (size < 10000) sizeBuckets['large (2000-10000)']++;
    else if (size < 50000) sizeBuckets['very large (10000-50000)']++;
    else sizeBuckets['mega (50000+)']++;
  });

  console.log(`SIZE DISTRIBUTION:`);
  Object.entries(sizeBuckets).forEach(([bucket, count]) => {
    const pct = (count / points.length * 100).toFixed(1);
    const bar = '#'.repeat(Math.round(count / points.length * 40));
    console.log(`  ${bucket.padEnd(25)} ${count.toString().padStart(5)} (${pct.padStart(5)}%) ${bar}`);
  });
  console.log();

  // === INTER-CLUSTER DISTANCE ANALYSIS ===
  console.log(`INTER-CLUSTER DISTANCE ANALYSIS:`);

  // Find nearest neighbor for each cluster
  const nearestDistances = [];
  const allDistances = [];

  for (let i = 0; i < points.length; i++) {
    let minDist = Infinity;
    for (let j = 0; j < points.length; j++) {
      if (i === j) continue;
      const dist = haversineDistance(points[i].location, points[j].location);
      allDistances.push(dist);
      if (dist < minDist) minDist = dist;
    }
    if (minDist < Infinity) nearestDistances.push(minDist);
  }

  const nearestStats = stats(nearestDistances);

  console.log(`  Nearest neighbor distances (meters):`);
  console.log(`    Min: ${Math.round(nearestStats.min).toLocaleString()}m`);
  console.log(`    Max: ${Math.round(nearestStats.max).toLocaleString()}m`);
  console.log(`    Mean: ${Math.round(nearestStats.mean).toLocaleString()}m`);
  console.log(`    Median: ${Math.round(nearestStats.median).toLocaleString()}m`);
  console.log(`    Std Dev: ${Math.round(nearestStats.std).toLocaleString()}m`);
  console.log();

  // Distance distribution
  const distBuckets = {
    'very close (<200m)': 0,
    'close (200-500m)': 0,
    'medium (500-1000m)': 0,
    'far (1-2km)': 0,
    'very far (2-5km)': 0,
    'extremely far (5km+)': 0,
  };

  nearestDistances.forEach(dist => {
    if (dist < 200) distBuckets['very close (<200m)']++;
    else if (dist < 500) distBuckets['close (200-500m)']++;
    else if (dist < 1000) distBuckets['medium (500-1000m)']++;
    else if (dist < 2000) distBuckets['far (1-2km)']++;
    else if (dist < 5000) distBuckets['very far (2-5km)']++;
    else distBuckets['extremely far (5km+)']++;
  });

  console.log(`  Nearest neighbor distance distribution:`);
  Object.entries(distBuckets).forEach(([bucket, count]) => {
    const pct = (count / nearestDistances.length * 100).toFixed(1);
    const bar = '#'.repeat(Math.round(count / nearestDistances.length * 40));
    console.log(`  ${bucket.padEnd(25)} ${count.toString().padStart(5)} (${pct.padStart(5)}%) ${bar}`);
  });
  console.log();

  // === CONNECTION ANALYSIS ===
  const connectionDistances = connections.map(c => c.drivingDistance);
  const connectionSizes = connections.map(c => c.size);
  const connDistStats = stats(connectionDistances);
  const connSizeStats = stats(connectionSizes);

  console.log(`CONNECTION ANALYSIS:`);
  console.log(`  Connection distances (meters):`);
  console.log(`    Min: ${Math.round(connDistStats.min).toLocaleString()}m`);
  console.log(`    Max: ${Math.round(connDistStats.max).toLocaleString()}m`);
  console.log(`    Mean: ${Math.round(connDistStats.mean).toLocaleString()}m`);
  console.log(`    Median: ${Math.round(connDistStats.median).toLocaleString()}m`);
  console.log();
  console.log(`  Connection sizes (people):`);
  console.log(`    Min: ${connSizeStats.min}, Max: ${connSizeStats.max}`);
  console.log(`    Mean: ${Math.round(connSizeStats.mean)}, Median: ${connSizeStats.median}`);
  console.log();

  // === PROBLEMATIC CLUSTERS ===
  console.log(`PROBLEMATIC CLUSTERS:`);

  // Mega clusters (too large)
  const megaClusters = points.filter(p => (p.residents + p.jobs) > 50000);
  if (megaClusters.length > 0) {
    console.log(`  ⚠️  ${megaClusters.length} MEGA CLUSTERS (>50k pop+jobs):`);
    megaClusters.slice(0, 5).forEach(p => {
      console.log(`      ${p.id}: ${(p.residents + p.jobs).toLocaleString()} (${p.residents.toLocaleString()} residents, ${p.jobs.toLocaleString()} jobs)`);
    });
    if (megaClusters.length > 5) console.log(`      ... and ${megaClusters.length - 5} more`);
    console.log();
  }

  // Isolated clusters (far from others)
  const isolatedClusters = points.filter((p, i) => nearestDistances[i] > 3000);
  if (isolatedClusters.length > 0) {
    console.log(`  ⚠️  ${isolatedClusters.length} ISOLATED CLUSTERS (>3km from nearest):`);
    isolatedClusters.slice(0, 5).forEach((p, idx) => {
      const i = points.indexOf(p);
      console.log(`      ${p.id}: ${Math.round(nearestDistances[i]).toLocaleString()}m to nearest`);
    });
    if (isolatedClusters.length > 5) console.log(`      ... and ${isolatedClusters.length - 5} more`);
    console.log();
  }

  // === QUALITY SCORES ===
  console.log(`QUALITY SCORES (0-100, higher is better):`);

  // Size uniformity score (lower std relative to mean = better)
  const sizeUniformity = Math.max(0, 100 - (sizeStats.std / sizeStats.mean * 50));

  // Distance score (closer clusters = better for transit)
  const targetMedianDist = 500; // Ideal median distance in meters
  const distScore = Math.max(0, 100 - Math.abs(nearestStats.median - targetMedianDist) / 20);

  // Coverage score (more clusters relative to area = better)
  const place = config.places.find(p => p.code === placeCode);
  let coverageScore = 50; // Default
  if (place && place.bbox) {
    const areaKm2 = (place.bbox[2] - place.bbox[0]) * 111 * (place.bbox[3] - place.bbox[1]) * 111 * Math.cos((place.bbox[1] + place.bbox[3]) / 2 * Math.PI / 180);
    const clustersPerKm2 = points.length / areaKm2;
    coverageScore = Math.min(100, clustersPerKm2 * 20); // 5 clusters per km² = 100
  }

  // No mega clusters score
  const noMegaScore = Math.max(0, 100 - megaClusters.length * 20);

  // Overall score
  const overallScore = (sizeUniformity + distScore + coverageScore + noMegaScore) / 4;

  console.log(`  Size uniformity:    ${sizeUniformity.toFixed(1).padStart(5)}/100`);
  console.log(`  Distance score:     ${distScore.toFixed(1).padStart(5)}/100 (target median: ${targetMedianDist}m)`);
  console.log(`  Coverage density:   ${coverageScore.toFixed(1).padStart(5)}/100`);
  console.log(`  No mega clusters:   ${noMegaScore.toFixed(1).padStart(5)}/100`);
  console.log(`  ────────────────────────────`);
  console.log(`  OVERALL SCORE:      ${overallScore.toFixed(1).padStart(5)}/100`);
  console.log();

  // === RECOMMENDATIONS ===
  console.log(`RECOMMENDATIONS:`);

  if (megaClusters.length > 0) {
    console.log(`  • Enable cluster splitting: set "max-place-size-for-splitting": 10000 in config.js`);
  }

  if (nearestStats.median > 1000) {
    console.log(`  • Clusters are too far apart. Enable synthetic cluster generation:`);
    console.log(`    set "synthetic-cluster-target-size": 5000 in config.js`);
  }

  if (sizeStats.std / sizeStats.mean > 2) {
    console.log(`  • High size variance. Consider enabling k-means clustering for more uniform sizes.`);
  }

  if (points.length < 20) {
    console.log(`  • Very few clusters. The OSM data may lack neighborhood definitions.`);
    console.log(`    Consider using synthetic cluster generation based on building density.`);
  }

  if (coverageScore < 30) {
    console.log(`  • Low cluster density. Consider reducing "synthetic-cluster-target-size" for more clusters.`);
  }

  console.log();
  console.log(`${'='.repeat(60)}\n`);

  return {
    placeCode,
    clusterCount: points.length,
    connectionCount: connections.length,
    sizeStats,
    nearestDistanceStats: nearestStats,
    scores: {
      sizeUniformity,
      distScore,
      coverageScore,
      noMegaScore,
      overall: overallScore
    },
    issues: {
      megaClusters: megaClusters.length,
      isolatedClusters: isolatedClusters.length
    }
  };
}

// Main execution
const args = process.argv.slice(2);

if (args.length > 0) {
  // Analyze specific city
  analyzeClusterQuality(args[0]);
} else {
  // Analyze all configured cities
  console.log(`Analyzing all configured cities...\n`);

  const results = [];
  for (const place of config.places) {
    const result = analyzeClusterQuality(place.code);
    if (result) results.push(result);
  }

  if (results.length > 1) {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`SUMMARY ACROSS ALL CITIES`);
    console.log(`${'='.repeat(60)}\n`);

    console.log(`City`.padEnd(10) + `Clusters`.padStart(10) + `Median Dist`.padStart(12) + `Score`.padStart(8));
    console.log(`${'─'.repeat(40)}`);

    results.forEach(r => {
      console.log(
        r.placeCode.padEnd(10) +
        r.clusterCount.toString().padStart(10) +
        `${Math.round(r.nearestDistanceStats.median)}m`.padStart(12) +
        r.scores.overall.toFixed(1).padStart(8)
      );
    });
  }
}
