/**
 * Compare Features
 *
 * Compares generated features against ground truth features
 * and computes error metrics for each city and overall.
 *
 * Usage: node compare_features.js
 */

import fs from 'fs';
import path from 'path';

// Map generated city codes to ground truth city codes
// Generated codes should be suffixed with _GEN to avoid conflicts
const CITY_CODE_MAPPING = {
  'HNL_GEN': 'HNL',
  'CHI_GEN': 'CHI',
  'SF_GEN': 'SF',
  'BOS_GEN': 'BOS',
  'DC_GEN': 'DC',
  'ATL_GEN': 'ATL',
  'SEA_GEN': 'SEA',
  'DEN_GEN': 'DEN',
  // Add more mappings as needed
};

// Metrics to compare and their weights in the aggregate error
const METRIC_CONFIG = {
  // Stage 1: Total Population/Jobs (weight: 0.25 total) - CRITICAL for map scale
  total_population: { weight: 0.15, displayName: 'Total Population' },
  total_jobs: { weight: 0.10, displayName: 'Total Jobs' },

  // Stage 2: Cluster Distribution (weight: 0.35 total)
  cluster_count: { weight: 0.08, displayName: 'Cluster Count' },
  size_mean: { weight: 0.05, displayName: 'Size Mean' },
  size_median: { weight: 0.05, displayName: 'Size Median' },
  size_max: { weight: 0.10, displayName: 'Size Max' },  // Critical: prevents mega-clusters
  size_std: { weight: 0.02, displayName: 'Size Std Dev' },
  nn_dist_median: { weight: 0.05, displayName: 'NN Distance Median' },

  // Stage 3: Connection Distribution (weight: 0.40 total)
  connections_per_cluster: { weight: 0.12, displayName: 'Connections/Cluster' },
  conn_dist_median: { weight: 0.10, displayName: 'Conn Distance Median' },
  conn_size_mean: { weight: 0.08, displayName: 'Conn Size Mean' },
  graph_density: { weight: 0.10, displayName: 'Graph Density' },
};

// Calculate relative error between generated and ground truth
function relativeError(generated, groundTruth) {
  if (groundTruth === 0) {
    return generated === 0 ? 0 : 1; // 100% error if ground truth is 0 but generated isn't
  }
  return Math.abs(generated - groundTruth) / Math.abs(groundTruth);
}

// Load JSON features file
function loadFeatures(filePath) {
  if (!fs.existsSync(filePath)) {
    return null;
  }
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

// Main execution
async function main() {
  const groundTruthPath = path.join(import.meta.dirname, 'ground_truth_features.json');
  const generatedPath = path.join(import.meta.dirname, 'generated_features.json');

  console.log('Loading feature files...');

  const groundTruth = loadFeatures(groundTruthPath);
  const generated = loadFeatures(generatedPath);

  if (!groundTruth) {
    console.error('Ground truth features not found. Run extract_ground_truth.js first.');
    process.exit(1);
  }

  if (!generated) {
    console.error('Generated features not found. Run extract_generated.js first.');
    process.exit(1);
  }

  console.log(`Ground truth: ${groundTruth.length} cities`);
  console.log(`Generated: ${generated.length} cities`);

  // Index ground truth by city code
  const gtByCity = {};
  groundTruth.forEach(gt => {
    gtByCity[gt.city] = gt;
  });

  // Find matching cities (using code mapping)
  const matchingCities = generated.filter(g => {
    const gtCode = CITY_CODE_MAPPING[g.city] || g.city;
    return gtByCity[gtCode];
  });
  console.log(`Matching cities: ${matchingCities.length}`);

  if (matchingCities.length === 0) {
    console.error('\nNo matching cities found between generated and ground truth data.');
    console.log('Generated cities:', generated.map(g => g.city).join(', '));
    console.log('Ground truth cities:', groundTruth.map(g => g.city).join(', '));
    process.exit(1);
  }

  // Compare each matching city
  const comparisons = [];

  for (const gen of matchingCities) {
    const gtCode = CITY_CODE_MAPPING[gen.city] || gen.city;
    const gt = gtByCity[gtCode];
    const errors = {};
    let aggregateError = 0;

    for (const [metric, config] of Object.entries(METRIC_CONFIG)) {
      const genValue = gen[metric];
      const gtValue = gt[metric];
      const error = relativeError(genValue, gtValue);

      errors[metric] = {
        generated: genValue,
        groundTruth: gtValue,
        relativeError: error,
        percentError: (error * 100).toFixed(1) + '%',
      };

      aggregateError += error * config.weight;
    }

    comparisons.push({
      city: gen.city,
      gtCity: gtCode,
      errors,
      aggregateError,
    });
  }

  // Sort by aggregate error
  comparisons.sort((a, b) => a.aggregateError - b.aggregateError);

  // Print per-city results
  console.log('\n' + '='.repeat(80));
  console.log('PER-CITY COMPARISON');
  console.log('='.repeat(80));

  for (const comp of comparisons) {
    const gtLabel = comp.city !== comp.gtCity ? ` -> ${comp.gtCity}` : '';
    console.log(`\n${comp.city}${gtLabel} (Aggregate Error: ${(comp.aggregateError * 100).toFixed(1)}%)`);
    console.log('-'.repeat(60));

    for (const [metric, config] of Object.entries(METRIC_CONFIG)) {
      const e = comp.errors[metric];
      const status = e.relativeError < 0.25 ? '✓' : e.relativeError < 0.50 ? '~' : '✗';
      console.log(`  ${status} ${config.displayName.padEnd(25)} Gen: ${e.generated.toFixed(2).padStart(10)} | GT: ${e.groundTruth.toFixed(2).padStart(10)} | Err: ${e.percentError.padStart(7)}`);
    }
  }

  // Calculate overall metrics across all cities
  console.log('\n' + '='.repeat(80));
  console.log('OVERALL METRICS');
  console.log('='.repeat(80));

  const overallErrors = {};
  for (const metric of Object.keys(METRIC_CONFIG)) {
    const errors = comparisons.map(c => c.errors[metric].relativeError);
    overallErrors[metric] = {
      mean: errors.reduce((a, b) => a + b, 0) / errors.length,
      median: [...errors].sort((a, b) => a - b)[Math.floor(errors.length / 2)],
      max: Math.max(...errors),
      min: Math.min(...errors),
    };
  }

  console.log('\nMetric                       Mean Err    Median Err    Max Err');
  console.log('-'.repeat(70));

  for (const [metric, config] of Object.entries(METRIC_CONFIG)) {
    const e = overallErrors[metric];
    const status = e.mean < 0.25 ? '✓' : e.mean < 0.50 ? '~' : '✗';
    console.log(`${status} ${config.displayName.padEnd(25)} ${(e.mean * 100).toFixed(1).padStart(8)}%    ${(e.median * 100).toFixed(1).padStart(8)}%    ${(e.max * 100).toFixed(1).padStart(8)}%`);
  }

  const overallAggregateError = comparisons.reduce((sum, c) => sum + c.aggregateError, 0) / comparisons.length;
  console.log('-'.repeat(70));
  console.log(`AGGREGATE ERROR:             ${(overallAggregateError * 100).toFixed(1)}%`);

  // Identify worst metrics
  console.log('\n' + '='.repeat(80));
  console.log('RECOMMENDATIONS');
  console.log('='.repeat(80));

  const sortedMetrics = Object.entries(overallErrors)
    .map(([metric, e]) => ({ metric, ...e, config: METRIC_CONFIG[metric] }))
    .sort((a, b) => b.mean - a.mean);

  console.log('\nMetrics to focus on (highest error first):');
  for (let i = 0; i < Math.min(3, sortedMetrics.length); i++) {
    const m = sortedMetrics[i];
    console.log(`  ${i + 1}. ${m.config.displayName} (${(m.mean * 100).toFixed(1)}% avg error)`);
  }

  // Save comparison results
  const resultsPath = path.join(import.meta.dirname, 'comparison_results.json');
  fs.writeFileSync(resultsPath, JSON.stringify({
    comparisons,
    overallErrors,
    overallAggregateError,
    timestamp: new Date().toISOString(),
  }, null, 2));
  console.log(`\nSaved detailed results to: ${resultsPath}`);

  // Return success/failure based on aggregate error
  if (overallAggregateError < 0.25) {
    console.log('\n✓ SUCCESS: Aggregate error is below 25% target');
  } else {
    console.log('\n✗ NEEDS IMPROVEMENT: Aggregate error exceeds 25% target');
  }
}

main().catch(console.error);
