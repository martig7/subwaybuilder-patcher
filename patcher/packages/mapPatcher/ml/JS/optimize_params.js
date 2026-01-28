/**
 * Parameter Optimization Script
 *
 * Uses evolutionary/genetic algorithm to find optimal parameters
 * that minimize the error between generated and ground truth data.
 *
 * Usage: node optimize_params.js [--method METHOD] [--iterations N] [--city CITY_CODE]
 *   --method METHOD   Optimization method: 'genetic', 'random', or 'grid' (default: genetic)
 *   --iterations N    Number of iterations/generations (default: 50)
 *   --city CODE       Only optimize for specific city (default: all matching cities)
 *   --population N    Population size for genetic algorithm (default: 20)
 *   --evals N         Evaluations per candidate to reduce noise (default: 1)
 */

import fs from 'fs';
import path from 'path';
import { pathToFileURL } from 'url';
import { execSync } from 'child_process';

// Parameter search space
const PARAM_SPACE = {
  // Stage 1: Building → Population
  'residential-sqft-multiplier': [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
  'commercial-sqft-multiplier': [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
  'mixed-use-residential-ratio': [0.3, 0.4, 0.5, 0.6, 0.7],

  // Stage 2: Cluster management
  'max-place-size-for-splitting': [500, 1000, 2000, 5000, 10000],
  'merge-places-distance-meters': [0, 50, 100, 200],
  'splitting-num-clusters-divisor': [625, 1250, 2500, 5000], // totalSize / divisor = numClusters
  'splitting-cluster-min-separation': [0.001, 0.002, 0.003, 0.004], // degrees (~100m, 200m, 300m, 400m)

  // Stage 3: Connection generation
  'gravity-exponent': [0.3, 0.5, 0.7, 1.0, 1.5],
  'min-connections-per-cluster': [1, 2, 3, 5],
  'max-connections-per-cluster': [5, 10, 15, 25],
  'connection-scaling-divisor': [20, 40, 80, 160],
  'population-scale-factor': [0.25, 0.5, 0.75, 1.0],
};

// Parameters to hold constant during optimization (not in search space)
const FIXED_PARAMS = {
  'tile-zoom-level': 16,
  'connection-size-cap': 200,
};

// Weights for aggregate error calculation (must sum to 1.0)
const ERROR_WEIGHTS = {
  // Stage 1: Total Population/Jobs - CRITICAL for map scale
  total_population: 0.15,
  total_jobs: 0.10,

  // Stage 2: Cluster Distribution
  cluster_count: 0.08,
  size_mean: 0.05,
  size_median: 0.05,
  size_max: 0.10,  // Critical: prevents mega-clusters (70k vs 3k issue)
  size_std: 0.02,
  nn_dist_median: 0.05,

  // Stage 3: Connection Distribution
  connections_per_cluster: 0.12,
  conn_dist_median: 0.10,
  conn_size_mean: 0.08,
  graph_density: 0.10,
};

const mapPatcherDir = path.join(import.meta.dirname, '..');

// Load ground truth features
function loadGroundTruth() {
  const gtPath = path.join(import.meta.dirname, 'ground_truth_features.json');
  if (!fs.existsSync(gtPath)) {
    console.error('Ground truth not found. Run extract_ground_truth.js first.');
    process.exit(1);
  }
  return JSON.parse(fs.readFileSync(gtPath, 'utf8'));
}

// Load generated features
function loadGenerated() {
  const genPath = path.join(import.meta.dirname, 'generated_features.json');
  if (!fs.existsSync(genPath)) {
    return null;
  }
  return JSON.parse(fs.readFileSync(genPath, 'utf8'));
}

// Calculate relative error
function relativeError(generated, groundTruth) {
  if (groundTruth === 0) {
    return generated === 0 ? 0 : 1;
  }
  return Math.abs(generated - groundTruth) / Math.abs(groundTruth);
}

// Calculate aggregate error for a city comparison
function calculateAggregateError(gen, gt) {
  let totalError = 0;
  for (const [metric, weight] of Object.entries(ERROR_WEIGHTS)) {
    const error = relativeError(gen[metric], gt[metric]);
    totalError += error * weight;
  }
  return totalError;
}

// Write config file with given parameters
function writeConfig(params, places) {
  const config = {
    ...FIXED_PARAMS,
    ...params,
    places,
  };

  const configContent = `const config = ${JSON.stringify(config, null, 2)};
export default config;`;

  fs.writeFileSync(path.join(mapPatcherDir, 'config.js'), configContent);
}

// Run process_data.js and extract_generated.js
function runPipeline() {
  try {
    execSync('node process_data.js', {
      cwd: mapPatcherDir,
      stdio: 'pipe',
      timeout: 300000, // 5 minute timeout
    });

    execSync('node extract_generated.js', {
      cwd: path.join(mapPatcherDir, 'ml'),
      stdio: 'pipe',
      timeout: 60000,
    });

    return true;
  } catch (err) {
    console.error('Pipeline failed:', err.message);
    return false;
  }
}

// Generate all combinations for grid search
function* gridSearchCombinations(paramSpace) {
  const keys = Object.keys(paramSpace);
  const indices = keys.map(() => 0);
  const lengths = keys.map(k => paramSpace[k].length);

  while (true) {
    // Build current combination
    const combo = {};
    for (let i = 0; i < keys.length; i++) {
      combo[keys[i]] = paramSpace[keys[i]][indices[i]];
    }
    yield combo;

    // Increment indices
    let carry = 1;
    for (let i = keys.length - 1; i >= 0 && carry; i--) {
      indices[i] += carry;
      if (indices[i] >= lengths[i]) {
        indices[i] = 0;
        carry = 1;
      } else {
        carry = 0;
      }
    }

    if (carry) break; // All combinations exhausted
  }
}

// Generate random combinations
function* randomSearchCombinations(paramSpace, n) {
  for (let i = 0; i < n; i++) {
    const combo = {};
    for (const [key, values] of Object.entries(paramSpace)) {
      combo[key] = values[Math.floor(Math.random() * values.length)];
    }
    yield combo;
  }
}

// City code mapping
const CITY_CODE_MAPPING = {
  'HNL_GEN': 'HNL',
  'CHI_GEN': 'CHI',
  'SF_GEN': 'SF',
  'BOS_GEN': 'BOS',
  'DC_GEN': 'DC',
  'ATL_GEN': 'ATL',
  'SEA_GEN': 'SEA',
  'DEN_GEN': 'DEN',
};

// ============================================================================
// GENETIC ALGORITHM IMPLEMENTATION
// ============================================================================

// Create a random individual (parameter set)
function createRandomIndividual(paramSpace) {
  const individual = {};
  for (const [key, values] of Object.entries(paramSpace)) {
    individual[key] = values[Math.floor(Math.random() * values.length)];
  }
  return individual;
}

// Crossover: combine two parents to create a child
function crossover(parent1, parent2, paramSpace) {
  const child = {};
  for (const key of Object.keys(paramSpace)) {
    // Uniform crossover: randomly pick from either parent
    child[key] = Math.random() < 0.5 ? parent1[key] : parent2[key];
  }
  return child;
}

// Mutation: randomly change some parameters
function mutate(individual, paramSpace, mutationRate = 0.15) {
  const mutated = { ...individual };
  for (const [key, values] of Object.entries(paramSpace)) {
    if (Math.random() < mutationRate) {
      // Either pick a completely random value, or shift to adjacent value
      if (Math.random() < 0.5) {
        // Random value
        mutated[key] = values[Math.floor(Math.random() * values.length)];
      } else {
        // Adjacent value (local search)
        const currentIdx = values.indexOf(individual[key]);
        const direction = Math.random() < 0.5 ? -1 : 1;
        const newIdx = Math.max(0, Math.min(values.length - 1, currentIdx + direction));
        mutated[key] = values[newIdx];
      }
    }
  }
  return mutated;
}

// Tournament selection: pick best from random subset
function tournamentSelect(population, fitnesses, tournamentSize = 3) {
  const indices = [];
  for (let i = 0; i < tournamentSize; i++) {
    indices.push(Math.floor(Math.random() * population.length));
  }
  // Find best in tournament (lowest error = highest fitness)
  let bestIdx = indices[0];
  for (const idx of indices) {
    if (fitnesses[idx] < fitnesses[bestIdx]) {
      bestIdx = idx;
    }
  }
  return population[bestIdx];
}

// Evaluate a candidate (run pipeline and calculate error)
function evaluateCandidate(params, targetPlaces, gtByCity, numEvals = 1) {
  const errors = [];
  
  for (let e = 0; e < numEvals; e++) {
    writeConfig(params, targetPlaces);
    
    if (!runPipeline()) {
      return { error: Infinity, valid: false };
    }
    
    const generated = loadGenerated();
    if (!generated) {
      return { error: Infinity, valid: false };
    }
    
    let totalError = 0;
    let matchCount = 0;
    
    for (const gen of generated) {
      const gtCode = CITY_CODE_MAPPING[gen.city] || gen.city;
      const gt = gtByCity[gtCode];
      if (!gt) continue;
      
      const error = calculateAggregateError(gen, gt);
      totalError += error;
      matchCount++;
    }
    
    if (matchCount === 0) {
      return { error: Infinity, valid: false };
    }
    
    errors.push(totalError / matchCount);
  }
  
  // Return average error across evaluations (reduces noise)
  const avgError = errors.reduce((a, b) => a + b, 0) / errors.length;
  return { error: avgError, valid: true };
}

// Run genetic algorithm
async function runGeneticAlgorithm(paramSpace, targetPlaces, gtByCity, options) {
  const {
    populationSize = 20,
    generations = 50,
    eliteCount = 2,      // Top N individuals survive unchanged
    mutationRate = 0.15,
    evalsPerCandidate = 1,
  } = options;
  
  console.log(`\nGenetic Algorithm Configuration:`);
  console.log(`  Population size: ${populationSize}`);
  console.log(`  Generations: ${generations}`);
  console.log(`  Elite count: ${eliteCount}`);
  console.log(`  Mutation rate: ${(mutationRate * 100).toFixed(0)}%`);
  console.log(`  Evaluations per candidate: ${evalsPerCandidate}`);
  console.log(`  Total evaluations: ~${populationSize * generations * evalsPerCandidate}`);
  
  // Initialize population
  console.log(`\nInitializing population...`);
  let population = [];
  let fitnesses = [];
  
  for (let i = 0; i < populationSize; i++) {
    const individual = createRandomIndividual(paramSpace);
    population.push(individual);
    
    process.stdout.write(`\r  Evaluating initial individual ${i + 1}/${populationSize}...`);
    const result = evaluateCandidate(individual, targetPlaces, gtByCity, evalsPerCandidate);
    fitnesses.push(result.error);
    
    if (result.valid && result.error !== Infinity) {
      process.stdout.write(` error: ${(result.error * 100).toFixed(1)}%\n`);
    } else {
      process.stdout.write(` FAILED\n`);
    }
  }
  
  // Track best overall
  let bestError = Math.min(...fitnesses);
  let bestIdx = fitnesses.indexOf(bestError);
  let bestParams = { ...population[bestIdx] };
  let bestGeneration = 0;
  
  console.log(`\nInitial best: ${(bestError * 100).toFixed(1)}%`);
  console.log(`\nStarting evolution...`);
  
  const history = [{ generation: 0, bestError, avgError: fitnesses.reduce((a, b) => a + b, 0) / fitnesses.length }];
  
  for (let gen = 1; gen <= generations; gen++) {
    // Sort population by fitness
    const sorted = population
      .map((ind, i) => ({ ind, fitness: fitnesses[i] }))
      .sort((a, b) => a.fitness - b.fitness);
    
    // New generation
    const newPopulation = [];
    const newFitnesses = [];
    
    // Elitism: keep top performers
    for (let i = 0; i < eliteCount; i++) {
      newPopulation.push(sorted[i].ind);
      newFitnesses.push(sorted[i].fitness);
    }
    
    // Generate rest of population through selection, crossover, mutation
    while (newPopulation.length < populationSize) {
      const parent1 = tournamentSelect(population, fitnesses);
      const parent2 = tournamentSelect(population, fitnesses);
      
      let child = crossover(parent1, parent2, paramSpace);
      child = mutate(child, paramSpace, mutationRate);
      
      // Evaluate child
      const result = evaluateCandidate(child, targetPlaces, gtByCity, evalsPerCandidate);
      
      newPopulation.push(child);
      newFitnesses.push(result.valid ? result.error : Infinity);
    }
    
    population = newPopulation;
    fitnesses = newFitnesses;
    
    // Track best
    const genBestError = Math.min(...fitnesses);
    const genBestIdx = fitnesses.indexOf(genBestError);
    const avgError = fitnesses.filter(f => f !== Infinity).reduce((a, b) => a + b, 0) / 
                     fitnesses.filter(f => f !== Infinity).length;
    
    history.push({ generation: gen, bestError: genBestError, avgError });
    
    if (genBestError < bestError) {
      bestError = genBestError;
      bestParams = { ...population[genBestIdx] };
      bestGeneration = gen;
      console.log(`Gen ${gen}: NEW BEST ${(bestError * 100).toFixed(1)}% (avg: ${(avgError * 100).toFixed(1)}%)`);
    } else if (gen % 5 === 0) {
      console.log(`Gen ${gen}: best ${(genBestError * 100).toFixed(1)}% (avg: ${(avgError * 100).toFixed(1)}%)`);
    }
    
    // Early stopping if no improvement for many generations
    if (gen - bestGeneration > 15) {
      console.log(`\nEarly stopping: no improvement for 15 generations`);
      break;
    }
  }
  
  return { bestParams, bestError, history, bestGeneration };
}

// ============================================================================
// MAIN
// ============================================================================

async function main() {
  const args = process.argv.slice(2);
  
  // Parse arguments
  const getArg = (name, defaultValue) => {
    const idx = args.indexOf(`--${name}`);
    return idx !== -1 ? args[idx + 1] : defaultValue;
  };
  const hasArg = (name) => args.indexOf(`--${name}`) !== -1;
  
  const method = getArg('method', 'genetic');
  const iterations = parseInt(getArg('iterations', '50'));
  const targetCity = getArg('city', null);
  const populationSize = parseInt(getArg('population', '20'));
  const evalsPerCandidate = parseInt(getArg('evals', '1'));
  
  // Legacy support for --random
  const useRandom = hasArg('random');
  const randomN = useRandom ? parseInt(getArg('random', '100')) : iterations;

  console.log('Loading ground truth...');
  const groundTruth = loadGroundTruth();
  const gtByCity = {};
  groundTruth.forEach(gt => { gtByCity[gt.city] = gt; });

  // Get places from current config to know what cities to process
  const configPath = pathToFileURL(path.join(mapPatcherDir, 'config.js')).href;
  const currentConfig = await import(configPath);
  const places = currentConfig.default.places;

  console.log(`Cities to optimize: ${places.map(p => p.code).join(', ')}`);

  // Filter to target city if specified
  const targetPlaces = targetCity
    ? places.filter(p => p.code === targetCity || CITY_CODE_MAPPING[p.code] === targetCity)
    : places;

  if (targetPlaces.length === 0) {
    console.error(`No matching places found for city: ${targetCity}`);
    process.exit(1);
  }

  let bestParams = null;
  let bestError = Infinity;
  let results = [];
  let history = [];

  // Choose optimization method
  if (method === 'genetic' && !useRandom) {
    // Genetic Algorithm
    const gaResult = await runGeneticAlgorithm(PARAM_SPACE, targetPlaces, gtByCity, {
      populationSize,
      generations: iterations,
      evalsPerCandidate,
    });
    
    bestParams = gaResult.bestParams;
    bestError = gaResult.bestError;
    history = gaResult.history;
    
  } else {
    // Random or Grid search (legacy)
    const totalCombinations = useRandom || method === 'random'
      ? randomN
      : Object.values(PARAM_SPACE).reduce((acc, v) => acc * v.length, 1);

    console.log(`\nSearch mode: ${useRandom || method === 'random' ? `Random (${randomN} iterations)` : 'Grid'}`);
    console.log(`Total combinations to try: ${totalCombinations}`);
    console.log(`Parameters: ${Object.keys(PARAM_SPACE).join(', ')}`);

    const combinations = useRandom || method === 'random'
      ? randomSearchCombinations(PARAM_SPACE, randomN)
      : gridSearchCombinations(PARAM_SPACE);

    let iteration = 0;

    for (const params of combinations) {
      iteration++;
      process.stdout.write(`\r[${iteration}/${totalCombinations}] Testing... `);

      const result = evaluateCandidate(params, targetPlaces, gtByCity, evalsPerCandidate);

      if (!result.valid) {
        console.log('FAILED');
        continue;
      }

      results.push({ params, error: result.error });

      if (result.error < bestError) {
        bestError = result.error;
        bestParams = { ...params };
        process.stdout.write(`NEW BEST: ${(result.error * 100).toFixed(1)}%\n`);
      }
    }
  }

  console.log('\n' + '='.repeat(80));
  console.log('OPTIMIZATION COMPLETE');
  console.log('='.repeat(80));

  if (bestParams) {
    console.log(`\nBest aggregate error: ${(bestError * 100).toFixed(1)}%`);
    console.log('\nBest parameters:');
    for (const [key, value] of Object.entries(bestParams)) {
      console.log(`  "${key}": ${value},`);
    }

    // Write best config
    writeConfig(bestParams, places);
    console.log('\nBest config written to config.js');

    // Save full results
    if (results.length > 0) {
      results.sort((a, b) => a.error - b.error);
    }
    const resultsPath = path.join(import.meta.dirname, 'optimization_results.json');
    fs.writeFileSync(resultsPath, JSON.stringify({
      method,
      bestParams,
      bestError,
      topResults: results.slice(0, 20),
      history,
      timestamp: new Date().toISOString(),
    }, null, 2));
    console.log(`Full results saved to: ${resultsPath}`);
  } else {
    console.log('\nNo valid results found.');
  }
}

main().catch(console.error);
