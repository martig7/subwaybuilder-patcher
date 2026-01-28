#!/usr/bin/env node

/**
 * Copy a map's data from mapPatcher to a top-level output folder.
 *
 * Usage:
 *   node copy_map_data.js [--output <folder>] [--city <code>]
 *
 * Options:
 *   --output <folder>  Destination folder name (default: "map_output")
 *   --city <code>      City code to copy (default: first city in config.json)
 *
 * Examples:
 *   node copy_map_data.js
 *   node copy_map_data.js --city FUK_GEN --output my_map
 */

import fs from 'fs';
import path from 'path';

const ROOT = import.meta.dirname;
const MAP_PATCHER = path.join(ROOT, 'patcher', 'packages', 'mapPatcher');

// Parse args
const args = process.argv.slice(2);
function getArg(name, fallback) {
  const idx = args.indexOf(`--${name}`);
  return idx !== -1 && args[idx + 1] ? args[idx + 1] : fallback;
}

// Load config.json to get available cities
const configPath = path.join(MAP_PATCHER, 'config.json');
if (!fs.existsSync(configPath)) {
  console.error('Error: config.json not found at', configPath);
  process.exit(1);
}
const config = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
const availableCities = config.places.map(p => p.code);

const cityCode = getArg('city', availableCities[0]);
const outputFolder = getArg('output', 'map_output');

if (!cityCode) {
  console.error('Error: No city specified and no cities found in config.json');
  process.exit(1);
}

console.log(`Copying map data for ${cityCode} to ${outputFolder}/`);

const processedDir = path.join(MAP_PATCHER, 'processed_data', cityCode);
const tilesFile = path.join(MAP_PATCHER, 'map_tiles', `${cityCode}.pmtiles`);
const destDir = path.join(ROOT, outputFolder, cityCode);

// Verify source exists
if (!fs.existsSync(processedDir)) {
  console.error(`Error: No processed data found for ${cityCode} at ${processedDir}`);
  console.error(`Available cities with processed data:`);
  const processed = path.join(MAP_PATCHER, 'processed_data');
  if (fs.existsSync(processed)) {
    fs.readdirSync(processed, { withFileTypes: true })
      .filter(d => d.isDirectory())
      .forEach(d => console.error(`  - ${d.name}`));
  }
  process.exit(1);
}

// Create destination
fs.mkdirSync(destDir, { recursive: true });

// Copy processed data files
const processedFiles = fs.readdirSync(processedDir);
for (const file of processedFiles) {
  const src = path.join(processedDir, file);
  const dst = path.join(destDir, file);
  fs.copyFileSync(src, dst);
  console.log(`  Copied ${file}`);
}

// Copy tiles if present
if (fs.existsSync(tilesFile)) {
  const dst = path.join(destDir, `${cityCode}.pmtiles`);
  fs.copyFileSync(tilesFile, dst);
  console.log(`  Copied ${cityCode}.pmtiles`);
} else {
  console.log(`  Warning: No tiles file found at ${tilesFile}`);
}

console.log(`Done. Output at ${path.relative(ROOT, destDir)}`);
