import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { SphericalMercator } from '@mapbox/sphericalmercator';
import { VectorTile } from '@mapbox/vector-tile';
import Pbf from 'pbf';
import zlib from 'zlib';

// Load config from config.json (fallback to config.js)
let config;
const configJsonPath = path.join(import.meta.dirname, 'config.json');
if (fs.existsSync(configJsonPath)) {
  config = JSON.parse(fs.readFileSync(configJsonPath, 'utf-8'));
  console.log('Loaded config from config.json');
} else {
  const configModule = await import('./config.js');
  config = configModule.default;
  console.log('Loaded config from config.js');
}

const mercator = new SphericalMercator({size: 256});
const pmtilesPath = path.join(import.meta.dirname, 'map_tiles', 'pmtiles');

const tryDownloadWithDate = (date, place) => {
    const dateStr = `${date.getFullYear()}${(date.getMonth()+1).toString().padStart(2, '0')}${date.getDate().toString().padStart(2, '0')}`;
    const protomapsBucket = `https://build.protomaps.com/${dateStr}.pmtiles`;
    const outputPath = path.resolve(import.meta.dirname, 'map_tiles', `${place.code}.pmtiles`);
    
    // Delete existing file if it exists to ensure clean extraction
    if (fs.existsSync(outputPath)) {
        fs.unlinkSync(outputPath);
    }
    
    try {
        execSync(`"${pmtilesPath}" extract ${protomapsBucket} --maxzoom=${config['tile-zoom-level']} --bbox="${place.bbox.join(',')}" "${outputPath}"`, {
            stdio: 'inherit' // Show pmtiles output directly
        });
        return fs.existsSync(outputPath); // Verify file was created
    } catch (e) {
        // pmtiles sometimes returns exit code 1 even on success - check if file exists
        if (fs.existsSync(outputPath) && fs.statSync(outputPath).size > 0) {
            console.log(`  Warning: pmtiles returned error code but file was created successfully`);
            return true;
        }
        return false;
    }
};

const extractWater = (place) => {
    console.log(`Extracting water layer for ${place.name}`);
    
    const pmtilesFile = path.join(import.meta.dirname, 'map_tiles', `${place.code}.pmtiles`);
    const outputDir = path.join(import.meta.dirname, 'raw_data', place.code);
    if (!fs.existsSync(outputDir)) {fs.mkdirSync(outputDir);}

    const xyz = mercator.xyz(place.bbox, 13);
    const features = [];
    
    let tilesChecked = 0;
    let layersFound = new Set();
    
    for (let x = xyz.minX; x <= xyz.maxX; x++) {
        for (let y = xyz.minY; y <= xyz.maxY; y++) {
            tilesChecked++;
            try {
                // Get tile data from pmtiles
                let buffer;
                try {
                    buffer = execSync(`"${pmtilesPath}" tile "${pmtilesFile}" 13 ${x} ${y}`, { stdio: ['ignore', 'pipe', 'ignore'], maxBuffer: 10 * 1024 * 1024 });
                } catch (e) {
                    // Skip if tile doesn't exist
                    continue;
                }
                
                if (!buffer.length) continue;
                
                // Decompress with Node's zlib
                let decompressed;
                try {
                    decompressed = zlib.gunzipSync(buffer);
                } catch (e) {
                    // If decompression fails, try using buffer as-is
                    decompressed = buffer;
                }
                
                const tile = new VectorTile(new Pbf(decompressed));
                
                // Collect all layer names
                Object.keys(tile.layers).forEach(layer => layersFound.add(layer));
                
                if (tile.layers.water) {
                    for (let i = 0; i < tile.layers.water.length; i++) {
                        const feature = tile.layers.water.feature(i);
                        if (feature.properties.kind === 'ocean' || feature.properties.kind === 'basin' || feature.properties.kind === 'river' || feature.properties.kind === 'canal' || feature.properties.kind === 'lake' || feature.properties.kind === 'dock' || feature.properties.kind === 'water') {
                            features.push(feature.toGeoJSON(x, y, 13));
                        }
                    }
                }
            } catch (e) {}
        }
    }
    
    console.log(`Extracted ${features.length} water features.`);
    fs.writeFileSync(path.join(outputDir, 'water.geojson'), JSON.stringify({ type: "FeatureCollection", features }));
};

console.log("Downloading map tiles");
for(var place of config.places) {
    console.log(`Fetching tiles for ${place.name} (${place.code})`);
    
    // Try today's date first, then yesterday
    const today = new Date();
    const yesterday = new Date(Date.now() - 86400000);
    
    let success = tryDownloadWithDate(today, place);
    if (!success) {
        console.log(`  Today's build not available, trying yesterday...`);
        success = tryDownloadWithDate(yesterday, place);
    }
    
    if (!success) {
        throw new Error(`Failed to download tiles for ${place.name}. Builds for today and yesterday are not available at https://build.protomaps.com/`);
    }
    
    extractWater(place);
}