/**
 * Download OSM data for cities in config.json
 *
 * Usage: node download_data.js [--no-redownload] [--resume]
 *   --no-redownload  Skip cities that already have data in raw_data/
 *   --resume         Resume from partial download (uses temp files)
 */

import fs from 'fs';
import { createParseStream, createStringifyStream } from 'big-json';
import { Readable } from "stream";
import * as turf from '@turf/turf';
import { encode as msgpackEncode, decode as msgpackDecode } from '@msgpack/msgpack';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load config from config.json (fallback to config.js)
let config;
const configJsonPath = join(__dirname, 'config.json');
if (fs.existsSync(configJsonPath)) {
  config = JSON.parse(fs.readFileSync(configJsonPath, 'utf-8'));
  console.log('Loaded config from config.json');
} else {
  const configModule = await import('./config.js');
  config = configModule.default;
  console.log('Loaded config from config.js');
}

const convertBbox = (bbox) => [bbox[1], bbox[0], bbox[3], bbox[2]];

const runQuery = async (query) => {
  const endpoint = "https://maps.mail.ru/osm/tools/overpass/api/interpreter";
  
  const res = await fetch(endpoint, {
    "credentials": "omit",
    "headers": {
      "User-Agent": "SubwayBuilder-Patcher (https://github.com/piemadd/subwaybuilder-patcher)",
      "Accept": "*/*",
      "Accept-Language": "en-US,en;q=0.5"
    },
    "body": `data=${encodeURIComponent(query)}`,
    "method": "POST",
    "mode": "cors"
  });

  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`HTTP ${res.status}: ${errorText.substring(0, 200)}`);
  }
  
  // Check if response is JSON
  const contentType = res.headers.get('content-type');
  if (!contentType || !contentType.includes('application/json')) {
    const errorText = await res.text();
    throw new Error(`Non-JSON response (${contentType}): ${errorText.substring(0, 200)}`);
  };

  let finalData = null;

  const parseStream = createParseStream();
  Readable.fromWeb(res.body).pipe(parseStream);


  // Listen for parsed objects
  parseStream.on('data', (data) => {
    finalData = data;
  });

  parseStream.on('error', (error) => {
    console.error('Error parsing JSON stream:', error.message);
    throw error;
  });

  await new Promise((resolve, reject) => {
    parseStream.on('end', resolve);
    parseStream.on('error', reject);
  });

  return finalData;
};

const getStreetName = (tags, preferLocale = 'en') => {
  if (tags.noname === 'yes') return '';
  const localized = tags[`name:${preferLocale}`];
  if (localized && localized.trim()) return localized.trim();
  if (tags.name && tags.name.trim()) return tags.name.trim();
  if (tags.ref && tags.ref.trim()) {
    return tags.ref.trim();
  }
  return '';
};

const fetchRunwayTaxiwayData = async (bbox) => {
  const runwayTaxiwayQuery = `
[out:json][timeout:180];
(
  way["aeroway"="runway"](${bbox.join(',')});
  way["aeroway"="taxiway"](${bbox.join(',')});
  way["aeroway"="apron"](${bbox.join(',')});
);
out geom;`;
  const data = await runQuery(runwayTaxiwayQuery);
  return {
    "type": "FeatureCollection", "features": data.elements.map((element) => {
      if(element.tags.aeroway !== "runway" && element.tags.aeroway !== "taxiway") {
      return {
        "type": "Feature",
        "properties": {
          roadType: "runway",
          aeroway: element.tags.aeroway,
          osm_way_id: new String(element.id),
          area: turf.area(turf.polygon([element.geometry.map((coord) => [coord.lon, coord.lat])])),
        },
        "geometry": {
          "type": "Polygon",
          "coordinates": [element.geometry.map((coord) => [coord.lon, coord.lat])],
        }
      }
   } else {
      return {
        "type": "Feature",
        "properties": {
          roadType: "runway",
          z_order: 0,
          osm_way_id: new String(element.id),
          area: turf.area(turf.lineString(element.geometry.map((coord) => [coord.lon, coord.lat]))),
        },
        "geometry": {
          "type": "Polygon",
          "coordinates": [turf.buffer(turf.lineString(element.geometry.map((coord) => [coord.lon, coord.lat])), element.tags.aeroway == "runway" ? 30 : 10, { units: 'meters' }).geometry.coordinates[0]],
        }
      }
    }
    })
  }
};

const fetchRoadData = async (bbox) => {
  const roadQuery = `
[out:json][timeout:180];
(
  way["highway"="motorway"](${bbox.join(',')});
  way["highway"="motorway_link"](${bbox.join(',')});
  way["highway"="trunk"](${bbox.join(',')});
  way["highway"="trunk_link"](${bbox.join(',')});
  way["highway"="primary"](${bbox.join(',')});
  way["highway"="primary_link"](${bbox.join(',')});
  way["highway"="secondary"](${bbox.join(',')});
  way["highway"="secondary_link"](${bbox.join(',')});
  way["highway"="tertiary"](${bbox.join(',')});
  way["highway"="tertiary_link"](${bbox.join(',')});
  way["highway"="unclassified"](${bbox.join(',')});
  way["highway"="residential"](${bbox.join(',')});
);
out geom;`

  const data = await runQuery(roadQuery);

  const roadTypes = {
    motorway: 'highway',
    motorway_link: 'highway',
    trunk: 'major',
    trunk_link: 'major',
    primary: 'major',
    primary_link: 'major',
    secondary: 'minor',
    secondary_link: 'minor',
    tertiary: 'minor',
    tertiary_link: 'minor',
    unclassified: 'minor',
    residential: 'minor',

  };

  return {
    "type": "FeatureCollection", "features": data.elements.map((element) => {

      return {
        "type": "Feature",
        "properties": {
        roadClass: roadTypes[element.tags.highway],
        structure: "normal",
        name: getStreetName(element.tags, (config.locale || 'en')),
         },
            "geometry": {
            "coordinates": element.geometry.map((coord) => [coord.lon, coord.lat]),
            "type": "LineString"
       }
      }
    })
  }
};

const fetchBuildingsData = async (bbox, placeCode) => {
  // Split large bbox into tiles to avoid memory issues
  const [minLat, minLon, maxLat, maxLon] = bbox;
  const latDiff = maxLat - minLat;
  const lonDiff = maxLon - minLon;
  
  // Use smaller tiles if area is large (> 0.1 degrees ~11km)
  const shouldSplit = latDiff > 0.1 || lonDiff > 0.1;
  
  // Temp file for partial results
  const tempFile = `${import.meta.dirname}/raw_data/${placeCode}/.buildings_temp.msgpack`;
  const progressFile = `${import.meta.dirname}/raw_data/${placeCode}/.progress.json`;
  const resumeMode = process.argv.includes('--resume');
  
  if (shouldSplit) {
    // Split into 4x4 grid (16 tiles)
    const tilesPerSide = Math.ceil(Math.max(latDiff, lonDiff) / 0.1);
    const latStep = latDiff / tilesPerSide;
    const lonStep = lonDiff / tilesPerSide;
    
    console.log(`Large area detected! Splitting into ${tilesPerSide}x${tilesPerSide} = ${tilesPerSide * tilesPerSide} tiles...`);
    
    // Load existing data if resuming
    let allBuildings = [];
    let completedTileIndices = new Set();
    
    if (resumeMode) {
      if (fs.existsSync(tempFile) && fs.existsSync(progressFile)) {
        console.log(`--resume mode: Loading existing data from ${tempFile}...`);
        const existingData = msgpackDecode(fs.readFileSync(tempFile));
        const progress = JSON.parse(fs.readFileSync(progressFile, 'utf-8'));
        allBuildings = existingData;
        completedTileIndices = new Set(progress.completedTiles);
        console.log(`Loaded ${allBuildings.length} existing buildings from ${completedTileIndices.size} completed tiles. Continuing download...`);
      } else if (fs.existsSync(tempFile) && !fs.existsSync(progressFile)) {
        console.log(`⚠️  WARNING: Found old temp file without progress tracking.`);
        console.log(`Cannot resume from old format - delete ${tempFile} and restart, or continue without --resume`);
        process.exit(1);
      } else {
        console.log(`--resume mode: No existing data found, starting fresh...`);
      }
    }
    
    // Generate all tiles
    const allTiles = [];
    for (let i = 0; i < tilesPerSide; i++) {
      for (let j = 0; j < tilesPerSide; j++) {
        allTiles.push({
          bbox: [
            minLat + i * latStep,
            minLon + j * lonStep,
            minLat + (i + 1) * latStep,
            minLon + (j + 1) * lonStep
          ],
          index: i * tilesPerSide + j
        });
      }
    }
    
    const totalTiles = allTiles.length;
    const failedTiles = [];
    
    // Download tiles sequentially
    for (let idx = 0; idx < allTiles.length; idx++) {
      const tile = allTiles[idx];
      const tileNum = tile.index + 1;
      
      // Skip already completed tiles
      if (completedTileIndices.has(tile.index)) {
        console.log(`Skipping tile ${tileNum}/${totalTiles} (already completed)`);
        continue;
      }
      
      console.log(`Fetching tile ${tileNum}/${totalTiles} (${((tileNum/totalTiles)*100).toFixed(1)}%)...`);
      
      const buildingQuery = `
[out:json][timeout:180];
(
  way["building"](${tile.bbox.join(',')});
);
out geom;`;
      
      try {
        const data = await runQuery(buildingQuery);
        if (data.elements && data.elements.length > 0) {
          console.log(`  Tile ${tileNum}: ${data.elements.length} buildings`);
          const elements = data.elements;
          allBuildings = allBuildings.concat(elements);
          completedTileIndices.add(tile.index);
          
          // Save progress every 10 tiles
          if (tileNum % 10 === 0) {
            const tempData = msgpackEncode(allBuildings);
            fs.writeFileSync(tempFile, tempData);
            fs.writeFileSync(progressFile, JSON.stringify({
              completedTiles: Array.from(completedTileIndices),
              totalTiles: totalTiles
            }));
            console.log(`  Saved progress (${allBuildings.length} total buildings, ${completedTileIndices.size}/${totalTiles} tiles)`);
          }
          
          data.elements = null;
        } else {
          console.log(`  Tile ${tileNum}: 0 buildings`);
          completedTileIndices.add(tile.index);
        }
        
        // Delay between requests
        await new Promise(resolve => setTimeout(resolve, 1000));
      } catch (err) {
        console.error(`  Tile ${tileNum} failed:`, err.message);
        
        // Retry once
        console.log(`  Retrying tile ${tileNum} in 5 seconds...`);
        await new Promise(resolve => setTimeout(resolve, 5000));
        try {
          const data = await runQuery(buildingQuery);
          if (data.elements && data.elements.length > 0) {
            console.log(`  Tile ${tileNum} RETRY SUCCESS: ${data.elements.length} buildings`);
            const elements = data.elements;
            allBuildings = allBuildings.concat(elements);
            completedTileIndices.add(tile.index);
            data.elements = null;
          } else {
            completedTileIndices.add(tile.index);
          }
        } catch (retryErr) {
          console.error(`  Tile ${tileNum} RETRY FAILED - skipping:`, retryErr.message);
          failedTiles.push({ tile: tileNum, bbox: tile.bbox, error: retryErr.message });
        }
      }
      
      // Periodic GC
      if (tileNum % 25 === 0 && global.gc) {
        global.gc();
        console.log(`  Memory cleanup at tile ${tileNum}`);
      }
    }
    
    console.log(`Total buildings fetched: ${allBuildings.length}`);
    
    // Log summary
    if (failedTiles.length > 0) {
      console.log(`\n⚠️  WARNING: ${failedTiles.length} tiles failed and were skipped:`);
      failedTiles.forEach(f => {
        console.log(`  - Tile ${f.tile}: ${f.bbox.join(',')} - ${f.error}`);
      });
      console.log(`Temp files kept at: ${tempFile} and ${progressFile}`);
      console.log(`Run with --resume to try failed tiles again, or continue with incomplete data\n`);
    } else {
      // Clean up temp files only if ALL tiles succeeded
      if (fs.existsSync(tempFile)) {
        fs.unlinkSync(tempFile);
      }
      if (fs.existsSync(progressFile)) {
        fs.unlinkSync(progressFile);
      }
      console.log(`✓ All tiles successful - cleaned up temp files`);
    }
    
    return allBuildings;
  } else {
    // Small area, fetch directly
    const buildingQuery = `
[out:json][timeout:180];
(
  way["building"](${bbox.join(',')});
);
out geom;`;

    console.log(`Building query bbox: ${bbox.join(',')}`);
    console.log('Fetching buildings... (this may take several minutes for large areas)');
    
    const data = await runQuery(buildingQuery);
    
    console.log(`Buildings returned: ${data.elements ? data.elements.length : 0}`);
    if (data.elements && data.elements.length > 0) {
      console.log(`First building sample:`, JSON.stringify(data.elements[0]).substring(0, 200));
    }
    if (data.remark) {
      console.log(`Overpass API remark: ${data.remark}`);
    }

    return data.elements || [];
  }
};

const fetchPlacesData = async (bbox) => {
  const neighborhoodsQuery = `
[out:json][timeout:180];
(
  nwr["place"="neighbourhood"](${bbox.join(',')});
  nwr["place"="quarter"](${bbox.join(',')});
  nwr["aeroway"="terminal"](${bbox.join(',')});
  nwr["amenity"](${bbox.join(',')});
);
out geom;`

  const data = await runQuery(neighborhoodsQuery);

  return data.elements;
};

/*
[out:json][timeout:180];
nwr["place"="neighbourhood"]({{bbox}});
out geom;
*/

const fetchAllData = async (place) => {
  if (!fs.existsSync(`${import.meta.dirname}/raw_data/${place.code}`)) fs.mkdirSync(`${import.meta.dirname}/raw_data/${place.code}`);
  console.log(`Fetching ${place.name} (${place.code}) - May take a while`);
  const convertedBoundingBox = convertBbox(place.bbox);
  console.time(`${place.name} (${place.code}) Road Data Fetch`);
  const roadData = await fetchRoadData(convertedBoundingBox);
  console.timeEnd(`${place.name} (${place.code}) Road Data Fetch`);
  console.time(`${place.name} (${place.code}) Building Data Fetch`);
  const buildingData = await fetchBuildingsData(convertedBoundingBox, place.code);
  console.timeEnd(`${place.name} (${place.code}) Building Data Fetch`);
  console.time(`${place.name} (${place.code}) Places Data Fetch`);
  const placesData = await fetchPlacesData(convertedBoundingBox);
  console.timeEnd(`${place.name} (${place.code}) Places Data Fetch`);
  console.time(`${place.name} (${place.code}) Runway Data Fetch`);
  const runwayTaxiwayData = await fetchRunwayTaxiwayData(convertedBoundingBox);
  console.timeEnd(`${place.name} (${place.code}) Runway Data Fetch`);

  // Write GeoJSON files (keep as JSON for compatibility)
  console.time(`Writing roads for ${place.name} (${place.code})`);
  fs.writeFileSync(`${import.meta.dirname}/raw_data/${place.code}/roads.geojson`, JSON.stringify(roadData), { encoding: 'utf8' });
  console.timeEnd(`Writing roads for ${place.name} (${place.code})`);

  console.time(`Writing runways/taxiways for ${place.name} (${place.code})`);
  fs.writeFileSync(`${import.meta.dirname}/raw_data/${place.code}/runways_taxiways.geojson`, JSON.stringify(runwayTaxiwayData), { encoding: 'utf8' });
  console.timeEnd(`Writing runways/taxiways for ${place.name} (${place.code})`);

  // Write large data files as MessagePack (5-10x faster than JSON)
  console.time(`Writing buildings for ${place.name} (${place.code})`);
  const buildingsMsgpack = msgpackEncode(buildingData);
  fs.writeFileSync(`${import.meta.dirname}/raw_data/${place.code}/buildings.msgpack`, buildingsMsgpack);
  console.timeEnd(`Writing buildings for ${place.name} (${place.code})`);
  console.log(`  Buildings: ${(buildingsMsgpack.length / 1024 / 1024).toFixed(1)}MB (MessagePack)`);

  console.time(`Writing places for ${place.name} (${place.code})`);
  const placesMsgpack = msgpackEncode(placesData);
  fs.writeFileSync(`${import.meta.dirname}/raw_data/${place.code}/places.msgpack`, placesMsgpack);
  console.timeEnd(`Writing places for ${place.name} (${place.code})`);
  console.log(`  Places: ${(placesMsgpack.length / 1024 / 1024).toFixed(2)}MB (MessagePack)`);

  console.log(`Done downloading ${place.name} (${place.code})`);
};

// Check for --no-redownload flag
const noRedownload = process.argv.includes('--no-redownload');

const hasExistingData = (placeCode) => {
  const rawDataDir = `${import.meta.dirname}/raw_data/${placeCode}`;
  if (!fs.existsSync(rawDataDir)) return false;

  // Check for required files (msgpack preferred, json fallback)
  const hasBuildings = fs.existsSync(`${rawDataDir}/buildings.msgpack`) ||
                       fs.existsSync(`${rawDataDir}/buildings.json`);
  const hasPlaces = fs.existsSync(`${rawDataDir}/places.msgpack`) ||
                    fs.existsSync(`${rawDataDir}/places.json`);
  const hasRoads = fs.existsSync(`${rawDataDir}/roads.geojson`);

  return hasBuildings && hasPlaces && hasRoads;
};

if (!fs.existsSync(`${import.meta.dirname}/raw_data`)) fs.mkdirSync(`${import.meta.dirname}/raw_data`);

for (const place of config.places) {
  if (noRedownload && hasExistingData(place.code)) {
    console.log(`Skipping ${place.name} (${place.code}) - data already exists (use without --no-redownload to force)`);
    continue;
  }
  await fetchAllData(place);
}
