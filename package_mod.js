import fs from 'fs';
import path from 'path';
import { createGzip } from 'zlib';
import { pipeline } from 'stream/promises';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// --- CLI args ---
const args = process.argv.slice(2);
function getArg(name, defaultVal) {
    const idx = args.indexOf(`--${name}`);
    if (idx === -1 || idx + 1 >= args.length) return defaultVal;
    return args[idx + 1];
}

const outputFolder = getArg('output', 'mod_output');
const cityFilter = getArg('city', null);
const authorName = getArg('author', 'subwaybuilder-patcher');

// --- Paths ---
const mapPatcherDir = path.join(__dirname, 'patcher', 'packages', 'mapPatcher');
const processedDataDir = path.join(mapPatcherDir, 'processed_data');
const mapTilesDir = path.join(mapPatcherDir, 'map_tiles');
const cphDir = path.join(__dirname, 'SubwayBuilder-CPH-main', 'SubwayBuilder-CPH-main');
const outputDir = path.join(__dirname, outputFolder);

// --- Load config ---
let config;
const configJsonPath = path.join(mapPatcherDir, 'config.json');
const configJsPath = path.join(mapPatcherDir, 'config.js');

if (fs.existsSync(configJsonPath)) {
    config = JSON.parse(fs.readFileSync(configJsonPath, 'utf8'));
} else {
    // Fallback: import config.js
    const mod = await import(configJsPath);
    config = mod.default;
}

const places = config.places || [];

// Filter cities
const citiesToPackage = cityFilter
    ? places.filter(p => p.code === cityFilter)
    : places;

if (citiesToPackage.length === 0) {
    console.error(`No cities found${cityFilter ? ` matching "${cityFilter}"` : ''}. Check config.json.`);
    process.exit(1);
}

// --- Data files to gzip ---
const DATA_FILES = [
    'buildings_index.json',
    'demand_data.json',
    'roads.geojson',
    'runways_taxiways.geojson',
    'ocean_depth_index.json'
];

// --- Gzip a file ---
async function gzipFile(srcPath, destPath) {
    const src = fs.createReadStream(srcPath);
    const dest = fs.createWriteStream(destPath);
    const gzip = createGzip();
    await pipeline(src, gzip, dest);
}

// --- Compute initial view state from bbox ---
function bboxToViewState(bbox) {
    const [minLon, minLat, maxLon, maxLat] = bbox;
    const latitude = (minLat + maxLat) / 2;
    const longitude = (minLon + maxLon) / 2;

    // Estimate zoom from bbox span
    const latSpan = maxLat - minLat;
    const lonSpan = maxLon - minLon;
    const maxSpan = Math.max(latSpan, lonSpan);
    // Rough heuristic: zoom ≈ log2(360/span)
    const zoom = Math.min(17, Math.max(8, Math.round(Math.log2(360 / maxSpan) - 1)));

    return { zoom, latitude, longitude, bearing: 0 };
}

// --- Generate index.js content ---
function generateIndexJs(city) {
    const viewState = bboxToViewState(city.bbox);
    return `// Auto-generated mod entry point for ${city.name}
function waitForAPI() {
    return new Promise((resolve) => {
        function check() {
            if (window.SubwayBuilderAPI) resolve(window.SubwayBuilderAPI);
            else setTimeout(check, 500);
        }
        check();
    });
}

async function initMod() {
    try {
        console.log("Starting ${city.code} mod initialization...");
        const api = await waitForAPI();

        // 1. Register city
        api.registerCity({
            name: ${JSON.stringify(city.name)},
            code: ${JSON.stringify(city.code)},
            description: ${JSON.stringify(city.description || `Generated data for ${city.name}`)},
            population: ${city.population},
            initialViewState: ${JSON.stringify(viewState, null, 16).replace(/\n {16}/g, '\n                ')},
            minZoom: 8,
            maxZoom: 17
        });

        // 2. Configure map tiles (PMTiles via local server)
        const mapServerUrl = 'http://127.0.0.1:8081';

        if (api.map.setTileURLOverride) {
            api.map.setTileURLOverride({
                cityCode: ${JSON.stringify(city.code)},
                tilesUrl: \`\${mapServerUrl}/general-tiles/{z}/{x}/{y}.mvt\`,
                tileType: 'vector',
                foundationTilesUrl: 'https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
                maxZoom: 17,
                minZoom: 8
            });
            console.log("${city.code}: PMTiles (Local) + CartoDB (Foundation)");
        }

        // 3. Layer visibility
        if (api.hooks && api.hooks.onCityLoad) {
            api.hooks.onCityLoad((cityCode) => {
                if (cityCode === ${JSON.stringify(city.code)}) {
                    if (api.map.setDefaultLayerVisibility) {
                        api.map.setDefaultLayerVisibility(${JSON.stringify(city.code)}, {
                            buildingFoundations: false,
                            oceanFoundations: true,
                            trackElevations: true
                        });
                    }
                    api.ui.showNotification('${city.name} loaded successfully!', 'success');
                }
            });
        }

    } catch (error) {
        console.error("${city.code} mod error:", error);
    }
}

console.log("${city.code} mod loading...");
setTimeout(() => { initMod(); }, 100);
`;
}

// --- Generate manifest.json ---
function generateManifest(city) {
    const modId = `com.${authorName}.${city.code}`;
    return {
        id: modId,
        name: city.name,
        description: city.description || `Generated data for ${city.name}`,
        version: '1.0.0',
        author: { name: authorName },
        main: 'index.js',
        license: 'MIT',
        subwayBuilderVersion: '>=0.11.0'
    };
}

// --- Package one city ---
async function packageCity(city) {
    const modId = `com.${authorName}.${city.code}`;
    const modDir = path.join(outputDir, modId);
    const dataDir = path.join(modDir, 'data');
    const scriptsDir = path.join(modDir, 'scripts');

    // Create directories
    fs.mkdirSync(dataDir, { recursive: true });
    fs.mkdirSync(scriptsDir, { recursive: true });

    // 1. Write manifest.json
    const manifest = generateManifest(city);
    fs.writeFileSync(path.join(modDir, 'manifest.json'), JSON.stringify(manifest, null, 4));
    console.log(`  [OK] manifest.json`);

    // 2. Write index.js
    fs.writeFileSync(path.join(modDir, 'index.js'), generateIndexJs(city));
    console.log(`  [OK] index.js`);

    // 3. Gzip data files
    const cityDataDir = path.join(processedDataDir, city.code);
    if (!fs.existsSync(cityDataDir)) {
        console.error(`  [WARN] No processed data found at ${cityDataDir} — skipping data files`);
    } else {
        for (const file of DATA_FILES) {
            const srcPath = path.join(cityDataDir, file);
            if (!fs.existsSync(srcPath)) {
                if (file === 'ocean_depth_index.json') {
                    // Optional file
                    continue;
                }
                console.warn(`  [WARN] Missing: ${file}`);
                continue;
            }
            const destPath = path.join(dataDir, file + '.gz');
            await gzipFile(srcPath, destPath);
            console.log(`  [OK] ${file}.gz`);
        }
    }

    // 4. Copy pmtiles
    const pmtilesSource = path.join(mapTilesDir, `${city.code}.pmtiles`);
    const pmtilesDest = path.join(scriptsDir, 'general-tiles.pmtiles');
    if (fs.existsSync(pmtilesSource)) {
        fs.copyFileSync(pmtilesSource, pmtilesDest);
        console.log(`  [OK] general-tiles.pmtiles`);
    } else {
        console.warn(`  [WARN] No pmtiles found at ${pmtilesSource}`);
    }

    // 5. Copy installer scripts from CPH example
    // Copy install.js as .cjs to avoid ESM conflicts
    const installJsSrc = path.join(cphDir, 'install.js');
    if (fs.existsSync(installJsSrc)) {
        fs.copyFileSync(installJsSrc, path.join(modDir, 'install.cjs'));
        console.log(`  [OK] install.cjs`);
    } else {
        console.warn(`  [WARN] CPH reference file not found: ${installJsSrc}`);
    }

    // Write install.bat (calls install.cjs)
    const installBat = `@echo off
title Install Map Pack
echo Starting installation with Node.js...
echo.

echo Step 1: Checking for Node.js...
where node >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Node.js is not installed or not in PATH!
    echo Please download and install Node.js from: https://nodejs.org/
    pause
    exit /b 1
)

echo Node.js found.

echo.
echo Step 2: Checking/installing required npm packages...
echo Installing adm-zip...
call npm install adm-zip
if %errorlevel% neq 0 (
    echo ERROR: Failed to install adm-zip!
    echo Please check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo Step 3: Running installer...
node install.cjs
set EXIT_CODE=%errorlevel%

echo.
if %EXIT_CODE% neq 0 (
    echo An error occurred during installation.
) else (
    echo Installation completed successfully!
)

echo Press any key to close...
pause >nul
exit /b %EXIT_CODE%
`;
    fs.writeFileSync(path.join(modDir, 'install.bat'), installBat);
    console.log(`  [OK] install.bat`);

    // Write install.sh (calls install.cjs)
    const installSh = `#!/bin/bash

echo "Starting installation with Node.js..."
echo

echo "Checking for Node.js..."
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js is not installed or not in PATH!"
    echo "Please install Node.js first from: https://nodejs.org/"
    exit 1
fi

echo "Node.js found."
echo

echo "Installing required packages (adm-zip)..."
npm install adm-zip --no-progress --silent 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Warning: Could not install adm-zip automatically."
    echo "The installer will try alternative methods."
    echo
fi

echo "Running installer..."
node install.cjs
EXIT_CODE=$?

echo
if [ $EXIT_CODE -eq 0 ]; then
    echo "Installation completed successfully!"
else
    echo "An error occurred during installation."
fi

echo "Press Enter to close..."
read -r
exit $EXIT_CODE
`;
    fs.writeFileSync(path.join(modDir, 'install.sh'), installSh);
    console.log(`  [OK] install.sh`);

    console.log(`  Done: ${modDir}\n`);
}

// --- Main ---
console.log(`Packaging ${citiesToPackage.length} city/cities into "${outputFolder}/"...\n`);

for (const city of citiesToPackage) {
    console.log(`Packaging ${city.name} (${city.code})...`);
    await packageCity(city);
}

console.log('All done!');
