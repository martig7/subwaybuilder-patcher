import fs from 'fs';
import config from './config.js';
import * as turf from '@turf/turf';
import { createParseStream } from 'big-json';

const optimizeBuilding = (unOptimizedBuilding) => {
  return {
    b: [unOptimizedBuilding.minX, unOptimizedBuilding.minY, unOptimizedBuilding.maxX, unOptimizedBuilding.maxY],
    f: unOptimizedBuilding.foundationDepth,
    p: unOptimizedBuilding.polygon,
  }
};

const CS = 0.0009; // This is what the cell size always is in the base game

const optimizeIndex = (unOptimizedIndex) => {
  return {
    cs: CS,
    bbox: [unOptimizedIndex.minLon, unOptimizedIndex.minLat, unOptimizedIndex.maxLon, unOptimizedIndex.maxLat],
    grid: [unOptimizedIndex.cols, unOptimizedIndex.rows],
    cells: Object.keys(unOptimizedIndex.cells).map((key) => [...key.split(',').map((n) => Number(n)), ...unOptimizedIndex.cells[key]]),
    buildings: unOptimizedIndex.buildings.map((unOptimizedBuilding) => optimizeBuilding(unOptimizedBuilding)),
    stats: {
      count: unOptimizedIndex.buildings.length,
      maxDepth: unOptimizedIndex.maxDepth,
    }
  }
};

// how much square footage we should probably expect per resident of this housing type
// later on ill calculate the cross section of the building's square footage, 
// then multiply that but the total number of floors to get an approximate full square footage number
// i can then divide by the below number to get a rough populaion stat
const squareFeetPerPopulation = {
  // NOTE: 'yes' is handled separately as mixed-use based on building size
  apartments: 240,
  barracks: 100, // google said 70-90, but imma bump it up a bit tbh
  bungalow: 600, // sfh
  cabin: 600, // sfh
  detached: 600, // sfh
  annexe: 240, // kinda like apartments
  dormitory: 125, // good lord
  farm: 600, // sfh
  ger: 240, // technically sfh, but generally usually smaller and more compact. honorary apartment. TIL "ger" is mongolian for the english word "yurt"
  hotel: 240, // gonna count these as apartments because hotel guests use transit too
  house: 600, // sfh
  houseboat: 600, // interdasting
  residential: 600, // could be anything, but im assuimg sfh here
  semidetached_house: 400, // duplex
  static_caravan: 500,
  stilt_house: 600,
  terrace: 500, // townhome
  tree_house: 240, // fuck it
  trullo: 240, // there is nothing scientific here, its all fucking vibes
};

const squareFeetPerJob = {
  commercial: 150, // non specific, restaraunts i guess?
  industrial: 500, // vibes vibes vibes vibes!!!!!,
  kiosk: 50, // its all vibes baby
  office: 150, // all of my vibes are 100% meat created
  retail: 300,
  supermarket: 300,
  warehouse: 500,
  // the following are all religious and im assuming ~100 square feet, not for job purposes, 
  // but for the fact that people go to religious institutions
  // might use a similar trick for sports stadiums
  religious: 100,
  cathedral: 100,
  chapel: 100,
  church: 100,
  kingdom_hall: 100,
  monastery: 100,
  mosque: 100,
  presbytery: 100,
  shrine: 100,
  synagogue: 100,
  temple: 100,
  // end of religious
  bakehouse: 300,
  college: 250, // collge/uni is a job
  fire_station: 500,
  government: 150,
  gatehouse: 150,
  hospital: 150,
  kindergarten: 100,
  museum: 300,
  public: 300,
  school: 100,
  train_station: 1000,
  transportation: 1000,
  university: 250,
  // sports time! im going to treat these like offices because i said so.
  // i think itll end up creating demand thats on average what stadiums see traffic wise. not sure
  grandstand: 150,
  pavilion: 150,
  riding_hall: 150,
  sports_hall: 150,
  sports_centre: 150,
  stadium: 150,
};

let terminalTicker = 0;
let uniTicker = 0;

const processPlaceConnections = (place, rawBuildings, rawPlaces) => {
  let neighborhoods = {};
  let centersOfNeighborhoods = {};
  let calculatedBuildings = {};

  // finding areas of neighborhoods
  rawPlaces.forEach((place) => {
    if (place.tags.place && (place.tags.place == 'quarter' || place.tags.place == 'neighbourhood') || (place.tags.aeroway && place.tags.aeroway == 'terminal') || (place.tags.amenity && place.tags.amenity == 'university')) {
      neighborhoods[place.id] = place;
      if (place.type == 'node') {
        centersOfNeighborhoods[place.id] = [place.lon, place.lat];
      } else if (place.type == 'way' || place.type == 'relation') {
        const center = [(place.bounds.minlon + place.bounds.maxlon) / 2, (place.bounds.minlat + place.bounds.maxlat) / 2];
        centersOfNeighborhoods[place.id] = center;
      }
    }
  });

  // Note: We'll merge places AFTER calculating population, so we can protect large neighborhoods
  const totalPlaces = Object.keys(neighborhoods).length;
  console.log(`Total places before population-aware merging: ${totalPlaces}`);

  const centersOfNeighborhoodsFeatureCollection = turf.featureCollection(
    Object.keys(centersOfNeighborhoods).map((placeID) =>
      turf.point(centersOfNeighborhoods[placeID], {
        placeID,
        name: neighborhoods[placeID].tags.name
      })
    )
  );

  // splitting everything into areas
  const voronoi = turf.voronoi(centersOfNeighborhoodsFeatureCollection, {
    bbox: place.bbox,
  })
  voronoi.features = voronoi.features.filter((feature) => feature);
  console.log(`Total voronoi features: ${voronoi.features.length}`);

  // sorting buildings between residential and commercial
  // OPTIMIZATION: Filter by area first, avoid creating polygons for tiny buildings
  const buildingTypeCounts = {};
  const residentialCounts = {};
  const commercialCounts = {};
  const mixedUseCounts = {};
  const unclassifiedCounts = {};
  
  rawBuildings.forEach((building) => {
    if (building.tags.building) { // should always be true, but why not
      const buildingType = building.tags.building;
      buildingTypeCounts[buildingType] = (buildingTypeCounts[buildingType] || 0) + 1;
      
      const __coords = building.geometry.map((point) => [point.lon, point.lat]);
      if (__coords.length < 3) return;
      if (__coords[0][0] !== __coords[__coords.length - 1][0] || __coords[0][1] !== __coords[__coords.length - 1][1]) __coords.push(__coords[0]);
      const buildingGeometry = turf.polygon([__coords]);
      let buildingAreaMultiplier = Math.max(Number(building.tags['building:levels']), 1); // assuming a single story if no level data
      if (isNaN(buildingAreaMultiplier)) buildingAreaMultiplier = 1;
      const buildingArea = turf.area(buildingGeometry) * buildingAreaMultiplier * 10.7639; // that magic number converts from square meters to square feet
      const buildingCenter = [(building.bounds.minlon + building.bounds.maxlon) / 2, (building.bounds.minlat + building.bounds.maxlat) / 2];

      // Special handling for building=yes (mixed-use based on size)
      if (building.tags.building === 'yes') {
        mixedUseCounts[buildingType] = (mixedUseCounts[buildingType] || 0) + 1;
        
        // Calculate residential/commercial split based on building size
        // Small buildings (<5000 sqft): 80% residential, 20% commercial
        // Medium buildings (5000-20000 sqft): 50% residential, 50% commercial
        // Large buildings (>20000 sqft): 30% residential, 70% commercial
        let residentialRatio, commercialRatio;
        if (buildingArea < 5000) {
          residentialRatio = 0.8;
          commercialRatio = 0.2;
        } else if (buildingArea < 20000) {
          // Linear interpolation between 5000 and 20000
          const t = (buildingArea - 5000) / 15000; // 0 to 1
          residentialRatio = 0.8 - (0.3 * t); // 0.8 -> 0.5
          commercialRatio = 0.2 + (0.3 * t); // 0.2 -> 0.5
        } else {
          // Linear interpolation from 20000 to 100000 (cap at 70% commercial)
          const t = Math.min(1, (buildingArea - 20000) / 80000); // 0 to 1
          residentialRatio = 0.5 - (0.2 * t); // 0.5 -> 0.3
          commercialRatio = 0.5 + (0.2 * t); // 0.5 -> 0.7
        }
        
        const residentialArea = buildingArea * residentialRatio;
        const commercialArea = buildingArea * commercialRatio;
        
        const approxPop = Math.floor(residentialArea / 400); // Mixed-use residential: smaller units (apartments)
        const approxJobs = Math.floor(commercialArea / 200); // Mixed-use commercial: diverse (retail/office mix)
        
        calculatedBuildings[building.id] = {
          ...building,
          approxPop,
          approxJobs,
          buildingCenter,
          isMixedUse: true
        };
      } else if (squareFeetPerPopulation[building.tags.building]) { // residential
        residentialCounts[buildingType] = (residentialCounts[buildingType] || 0) + 1;
        const approxPop = Math.floor(buildingArea / squareFeetPerPopulation[building.tags.building]);
        calculatedBuildings[building.id] = {
          ...building,
          approxPop,
          buildingCenter,
        };
      } else if (squareFeetPerJob[building.tags.building]) { // commercial/jobs
        commercialCounts[buildingType] = (commercialCounts[buildingType] || 0) + 1;
        let approxJobs = Math.floor(buildingArea / squareFeetPerJob[building.tags.building]);

        if(building.tags.aeroway && building.tags.aeroway == 'terminal')
          approxJobs *= 20;

        calculatedBuildings[building.id] = {
          ...building,
          approxJobs,
          buildingCenter,
        };
      } else {
        unclassifiedCounts[buildingType] = (unclassifiedCounts[buildingType] || 0) + 1;
      }
    }
  });

  console.log(`Calculated buildings with population/jobs: ${Object.keys(calculatedBuildings).length} / ${rawBuildings.length}`);
  console.log('\n=== BUILDING TYPE ANALYSIS ===');
  console.log(`Total building types found: ${Object.keys(buildingTypeCounts).length}`);
  console.log(`\nTop 20 most common building types:`);
  Object.entries(buildingTypeCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 20)
    .forEach(([type, count]) => {
      const classification = mixedUseCounts[type] ? 'MIXED-USE' :
                           residentialCounts[type] ? 'RESIDENTIAL' : 
                           commercialCounts[type] ? 'COMMERCIAL' : 
                           'UNCLASSIFIED';
      console.log(`  ${type.padEnd(20)} - ${count.toString().padStart(7)} buildings (${classification})`);
    });
  
  const totalResidential = Object.values(residentialCounts).reduce((sum, n) => sum + n, 0);
  const totalCommercial = Object.values(commercialCounts).reduce((sum, n) => sum + n, 0);
  const totalMixedUse = Object.values(mixedUseCounts).reduce((sum, n) => sum + n, 0);
  const totalUnclassified = Object.values(unclassifiedCounts).reduce((sum, n) => sum + n, 0);
  
  console.log(`\nClassification summary:`);
  console.log(`  Residential:   ${totalResidential.toString().padStart(7)} buildings (${(totalResidential/rawBuildings.length*100).toFixed(1)}%)`);
  console.log(`  Commercial:    ${totalCommercial.toString().padStart(7)} buildings (${(totalCommercial/rawBuildings.length*100).toFixed(1)}%)`);
  console.log(`  Mixed-Use:     ${totalMixedUse.toString().padStart(7)} buildings (${(totalMixedUse/rawBuildings.length*100).toFixed(1)}%)`);
  console.log(`  Unclassified:  ${totalUnclassified.toString().padStart(7)} buildings (${(totalUnclassified/rawBuildings.length*100).toFixed(1)}%)`);
  
  if (totalUnclassified > 0) {
    console.log(`\nTop 10 unclassified building types:`);
    Object.entries(unclassifiedCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .forEach(([type, count]) => {
        console.log(`  ${type.padEnd(20)} - ${count.toString().padStart(7)} buildings`);
      });
  }
  console.log('==============================\n');

  // so we can do like, stuff with it
  const buildingsAsFeatureCollection = turf.featureCollection(
    Object.values(calculatedBuildings).map((building) =>
      turf.point(building.buildingCenter, { buildingID: building.id })
    )
  );

  console.log(`Assigning buildings to neighborhoods and calculating connections...`);

  let totalPopulation = 0;
  let totalJobs = 0;
  let finalVoronoiMembers = {}; // what buildings are in each voronoi
  let finalVoronoiMetadata = {}; // additional info on population and jobs

  // OPTIMIZATION: Build a spatial grid for O(1) neighborhood lookups
  // Instead of O(buildings × neighborhoods), we do O(grid_cells × neighborhoods + buildings)
  const bbox = place.bbox; // [minLon, minLat, maxLon, maxLat]
  const GRID_SIZE = Math.floor(centersOfNeighborhoodsFeatureCollection.features.length ** 0.5); //grid - tune for accuracy vs build time
  const cellWidth = (bbox[2] - bbox[0]) / GRID_SIZE;
  const cellHeight = (bbox[3] - bbox[1]) / GRID_SIZE;

  console.log(`Building ${GRID_SIZE}x${GRID_SIZE} spatial index for fast neighborhood lookups...`);
  
  // Precompute nearest neighborhood for each grid cell center
  const neighborhoodGrid = new Array(GRID_SIZE);
  for (let x = 0; x < GRID_SIZE; x++) {
    neighborhoodGrid[x] = new Array(GRID_SIZE);
    for (let y = 0; y < GRID_SIZE; y++) {
      const cellCenterLon = bbox[0] + (x + 0.5) * cellWidth;
      const cellCenterLat = bbox[1] + (y + 0.5) * cellHeight;
      const cellCenter = turf.point([cellCenterLon, cellCenterLat]);
      const nearest = turf.nearestPoint(cellCenter, centersOfNeighborhoodsFeatureCollection);
      neighborhoodGrid[x][y] = nearest.properties.placeID;
    }
  }
  console.log(`Spatial index built.`);

  // Fast building assignment using grid lookup - O(1) per building
  const buildingAssignments = {}; // placeID -> array of building features
  let count = 0;
  Object.values(calculatedBuildings).forEach((building) => {
    const [lon, lat] = building.buildingCenter;
    const gridX = Math.min(GRID_SIZE - 1, Math.max(0, Math.floor((lon - bbox[0]) / cellWidth)));
    const gridY = Math.min(GRID_SIZE - 1, Math.max(0, Math.floor((lat - bbox[1]) / cellHeight)));
    const placeID = neighborhoodGrid[gridX][gridY];
    
    if (!buildingAssignments[placeID]) buildingAssignments[placeID] = [];
    const buildingPoint = turf.point(building.buildingCenter, { buildingID: building.id });
    buildingAssignments[placeID].push(buildingPoint);
    count++;
    if (count % 10000 === 0) {
      console.log(`Assigned ${count} buildings to neighborhoods...`);
    }
  });

  console.log(`Assigned ${count} buildings to neighborhoods.`);

  voronoi.features.forEach((feature) => {
    const placeID = feature.properties.placeID;
    const assignedBuildings = buildingAssignments[placeID] || [];
    finalVoronoiMembers[placeID] = assignedBuildings;

    const finalFeature = {
      ...feature.properties,
      totalPopulation: 0,
      totalJobs: 0,
      percentOfTotalPopulation: null,
      percentOfTotalJobs: null,
    };
      
    assignedBuildings.forEach((buildingFeature) => {
      const building = calculatedBuildings[buildingFeature.properties.buildingID];
      const pop = building.approxPop ?? 0;
      const jobs = building.approxJobs ?? 0;
      finalFeature.totalPopulation += pop;
      finalFeature.totalJobs += jobs;
      totalPopulation += pop;
      totalJobs += jobs;
    });

    finalVoronoiMetadata[placeID] = finalFeature;
  });
  console.log(`Total calculated population: ${totalPopulation}, Total calculated jobs: ${totalJobs}`);

  // Scale down population if configured (reduces connection counts while maintaining proportions)
  const POPULATION_SCALE = config['population-scale-factor'] || 1.0;
  if (POPULATION_SCALE !== 1.0) {
    console.log(`Scaling population by ${POPULATION_SCALE * 100}%...`);
    totalPopulation = Math.round(totalPopulation * POPULATION_SCALE);
    totalJobs = Math.round(totalJobs * POPULATION_SCALE);
    
    Object.keys(finalVoronoiMetadata).forEach(placeID => {
      finalVoronoiMetadata[placeID].totalPopulation = Math.round(finalVoronoiMetadata[placeID].totalPopulation * POPULATION_SCALE);
      finalVoronoiMetadata[placeID].totalJobs = Math.round(finalVoronoiMetadata[placeID].totalJobs * POPULATION_SCALE);
    });
    console.log(`Scaled population: ${totalPopulation}, Scaled jobs: ${totalJobs}`);
  }

  // Smart merging: Only merge SMALL neighborhoods that are close together
  // This keeps large population centers separate even if they're nearby
  const MERGE_DISTANCE = config['merge-places-distance-meters'] || 0;
  const MIN_SIZE_FOR_PROTECTION = config['min-place-size-for-protection'] || 5000;
  
  if (MERGE_DISTANCE > 0) {
    console.log(`Merging small neighborhoods (<${MIN_SIZE_FOR_PROTECTION} residents+jobs) within ${MERGE_DISTANCE}m...`);
    const placeIds = Object.keys(finalVoronoiMetadata);
    const merged = new Set();
    let mergedCount = 0;
    
    for (let i = 0; i < placeIds.length; i++) {
      const id1 = placeIds[i];
      if (merged.has(id1)) continue;
      
      const place1 = finalVoronoiMetadata[id1];
      const size1 = place1.totalPopulation + place1.totalJobs;
      
      // Skip if this neighborhood is large (protected)
      if (size1 >= MIN_SIZE_FOR_PROTECTION) continue;
      
      const cluster = [id1];
      let clusterPop = place1.totalPopulation;
      let clusterJobs = place1.totalJobs;
      
      for (let j = i + 1; j < placeIds.length; j++) {
        const id2 = placeIds[j];
        if (merged.has(id2)) continue;
        
        const place2 = finalVoronoiMetadata[id2];
        const size2 = place2.totalPopulation + place2.totalJobs;
        
        // Only merge if BOTH neighborhoods are small
        if (size2 >= MIN_SIZE_FOR_PROTECTION) continue;
        
        const dist = turf.distance(
          turf.point(centersOfNeighborhoods[id1]),
          turf.point(centersOfNeighborhoods[id2]),
          { units: 'meters' }
        );
        
        if (dist <= MERGE_DISTANCE) {
          cluster.push(id2);
          merged.add(id2);
          clusterPop += place2.totalPopulation;
          clusterJobs += place2.totalJobs;
        }
      }
      
      if (cluster.length > 1) {
        // Merge cluster into first ID
        const mainId = cluster[0];
        const avgLon = cluster.reduce((sum, id) => sum + centersOfNeighborhoods[id][0], 0) / cluster.length;
        const avgLat = cluster.reduce((sum, id) => sum + centersOfNeighborhoods[id][1], 0) / cluster.length;
        centersOfNeighborhoods[mainId] = [avgLon, avgLat];
        
        // Update merged population/jobs
        finalVoronoiMetadata[mainId].totalPopulation = clusterPop;
        finalVoronoiMetadata[mainId].totalJobs = clusterJobs;
        
        // Remove merged places
        cluster.slice(1).forEach(id => {
          delete neighborhoods[id];
          delete centersOfNeighborhoods[id];
          delete finalVoronoiMetadata[id];
        });
        mergedCount += cluster.length - 1;
      }
    }
    
    console.log(`Merged ${mergedCount} small neighborhoods. Remaining: ${Object.keys(finalVoronoiMetadata).length}`);
  }

  // Smart splitting: Break up LARGE neighborhoods into smaller sub-neighborhoods
  // This improves granularity in dense areas and reduces mega-connection sizes
  const MAX_SIZE_FOR_SPLITTING = config['max-place-size-for-splitting'] || 0;
  
  if (MAX_SIZE_FOR_SPLITTING > 0) {
    console.log(`Splitting large neighborhoods (>${MAX_SIZE_FOR_SPLITTING} residents+jobs)...`);
    const placeIds = Object.keys(finalVoronoiMetadata);
    let splitCount = 0;
    let newPlaceIdCounter = Date.now(); // Use timestamp to ensure unique IDs
    
    placeIds.forEach(placeID => {
      const place = finalVoronoiMetadata[placeID];
      const totalSize = place.totalPopulation + place.totalJobs;
      
      if (totalSize <= MAX_SIZE_FOR_SPLITTING) return; // Not large enough to split
      
      // Determine split grid size based on how large the neighborhood is
      let gridSize = 2; // Default: 2x2 = 4 sub-neighborhoods
      if (totalSize > MAX_SIZE_FOR_SPLITTING * 4) gridSize = 3; // 3x3 = 9 sub-neighborhoods
      if (totalSize > MAX_SIZE_FOR_SPLITTING * 9) gridSize = 4; // 4x4 = 16 sub-neighborhoods
      
      const assignedBuildings = finalVoronoiMembers[placeID] || [];
      if (assignedBuildings.length === 0) return; // No buildings to redistribute
      
      // Get the bounds of this neighborhood's buildings
      const buildingCoords = assignedBuildings.map(b => b.geometry.coordinates);
      const minLon = Math.min(...buildingCoords.map(c => c[0]));
      const maxLon = Math.max(...buildingCoords.map(c => c[0]));
      const minLat = Math.min(...buildingCoords.map(c => c[1]));
      const maxLat = Math.max(...buildingCoords.map(c => c[1]));
      
      // Create grid of sub-centers
      const subCenters = [];
      const cellWidth = (maxLon - minLon) / gridSize;
      const cellHeight = (maxLat - minLat) / gridSize;
      
      for (let x = 0; x < gridSize; x++) {
        for (let y = 0; y < gridSize; y++) {
          const centerLon = minLon + (x + 0.5) * cellWidth;
          const centerLat = minLat + (y + 0.5) * cellHeight;
          const subPlaceID = `${placeID}_split_${newPlaceIdCounter++}`;
          
          subCenters.push({
            id: subPlaceID,
            location: [centerLon, centerLat],
            buildings: [],
            totalPopulation: 0,
            totalJobs: 0
          });
        }
      }
      
      // Redistribute buildings to nearest sub-center
      assignedBuildings.forEach(buildingFeature => {
        const building = calculatedBuildings[buildingFeature.properties.buildingID];
        const [bLon, bLat] = building.buildingCenter;
        
        let nearestSub = subCenters[0];
        let minDist = Infinity;
        
        subCenters.forEach(sub => {
          const dist = Math.sqrt(
            Math.pow(bLon - sub.location[0], 2) + 
            Math.pow(bLat - sub.location[1], 2)
          );
          if (dist < minDist) {
            minDist = dist;
            nearestSub = sub;
          }
        });
        
        nearestSub.buildings.push(buildingFeature);
        nearestSub.totalPopulation += building.approxPop ?? 0;
        nearestSub.totalJobs += building.approxJobs ?? 0;
      });
      
      // Remove original large neighborhood
      delete neighborhoods[placeID];
      delete centersOfNeighborhoods[placeID];
      delete finalVoronoiMetadata[placeID];
      delete finalVoronoiMembers[placeID];
      
      // Add sub-neighborhoods
      subCenters.forEach(sub => {
        if (sub.totalPopulation > 0 || sub.totalJobs > 0) { // Only add if has population
          neighborhoods[sub.id] = neighborhoods[placeID] || { tags: {} }; // Copy original tags
          centersOfNeighborhoods[sub.id] = sub.location;
          finalVoronoiMembers[sub.id] = sub.buildings;
          finalVoronoiMetadata[sub.id] = {
            placeID: sub.id,
            name: place.name ? `${place.name} (${sub.id.split('_').pop()})` : sub.id,
            totalPopulation: sub.totalPopulation,
            totalJobs: sub.totalJobs,
            percentOfTotalPopulation: sub.totalPopulation / totalPopulation,
            percentOfTotalJobs: sub.totalJobs / totalJobs
          };
          splitCount++;
        }
      });
    });
    
    console.log(`Split large neighborhoods into ${splitCount} smaller neighborhoods. Total: ${Object.keys(finalVoronoiMetadata).length}`);
  }

  let finalNeighborhoods = {};
  let neighborhoodConnections = [];

  // creating total percents and setting up final dicts
  Object.values(finalVoronoiMetadata).forEach((place) => {
    finalVoronoiMetadata[place.placeID].percentOfTotalPopulation = place.totalPopulation / totalPopulation;
    finalVoronoiMetadata[place.placeID].percentOfTotalJobs = place.totalJobs / totalJobs;

    let id = place.placeID;

    if(neighborhoods[id] && neighborhoods[id].tags && neighborhoods[id].tags.aeroway && neighborhoods[id].tags.aeroway == 'terminal'){
      id = "AIR_Terminal_" + terminalTicker;
      terminalTicker++;
      console.log("New terminal added:", id);
    }
    else if(neighborhoods[id] && neighborhoods[id].tags && neighborhoods[id].tags.amenity && neighborhoods[id].tags.amenity == 'university'){
      id = "UNI_" + uniTicker;
      uniTicker++;
      console.log("New university added:", id);
    }


    finalNeighborhoods[place.placeID] = {
      id: id,
      location: centersOfNeighborhoods[place.placeID],
      jobs: place.totalJobs,
      residents: place.totalPopulation,
      popIds: [],
    }
  });

  console.log('Generating connections using gravity model with sampling...');
  
  Object.values(finalVoronoiMetadata).forEach((outerPlace) => {
    const totalDemand = outerPlace.totalPopulation;
    if (totalDemand <= 5) return; // Skip very small places
    
    // Calculate gravity scores to all potential job destinations
    const gravityScores = [];
    let totalGravity = 0;
    
    Object.values(finalVoronoiMetadata).forEach((innerPlace) => {
      if (innerPlace.totalJobs <= 0) return; // Skip places with no jobs
      if (innerPlace.placeID === outerPlace.placeID) return; // Skip same place (no live-work in same location)
      
      const connectionDistance = turf.length(turf.lineString([
        centersOfNeighborhoods[outerPlace.placeID],
        centersOfNeighborhoods[innerPlace.placeID],
      ]), { units: 'meters' });
      
      // Gravity model: (Jobs * Population) / distance^0.5
      // Lower exponent = less distance penalty = more long-distance connections (better for transit)
      // Use minimum distance to avoid division by zero for same-location connections
      const effectiveDistance = Math.max(connectionDistance, 100);
      const gravity = (innerPlace.totalJobs * outerPlace.totalPopulation) / Math.pow(effectiveDistance, 0.5);
      
      if (gravity > 0) {
        gravityScores.push({
          innerPlace,
          gravity,
          distance: connectionDistance
        });
        totalGravity += gravity;
      }
    });
    
    if (gravityScores.length === 0) return; // No valid destinations
    
    // Normalize gravity scores to probabilities
    gravityScores.forEach(item => {
      item.probability = item.gravity / totalGravity;
    });
    
    // Sort by gravity (highest first) for deterministic top selection
    gravityScores.sort((a, b) => b.gravity - a.gravity);
    
    // Determine number of connections: scales with population (min 5, max 25)
    // Smaller places get fewer connections, larger places get more
    const numConnections = Math.min(25, Math.max(5, Math.floor(Math.sqrt(totalDemand / 40))));
    
    const selectedDestinations = new Set();
    
    // Sample ALL destinations using weighted random selection (no guaranteed top picks)
    // This creates more diverse, transit-appropriate connections
    let attempts = 0;
    const maxAttempts = gravityScores.length * 5; // Prevent infinite loops
    while (selectedDestinations.size < numConnections && selectedDestinations.size < gravityScores.length && attempts < maxAttempts) {
      attempts++;
      const rand = Math.random();
      let cumulative = 0;
      
      for (const item of gravityScores) {
        if (selectedDestinations.has(item.innerPlace.placeID)) continue;
        cumulative += item.probability;
        if (rand <= cumulative) {
          selectedDestinations.add(item.innerPlace.placeID);
          break;
        }
      }
    }
    
    // Distribute total demand across selected destinations based on their gravity
    const selectedScores = gravityScores.filter(item => 
      selectedDestinations.has(item.innerPlace.placeID)
    );
    const selectedTotalGravity = selectedScores.reduce((sum, item) => sum + item.gravity, 0);
    
    selectedScores.forEach(item => {
      // Proportionally allocate demand based on gravity weight
      const connectionSize = Math.round((item.gravity / selectedTotalGravity) * totalDemand);
      if (connectionSize <= 0) return;
      
      const connectionDistance = item.distance;
      const connectionSeconds = connectionDistance * 0.12; // very scientific (hey, this is something i got from the subwaybuilder data)
      
      // Split large connections into chunks of 200
      const MAX_CONNECTION_SIZE = 200;
      let remainingSize = connectionSize;
      
      while (remainingSize > 0) {
        const chunkSize = Math.min(MAX_CONNECTION_SIZE, remainingSize);
        neighborhoodConnections.push({
          residenceId: outerPlace.placeID,
          jobId: item.innerPlace.placeID,
          size: chunkSize,
          drivingDistance: Math.round(connectionDistance),
          drivingSeconds: Math.round(connectionSeconds),
        });
        remainingSize -= chunkSize;
      }
    });
  });
  
  console.log(`Generated ${neighborhoodConnections.length} connections using gravity-based sampling.`);

  // need to populate popIds within finalNeighborhoods
  neighborhoodConnections = neighborhoodConnections
    .filter((connection) => {
      return connection.size > 0;
    })
    .map((connection, i) => {
      const id = i.toString();
      finalNeighborhoods[connection.jobId].popIds.push(id);
      finalNeighborhoods[connection.residenceId].popIds.push(id);
      return {
        ...connection,
        id,
      }
    });

  // handle airport terminals
  neighborhoodConnections.forEach((connection) =>{
    connection.residenceId = finalNeighborhoods[connection.residenceId].id;
    connection.jobId = finalNeighborhoods[connection.jobId].id;
  });

  // Filter out demand points with no connections (empty popIds arrays)
  const pointsWithConnections = Object.values(finalNeighborhoods).filter(point => point.popIds.length > 0);
  console.log(`Filtered demand points: ${Object.keys(finalNeighborhoods).length} -> ${pointsWithConnections.length} (removed ${Object.keys(finalNeighborhoods).length - pointsWithConnections.length} points with no connections)`);

  return {
    points: pointsWithConnections,
    pops: neighborhoodConnections,
  }
};

const processBuildings = (place, rawBuildings) => {
  // looking at the sample data, cells are approximately 100 meters long and wide, so thats what im gonna go with
  let minLon = 9999;
  let minLat = 9999;
  let maxLon = -999;
  let maxLat = -999;

  let buildingsRemoved = 0;
  let buildingsToProcess = [];

  rawBuildings.forEach((building, i) => {
    const __points = building.geometry.map((coord) => [coord.lon, coord.lat]);
    if (__points[0][0] !== __points[__points.length - 1][0] || __points[0][1] !== __points[__points.length - 1][1]) __points.push(__points[0]);
    const buildingPolygon = turf.polygon([__points]);

    buildingsToProcess.push({
      geometry: buildingPolygon,
      tags: building.tags,
      id: i
    });
  });

  const MERGE_DISTANCE = config['merge-buildings-distance-meters'] || 0;
  if (MERGE_DISTANCE > 0 && buildingsToProcess.length > 0) {
    console.log(`Merging buildings within ${MERGE_DISTANCE}m using spatial grid...`);
    const initialCount = buildingsToProcess.length;
    
    // Convert distance to degrees (approximate)
    const distDegrees = MERGE_DISTANCE / 111320;
    const cellSize = distDegrees * 2; // Grid cells are 2x merge distance
    
    // Build spatial grid for O(1) neighbor lookups
    const grid = {};
    const buildingsWithMeta = buildingsToProcess.map((b, idx) => {
      const bbox = turf.bbox(b.geometry);
      // Expand bbox by merge distance for neighbor detection
      const expandedBbox = [
        bbox[0] - distDegrees,
        bbox[1] - distDegrees,
        bbox[2] + distDegrees,
        bbox[3] + distDegrees
      ];
      return { 
        geometry: b.geometry,
        tags: b.tags,
        id: b.id,
        bbox,
        expandedBbox,
        idx,
        merged: false 
      };
    });
    
    // Populate grid - each building goes into all cells it touches
    console.log(`  Building spatial index...`);
    buildingsWithMeta.forEach((b) => {
      const minGx = Math.floor(b.bbox[0] / cellSize);
      const maxGx = Math.floor(b.bbox[2] / cellSize);
      const minGy = Math.floor(b.bbox[1] / cellSize);
      const maxGy = Math.floor(b.bbox[3] / cellSize);
      for (let gx = minGx; gx <= maxGx; gx++) {
        for (let gy = minGy; gy <= maxGy; gy++) {
          const key = `${gx},${gy}`;
          if (!grid[key]) grid[key] = [];
          grid[key].push(b);
        }
      }
    });
    console.log(`  Spatial index built with ${Object.keys(grid).length} cells.`);

    const mergedResults = [];
    let mergeCount = 0;
    let lastProgress = 0;

    // Process each building
    for (let i = 0; i < buildingsWithMeta.length; i++) {
      const progress = Math.floor((i / buildingsWithMeta.length) * 100);
      if (progress >= lastProgress + 5) {
        console.log(`  Merging progress: ${progress}% (${i}/${buildingsWithMeta.length})`);
        lastProgress = progress;
      }
      
      const building = buildingsWithMeta[i];
      if (building.merged) continue;
      building.merged = true;

      // Find all neighbors using spatial grid
      const checked = new Set();
      checked.add(building.idx);
      const toMerge = [building];
      
      // Get all cells this building's expanded bbox touches
      const minGx = Math.floor(building.expandedBbox[0] / cellSize);
      const maxGx = Math.floor(building.expandedBbox[2] / cellSize);
      const minGy = Math.floor(building.expandedBbox[1] / cellSize);
      const maxGy = Math.floor(building.expandedBbox[3] / cellSize);
      
      for (let gx = minGx; gx <= maxGx; gx++) {
        for (let gy = minGy; gy <= maxGy; gy++) {
          const key = `${gx},${gy}`;
          const cell = grid[key];
          if (!cell) continue;
          
          for (const candidate of cell) {
            if (candidate.merged || checked.has(candidate.idx)) continue;
            checked.add(candidate.idx);
            
            // Fast bbox overlap check (expanded bbox vs actual bbox)
            if (building.expandedBbox[0] <= candidate.bbox[2] &&
                building.expandedBbox[2] >= candidate.bbox[0] &&
                building.expandedBbox[1] <= candidate.bbox[3] &&
                building.expandedBbox[3] >= candidate.bbox[1]) {
              toMerge.push(candidate);
              candidate.merged = true;
              mergeCount++;
            }
          }
        }
      }

      // Output result (use largest building as representative)
      if (toMerge.length === 1) {
        mergedResults.push({
          geometry: building.geometry,
          tags: building.tags,
          id: building.id
        });
      } else {
        // For merged groups, take the largest building's geometry as representative
        let largest = toMerge[0];
        let largestArea = (largest.bbox[2] - largest.bbox[0]) * (largest.bbox[3] - largest.bbox[1]);
        let maxDepth = largest.tags['building:levels:underground'] ? Number(largest.tags['building:levels:underground']) : 1;
        
        for (let j = 1; j < toMerge.length; j++) {
          const area = (toMerge[j].bbox[2] - toMerge[j].bbox[0]) * (toMerge[j].bbox[3] - toMerge[j].bbox[1]);
          if (area > largestArea) {
            largest = toMerge[j];
            largestArea = area;
          }
          const depth = toMerge[j].tags['building:levels:underground'] ? Number(toMerge[j].tags['building:levels:underground']) : 1;
          maxDepth = Math.max(maxDepth, depth);
        }
        
        mergedResults.push({
          geometry: largest.geometry,
          tags: { ...largest.tags, 'building:levels:underground': maxDepth },
          id: largest.id
        });
      }
    }
    
    console.log(`Merged ${initialCount} buildings into ${mergedResults.length} buildings (${mergeCount} absorbed).`);
    buildingsToProcess = mergedResults;
  }

  let processedBuildings = {};
  buildingsToProcess.forEach((b, i) => {
    const bbox = turf.bbox(b.geometry);
    const center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2];

    // overall bbox update
    if (bbox[0] < minLon) minLon = bbox[0];
    if (bbox[1] < minLat) minLat = bbox[1];
    if (bbox[2] > maxLon) maxLon = bbox[2];
    if (bbox[3] > maxLat) maxLat = bbox[3];

    processedBuildings[i] = {
      bbox: {
        minLon: bbox[0],
        minLat: bbox[1],
        maxLon: bbox[2],
        maxLat: bbox[3],
      },
      center: center,
      tags: b.tags,
      id: i,
      geometry: b.geometry.geometry.coordinates,
    }
  });

  console.log(`Processed ${rawBuildings.length} buildings, removed ${buildingsRemoved} (${Object.keys(processedBuildings).length} remaining)`);

// === Cell size taken from R analysis ===
const cs  = 0.0009; // latitude (deg)
const latMid = (minLat + maxLat) / 2;
const distortionFactor = 1 / Math.cos(latMid * Math.PI / 180);
const cs_x = cs * distortionFactor;

// Compute grid dimensions
const grid_x = Math.ceil((maxLon - minLon) / cs_x);
const grid_y = Math.ceil((maxLat - minLat) / cs);

// Build boundary coordinate arrays
let columnCoords = [];
for (let i = 0; i <= grid_x; i++) {
  columnCoords.push(minLon + i * cs_x);
}

let rowCoords = [];
for (let j = 0; j <= grid_y; j++) {
  rowCoords.push(minLat + j * cs);
}

// Assign buildings → X cell
Object.values(processedBuildings).forEach(b => {
  for (let x = 0; x < columnCoords.length - 1; x++) {
    const xMin = columnCoords[x];
    const xMax = columnCoords[x + 1];
    if (b.center[0] >= xMin && b.center[0] < xMax) {
      b.xCellCoord = x;
      break;
    }
  }
});

// Assign buildings → Y cell
Object.values(processedBuildings).forEach(b => {
  for (let y = 0; y < rowCoords.length - 1; y++) {
    const yMin = rowCoords[y];
    const yMax = rowCoords[y + 1];
    if (b.center[1] >= yMin && b.center[1] < yMax) {
      b.yCellCoord = y;
      break;
    }
  }
});

// Build cell dictionary
let cellsDict = {};
Object.values(processedBuildings).forEach(b => {
  const key = `${b.xCellCoord},${b.yCellCoord}`;
  if (!cellsDict[key]) cellsDict[key] = [];
  cellsDict[key].push(b.id);
});


  let maxDepth = 1;

  const optimizedIndex = optimizeIndex({
    cellHeightCoords: cs,
    minLon,
    minLat,
    maxLon,
    maxLat,
    cols: columnCoords.length,
    rows: rowCoords.length,
    cells: cellsDict,
    buildings: Object.values(processedBuildings).map((building) => {
      if (
        building.tags['building:levels:underground'] &&
        Number(building.tags['building:levels:underground']) > maxDepth
      )
        maxDepth = Number(building.tags['building:levels:underground']);

      return {
        minX: building.bbox.minLon,
        minY: building.bbox.minLat,
        maxX: building.bbox.maxLon,
        maxY: building.bbox.maxLat,
        foundationDepth: building.tags['building:levels:underground'] ? Number(building.tags['building:levels:underground']) : 1,
        polygon: building.geometry,
      }
    }),
    maxDepth,
  });

  return optimizedIndex;
}
// Converts water.geojson to ocean_depth_index.json
const processWater = (place) => {

  let minLon = Infinity, minLat = Infinity, maxLon = -Infinity, maxLat = -Infinity;
  const inputPath = `${import.meta.dirname}/raw_data/${place.code}/water.geojson`;
  if (!fs.existsSync(inputPath)) {
    console.warn(`No water.geojson found for ${place.code}`);
    return null;
  }
  const geojson = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
  if (!geojson.features || geojson.features.length === 0) {
    console.warn(`No water features found in ${inputPath}`);
    return null;
  }

  // Calculate global bounds
  geojson.features.forEach(f => {
    const bbox = turf.bbox(f);
    if (bbox[0] < minLon) minLon = bbox[0];
    if (bbox[1] < minLat) minLat = bbox[1];
    if (bbox[2] > maxLon) maxLon = bbox[2];
    if (bbox[3] > maxLat) maxLat = bbox[3];
  });
  const bounds = [minLon, minLat, maxLon, maxLat];

  // Cell size (from ocean_depth.py, cs_y = 0.0027)
  const cs_y = 0.0027;
  const center_lat = (minLat + maxLat) / 2.0;
  const cs_x = cs_y / Math.cos(center_lat * Math.PI / 180);
  const width = maxLon - minLon;
  const height = maxLat - minLat;
  const cols = Math.ceil(width / cs_x);
  const rows = Math.ceil(height / cs_y);

  // Build polygons array
  const polygons = [];
  
  geojson.features.forEach(f => {
    const type = f.geometry.type;
    const coordinates = f.geometry.coordinates;

    if (type === 'Polygon') {
      const rings = coordinates;
      // Calculate bbox for this polygon (from exterior ring)
      const polyFeature = turf.polygon(rings);
      const bbox = turf.bbox(polyFeature);
      
      polygons.push({
        b: bbox,
        d: -4,
        p: rings
      });
    } else if (type === 'MultiPolygon') {
      // Flatten MultiPolygon into individual Polygons
      coordinates.forEach(polyCoords => {
        const rings = polyCoords;
        const polyFeature = turf.polygon(rings);
        const bbox = turf.bbox(polyFeature);
        
        polygons.push({
          b: bbox,
          d: -4,
          p: rings
        });
      });
    }
  });

  // Assign polygons to grid cells
  let cellsDict = {};
  polygons.forEach((poly, i) => {
    const bbox = poly.b;
    let start_col = Math.floor((bbox[0] - minLon) / cs_x);
    let end_col = Math.floor((bbox[2] - minLon) / cs_x);
    let start_row = Math.floor((bbox[1] - minLat) / cs_y);
    let end_row = Math.floor((bbox[3] - minLat) / cs_y);
    
    start_col = Math.max(0, start_col);
    end_col = Math.min(cols - 1, end_col);
    start_row = Math.max(0, start_row);
    end_row = Math.min(rows - 1, end_row);
    
    for (let c = start_col; c <= end_col; c++) {
      for (let r = start_row; r <= end_row; r++) {
        const key = `${c},${r}`;
        if (!cellsDict[key]) cellsDict[key] = [];
        cellsDict[key].push(i);
      }
    }
  });

  // Convert cellsDict to list
  const cells_list = Object.keys(cellsDict).map((key) => {
    const [c, r] = key.split(',').map(Number);
    return [c, r, ...cellsDict[key]];
  });

  // Stats
  const stats = {
    count: polygons.length,
    minDepth: -4,
    maxDepth: -4
  };

  // Output structure
  return {
    cs: cs_y,
    bbox: bounds,
    grid: [cols, rows],
    cells: cells_list,
    depths: polygons,
    stats
  };
};

const processAllData = async (place) => {
  const readJsonFile = (filePath) => {
    return new Promise((resolve, reject) => {
      const parseStream = createParseStream();
      let jsonData;

      parseStream.on('data', (data) => {
        jsonData = data;
      });

      parseStream.on('end', () => {
        resolve(jsonData);
      });

      parseStream.on('error', (err) => {
        reject(err);
      });

      fs.createReadStream(filePath).pipe(parseStream);
    });
  };

  console.log('Reading raw data for', place.code);
  const rawBuildings = await readJsonFile(`${import.meta.dirname}/raw_data/${place.code}/buildings.json`);
  const rawPlaces = await readJsonFile(`${import.meta.dirname}/raw_data/${place.code}/places.json`);

  console.log('Processing Buildings for', place.code)
  const processedBuildings = config['disable-3d-buildings'] 
    ? { cs: 0.0009, bbox: place.bbox, grid: [0, 0], cells: [], buildings: [], stats: { count: 0, maxDepth: 1 } }
    : processBuildings(place, rawBuildings);
  console.log('Processing Connections/Demand for', place.code)
  const processedConnections = processPlaceConnections(place, rawBuildings, rawPlaces);
  console.log('Processing Water for', place.code)
  const processedWater = processWater(place);

  console.log('Writing finished data for', place.code)
  // Use streaming write for large buildings_index.json to avoid string length limits
  const buildingsPath = `${import.meta.dirname}/processed_data/${place.code}/buildings_index.json`;
  const writeStream = fs.createWriteStream(buildingsPath, { encoding: 'utf8' });
  
  // Write header
  writeStream.write('{');
  writeStream.write(`"cs":${JSON.stringify(processedBuildings.cs)},`);
  writeStream.write(`"bbox":${JSON.stringify(processedBuildings.bbox)},`);
  writeStream.write(`"grid":${JSON.stringify(processedBuildings.grid)},`);
  writeStream.write(`"cells":${JSON.stringify(processedBuildings.cells)},`);
  writeStream.write(`"stats":${JSON.stringify(processedBuildings.stats)},`);
  
  // Write buildings array in chunks
  writeStream.write('"buildings":[');
  const buildings = processedBuildings.buildings;
  const chunkSize = 1000;
  for (let i = 0; i < buildings.length; i += chunkSize) {
    if (i > 0) writeStream.write(',');
    const chunk = buildings.slice(i, Math.min(i + chunkSize, buildings.length));
    writeStream.write(chunk.map(b => JSON.stringify(b)).join(','));
    if (i % 100000 === 0 && i > 0) console.log(`  Written ${i}/${buildings.length} buildings...`);
  }
  writeStream.write(']}');
  writeStream.end();
  
  // Wait for write to complete
  await new Promise((resolve, reject) => {
    writeStream.on('finish', resolve);
    writeStream.on('error', reject);
  });
  console.log(`  buildings_index.json written (${buildings.length} buildings)`);
  fs.cpSync(`${import.meta.dirname}/raw_data/${place.code}/roads.geojson`, `${import.meta.dirname}/processed_data/${place.code}/roads.geojson`);
  fs.cpSync(`${import.meta.dirname}/raw_data/${place.code}/runways_taxiways.geojson`, `${import.meta.dirname}/processed_data/${place.code}/runways_taxiways.geojson`);
  fs.writeFileSync(`${import.meta.dirname}/processed_data/${place.code}/demand_data.json`, JSON.stringify(processedConnections), { encoding: 'utf8' });
  if (processedWater) {fs.writeFileSync(`${import.meta.dirname}/processed_data/${place.code}/ocean_depth_index.json`, JSON.stringify(processedWater), { encoding: 'utf8' });}
};

if (!fs.existsSync(`${import.meta.dirname}/processed_data`)) fs.mkdirSync(`${import.meta.dirname}/processed_data`);

(async () => {
  for (const place of config.places) {
    if (fs.existsSync(`${import.meta.dirname}/processed_data/${place.code}`)) fs.rmSync(`${import.meta.dirname}/processed_data/${place.code}`, { recursive: true, force: true });
    fs.mkdirSync(`${import.meta.dirname}/processed_data/${place.code}`)
    await processAllData(place);
    console.log(`Finished processing ${place.code}.`);
  }
  console.log('All places processed successfully!');
})();
