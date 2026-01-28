const config = {
    "tile-zoom-level": 16, // zoom level for map tiles to download

    // === CLUSTERING OPTIONS ===
    // These control how demand points (neighborhoods) are generated

    // Synthetic clustering: Generate clusters based on building density using k-means
    // Set to target population+jobs per cluster (e.g., 5000 for ~5000 people per cluster)
    // Set to 0 to use only OSM neighborhood data (default)
    "synthetic-cluster-target-size": 5000,

    // Minimum number of clusters to generate (used with synthetic clustering)
    "min-clusters": 20,

    // Merge small neighborhoods within this distance (meters). Set to 0 to disable.
    "merge-places-distance-meters": 0,

    // Neighborhoods below this size (pop+jobs) can be merged
    "min-place-size-for-protection": 5000,

    // Split neighborhoods larger than this (pop+jobs). Set to 0 to disable.
    "max-place-size-for-splitting": 15000,

    // Scale all population/job numbers by this factor (e.g., 0.5 = 50%)
    "population-scale-factor": 1.0,

    // === BUILDING OPTIONS ===
    // Merge buildings within this distance (meters). Set to 0 to disable.
    "merge-buildings-distance-meters": 0,

    // Skip 3D building processing (faster, no underground stations)
    "disable-3d-buildings": false,

    // === PLACES TO PROCESS ===
    "places": [
      {
        "code": "YYZ",
        "name": "Toronto",
        "description": "sideways chicago. da windy city babayyyyy",
        "bbox": [-79.671478, 43.571686, -79.232368, 43.788693],
        "population": 2700000,
        "initialViewState": { // OPTIONAL: CUSTOM INITIAL VIEW STATE FOR THE CITY IN THE GAME
          "zoom": 12.5,
          "latitude": 43.70011,
          "longitude": -79.4163,
          "bearing": 0
        },
        "thumbnailBbox": [-79.630597,43.643208,-79.276123,43.772823] // OPTIONAL: CUSTOM BBOX FOR THUMBNAIL GENERATION
      },
    ],
  };
  

  export default config;

