# CLAUDE.md — Project Context for Claude Code

## Project Overview
Geospatial pathfinding system that processes OpenStreetMap (OSM) data and runs pathfinding algorithms (A*, ARA*, MHA*, NAMOA*) over raster road/environment bitmaps. Includes OSM filtering, data pipeline, cost map generation, and animation/visualization tools.

---

## Project Structure

```
./
├── src/                        # PRIMARY SOURCE CODE — make changes here
│   ├── pathfinding.py          # Core pathfinding logic
│   ├── pathfinding2.py         # Pathfinding v2 / experiments
│   ├── pathfinding_texas.py    # Texas-specific pathfinding
│   ├── generate_bitmap.py      # Raster bitmap generation from OSM data
│   ├── animate_astar.py        # A* animation utilities
│   ├── lichtenstein_roads.py   # Liechtenstein test runner
│   ├── road-rail.py            # Road/rail OSM extraction
│   ├── austin-dallas_*.py      # Austin-Dallas corridor scripts
│   │
│   ├── data_pipeline/          # Data ingestion and feature extraction
│   │   ├── create_sample_test_data.py
│   │   └── system/             # Core pipeline modules
│   │       ├── feature_extraction.py       # OSM feature extraction
│   │       ├── construction_cost_map.py    # Cost map: construction
│   │       ├── geometry_cost_map.py        # Cost map: geometry
│   │       ├── environmental_cost_map.py   # Cost map: environment
│   │       ├── cost_map_base.py            # Base cost map class
│   │       ├── crs_detection.py            # CRS/projection detection
│   │       └── example_usage.py            # Usage examples
│   │
│   ├── simple_astar_results/   # OUTPUT DATA — algorithm results, bitmaps, animations
│   │   └── advanced/           # Advanced algorithm comparisons (ARA*, MHA*, NAMOA*)
│   │       ├── compare_all_algorithms.py
│   │       ├── compare_astar_ara.py
│   │       ├── astar_animator.py
│   │       ├── hyperloop_astar_from_npz.py
│   │       ├── hyperloop_ara_from_npz.py
│   │       ├── hyperloop_mha_from_npz.py
│   │       ├── hyperloop_namoa_from_npz.py
│   │       └── [city]_*/       # Per-city output folders (Austin, Seattle, Portland)
│   │
│   └── sample-test-set/        # Test rasters and basemap images
│       ├── austin_test_raster.npz
│       ├── seattle_test_raster.npz
│       ├── portland_test_raster.npz
│       └── images/             # Basemap PNGs for visualization
│
├── modified_files_osm/         # Patched OSM filter files (top-level overrides)
│   ├── element_filter.py
│   ├── osm_filter.py
│   └── pre_filter.py
│
├── docker-envr/                # Docker environment
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements.txt
│   ├── esy-osmfilter/          # Vendored esy-osmfilter library
│   │   └── esy/osmfilter/      # Core OSM filtering logic (element_filter, osm_filter, pre_filter)
│   ├── modified_files/         # Docker-specific modified OSM files
│   └── scripts/                # Scripts to run inside Docker
│
├── tests/                      # Test suite
│   ├── test_osmfilter.py       # Main OSM filter tests
│   └── input/output/           # Test data (Liechtenstein PBF)
│
├── LI_data/                    # Liechtenstein processed data
├── geoai_tests/                # GeoAI/building detection experiments
├── overpass_tests/             # Overpass API query tests
├── pbfsize-reduction/          # PBF file size reduction experiments
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Key Files to Know

| File | Purpose |
|------|---------|
| `src/pathfinding.py` | Main pathfinding entry point |
| `src/generate_bitmap.py` | Converts OSM/GeoJSON → raster `.npz` bitmaps |
| `src/data_pipeline/system/feature_extraction.py` | Extracts road/env features from OSM |
| `src/data_pipeline/system/cost_map_base.py` | Base class for all cost maps |
| `src/simple_astar_results/advanced/astar_animator.py` | Animation/visualization logic |
| `src/simple_astar_results/advanced/compare_all_algorithms.py` | Runs A* vs ARA* vs MHA* vs NAMOA* |
| `modified_files_osm/osm_filter.py` | Top-level patched OSM filter (overrides vendored lib) |
| `docker-envr/esy-osmfilter/esy/osmfilter/` | Vendored OSM filtering library |

---

## Algorithms Implemented
- **A\*** — standard heuristic search
- **ARA\*** — Anytime Repairing A* (iterative suboptimal → optimal)
- **MHA\*** — Multi-Heuristic A*
- **NAMOA\*** — Multi-objective A* (Pareto-optimal paths)

---

## Data Formats
- **Input maps**: `.npz` raster bitmaps (road bitmap + protected areas bitmap)
- **OSM input**: `.osm.pbf` (protobuf binary format)
- **Intermediate**: `.pickle` / `.pkl` for serialized Python objects
- **Output paths**: `.npy` arrays, `.json` for Pareto fronts, `.geojson` for road networks
- **Animations**: `.gif`, `.png` comparison images

---

## Test Cities
- **Liechtenstein (LI)** — small test case, fast iteration
- **Austin, TX** — primary development target
- **Seattle, WA** — comparison city
- **Portland, OR** — comparison city

---

## Common Tasks

### Run pathfinding on a test raster
```bash
cd src
python pathfinding.py
```

### Compare all algorithms
```bash
cd src/simple_astar_results/advanced
python compare_all_algorithms.py
```

### Run OSM filter tests
```bash
python -m pytest tests/test_osmfilter.py
```

### Build & run Docker environment
```bash
cd docker-envr
docker-compose up
```

---

## Conventions
- Modified/patched versions of `esy-osmfilter` live in `modified_files_osm/` (root) and `docker-envr/modified_files/` — these override the vendored library files
- Per-city output folders follow pattern: `[algorithm]_output_[city]/` or `[city]_data/`
- Test data uses Liechtenstein (small country = fast tests)
- `__pycache__/` and `.DS_Store` files should be ignored
