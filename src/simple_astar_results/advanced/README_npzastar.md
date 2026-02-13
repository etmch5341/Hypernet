# A* Pathfinding from NPZ Test Sets

This system runs multi-goal A* pathfinding directly from NPZ raster files and generates animations showing the search process.

## Files

- **`hyperloop_astar_from_npz.py`** - Main script that loads NPZ files, runs A*, and generates animations
- **`run_all_tests.py`** - Batch script to process all test sets (Austin, Seattle, Portland)
- **`astar_animator.py`** - Animation generation module (unchanged)
- **`generate_animations.py`** - CLI tool to regenerate animations from existing output

## Quick Start

### Run a Single Test Set

```bash
# Austin test
python hyperloop_astar_from_npz.py --input ./src/sample-test-set/austin_test_raster.npz

# Seattle test
python hyperloop_astar_from_npz.py --input ./src/sample-test-set/seattle_test_raster.npz

# Portland test
python hyperloop_astar_from_npz.py --input ./src/sample-test-set/portland_test_raster.npz
```

### Run All Test Sets

```bash
python run_all_tests.py
```

This will process all three test sets sequentially and create separate output directories for each.

## Command Line Options

### `astar_from_npz.py`

```bash
python astar_from_npz.py --input <npz_file> [options]

Required:
  --input PATH              Path to NPZ test set file

Optional:
  --output DIR              Output directory (default: ./astar_output_<testname>)
  --no-animation            Skip creating animations (save data only)
  --fps N                   Animation frames per second (default: 15)
  --frames N                Number of animation frames (default: 150)
  --max-expansions N        Maximum node expansions (default: 5,000,000)
  --no-diagonal             Disable diagonal movement (4-connected instead of 8-connected)
```

### Examples

```bash
# Custom output directory
python astar_from_npz.py --input austin_test_raster.npz --output ./results/austin_run1

# Skip animation (just run pathfinding)
python astar_from_npz.py --input austin_test_raster.npz --no-animation

# Slower, more detailed animation
python astar_from_npz.py --input austin_test_raster.npz --fps 30 --frames 300

# Limit search space (useful for testing)
python astar_from_npz.py --input austin_test_raster.npz --max-expansions 100000

# Disable diagonal movement (only 4 directions)
python astar_from_npz.py --input austin_test_raster.npz --no-diagonal
```

## Output Structure

Each run creates an output directory with the following files:

```
astar_output_<testname>/
├── road_bitmap.npz              # Road network raster
├── protected_bitmap.npz         # Empty (for compatibility)
├── meta.pkl                     # Metadata and transform info
├── astar_sparse_frames.pkl      # Animation frame data
├── astar_final_path.npy         # Final path waypoints
├── astar_animation.gif          # Animated search visualization
└── astar_comparison.png         # Static before/during/after comparison
```

## NPZ File Format

The input NPZ files should contain:

- **`raster`** - 2D numpy array where 1=road, 0=off-road
- **`goal_points`** - Array of (x, y, name) tuples in pixel coordinates
- **`width`** - Raster width in pixels
- **`height`** - Raster height in pixels
- **`bbox`** - Bounding box [min_lon, min_lat, max_lon, max_lat]
- **`target_crs`** - Target coordinate reference system (e.g., "EPSG:3083")

## Cost Model

The pathfinding uses a simple terrain-based cost model:

- **On-road travel** (raster value = 1): Cost multiplier = 1.0
- **Off-road travel** (raster value = 0): Cost multiplier = 5.0

Diagonal moves cost √2 times the base terrain cost (Euclidean distance).

## Algorithm Details

- **Algorithm**: A* with multi-goal support via bitmask states
- **Heuristic**: Minimum distance to nearest unvisited goal + MST of remaining goals
- **State space**: (position, visited_goals_bitmask)
- **Movement**: 8-connected (diagonal allowed by default) or 4-connected

## Regenerating Animations

If you already ran pathfinding and want to regenerate animations with different settings:

```bash
# Regenerate with different FPS
python generate_animations.py --output-dir ./astar_output_austin --fps 30

# Create only the animation (skip comparison image)
python generate_animations.py --output-dir ./astar_output_austin --skip-comparison

# Create only the comparison (skip animation)
python generate_animations.py --output-dir ./astar_output_austin --skip-animation
```

## Performance Notes

- **Austin test**: ~200x300 pixels, typically completes in seconds
- **Seattle test**: ~200x300 pixels, typically completes in seconds  
- **Portland test**: ~1000x1000 pixels, may take several minutes

For larger test sets:
- Use `--max-expansions` to limit search time during development
- Consider using `--no-animation` first to verify the path exists
- The search will print progress updates every 5,000 expansions

## Integration with HYPERNET

This simplified system is designed for quick testing. For the full HYPERNET project:

1. These test sets validate the core A* implementation
2. The animation system helps visualize algorithm behavior
3. The cost model can be extended to multiple objectives (geometry, construction cost, environmental impact)
4. The multi-goal approach demonstrates routing through multiple waypoints

To scale to full Texas Triangle analysis:
- Increase raster resolution to 10m (currently using test resolution)
- Add multiple cost maps for different objectives
- Implement A-MHA* for Pareto-optimal solutions
- Use TACC resources for distributed computation

## Troubleshooting

**Problem**: `FileNotFoundError: road_bitmap.npz`
- **Solution**: The NPZ file doesn't exist or path is wrong. Check `--input` path.

**Problem**: `No path found (or search aborted)`
- **Solution**: Goals may be unreachable or search hit max expansions. Try:
  - Increase `--max-expansions`
  - Check that goals are accessible in the raster
  - Verify raster data is correct (use `load_npz_raster.py` to visualize)

**Problem**: Animation generation fails but path was found
- **Solution**: This is usually a memory issue. The data is still saved, you can:
  - Reduce `--frames` to use fewer animation frames
  - Use `generate_animations.py` separately with lower settings
  - Skip animation with `--no-animation` flag

## Dependencies

```bash
pip install numpy matplotlib geopandas rasterio pyproj
```

(These should already be installed in your HYPERNET environment)