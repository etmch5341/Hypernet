# Pure A*pex Multi-Objective Search

**Date**: February 2026  
**Author**: Pranav Krishnan

---

## Summary

**Pure multi-objective A*pex algorithm** for hyperloop route optimization with 3 objectives:
1. **Distance** — Weighted path length (on-road vs off-road)
2. **Elevation** — Cumulative absolute elevation change
3. **Slope** — Cumulative terrain gradient

This is a true multi-objective search (not weighted scalarization) that finds the **Pareto-optimal frontier** of routes using ε-dominance. See [APEX_FEATURES.md](APEX_FEATURES.md) for detailed objective calculations.

---

## Quick Start

### Run A*pex on all test cities (Seattle, Austin, Portland)
```bash
cd /Users/pranavkrishnan/Development/Hypernet

# Full run: search + static visualizations + animated GIFs
python3 src/run_apex_all.py

# Search + visualizations only (faster, no GIF)
python3 src/run_apex_all.py --no-animation

# Run specific cities
python3 src/run_apex_all.py --cities seattle austin
```

Output goes to `src/sample-test-set/apex_results/{city}/`.

### Run on a single raster
```bash
python3 src/apex_pure.py \
  --input src/sample-test-set/austin_test_raster.npz \
  --output src/apex_results/austin \
  --eps 0.1 0.1 0.1 \
  --max-expansions 5000000
```

### Generate visualizations from existing results
```bash
python3 src/apex_visualizer.py \
  --results src/sample-test-set/apex_results/austin/apex_results.json \
  --raster src/sample-test-set/austin_test_raster.npz \
  --output src/sample-test-set/apex_results/austin
```

### Generate animated search GIF only
```bash
python3 src/apex_animate.py \
  --input src/sample-test-set/seattle_test_raster.npz \
  --output src/sample-test-set/apex_results/seattle/apex_animation.gif \
  --eps 0.1 \
  --frame-interval 5000 \
  --fps 4
```

---

## Files

| File | Description |
|------|-------------|
| [apex_pure.py](file:///Users/pranavkrishnan/Development/Hypernet/src/apex_pure.py) | Core A*pex algorithm with ε-dominance |
| [apex_visualizer.py](file:///Users/pranavkrishnan/Development/Hypernet/src/apex_visualizer.py) | Pareto plots, route overlays, progress charts |
| [apex_animate.py](file:///Users/pranavkrishnan/Development/Hypernet/src/apex_animate.py) | Animated GIF of search wavefront |
| [run_apex_all.py](file:///Users/pranavkrishnan/Development/Hypernet/src/run_apex_all.py) | Batch runner for all test cities |
| [apex_comparison.py](file:///Users/pranavkrishnan/Development/Hypernet/src/apex_comparison.py) | A*pex vs EMO+A* comparison framework |
| [APEX_FEATURES.md](file:///Users/pranavkrishnan/Development/Hypernet/docs/APEX_FEATURES.md) | How each objective is calculated |

---

## Output Structure

Each city produces:
```
src/sample-test-set/apex_results/
├── seattle/
│   ├── apex_results.json      # Full results + paths
│   ├── routes_overlay.png     # Pareto routes on raster
│   ├── pareto_3d.png          # 3D Pareto front scatter
│   ├── pareto_2d.png          # 2D objective projections
│   ├── search_progress.png    # Expansion/queue/solutions over time
│   └── apex_animation.gif     # Animated search wavefront
├── austin/
│   └── ...
└── portland/
    └── ...
```

---

## Results (eps=0.1)

| City | Grid Size | ε | Solutions | Expansions | Runtime |
|------|-----------|---|-----------|------------|---------|
| Seattle | 102×142 | 0.1 | **8** | 170K | 7s |
| Austin | 540×864 | 0.1 | **6** | 5M | 203s |
| Portland | 1014×707 | 0.1 | **2** | 5M | 230s |

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--eps` | 0.1 | ε-dominance tolerance per objective. Lower = more solutions but exponentially slower |
| `--max-expansions` | 500K–5M | Safety limit on node expansions. Larger grids need more |
| `--frame-interval` | 5K–50K | Animation: capture a frame every N expansions |
| `--fps` | 4 | Animation: frames per second in output GIF |

> **Tuning ε**: Use 0.1 as a baseline. At 0.01, queue size explodes (100K+ items) and search struggles to converge on grids larger than ~100×100.

---

## Architecture: A*pex vs EMO+A*

| Aspect | Pure A*pex | EMO+A* |
|--------|------------|--------|
| Search type | Multi-objective | Weighted single-objective |
| Labels per node | Multiple (Pareto set) | One (best scalarized) |
| Guarantees | ε-Pareto optimal | Convex hull only |
| Speed | Slower | Faster |
| Use case | Theoretical completeness | Practical large grids |

See [APEX_EMO_WALKTHROUGH.md](APEX_EMO_WALKTHROUGH.md) for EMO details.
