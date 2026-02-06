# Pure A*pex Multi-Objective Search Implementation

**Date**: February 6, 2026  
**Author**: Pranav Krishnan

---

## Summary

Implemented **pure multi-objective A*pex algorithm** for hyperloop route optimization with 3 objectives:
1. **Distance** - Euclidean path length
2. **Turns** - Number of direction changes  
3. **Elevation** - Cumulative elevation change

This is a true multi-objective search (not weighted scalarization) that finds the **Pareto-optimal frontier** of routes.

---

## New Files

| File | Description |
|------|-------------|
| [apex_pure.py](file:///Users/pranavkrishnan/Development/Hypernet/src/apex_pure.py) | Core A*pex algorithm with ε-dominance |
| [apex_visualizer.py](file:///Users/pranavkrishnan/Development/Hypernet/src/apex_visualizer.py) | Pareto plots, route overlays, progress charts |
| [apex_comparison.py](file:///Users/pranavkrishnan/Development/Hypernet/src/apex_comparison.py) | Comparison framework: A*pex vs EMO+A* |

---

## How to Run

### 1. Pure A*pex Search
```bash
cd /Users/pranavkrishnan/Development/Hypernet

python3 src/apex_pure.py \
  --input src/sample-test-set/austin_test_raster.npz \
  --output src/apex_results/austin \
  --eps 0.2 0.2 0.2 \
  --max-expansions 2000000
```

### 2. Generate Visualizations
```bash
python3 src/apex_visualizer.py \
  --results src/apex_results/austin/apex_results.json \
  --raster src/sample-test-set/austin_test_raster.npz \
  --output src/apex_results/austin
```

### 3. Algorithm Comparison
```bash
python3 src/apex_comparison.py \
  --input src/sample-test-set/austin_test_raster.npz \
  --output src/apex_results/comparison
```

---

## Results (Austin Test Set)

### Pareto-Optimal Solutions Found

| Route | Distance | Turns | Elevation |
|-------|----------|-------|-----------|
| 1 | 1,064 | 571 | 11,275 |
| 2 | 1,369 | 475 | 11,871 |

**Trade-off**: Route 2 is 29% longer but has 17% fewer turns.

### Performance Comparison

| Algorithm | Runtime | Expansions | Solutions |
|-----------|---------|------------|-----------|
| Single-Objective A* | 0.7s | 211k | 1 |
| Pure A*pex (ε=0.2) | 70s | 2M | 2 |
| EMO+A* | 17s | 2.5M | 0* |

*EMO+A* hit expansion limit without finding solutions due to turn/elevation cost inflation

---

## Output Files

Generated in `src/apex_results/austin/`:
- `routes_overlay.png` - Both routes on road network
- `pareto_2d.png` - 2D projections of Pareto front
- `pareto_3d.png` - 3D Pareto front visualization
- `search_progress.png` - Expansion progress over time
- `apex_results.json` - Full results with paths

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--eps` | 0.01 0.01 0.01 | ε-dominance tolerance (higher = more pruning) |
| `--max-expansions` | 500,000 | Safety limit on node expansions |

> **Recommendation**: Use `--eps 0.2 0.2 0.2` for large grids (~500×800 pixels) to make search tractable.

---

## Architecture Difference: A*pex vs EMO+A*

| Aspect | Pure A*pex | EMO+A* |
|--------|------------|--------|
| Search type | Multi-objective | Weighted single-objective |
| Labels per node | Multiple (Pareto set) | One (best scalarized) |
| Guarantees | ε-Pareto optimal | Convex hull only |
| Speed | Slower | Faster |
| Use case | Theoretical completeness | Practical large grids |
