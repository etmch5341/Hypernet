# A*pex Feature / Objective Calculations

The Pure A*pex search (`apex_pure.py`) optimizes 3 objectives simultaneously.
Here's exactly how each one is computed during the search.

---

## 1. Distance (Weighted Path Length)

**What it measures**: Total path length in grid cells, penalized for going off-road.

**Per-edge cost**:
```
cost_dist = hypot(dr, dc) × terrain_multiplier
```

| Terrain | Multiplier |
|---|---|
| On-road (`raster[r,c] == 1`) | `road_cost` (default 1.0) |
| Off-road (`raster[r,c] == 0`) | `offroad_cost` (default 5.0) |

For cardinal moves `hypot(1,0) = 1.0`, for diagonal moves `hypot(1,1) ≈ 1.414`.

**Heuristic**: Euclidean distance to goal — `hypot(goal_r - r, goal_c - c)`. Admissible because straight-line is always ≤ actual path.

---

## 2. Elevation Change (Cumulative Absolute Difference)

**What it measures**: Total meters of elevation gain and loss along the path.

**Per-edge cost**:
```
cost_elev = |elevation[neighbor] - elevation[current]|
```

This uses the absolute difference, so a path going down 100m then up 100m accumulates 200m of elevation change. A flat path accumulates 0.

**Where elevation comes from**: 
- Real elevation: fetched via `extract_elevation(bbox, resolution)` from USGS/SRTM data
- Fallback: multi-octave synthetic noise (Perlin-like), range 0–500m

**Heuristic**: Direct elevation difference to goal — `|elevation[goal] - elevation[pos]|`. Admissible because the actual cumulative change is always ≥ the net difference.

---

## 3. Slope (Terrain Gradient)

**What it measures**: Cumulative steepness encountered along the path.

**How slope is precomputed** (once, at initialization):
```python
grad_y, grad_x = np.gradient(elevation, resolution)  # meters rise per meter run
slope = sqrt(grad_x² + grad_y²)                       # magnitude, clamped to [0, 1]
```

This computes the gradient magnitude at every cell — essentially how steep the terrain is at that point, as a fraction (0 = flat, 1 = 45°).

**Per-edge cost**:
```
cost_slope = slope[neighbor_row, neighbor_col]
```

Simply the slope value at the destination cell. Steep cells cost more.

**Heuristic**: 0 (zero). This is always admissible since you can't know the minimum slope along the remaining path. It makes the slope objective the "loosest" heuristic — A*pex will explore more in this dimension.

---

## How They Interact (Multi-Objective)

These 3 costs are **not combined** into a single number. Instead, each edge produces a **cost vector**:

```
edge_cost = (cost_dist, cost_elev, cost_slope)
```

The search tracks these independently. Two paths are compared via **ε-dominance**: path A ε-dominates path B if A is at most (1+ε) worse in *every* objective. Only non-dominated paths are kept in the Pareto front.

This means the search naturally finds tradeoff solutions like:
- **Short but steep**: minimal distance, high elevation/slope
- **Flat but long**: detours to avoid hills, low elevation cost
- **Balanced**: moderate in all three
