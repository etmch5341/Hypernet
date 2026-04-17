# combined/ — HPA* clustering + NAMOA*-dr + APEX selection

End-to-end integration of the three dev branches of `Hypernet`:

| Source branch | Contribution kept here |
|---|---|
| `rishik-k`  | NAMOA*-dr multi-objective pathfinding on the three cost-map layers (construction, environmental, geometry) |
| `arvin-dev` | HPA* cluster + gateway abstraction (partition the raster, find cheap cells on cluster borders, build an abstract graph from start through every goal). The scalar A* searches at every stage are replaced with NAMOA*-dr. |
| `pranav-dev` | APEX-style ε-Pareto filter + distance-to-ideal ranking that picks the single best compromise path from the candidates produced upstream. |

## Pipeline

```
cost maps (3 layers) ──┐
                       ├─▶ HPA* cluster graph (arvin)
road raster, goals ────┘        │
                                │ every edge weighted with a Pareto set
                                ▼   of 3-D vectors via NAMOA*-dr (rishik)
                        abstract NAMOA*-dr search  ─┐
                                                    ▼
                              pixel-level refinement per Pareto abstract path
                                                    │
                                                    ▼
                                        APEX Pareto ranking (pranav)
                                                    │
                                                    ▼
                                         best compromise path
```

## Files

- `cost_map_loader.py` — loads the three normalised cost layers and the Austin raster, plus the composite scalar grid used only for cluster/gateway decisions.
- `clustered_namoa.py` — HPA*-style `ClusteredNamoaGraph` where every cluster-internal pair is connected with a Pareto set of 3-D cost vectors produced by `namoa_dr_segment` (NAMOA*-dr constrained to the cluster). `namoa_dr_abstract` runs multi-objective search over the gateway graph to produce a Pareto set of abstract paths; `refine_abstract_path` stitches segment-level NAMOA-dr pixel paths together.
- `apex_pareto.py` — `apex_rank` applies ε-Pareto dominance and ranks survivors by distance-to-ideal in normalised objective space. `format_ranking_summary` produces the human-readable report.
- `run_combined.py` — CLI runner. Loads data, invokes `run_clustered_namoa`, then `apex_rank`, then dumps JSON + `.npy` outputs.

## Usage

```bash
# From the project root
python combined/run_combined.py

# Tune epsilons / cluster size
python combined/run_combined.py --cluster-size 30 --segment-eps 0.2 \
                                --abstract-eps 0.15 --apex-eps 0.05
```

Outputs land in `combined/combined_output/` by default:

```
combined_output/
├── combined_results.json        # full candidate list + ranking metadata
├── combined_summary.txt         # human-readable summary
├── best_path.npy                # (N, 2) int32 row/col pixel path
├── pareto_paths/
│   └── pareto_path_XX.npy       # one file per Pareto candidate
└── graph_meta.pkl               # cluster-graph metadata
```

## Data expectations

The runner uses the same inputs as `src/simple_astar_results/advanced/namoa_costmap.py`:

- `npz-files/austin_construction_cost.npz` (`cost_map_normalized`)
- `npz-files/austin_environmental_impact.npz` (`env_map_normalized`)
- `npz-files/geometry_cost_map.npz` (`cost_map`)
- `src/sample-test-set/austin_test_raster.npz` (`raster`, `goal_points`, `width`, `height`, optional `bbox`, `target_crs`)

## Notes

- Intra-cluster NAMOA-dr expansion is capped per cluster area (`max(5000, cluster_size² · 8)`) so that a single pathological cluster cannot stall the graph build.
- The abstract search uses Euclidean distance replicated across the three objectives as an admissible heuristic; cost layers are non-negative so this lower bound is conservative in every component.
- Segment refinement picks the per-segment best-compromise Pareto point (closest to ideal in normalised space) rather than enumerating the cross-product of all per-segment Pareto sets — that would blow up combinatorially on multi-goal tours.
