#!/usr/bin/env python3
"""End-to-end runner for the combined pipeline.

Stages:
  1. Load the three cost-map layers + the Austin road raster (goal pts).
  2. Build an HPA*-style clustered graph (arvin-dev), but compute every
     cluster-internal and inter-cluster edge with NAMOA*-dr (rishik-k).
  3. Run a multi-objective NAMOA*-dr over the abstract graph, visiting
     every goal, to produce a Pareto set of candidate paths.
  4. APEX (pranav-dev) analyses the Pareto set and selects the
     best-compromise path.

Outputs under ``--output``:
  combined_results.json            — ideal/nadir, ranking, every candidate
  best_path.npy                    — (N, 2) int32 row/col pixels
  pareto_paths/pareto_path_XX.npy  — one file per Pareto candidate
  combined_summary.txt             — human-readable summary
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

# Allow running as either `python run_combined.py` (cwd=combined/) or
# `python combined/run_combined.py` from the project root.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from cost_map_loader import (load_cost_maps, load_raster_and_goals,
                             composite_cost_grid, DEFAULT_RASTER)
from clustered_namoa import run_clustered_namoa
from apex_pareto import apex_rank, format_ranking_summary
from visualize import visualize_combined_results


def main():
    parser = argparse.ArgumentParser(
        description="Clustered NAMOA-dr + APEX combined pipeline")
    parser.add_argument("--raster", default=DEFAULT_RASTER,
                        help="Road raster NPZ (default: Austin).")
    parser.add_argument("--output", default=os.path.join(_HERE,
                                                         "combined_output"),
                        help="Output directory.")
    parser.add_argument("--cluster-size", type=int, default=40)
    parser.add_argument("--segment-eps", type=float, default=0.2,
                        help="ε-dominance for intra-cluster NAMOA-dr.")
    parser.add_argument("--abstract-eps", type=float, default=0.15,
                        help="ε-dominance for abstract NAMOA-dr.")
    parser.add_argument("--apex-eps", type=float, default=0.05,
                        help="ε-dominance for APEX Pareto filter.")
    parser.add_argument("--max-abstract-expansions", type=int, default=500_000)
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers for cluster NAMOA-dr "
                             "(default: one per CPU core; pass 1 to force "
                             "sequential).")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip the visualisation stage.")
    parser.add_argument("--heatmap", action="store_true",
                        help="Also render the composite cost-map heatmap.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    t_start = time.time()

    print("=" * 70)
    print("Combined pipeline: HPA* clustering + NAMOA*-dr + APEX selection")
    print("=" * 70)

    print("\n[1/4] Loading cost maps + raster ...")
    cost_maps = load_cost_maps()
    raster, goals, meta = load_raster_and_goals(args.raster)
    composite = composite_cost_grid(cost_maps)
    start = goals[0]
    print(f"  Start: {start}")
    print(f"  Goals: {goals}")

    print("\n[2/4] Clustered NAMOA*-dr ...")
    candidates, graph = run_clustered_namoa(
        start, goals, cost_maps, composite,
        cluster_size=args.cluster_size,
        segment_eps=args.segment_eps,
        abstract_eps=args.abstract_eps,
        abstract_max_expansions=args.max_abstract_expansions,
        verbose=True,
        num_workers=args.workers,
    )

    if not candidates:
        print("\nNo candidate paths produced. Exiting.")
        return 1

    print("\n[3/4] APEX Pareto analysis ...")
    ranking = apex_rank(candidates, eps=args.apex_eps)
    print(format_ranking_summary(ranking))

    print("\n[4/4] Writing outputs ...")
    best = ranking.best
    if best is not None:
        best_path = np.asarray(best["path"], dtype=np.int32)
        np.save(os.path.join(args.output, "best_path.npy"), best_path)

    pareto_dir = os.path.join(args.output, "pareto_paths")
    os.makedirs(pareto_dir, exist_ok=True)
    for cand in ranking.pareto:
        arr = np.asarray(cand["path"], dtype=np.int32)
        np.save(os.path.join(pareto_dir,
                             f"pareto_path_{cand['candidate_id']:02d}.npy"),
                arr)

    json_payload = {
        "algorithm": "HPA* clustering + NAMOA*-dr + APEX selection",
        "objectives": ["construction", "environmental", "geometry"],
        "ideal": list(ranking.ideal),
        "nadir": list(ranking.nadir),
        "start": list(start),
        "goals": [list(g) for g in goals],
        "num_candidates": len(candidates),
        "num_pareto": len(ranking.pareto),
        "best_candidate_id": best.get("candidate_id") if best else None,
        "best_cost_vector": list(best["cost_vector"]) if best else None,
        "candidates": [
            {
                "candidate_id": c["candidate_id"],
                "construction": c["construction"],
                "environmental": c["environmental"],
                "geometry": c["geometry"],
                "num_waypoints": len(c["path"]),
                "apex_rank": c.get("apex_rank"),
                "apex_distance_to_ideal": c.get("apex_distance_to_ideal"),
                "on_pareto_front": c in ranking.pareto,
                "abstract_sequence": [list(p) for p in c["abstract_sequence"]],
                "path_row_col": [[int(r), int(cc)] for r, cc in c["path"]],
            }
            for c in candidates
        ],
        "total_runtime_seconds": time.time() - t_start,
    }
    with open(os.path.join(args.output, "combined_results.json"), "w") as f:
        json.dump(json_payload, f, indent=2)

    with open(os.path.join(args.output, "combined_summary.txt"), "w") as f:
        f.write(format_ranking_summary(ranking))
        f.write(f"\n\nTotal runtime: "
                f"{json_payload['total_runtime_seconds']:.2f}s\n")

    # Persist the clustered graph metadata for later inspection (nodes
    # only — edge Pareto sets would be verbose).
    with open(os.path.join(args.output, "graph_meta.pkl"), "wb") as f:
        pickle.dump({
            "cluster_size": args.cluster_size,
            "num_gateways": len(graph.nodes),
            "raster_shape": tuple(composite.shape),
            "goals": goals,
            "start": start,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    if not args.no_plots:
        print("\n[plots] Rendering visualisations ...")
        plots_dir = os.path.join(args.output, "plots")
        try:
            visualize_combined_results(
                os.path.join(args.output, "combined_results.json"),
                raster, plots_dir,
                cost_maps=cost_maps if args.heatmap else None,
                pareto_only=True,
            )
        except Exception as exc:
            print(f"[plots] skipped: {type(exc).__name__}: {exc}")

    print(f"\nSaved to: {args.output}")
    print(f"Total runtime: {time.time() - t_start:.2f}s")
    if best is not None:
        cv = best["cost_vector"]
        print(f"Best path: id={best['candidate_id']}, "
              f"constr={cv[0]:.1f}, env={cv[1]:.1f}, geo={cv[2]:.1f}, "
              f"waypoints={len(best['path'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
