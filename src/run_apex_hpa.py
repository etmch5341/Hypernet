#!/usr/bin/env python3
"""
Runner: APEX + HPA* combined pipeline

Usage:
    python3 src/run_apex_hpa.py
    python3 src/run_apex_hpa.py --city portland --cluster-size 30
    python3 src/run_apex_hpa.py --apex-eps 0.15 --filter-eps 0.05
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# Resolve project root (works whether invoked from root or from src/)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

from apex_hpa import load_cost_maps, run_apex_hpa

# ---------------------------------------------------------------------------
# City configurations
# ---------------------------------------------------------------------------
CITY_RASTERS = {
    "austin":   os.path.join(_ROOT, "src", "sample-test-set", "austin_test_raster.npz"),
    "seattle":  os.path.join(_ROOT, "src", "sample-test-set", "seattle_test_raster.npz"),
    "portland": os.path.join(_ROOT, "src", "sample-test-set", "portland_test_raster.npz"),
}


def load_raster(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    raster = data["raster"]
    goal_points = data["goal_points"]
    goals = [(int(y), int(x)) for x, y, *_ in goal_points]
    return raster, goals


def save_results(output_dir, candidates, pareto, best, city):
    os.makedirs(output_dir, exist_ok=True)

    payload = {
        "algorithm": "APEX + HPA*",
        "city": city,
        "objectives": ["construction", "environmental", "geometry"],
        "num_candidates": len(candidates),
        "num_pareto": len(pareto),
        "best_candidate_id": best.get("candidate_id") if best else None,
        "best_cost_vector": list(best["cost_vector"]) if best else None,
        "candidates": [
            {
                "candidate_id":   c["candidate_id"],
                "construction":   c["construction"],
                "environmental":  c["environmental"],
                "geometry":       c["geometry"],
                "num_waypoints":  len(c["path"]),
                "apex_rank":      c.get("apex_rank"),
                "apex_dist":      c.get("apex_dist"),
                "on_pareto":      c in pareto,
                "path_row_col":   [[int(r), int(cc)] for r, cc in c["path"]],
            }
            for c in candidates
        ],
    }

    results_path = os.path.join(output_dir, "apex_hpa_results.json")
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Results saved: {results_path}")

    if best:
        path_arr = np.array(best["path"], dtype=np.int32)
        np.save(os.path.join(output_dir, "best_path.npy"), path_arr)

    for c in pareto:
        arr = np.array(c["path"], dtype=np.int32)
        pareto_dir = os.path.join(output_dir, "pareto_paths")
        os.makedirs(pareto_dir, exist_ok=True)
        np.save(os.path.join(pareto_dir,
                             f"pareto_path_{c['candidate_id']:02d}.npy"), arr)

    return results_path


def visualize(results_path, raster, output_dir):
    """Simple route overlay using matplotlib — no external dependencies."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        with open(results_path) as f:
            data = json.load(f)

        candidates = [c for c in data["candidates"] if c["on_pareto"]]
        best_id = data.get("best_candidate_id")

        fig, ax = plt.subplots(figsize=(14, 11), facecolor="#0a0a0a")
        ax.set_facecolor("#111111")
        cmap = ListedColormap(["#1a1a2e", "#00d4ff"])
        ax.imshow(raster, cmap=cmap, alpha=0.55, origin="upper")

        colors = plt.cm.plasma(np.linspace(0.1, 0.9, max(len(candidates), 1)))
        for i, c in enumerate(candidates):
            path = c["path_row_col"]
            if not path:
                continue
            ys = [p[0] for p in path]
            xs = [p[1] for p in path]
            is_best = c["candidate_id"] == best_id
            lw = 3.5 if is_best else 1.8
            label = f"id={c['candidate_id']}" + (" (BEST)" if is_best else "")
            ax.plot(xs, ys, color=colors[i % len(colors)],
                    linewidth=lw, alpha=1.0 if is_best else 0.75, label=label)

        if candidates and candidates[0]["path_row_col"]:
            s = candidates[0]["path_row_col"][0]
            g = candidates[0]["path_row_col"][-1]
            ax.scatter(s[1], s[0], c="#00ff00", s=280, marker="*",
                       zorder=10, edgecolors="white", linewidths=1.5)
            ax.scatter(g[1], g[0], c="#ff00ff", s=280, marker="*",
                       zorder=10, edgecolors="white", linewidths=1.5)

        ax.set_title(f"APEX + HPA*  |  {data['city']}  "
                     f"|  {len(candidates)} Pareto paths",
                     color="white", fontsize=13, fontweight="bold")
        ax.tick_params(colors="white")
        ax.legend(loc="upper right", fontsize=8, facecolor="#222222",
                  edgecolor="white", labelcolor="white")
        plt.tight_layout()
        out = os.path.join(output_dir, "routes_overlay.png")
        plt.savefig(out, dpi=150, facecolor="#0a0a0a", bbox_inches="tight")
        plt.close()
        print(f"  Visualization: {out}")
    except Exception as exc:
        print(f"  [viz] skipped: {exc}")


def main():
    parser = argparse.ArgumentParser(description="APEX + HPA* combined pipeline")
    parser.add_argument("--city", default="austin",
                        choices=list(CITY_RASTERS.keys()))
    parser.add_argument("--cluster-size", type=int, default=40)
    parser.add_argument("--apex-eps",   type=float, default=0.1,
                        help="ε for APEX abstract search (default 0.1)")
    parser.add_argument("--filter-eps", type=float, default=0.05,
                        help="ε for final Pareto filter (default 0.05)")
    parser.add_argument("--max-expansions", type=int, default=500_000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_dir = args.output or os.path.join(
        _HERE, "sample-test-set", "apex_hpa_results", args.city)

    print("=" * 65)
    print("  APEX + HPA*  Combined Multi-Objective Pathfinding")
    print("=" * 65)
    print(f"  City          : {args.city}")
    print(f"  Cluster size  : {args.cluster_size}")
    print(f"  APEX ε        : {args.apex_eps}")
    print(f"  Filter ε      : {args.filter_eps}")
    print(f"  Max expansions: {args.max_expansions:,}")
    print(f"  Output        : {output_dir}")
    print("=" * 65)

    raster_path = CITY_RASTERS[args.city]
    raster, goals = load_raster(raster_path)
    cost_maps = load_cost_maps(_ROOT)

    print(f"\n  Raster : {raster.shape}")
    print(f"  Goals  : {goals}")

    candidates, pareto, best, graph = run_apex_hpa(
        raster, cost_maps, goals,
        cluster_size=args.cluster_size,
        apex_eps=args.apex_eps,
        filter_eps=args.filter_eps,
        max_expansions=args.max_expansions,
        verbose=True,
    )

    if not candidates:
        print("\nNo paths produced — exiting.")
        return 1

    results_path = save_results(output_dir, candidates, pareto, best, args.city)
    visualize(results_path, raster, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
