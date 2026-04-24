#!/usr/bin/env python3
"""
compare_apex_vs_hpa.py
======================
Runs plain APEX and APEX+HPA* on the same city raster and cost maps,
then produces a side-by-side visual comparison with timing stats.

Usage:
    python3 src/compare_apex_vs_hpa.py
    python3 src/compare_apex_vs_hpa.py --city portland
    python3 src/compare_apex_vs_hpa.py --apex-eps 0.1 --apex-max-exp 500000
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

from apex_pure import RasterApexSearch
from apex_hpa import load_cost_maps, run_apex_hpa

# ---------------------------------------------------------------------------
CITY_RASTERS = {
    "austin":   os.path.join(_ROOT, "src", "sample-test-set", "austin_test_raster.npz"),
    "seattle":  os.path.join(_ROOT, "src", "sample-test-set", "seattle_test_raster.npz"),
    "portland": os.path.join(_ROOT, "src", "sample-test-set", "portland_test_raster.npz"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_raster(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    raster = data["raster"]
    goal_points = data["goal_points"]
    goals = [(int(y), int(x)) for x, y, *_ in goal_points]
    bbox = data["bbox"].tolist() if "bbox" in data.files else None
    return raster, goals, bbox


def pick_best_pareto(solutions, objective_names):
    """Distance-to-ideal selection: normalise costs, pick closest to (0,0,...)."""
    if not solutions:
        return None
    if len(solutions) == 1:
        return solutions[0]
    costs = np.array([list(s.objectives) for s in solutions], dtype=np.float64)
    mn, mx = costs.min(0), costs.max(0)
    span = np.where(mx - mn > 1e-9, mx - mn, 1.0)
    norm = (costs - mn) / span
    dists = np.linalg.norm(norm, axis=1)
    return solutions[int(np.argmin(dists))]


# ---------------------------------------------------------------------------
# Run plain APEX
# ---------------------------------------------------------------------------

# Per-city tuned settings matching the original run_apex_all.py config
APEX_CITY_CONFIG = {
    "austin":   {"eps": (0.1, 0.1, 0.1, 0.3), "max_expansions": 5_000_000},
    "seattle":  {"eps": (0.1, 0.1, 0.1, 0.1), "max_expansions":   500_000},
    "portland": {"eps": (0.1, 0.1, 0.1, 0.5), "max_expansions": 5_000_000},
}


def run_plain_apex(raster, goals, city, bbox=None, eps_override=None,
                   max_exp_override=None):
    """
    Run RasterApexSearch exactly as run_apex_all.py does:
      - Native objectives: distance, elevation, slope, turn_angle
      - Synthetic elevation (no real SRTM needed)
      - Per-city tuned eps and max_expansions
    """
    cfg = APEX_CITY_CONFIG.get(city, APEX_CITY_CONFIG["austin"])
    eps          = eps_override or cfg["eps"]
    max_expansions = max_exp_override or cfg["max_expansions"]
    source, target = goals[0], goals[-1]

    print(f"\n{'='*60}")
    print("  PLAIN APEX (full-grid search)")
    print(f"{'='*60}")
    print(f"  Source : {source}")
    print(f"  Target : {target}")
    print(f"  Grid   : {raster.shape}")
    print(f"  ε      : {eps}")
    print(f"  Max exp: {max_expansions:,}")
    print(f"  Objectives: distance, elevation, slope, turn_angle")

    searcher = RasterApexSearch(
        raster=raster,
        bbox=bbox,          # lets it use real elevation if available
        eps=eps,
        max_expansions=max_expansions,
        log_interval=50_000,
        # No custom_costmaps → uses native distance/elevation/slope objectives
    )

    t0 = time.time()
    solutions, stats = searcher.search(source, target)
    elapsed = time.time() - t0

    best = pick_best_pareto(solutions, searcher.objective_names)
    return {
        "algorithm":    "Plain APEX",
        "elapsed":      elapsed,
        "expansions":   stats.nodes_expanded,
        "pareto_count": len(solutions),
        "solutions":    solutions,
        "best":         best,
        "stats":        stats,
    }


# ---------------------------------------------------------------------------
# Run APEX + HPA*
# ---------------------------------------------------------------------------

def run_apex_hpa_wrapper(raster, goals, cost_maps, apex_eps, filter_eps,
                         cluster_size, max_expansions):
    print(f"\n{'='*60}")
    print("  APEX + HPA* (hierarchical search)")
    print(f"{'='*60}")
    print(f"  Cluster size : {cluster_size}")
    print(f"  APEX ε       : {apex_eps}")
    print(f"  Filter ε     : {filter_eps}")
    print(f"  Max exp      : {max_expansions:,}")

    t0 = time.time()
    candidates, pareto, best, graph = run_apex_hpa(
        raster, cost_maps, goals,
        cluster_size=cluster_size,
        apex_eps=apex_eps,
        filter_eps=filter_eps,
        max_expansions=max_expansions,
        verbose=True,
    )
    elapsed = time.time() - t0

    return {
        "algorithm":    "APEX + HPA*",
        "elapsed":      elapsed,
        "pareto_count": len(pareto),
        "candidates":   candidates,
        "pareto":       pareto,
        "best":         best,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_comparison(result_apex, result_hpa, raster, output_dir, city):
    """
    4-panel figure:
      [0] Plain APEX best path + all Pareto paths
      [1] APEX+HPA* best path + all Pareto paths
      [2] Overlay: both best paths on same raster
      [3] Timing & stats bar chart
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    road_cmap = ListedColormap(["#1a1a2e", "#00d4ff"])

    fig, axes = plt.subplots(1, 4, figsize=(28, 9), facecolor="#0a0a0a")
    fig.suptitle(
        f"APEX vs APEX+HPA*  |  {city.upper()}",
        color="white", fontsize=16, fontweight="bold", y=1.01,
    )

    def base_raster(ax, title):
        ax.set_facecolor("#111111")
        ax.imshow(raster, cmap=road_cmap, alpha=0.5, origin="upper")
        ax.set_title(title, color="white", fontsize=11, fontweight="bold")
        ax.tick_params(colors="white")

    # ---- Panel 0: Plain APEX ------------------------------------------------
    ax0 = axes[0]
    base_raster(ax0, f"Plain APEX\n{result_apex['pareto_count']} Pareto  |  "
                     f"{result_apex['elapsed']:.1f}s  |  "
                     f"{result_apex['expansions']:,} exp")

    apex_sols = result_apex["solutions"]
    apex_best = result_apex["best"]
    if apex_sols:
        colors = plt.cm.plasma(np.linspace(0.15, 0.9, len(apex_sols)))
        for i, sol in enumerate(apex_sols):
            path = sol.path
            xs = [p[1] for p in path]
            ys = [p[0] for p in path]
            is_best = sol is apex_best
            ax0.plot(xs, ys, color=colors[i], linewidth=3.0 if is_best else 1.2,
                     alpha=1.0 if is_best else 0.5,
                     label="best" if is_best else None)
        if apex_best:
            s, g = apex_best.path[0], apex_best.path[-1]
            ax0.scatter(s[1], s[0], c="#00ff00", s=220, marker="*",
                        zorder=10, edgecolors="white", linewidths=1.5)
            ax0.scatter(g[1], g[0], c="#ff00ff", s=220, marker="*",
                        zorder=10, edgecolors="white", linewidths=1.5)
        ax0.legend(handles=[Patch(color="#ffd93d", label="best")],
                   loc="upper right", facecolor="#222", edgecolor="white",
                   labelcolor="white", fontsize=8)

    # ---- Panel 1: APEX+HPA* -------------------------------------------------
    ax1 = axes[1]
    base_raster(ax1, f"APEX + HPA*\n{result_hpa['pareto_count']} Pareto  |  "
                     f"{result_hpa['elapsed']:.1f}s")

    hpa_pareto = result_hpa["pareto"]
    hpa_best   = result_hpa["best"]
    if hpa_pareto:
        colors = plt.cm.plasma(np.linspace(0.15, 0.9, len(hpa_pareto)))
        for i, cand in enumerate(hpa_pareto):
            path = cand["path"]
            xs = [p[1] for p in path]
            ys = [p[0] for p in path]
            is_best = hpa_best and cand["candidate_id"] == hpa_best["candidate_id"]
            ax1.plot(xs, ys, color=colors[i], linewidth=3.0 if is_best else 1.2,
                     alpha=1.0 if is_best else 0.5)
        if hpa_best:
            s = hpa_best["path"][0]
            g = hpa_best["path"][-1]
            ax1.scatter(s[1], s[0], c="#00ff00", s=220, marker="*",
                        zorder=10, edgecolors="white", linewidths=1.5)
            ax1.scatter(g[1], g[0], c="#ff00ff", s=220, marker="*",
                        zorder=10, edgecolors="white", linewidths=1.5)

    # ---- Panel 2: Overlay of both best paths --------------------------------
    ax2 = axes[2]
    base_raster(ax2, "Best-path overlay\n(yellow=APEX, cyan=APEX+HPA*)")

    if apex_best:
        path = apex_best.path
        ax2.plot([p[1] for p in path], [p[0] for p in path],
                 color="#ffd93d", linewidth=2.5, alpha=0.9, label="Plain APEX")
    if hpa_best:
        path = hpa_best["path"]
        ax2.plot([p[1] for p in path], [p[0] for p in path],
                 color="#00e5ff", linewidth=2.5, alpha=0.9, label="APEX+HPA*")
    # start / goal from APEX+HPA (both share same endpoints)
    if hpa_best:
        s, g = hpa_best["path"][0], hpa_best["path"][-1]
        ax2.scatter(s[1], s[0], c="#00ff00", s=260, marker="*",
                    zorder=10, edgecolors="white", linewidths=1.5)
        ax2.scatter(g[1], g[0], c="#ff00ff", s=260, marker="*",
                    zorder=10, edgecolors="white", linewidths=1.5)
    ax2.legend(loc="upper right", facecolor="#222", edgecolor="white",
               labelcolor="white", fontsize=9)

    # ---- Panel 3: Stats bar chart -------------------------------------------
    ax3 = axes[3]
    ax3.set_facecolor("#111111")
    ax3.set_title("Timing & Pareto size", color="white",
                  fontsize=11, fontweight="bold")

    metrics = {
        "Runtime (s)":       [result_apex["elapsed"],
                               result_hpa["elapsed"]],
        "Pareto paths":      [result_apex["pareto_count"],
                               result_hpa["pareto_count"]],
        "Expansions (k)":    [result_apex["expansions"] / 1000,
                               0],  # HPA* abstract expansions not comparable
    }
    labels = ["Plain APEX", "APEX+HPA*"]
    colors_bar = ["#ffd93d", "#00e5ff"]
    x = np.arange(len(metrics))
    width = 0.3

    for i, (metric, vals) in enumerate(metrics.items()):
        bars = ax3.bar(x[i] + np.array([-width/2, width/2]), vals,
                       width=width, color=colors_bar, edgecolor="white",
                       linewidth=0.8)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax3.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + max(vals) * 0.02,
                         f"{val:.1f}", ha="center", va="bottom",
                         color="white", fontsize=8)

    ax3.set_xticks(x)
    ax3.set_xticklabels(list(metrics.keys()), color="white", fontsize=9)
    ax3.tick_params(colors="white")
    ax3.set_ylabel("Value", color="white")
    ax3.yaxis.label.set_color("white")
    for spine in ax3.spines.values():
        spine.set_edgecolor("#555")
    legend_handles = [Patch(color=c, label=l)
                      for c, l in zip(colors_bar, labels)]
    ax3.legend(handles=legend_handles, loc="upper right",
               facecolor="#222", edgecolor="white",
               labelcolor="white", fontsize=9)

    # ---- Cost summary below bar chart ----------------------------------------
    def fmt_cost(result, algo_label):
        if algo_label == "apex":
            best = result["best"]
            if best is None:
                return "No solution"
            o = best.objectives
            return (f"dist={o[0]:.1f}  elev={o[1]:.1f}  "
                    f"slope={o[2]:.3f}  waypts={len(best.path)}")
        else:
            b = result["best"]
            if b is None:
                return "No solution"
            cv = b["cost_vector"]
            return (f"constr={cv[0]:.1f}  env={cv[1]:.1f}  "
                    f"geo={cv[2]:.1f}  waypts={len(b['path'])}")

    cost_text = (
        f"Plain APEX best:  {fmt_cost(result_apex, 'apex')}\n"
        f"APEX+HPA*  best:  {fmt_cost(result_hpa, 'hpa')}"
    )
    ax3.text(0.01, -0.18, cost_text, transform=ax3.transAxes,
             color="#aaaaaa", fontsize=8, verticalalignment="top",
             fontfamily="monospace")

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"comparison_{city}.png")
    plt.savefig(out, dpi=150, facecolor="#0a0a0a", bbox_inches="tight")
    plt.close()
    print(f"\n  [viz] Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare Plain APEX vs APEX+HPA* on the same raster")
    parser.add_argument("--city", default="austin",
                        choices=list(CITY_RASTERS.keys()))
    parser.add_argument("--cluster-size",   type=int,   default=80)
    parser.add_argument("--apex-eps",       type=float, default=0.1,
                        help="ε for both APEX searches (default 0.1)")
    parser.add_argument("--filter-eps",     type=float, default=0.05,
                        help="ε for APEX+HPA* final filter (default 0.05)")
    parser.add_argument("--apex-max-exp",   type=int,   default=500_000,
                        help="Max expansions for plain APEX (default 500k)")
    parser.add_argument("--hpa-max-exp",    type=int,   default=500_000,
                        help="Max abstract expansions for APEX+HPA* (default 500k)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_dir = args.output or os.path.join(
        _HERE, "sample-test-set", "comparison_output", args.city)

    print("=" * 65)
    print("  APEX vs APEX+HPA*  Comparison")
    print("=" * 65)
    print(f"  City         : {args.city}")
    print(f"  Cluster size : {args.cluster_size}")
    print(f"  APEX ε       : {args.apex_eps}")
    print(f"  Filter ε     : {args.filter_eps}")

    # Load shared inputs
    raster, goals, bbox = load_raster(CITY_RASTERS[args.city])
    cost_maps = load_cost_maps(_ROOT)
    print(f"  Raster       : {raster.shape}")
    print(f"  Goals        : {goals}")

    # Run both algorithms
    result_apex = run_plain_apex(
        raster, goals, city=args.city, bbox=bbox,
    )

    result_hpa = run_apex_hpa_wrapper(
        raster, goals, cost_maps,
        apex_eps=args.apex_eps,
        filter_eps=args.filter_eps,
        cluster_size=args.cluster_size,
        max_expansions=args.hpa_max_exp,
    )

    # Print summary table
    print(f"\n{'='*65}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Metric':<28} {'Plain APEX':>14}  {'APEX+HPA*':>14}")
    print(f"  {'-'*58}")
    print(f"  {'Runtime (s)':<28} {result_apex['elapsed']:>14.2f}  "
          f"{result_hpa['elapsed']:>14.2f}")
    print(f"  {'Pareto paths found':<28} {result_apex['pareto_count']:>14}  "
          f"{result_hpa['pareto_count']:>14}")
    print(f"  {'Nodes expanded':<28} {result_apex['expansions']:>14,}  "
          f"{'(abstract)':>14}")

    apex_best = result_apex["best"]
    hpa_best  = result_hpa["best"]

    if apex_best:
        o = apex_best.objectives
        names = ["distance", "elevation", "slope", "turn"]
        print(f"\n  Plain APEX best path:")
        for nm, val in zip(names, o):
            print(f"    {nm}={val:.2f}", end="  ")
        print(f"  waypoints={len(apex_best.path)}")
    if hpa_best:
        cv = hpa_best["cost_vector"]
        print(f"\n  APEX+HPA* best path:")
        print(f"    construction={cv[0]:.1f}  environmental={cv[1]:.1f}  "
              f"geometry={cv[2]:.1f}  waypoints={len(hpa_best['path'])}")

    speedup = result_apex["elapsed"] / max(result_hpa["elapsed"], 0.001)
    print(f"\n  Speedup (APEX+HPA* vs Plain APEX): {speedup:.1f}×")
    print(f"{'='*65}")

    # Save JSON summary
    os.makedirs(output_dir, exist_ok=True)
    summary = {
        "city":   args.city,
        "plain_apex": {
            "elapsed_s":      result_apex["elapsed"],
            "expansions":     result_apex["expansions"],
            "pareto_count":   result_apex["pareto_count"],
            "objectives":     ["distance", "elevation", "slope", "turn_angle"],
            "best_cost":      [float(v) for v in apex_best.objectives] if apex_best else None,
            "best_waypoints": len(apex_best.path) if apex_best else None,
        },
        "apex_hpa": {
            "elapsed_s":      result_hpa["elapsed"],
            "pareto_count":   result_hpa["pareto_count"],
            "objectives":     ["construction", "environmental", "geometry"],
            "best_cost":      list(hpa_best["cost_vector"]) if hpa_best else None,
            "best_waypoints": len(hpa_best["path"]) if hpa_best else None,
        },
        "speedup_x": speedup,
    }
    summary_path = os.path.join(output_dir, f"summary_{args.city}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON : {summary_path}")

    # Generate visualization
    visualize_comparison(result_apex, result_hpa, raster, output_dir, args.city)

    return 0


if __name__ == "__main__":
    sys.exit(main())
