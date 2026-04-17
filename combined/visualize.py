#!/usr/bin/env python3
"""Visualisation for the combined HPA* + NAMOA*-dr + APEX pipeline.

Adapted from pranav-dev's ``ApexVisualizer``. Reads
``combined_results.json`` (the schema produced by ``run_combined.py``)
and writes:

  * ``routes_overlay.png`` — Pareto routes drawn on the Austin raster
  * ``pareto_3d.png``      — 3-D Pareto front with the BEST point flagged
  * ``pareto_2d.png``      — all pairwise 2-D projections
  * ``composite_heatmap.png`` — routes on top of the summed cost layers
    (only when ``--npz-dir`` is supplied so the layers can be loaded)

The input schema differs from A*pex — candidates live under ``candidates``
with keys ``path_row_col``, ``construction``, ``environmental``,
``geometry`` and an ``on_pareto_front`` flag — so this module normalises
those into the (path, objectives) shape pranav's plotting code expects.
"""

import argparse
import json
import os
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


OBJECTIVES = ("construction", "environmental", "geometry")


def _candidates_to_solutions(results: dict,
                             pareto_only: bool = True) -> List[dict]:
    """Flatten the combined pipeline's candidate list into pranav's shape."""
    raw = results.get("candidates", [])
    if pareto_only:
        raw = [c for c in raw if c.get("on_pareto_front")]
        if not raw:
            raw = results.get("candidates", [])
    best_id = results.get("best_candidate_id")

    sols = []
    for c in raw:
        sols.append({
            "path": c.get("path_row_col", []),
            "objectives": [float(c.get(o, 0.0)) for o in OBJECTIVES],
            "candidate_id": c.get("candidate_id"),
            "apex_rank": c.get("apex_rank"),
            "is_best": c.get("candidate_id") == best_id,
        })
    sols.sort(key=lambda s: (s.get("apex_rank") or 1_000_000))
    return sols


def plot_routes_on_raster(solutions: List[dict], raster: np.ndarray,
                          output_path: str, title: str = "Pareto routes",
                          max_routes: int = 12) -> None:
    if not solutions:
        print("  [viz] no solutions — skipping routes overlay")
        return
    sols = solutions[:max_routes]

    fig, ax = plt.subplots(figsize=(14, 12), facecolor="#0a0a0a")
    ax.set_facecolor("#111111")
    cmap = ListedColormap(["#1a1a2e", "#00d4ff"])
    ax.imshow(raster, cmap=cmap, alpha=0.6, origin="upper")

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(sols)))
    for i, sol in enumerate(sols):
        path = sol["path"]
        if not path:
            continue
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        lw = 3.5 if sol.get("is_best") else 2.0
        alpha = 1.0 if sol.get("is_best") else 0.8
        label = (f"id={sol['candidate_id']}"
                 + (" (BEST)" if sol.get("is_best") else ""))
        ax.plot(xs, ys, color=colors[i], linewidth=lw, alpha=alpha,
                label=label)

    # Start / goal markers from the first Pareto path.
    first = sols[0]["path"]
    if first:
        s = first[0]
        g = first[-1]
        ax.scatter(s[1], s[0], c="#00ff00", s=300, marker="*",
                   zorder=10, edgecolors="white", linewidths=2, label="Start")
        ax.scatter(g[1], g[0], c="#ff00ff", s=300, marker="*",
                   zorder=10, edgecolors="white", linewidths=2, label="Goal")

    ax.set_title(title, color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    ax.legend(loc="upper right", fontsize=8, facecolor="#222222",
              edgecolor="white", labelcolor="white")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor="#0a0a0a",
                bbox_inches="tight")
    plt.close()
    print(f"  [viz] wrote {output_path}")


def plot_pareto_3d(solutions: List[dict], output_path: str,
                   ideal: Optional[Tuple[float, float, float]] = None,
                   nadir: Optional[Tuple[float, float, float]] = None) -> None:
    if not solutions:
        return
    obj = np.asarray([s["objectives"] for s in solutions], dtype=np.float64)

    fig = plt.figure(figsize=(12, 10), facecolor="#0a0a0a")
    ax = fig.add_subplot(111, projection="3d", facecolor="#0a0a0a")

    scatter = ax.scatter(
        obj[:, 0], obj[:, 1], obj[:, 2],
        c=np.arange(len(obj)), cmap="plasma",
        s=120, alpha=0.9, edgecolors="white", linewidths=0.5,
    )

    best_idx = next((i for i, s in enumerate(solutions) if s.get("is_best")),
                    None)
    if best_idx is not None:
        bx, by, bz = obj[best_idx]
        ax.scatter([bx], [by], [bz], s=380, marker="*", c="#ffd93d",
                   edgecolors="black", linewidths=1.2, label="BEST")

    if ideal is not None:
        ax.scatter([ideal[0]], [ideal[1]], [ideal[2]],
                   s=220, marker="X", c="#00ff88",
                   edgecolors="white", linewidths=1.0, label="ideal")
    if nadir is not None:
        ax.scatter([nadir[0]], [nadir[1]], [nadir[2]],
                   s=220, marker="X", c="#ff5555",
                   edgecolors="white", linewidths=1.0, label="nadir")

    ax.set_xlabel("construction", color="white")
    ax.set_ylabel("environmental", color="white")
    ax.set_zlabel("geometry", color="white")
    ax.set_title("Pareto front (3D)", color="white",
                 fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
    if best_idx is not None or ideal is not None:
        ax.legend(loc="upper left", facecolor="#222222", edgecolor="white",
                  labelcolor="white")
    plt.colorbar(scatter, ax=ax, label="candidate index", pad=0.1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor="#0a0a0a",
                bbox_inches="tight")
    plt.close()
    print(f"  [viz] wrote {output_path}")


def plot_pareto_2d_projections(solutions: List[dict],
                               output_path: str) -> None:
    if not solutions:
        return
    obj = np.asarray([s["objectives"] for s in solutions], dtype=np.float64)
    pairs = list(combinations(range(len(OBJECTIVES)), 2))
    n_cols = len(pairs)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5),
                             facecolor="#111111")
    if n_cols == 1:
        axes = [axes]

    for ax, (i, j) in zip(axes, pairs):
        ax.set_facecolor("#1a1a1a")
        ax.scatter(obj[:, i], obj[:, j],
                   c=np.arange(len(obj)), cmap="viridis",
                   s=140, alpha=0.9, edgecolors="white", linewidths=1)
        for idx, s in enumerate(solutions):
            tag = str(s.get("candidate_id", idx + 1))
            ax.annotate(tag, (obj[idx, i], obj[idx, j]),
                        fontsize=9, color="#ffd93d",
                        xytext=(4, 4), textcoords="offset points")
            if s.get("is_best"):
                ax.scatter([obj[idx, i]], [obj[idx, j]],
                           s=260, facecolors="none",
                           edgecolors="#ffd93d", linewidths=2.0)
        ax.set_xlabel(OBJECTIVES[i], color="white")
        ax.set_ylabel(OBJECTIVES[j], color="white")
        ax.set_title(f"{OBJECTIVES[i]} vs {OBJECTIVES[j]}",
                     color="white", fontsize=12)
        ax.tick_params(colors="white")
        ax.grid(True, alpha=0.2, color="gray")

    plt.suptitle("Pareto front — 2D projections", color="white",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor="#111111",
                bbox_inches="tight")
    plt.close()
    print(f"  [viz] wrote {output_path}")


def plot_composite_heatmap(solutions: List[dict],
                           cost_maps: Tuple[np.ndarray, np.ndarray, np.ndarray],
                           output_path: str,
                           max_routes: int = 8) -> None:
    """Sum the three normalised cost layers and overlay the routes."""
    if not solutions or cost_maps is None:
        return
    composite = np.zeros(cost_maps[0].shape, dtype=np.float64)
    for arr in cost_maps:
        mn, mx = float(arr.min()), float(arr.max())
        if mx > mn:
            composite += (arr - mn) / (mx - mn)

    fig, ax = plt.subplots(figsize=(14, 12), facecolor="#0a0a0a")
    ax.set_facecolor("#111111")
    im = ax.imshow(composite, cmap="hot", origin="upper", alpha=0.9)
    plt.colorbar(im, ax=ax, label="composite penalty (sum of normalised layers)",
                 shrink=0.8)

    sols = solutions[:max_routes]
    colors = plt.cm.cool(np.linspace(0.1, 0.9, len(sols)))
    for i, sol in enumerate(sols):
        path = sol["path"]
        if not path:
            continue
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        lw = 3.0 if sol.get("is_best") else 2.0
        ax.plot(xs, ys, color=colors[i], linewidth=lw, alpha=0.9,
                label=f"id={sol['candidate_id']}"
                      + (" (BEST)" if sol.get("is_best") else ""))

    first = sols[0]["path"]
    if first:
        s, g = first[0], first[-1]
        ax.scatter(s[1], s[0], c="#00ff00", s=300, marker="*", zorder=10,
                   edgecolors="black", linewidths=1.5)
        ax.scatter(g[1], g[0], c="#ff00ff", s=300, marker="*", zorder=10,
                   edgecolors="black", linewidths=1.5)
    ax.set_title("Composite cost map with Pareto routes",
                 color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    ax.legend(loc="upper right", fontsize=8, facecolor="#222222",
              edgecolor="white", labelcolor="white")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor="#0a0a0a",
                bbox_inches="tight")
    plt.close()
    print(f"  [viz] wrote {output_path}")


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def visualize_combined_results(results_path: str, raster: np.ndarray,
                               output_dir: str,
                               cost_maps: Optional[Tuple[np.ndarray, np.ndarray,
                                                         np.ndarray]] = None,
                               pareto_only: bool = True) -> None:
    with open(results_path) as f:
        results = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    solutions = _candidates_to_solutions(results, pareto_only=pareto_only)
    ideal = tuple(results["ideal"]) if "ideal" in results else None
    nadir = tuple(results["nadir"]) if "nadir" in results else None

    plot_routes_on_raster(solutions, raster,
                          os.path.join(output_dir, "routes_overlay.png"))
    plot_pareto_3d(solutions,
                   os.path.join(output_dir, "pareto_3d.png"),
                   ideal=ideal, nadir=nadir)
    plot_pareto_2d_projections(solutions,
                               os.path.join(output_dir, "pareto_2d.png"))
    if cost_maps is not None:
        plot_composite_heatmap(solutions, cost_maps,
                               os.path.join(output_dir,
                                            "composite_heatmap.png"))


def main():
    parser = argparse.ArgumentParser(
        description="Render plots for combined_results.json")
    parser.add_argument("--results", default=None,
                        help="Path to combined_results.json (default: "
                             "combined/combined_output/combined_results.json)")
    parser.add_argument("--raster", default=None,
                        help="Austin raster NPZ (default: Austin test set)")
    parser.add_argument("--output", default=None,
                        help="Output dir (default: "
                             "<results_dir>/plots)")
    parser.add_argument("--all-candidates", action="store_true",
                        help="Plot every candidate, not just the Pareto "
                             "frontier.")
    parser.add_argument("--with-heatmap", action="store_true",
                        help="Also render composite_heatmap.png (requires "
                             "npz-files/*.npz to be present).")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    results_path = args.results or os.path.join(here, "combined_output",
                                                "combined_results.json")

    from cost_map_loader import DEFAULT_RASTER, load_cost_maps
    raster_path = args.raster or DEFAULT_RASTER
    raster = np.load(raster_path, allow_pickle=True)["raster"]

    output_dir = args.output or os.path.join(os.path.dirname(results_path),
                                             "plots")

    cost_maps = None
    if args.with_heatmap:
        cost_maps = load_cost_maps(verbose=False)

    visualize_combined_results(results_path, raster, output_dir,
                               cost_maps=cost_maps,
                               pareto_only=not args.all_candidates)
    print(f"\nSaved visualisations to {output_dir}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
