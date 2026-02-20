#!/usr/bin/env python3
"""
A*pex Algorithm Comparison Framework

Compares:
  1. Pure A*pex (true multi-objective with ε-dominance)  
  2. EMO+A* (evolutionary weight tuning on scalarized A*)
  3. Single-objective A* (baseline for timing comparison)

Outputs metrics, Pareto fronts, and visualizations for paper.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import json
import os
import heapq
import math
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from dataclasses import dataclass, field
import sys

# Add src to path to allow imports
sys.path.append(os.path.join(os.getcwd(), 'src'))

from apex_pure import RasterApexSearch

try:
    from data_pipeline.system.feature_extraction import extract_elevation
    HAS_ELEVATION_LIB = True
except ImportError:
    try:
        from .data_pipeline.system.feature_extraction import extract_elevation
        HAS_ELEVATION_LIB = True
    except ImportError:
        print("Warning: Could not import extract_elevation. Real elevation data will not be available.")
        HAS_ELEVATION_LIB = False


@dataclass
class ComparisonResult:
    """Results from one algorithm run."""
    algorithm: str
    runtime_seconds: float
    nodes_expanded: int
    solutions_found: int
    pareto_front: List[Tuple[float, ...]]  # List of objective vectors
    hypervolume: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "runtime_seconds": self.runtime_seconds,
            "nodes_expanded": self.nodes_expanded,
            "solutions_found": self.solutions_found,
            "pareto_front": [[float(v) for v in p] for p in self.pareto_front],
            "hypervolume": self.hypervolume
        }


def compute_hypervolume(pareto_front: List[Tuple[float, ...]], 
                         ref_point: Tuple[float, ...]) -> float:
    """
    Compute hypervolume indicator (2D only for simplicity).
    For 3D, this would need more complex calculation.
    """
    if not pareto_front or len(pareto_front[0]) != 2:
        return 0.0
    
    # Sort by first objective
    sorted_front = sorted(pareto_front, key=lambda x: x[0])
    
    hv = 0.0
    prev_y = ref_point[1]
    
    for point in sorted_front:
        x, y = point[0], point[1]
        if x < ref_point[0] and y < prev_y:
            hv += (ref_point[0] - x) * (prev_y - y)
            prev_y = y
    
    return hv

# =============================================================================
# Single-Objective A* (Baseline)
# =============================================================================

def single_objective_astar(raster: np.ndarray, 
                            source: Tuple[int, int], 
                            target: Tuple[int, int],
                            road_cost: float = 1.0,
                            offroad_cost: float = 5.0) -> ComparisonResult:
    """
    Standard single-objective A* for baseline comparison.
    Optimizes only distance.
    """
    height, width = raster.shape
    
    def heuristic(pos):
        return math.hypot(target[0] - pos[0], target[1] - pos[1])
    
    def get_neighbors(pos):
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        for dr, dc in directions:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < height and 0 <= nc < width:
                dist = math.hypot(dr, dc)
                mult = road_cost if raster[nr, nc] == 1 else offroad_cost
                yield (nr, nc), dist * mult
    
    start_time = time.time()
    
    # Priority queue: (f_score, g_score, position)
    heap = [(heuristic(source), 0.0, source)]
    g_scores = {source: 0.0}
    came_from = {}
    expansions = 0
    
    while heap:
        f, g, current = heapq.heappop(heap)
        
        if current == target:
            # Found solution
            elapsed = time.time() - start_time
            return ComparisonResult(
                algorithm="Single-Objective A*",
                runtime_seconds=elapsed,
                nodes_expanded=expansions,
                solutions_found=1,
                pareto_front=[(g,)]  # Just distance
            )
        
        if g > g_scores.get(current, float('inf')):
            continue
        
        expansions += 1
        
        for neighbor, cost in get_neighbors(current):
            new_g = g + cost
            
            if new_g < g_scores.get(neighbor, float('inf')):
                g_scores[neighbor] = new_g
                came_from[neighbor] = current
                heapq.heappush(heap, (new_g + heuristic(neighbor), new_g, neighbor))
    
    # No path found
    elapsed = time.time() - start_time
    return ComparisonResult(
        algorithm="Single-Objective A*",
        runtime_seconds=elapsed,
        nodes_expanded=expansions,
        solutions_found=0,
        pareto_front=[]
    )


# =============================================================================  
# Scalarized A* (for EMO comparison)
# =============================================================================

def scalarized_astar(raster: np.ndarray,
                      elevation: np.ndarray,
                      slope: np.ndarray,
                      source: Tuple[int, int],
                      target: Tuple[int, int],
                      weights: Tuple[float, float, float],
                      road_cost: float = 1.0,
                      offroad_cost: float = 5.0,
                      max_expansions: int = 1000000) -> Tuple[Optional[List], Tuple[float, float, float], int]:
    """
    A* with weighted scalarization of 3 objectives (distance, elevation, slope).
    
    Returns: (path, objective_vector, nodes_expanded)
    """
    height, width = raster.shape
    w_dist, w_elev, w_slope = weights
    
    def heuristic(pos):
        return math.hypot(target[0] - pos[0], target[1] - pos[1]) * w_dist
    
    def get_neighbors(pos):
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        for dr, dc in directions:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < height and 0 <= nc < width:
                dist = math.hypot(dr, dc)
                mult = road_cost if raster[nr, nc] == 1 else offroad_cost
                cost_dist = dist * mult
                cost_elev = abs(float(elevation[nr, nc]) - float(elevation[pos[0], pos[1]]))
                cost_slope = float(slope[nr, nc])
                total = w_dist * cost_dist + w_elev * cost_elev + w_slope * cost_slope
                yield (nr, nc), total, (cost_dist, cost_elev, cost_slope)
    
    heap = [(heuristic(source), 0.0, (0.0, 0.0, 0.0), source)]
    g_scores = {source: 0.0}
    came_from = {}
    expansions = 0
    
    while heap and expansions < max_expansions:
        f, g, g_vec, current = heapq.heappop(heap)
        
        if current == target:
            path = [current]
            pos = current
            while pos in came_from:
                pos = came_from[pos]
                path.append(pos)
            return path[::-1], g_vec, expansions
        
        if g > g_scores.get(current, float('inf')):
            continue
        
        expansions += 1
        
        for neighbor, cost, cost_vec in get_neighbors(current):
            new_g = g + cost
            new_g_vec = tuple(a + b for a, b in zip(g_vec, cost_vec))
            
            if new_g < g_scores.get(neighbor, float('inf')):
                g_scores[neighbor] = new_g
                came_from[neighbor] = current
                heapq.heappush(heap, (new_g + heuristic(neighbor), new_g, new_g_vec, neighbor))
    
    return None, (float('inf'), float('inf'), float('inf')), expansions


# =============================================================================
# EMO + A* (Weight Evolution)
# =============================================================================

def emo_astar(raster: np.ndarray,
              elevation: np.ndarray,
              slope: np.ndarray,
              source: Tuple[int, int],
              target: Tuple[int, int],
              n_samples: int = 20,
              max_expansions_per_run: int = 250000) -> ComparisonResult:
    """
    EMO + A*: Run A* with different weight configurations.
    Simple grid sampling of weight space instead of full NSGA-II for speed.
    """
    start_time = time.time()
    
    # Generate diverse weight configurations
    weight_configs = []
    
    # Dominant-objective configs (avoid pure 0 weights which cause degenerate search)
    weight_configs.append((1.0, 0.1, 0.1))  # Distance-dominant
    weight_configs.append((0.1, 1.0, 0.1))  # Elevation-dominant
    weight_configs.append((0.1, 0.1, 1.0))  # Slope-dominant
    
    # Balanced
    weight_configs.append((1.0, 0.5, 0.5))
    weight_configs.append((0.5, 1.0, 0.5))
    weight_configs.append((0.5, 0.5, 1.0))
    
    # Add random samples
    np.random.seed(42)
    while len(weight_configs) < n_samples:
        w = np.random.dirichlet([1, 1, 1])
        weight_configs.append(tuple(w))
    
    weight_configs = weight_configs[:n_samples]
    
    # Run A* for each weight configuration
    all_results = []  # (objectives, path) tuples
    total_expansions = 0
    
    for i, weights in enumerate(weight_configs):
        path, objectives, expansions = scalarized_astar(
            raster, elevation, slope, source, target, weights,
            max_expansions=max_expansions_per_run
        )
        total_expansions += expansions
        
        if path is not None:
            all_results.append((objectives, path))
            print(f"  EMO run {i+1}/{n_samples}: weights={weights}, obj={objectives}")
    
    # Filter to non-dominated solutions
    pareto_front = []
    pareto_paths = []
    results_objs = [r[0] for r in all_results]
    for obj, path in all_results:
        dominated = False
        for other in results_objs:
            if other != obj:
                # Check if other dominates obj
                if all(o <= ob for o, ob in zip(other, obj)) and any(o < ob for o, ob in zip(other, obj)):
                    dominated = True
                    break
        if not dominated and obj not in pareto_front:
            pareto_front.append(obj)
            pareto_paths.append(path)
    
    elapsed = time.time() - start_time
    
    return ComparisonResult(
        algorithm="EMO+A* (Weight Sampling)",
        runtime_seconds=elapsed,
        nodes_expanded=total_expansions,
        solutions_found=len(pareto_front),
        pareto_front=pareto_front
    ), pareto_paths

# =============================================================================
# Visualization
# =============================================================================

def visualize_pareto_front(apex_solutions, emo_front, output_dir: str):
    """
    Visualize the Pareto front as pairwise objective scatter plots + 3D view.
    
    Args:
        apex_solutions: list of ParetoSolution from A*pex
        emo_front: list of objective tuples from EMO
        output_dir: where to save images
    """
    obj_names = ["Distance", "Elevation", "Slope"]
    
    # Extract objective vectors
    apex_objs = [tuple(float(v) for v in sol.objectives) for sol in apex_solutions]
    emo_objs = [tuple(float(v) for v in obj) for obj in emo_front]
    
    # --- 2x2 figure: 3 pairwise plots + 3D ---
    fig = plt.figure(figsize=(16, 14))
    
    pairs = [(0, 1), (0, 2), (1, 2)]  # objective index pairs
    
    for idx, (i, j) in enumerate(pairs):
        ax = fig.add_subplot(2, 2, idx + 1)
        
        # EMO points
        if emo_objs:
            ex = [o[i] for o in emo_objs]
            ey = [o[j] for o in emo_objs]
            ax.scatter(ex, ey, c='#FFA500', s=80, marker='D', edgecolors='black',
                      linewidths=0.8, label=f'EMO ({len(emo_objs)})', alpha=0.8, zorder=5)
        
        # A*pex points
        if apex_objs:
            ax_vals = [o[i] for o in apex_objs]
            ay_vals = [o[j] for o in apex_objs]
            ax.scatter(ax_vals, ay_vals, c='#FF4444', s=120, marker='*', edgecolors='black',
                      linewidths=0.8, label=f'A*pex ({len(apex_objs)})', alpha=0.9, zorder=10)
            # Label each A*pex point
            for k, (x, y) in enumerate(zip(ax_vals, ay_vals)):
                ax.annotate(f'#{k+1}', (x, y), textcoords="offset points",
                           xytext=(8, 8), fontsize=8, fontweight='bold', color='#CC0000')
        
        ax.set_xlabel(obj_names[i], fontsize=11)
        ax.set_ylabel(obj_names[j], fontsize=11)
        ax.set_title(f'{obj_names[i]} vs {obj_names[j]}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # 3D scatter plot
    ax3d = fig.add_subplot(2, 2, 4, projection='3d')
    
    if emo_objs:
        ax3d.scatter([o[0] for o in emo_objs], [o[1] for o in emo_objs], 
                    [o[2] for o in emo_objs],
                    c='#FFA500', s=80, marker='D', edgecolors='black',
                    linewidths=0.5, label=f'EMO ({len(emo_objs)})', alpha=0.7)
    
    if apex_objs:
        ax3d.scatter([o[0] for o in apex_objs], [o[1] for o in apex_objs],
                    [o[2] for o in apex_objs],
                    c='#FF4444', s=150, marker='*', edgecolors='black',
                    linewidths=0.5, label=f'A*pex ({len(apex_objs)})', alpha=0.9)
        for k, o in enumerate(apex_objs):
            ax3d.text(o[0], o[1], o[2], f'  #{k+1}', fontsize=7, fontweight='bold', color='#CC0000')
    
    ax3d.set_xlabel('Distance', fontsize=9)
    ax3d.set_ylabel('Elevation', fontsize=9)
    ax3d.set_zlabel('Slope', fontsize=9)
    ax3d.set_title('3D Pareto Front', fontsize=13, fontweight='bold')
    ax3d.legend(fontsize=9)
    
    fig.suptitle('Pareto Front: A*pex vs EMO+A*', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    pareto_img = os.path.join(output_dir, "pareto_front.png")
    fig.savefig(pareto_img, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Pareto front saved to {pareto_img}")
    
    # --- A*pex-only Pareto front ---
    if apex_objs:
        fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        apex_colors = ['#FF4444', '#44FF44', '#4444FF', '#FFAA00', '#FF44FF', '#44FFFF',
                        '#FF8800', '#8844FF', '#44FF88', '#FF0088', '#0088FF', '#88FF00']
        
        for idx, (i, j) in enumerate(pairs):
            ax = axes[idx]
            for k, o in enumerate(apex_objs):
                c = apex_colors[k % len(apex_colors)]
                ax.scatter(o[i], o[j], c=c, s=150, marker='o', edgecolors='black',
                          linewidths=1, zorder=10)
                ax.annotate(f'#{k+1}', (o[i], o[j]), textcoords="offset points",
                           xytext=(10, 10), fontsize=9, fontweight='bold', color=c)
            
            # Connect points to show front shape
            sorted_pts = sorted(apex_objs, key=lambda o: o[i])
            ax.plot([o[i] for o in sorted_pts], [o[j] for o in sorted_pts],
                   '--', color='gray', alpha=0.4, linewidth=1)
            
            ax.set_xlabel(obj_names[i], fontsize=11)
            ax.set_ylabel(obj_names[j], fontsize=11)
            ax.set_title(f'{obj_names[i]} vs {obj_names[j]}', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        fig2.suptitle('A*pex Pareto Front (ε=0.1)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        apex_pareto_img = os.path.join(output_dir, "pareto_front_apex_only.png")
        fig2.savefig(apex_pareto_img, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f"  ✓ A*pex-only Pareto front saved to {apex_pareto_img}")

def visualize_paths(raster: np.ndarray,
                    paths: List[Tuple[List, str, str, float]],
                    source: Tuple[int, int],
                    target: Tuple[int, int],
                    output_dir: str,
                    title: str = "A*pex vs EMO+A* Route Comparison",
                    filename_prefix: str = "route_comparison"):
    """
    Plot algorithm paths on the raster grid.
    
    Args:
        raster: 2D road/obstacle grid
        paths: list of (path_coords, label, color, linewidth)
        source: start position (row, col)
        target: goal position (row, col)
        output_dir: where to save the image
        title: plot title
        filename_prefix: prefix for output filenames
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Draw raster background
    # 0 = obstacle (dark), 1 = road (light)
    cmap = mcolors.ListedColormap(['#1a1a2e', '#e0e0e0'])
    ax.imshow(raster, cmap=cmap, interpolation='nearest', alpha=0.6)
    
    # Draw each path
    for path_coords, label, color, lw in paths:
        rows = [p[0] for p in path_coords]
        cols = [p[1] for p in path_coords]
        ax.plot(cols, rows, color=color, linewidth=lw, label=label, alpha=0.85)
    
    # Mark source and target
    ax.plot(source[1], source[0], 'o', color='lime', markersize=14, 
            markeredgecolor='white', markeredgewidth=2, label='Source', zorder=10)
    ax.plot(target[1], target[0], '*', color='red', markersize=18,
            markeredgecolor='white', markeredgewidth=2, label='Target', zorder=10)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("Column (x)")
    ax.set_ylabel("Row (y)")
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    plt.tight_layout()
    path_img = os.path.join(output_dir, f"{filename_prefix}.png")
    fig.savefig(path_img, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Route comparison saved to {path_img}")
    
    # Also save a zoomed version around the paths
    if paths:
        all_rows = []
        all_cols = []
        for path_coords, _, _, _ in paths:
            all_rows.extend([p[0] for p in path_coords])
            all_cols.extend([p[1] for p in path_coords])
        
        if all_rows:
            margin = 20
            r_min, r_max = max(0, min(all_rows) - margin), min(raster.shape[0], max(all_rows) + margin)
            c_min, c_max = max(0, min(all_cols) - margin), min(raster.shape[1], max(all_cols) + margin)
            
            fig2, ax2 = plt.subplots(1, 1, figsize=(16, 10))
            ax2.imshow(raster[r_min:r_max, c_min:c_max], cmap=cmap, 
                      interpolation='nearest', alpha=0.6, 
                      extent=[c_min, c_max, r_max, r_min])
            
            for path_coords, label, color, lw in paths:
                rows = [p[0] for p in path_coords]
                cols = [p[1] for p in path_coords]
                ax2.plot(cols, rows, color=color, linewidth=lw+0.5, label=label, alpha=0.9)
            
            ax2.plot(source[1], source[0], 'o', color='lime', markersize=14,
                    markeredgecolor='white', markeredgewidth=2, label='Source', zorder=10)
            ax2.plot(target[1], target[0], '*', color='red', markersize=18,
                    markeredgecolor='white', markeredgewidth=2, label='Target', zorder=10)
            
            ax2.set_title(f"{title} (Zoomed)", fontsize=16, fontweight='bold')
            ax2.set_xlabel("Column (x)")
            ax2.set_ylabel("Row (y)")
            ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)
            
            plt.tight_layout()
            zoom_img = os.path.join(output_dir, f"{filename_prefix}_zoomed.png")
            fig2.savefig(zoom_img, dpi=150, bbox_inches='tight')
            plt.close(fig2)
            print(f"  ✓ Zoomed route comparison saved to {zoom_img}")


def run_comparison(npz_path: str, output_dir: str) -> Dict:
    """
    Run all algorithms on the same test case.
    """
    print(f"\n{'='*70}")
    print("A*pex Algorithm Comparison")
    print(f"{'='*70}")
    print(f"Test set: {npz_path}")
    print(f"{'='*70}\n")
    
    # Load data
    data = np.load(npz_path, allow_pickle=True)
    raster = data['raster']
    goal_points = data['goal_points']
    
    if len(goal_points) < 2:
        print("Error: Need at least 2 goal points")
        return {}
    
    source = (int(goal_points[0][1]), int(goal_points[0][0]))
    target = (int(goal_points[1][1]), int(goal_points[1][0]))
    
    print(f"Raster: {raster.shape}")
    print(f"Source: {source}")
    print(f"Target: {target}")
    
    # Extract bounding box for real elevation data
    bbox = tuple(data['bbox']) if 'bbox' in data else None
    
    # Load or Generate Elevation
    elevation = None
    if bbox is not None and HAS_ELEVATION_LIB:
        print(f"Attempting to fetch real elevation data for bbox {bbox}...")
        try:
            # Assume 30m resolution for consistency
            elevation = extract_elevation(bbox, 30.0)
            print(f"Successfully loaded real elevation data. Range: {elevation.min():.1f}m - {elevation.max():.1f}m")
        except Exception as e:
            print(f"Failed to fetch real elevation: {e}")
            print("Falling back to synthetic elevation.")
    
    if elevation is None:
        print("Generating synthetic elevation.")
        np.random.seed(42)
        # Use simple Perlin-like noise or uniform random (keeping uniform for consistency with old baseline unless updated)
        # Actually, let's use the better synthetic generator from RasterApexSearch if possible, but it's internal.
        # For fair comparison, we'll stick to uniform random for now as baseline, or copy logic.
        # Let's use uniform random as before to match strict comparison logic
        elevation = np.random.uniform(0, 500, raster.shape).astype(np.float32)
        
    # Resize elevation if needed (simple crop/pad)
    if elevation.shape != raster.shape:
        print(f"Warning: Elevation shape {elevation.shape} != Raster shape {raster.shape}")
        h, w = raster.shape
        eh, ew = elevation.shape
        # Simple crop
        elevation = elevation[:min(h, eh), :min(w, ew)]
        # Pad if needed
        if elevation.shape != (h, w):
             padded = np.zeros((h, w), dtype=np.float32)
             padded[:elevation.shape[0], :elevation.shape[1]] = elevation
             elevation = padded
    
    # Compute slope from elevation (same as extract_slope)
    resolution = 30.0
    grad_y, grad_x = np.gradient(elevation, resolution)
    slope = np.clip(np.sqrt(grad_x**2 + grad_y**2), 0, 1.0).astype(np.float32)
    print(f"Slope: mean={np.mean(slope)*100:.2f}%, max={np.max(slope)*100:.2f}%")

    results = {}
    
    # 1. Pure A*pex (Multi-Objective)
    print(f"\n{'='*50}")
    print("Running: Pure A*pex (Multi-Objective)")
    print(f"{'='*50}")
    
    apex_searcher = RasterApexSearch(
        raster=raster,
        elevation_array=elevation,
        resolution=30.0,
        eps=(0.1, 0.1, 0.1), # Finer epsilon for diverse Pareto front
        max_expansions=10_000_000
    )
    
    pure_solutions, pure_stats = apex_searcher.search(source, target)
    
    # Convert pure solutions to list of objective vectors
    pure_front = [tuple(sol.objectives) for sol in pure_solutions]
    
    results["pure_apex"] = ComparisonResult(
        algorithm="Pure A*pex",
        runtime_seconds=pure_stats.runtime_seconds,
        nodes_expanded=pure_stats.nodes_expanded,
        solutions_found=len(pure_solutions),
        pareto_front=pure_front
    )
    
    # 2. Single-objective baseline
    print(f"\n{'='*50}")
    print("Running: Single-Objective A*")
    print(f"{'='*50}")
    result_single = single_objective_astar(raster, source, target)
    print(f"  Runtime: {result_single.runtime_seconds:.2f}s")
    print(f"  Expansions: {result_single.nodes_expanded:,}")
    print(f"  Solutions: {result_single.solutions_found}")
    results["single_objective"] = result_single
    
    # 3. EMO + A*
    print(f"\n{'='*50}")
    print("Running: EMO+A* (Weight Sampling)")
    print(f"{'='*50}")
    result_emo, emo_paths = emo_astar(raster, elevation, slope, source, target, n_samples=10, max_expansions_per_run=1_000_000)
    print(f"  Runtime: {result_emo.runtime_seconds:.2f}s")
    print(f"  Total Expansions: {result_emo.nodes_expanded:,}")
    print(f"  Pareto Solutions: {result_emo.solutions_found}")
    results["emo_astar"] = result_emo
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output = {
        "test_set": npz_path,
        "source": source,
        "target": target,
        "raster_shape": list(raster.shape),
        "algorithms": {name: r.to_dict() for name, r in results.items()}
    }
    
    output_path = os.path.join(output_dir, "comparison_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    # =========================================================================
    # Visualize paths on raster
    # =========================================================================
    print(f"\nGenerating path visualizations...")
    
    # Collect all paths with labels and colors
    all_paths = []
    
    # A*pex paths
    apex_colors = ['#FF4444', '#44FF44', '#4444FF', '#FFAA00', '#FF44FF', '#44FFFF',
                    '#FF8800', '#8844FF', '#44FF88', '#FF0088', '#0088FF', '#88FF00']
    for i, sol in enumerate(pure_solutions):
        color = apex_colors[i % len(apex_colors)]
        obj_str = f"d={sol.objectives[0]:.0f}, e={float(sol.objectives[1]):.0f}, s={float(sol.objectives[2]):.2f}"
        all_paths.append((sol.path, f"A*pex #{i+1} ({obj_str})", color, 2.5))
    
    # EMO paths
    emo_colors = ['#FFD700', '#FFA500', '#FF6347', '#DA70D6', '#00CED1', '#32CD32']
    for i, path in enumerate(emo_paths):
        obj = result_emo.pareto_front[i] if i < len(result_emo.pareto_front) else None
        label = f"EMO #{i+1}"
        if obj:
            label += f" (d={obj[0]:.0f}, e={float(obj[1]):.0f}, s={float(obj[2]):.2f})"
        color = emo_colors[i % len(emo_colors)]
        all_paths.append((path, label, color, 1.5))
    
    if all_paths:
        visualize_paths(raster, all_paths, source, target, output_dir)
    else:
        print("  No paths to visualize.")
    
    # A*pex-only visualization
    apex_only_paths = []
    for i, sol in enumerate(pure_solutions):
        color = apex_colors[i % len(apex_colors)]
        obj_str = f"d={sol.objectives[0]:.0f}, e={float(sol.objectives[1]):.0f}, s={float(sol.objectives[2]):.2f}"
        apex_only_paths.append((sol.path, f"A*pex #{i+1} ({obj_str})", color, 2.5))
    if apex_only_paths:
        visualize_paths(raster, apex_only_paths, source, target, output_dir,
                       title="A*pex Pareto Front Routes", filename_prefix="apex_only")
    
    # Pareto front objective scatter plots
    visualize_pareto_front(pure_solutions, result_emo.pareto_front, output_dir)
    
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    for name, r in results.items():
        print(f"{name}:")
        print(f"  Runtime: {r.runtime_seconds:.2f}s")
        print(f"  Expansions: {r.nodes_expanded:,}")
        print(f"  Solutions: {r.solutions_found}")
    print(f"{'='*70}")
    print(f"\n✓ Results saved to {output_path}")
    
    return output


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare A*pex Algorithms")
    parser.add_argument("--input", type=str, required=True, help="Path to NPZ test set")
    parser.add_argument("--output", type=str, default="./comparison_results", help="Output directory")
    
    args = parser.parse_args()
    
    run_comparison(args.input, args.output)


if __name__ == "__main__":
    main()
