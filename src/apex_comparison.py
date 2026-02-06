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
import time
import json
import os
import heapq
import math
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from dataclasses import dataclass, field


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
            "pareto_front": [list(p) for p in self.pareto_front],
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
                      source: Tuple[int, int],
                      target: Tuple[int, int],
                      weights: Tuple[float, float, float],
                      road_cost: float = 1.0,
                      offroad_cost: float = 5.0,
                      max_expansions: int = 1000000) -> Tuple[Optional[List], Tuple[float, float, float], int]:
    """
    A* with weighted scalarization of objectives.
    
    Returns: (path, objective_vector, nodes_expanded)
    """
    height, width = raster.shape
    w_dist, w_turn, w_elev = weights
    
    def heuristic(pos):
        return math.hypot(target[0] - pos[0], target[1] - pos[1]) * w_dist
    
    def get_neighbors(pos, prev_dir):
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        for dr, dc in directions:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < height and 0 <= nc < width:
                # Distance cost
                dist = math.hypot(dr, dc)
                mult = road_cost if raster[nr, nc] == 1 else offroad_cost
                cost_dist = dist * mult
                
                # Turn cost
                direction = (dr, dc)
                cost_turn = 0.0 if prev_dir is None or direction == prev_dir else 1.0
                
                # Elevation cost
                cost_elev = abs(elevation[nr, nc] - elevation[pos])
                
                # Scalarized cost
                total = w_dist * cost_dist + w_turn * cost_turn + w_elev * cost_elev
                
                yield (nr, nc), total, (cost_dist, cost_turn, cost_elev), direction
    
    # State = (position, prev_direction)
    start_state = (source, None)
    heap = [(heuristic(source), 0.0, (0.0, 0.0, 0.0), source, None)]  # (f, g_scalar, g_vec, pos, prev_dir)
    g_scores = {start_state: 0.0}
    came_from = {}
    expansions = 0
    
    while heap and expansions < max_expansions:
        f, g, g_vec, current, prev_dir = heapq.heappop(heap)
        state = (current, prev_dir)
        
        if current == target:
            # Reconstruct path
            path = [current]
            s = state
            while s in came_from:
                s = came_from[s]
                path.append(s[0])
            return path[::-1], g_vec, expansions
        
        if g > g_scores.get(state, float('inf')):
            continue
        
        expansions += 1
        
        for neighbor, cost, cost_vec, direction in get_neighbors(current, prev_dir):
            new_g = g + cost
            new_g_vec = (g_vec[0] + cost_vec[0], g_vec[1] + cost_vec[1], g_vec[2] + cost_vec[2])
            new_state = (neighbor, direction)
            
            if new_g < g_scores.get(new_state, float('inf')):
                g_scores[new_state] = new_g
                came_from[new_state] = state
                heapq.heappush(heap, (new_g + heuristic(neighbor), new_g, new_g_vec, neighbor, direction))
    
    return None, (float('inf'), float('inf'), float('inf')), expansions


# =============================================================================
# EMO + A* (Weight Evolution)
# =============================================================================

def emo_astar(raster: np.ndarray,
              elevation: np.ndarray, 
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
    
    # Pure objectives
    weight_configs.append((1.0, 0.0, 0.0))  # Pure distance
    weight_configs.append((0.0, 1.0, 0.0))  # Pure turns
    weight_configs.append((0.0, 0.0, 1.0))  # Pure elevation
    
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
    results = []
    total_expansions = 0
    
    for i, weights in enumerate(weight_configs):
        path, objectives, expansions = scalarized_astar(
            raster, elevation, source, target, weights,
            max_expansions=max_expansions_per_run
        )
        total_expansions += expansions
        
        if path is not None:
            results.append(objectives)
            print(f"  EMO run {i+1}/{n_samples}: weights={weights}, obj={objectives}")
    
    # Filter to non-dominated solutions
    pareto_front = []
    for obj in results:
        dominated = False
        for other in results:
            if other != obj:
                # Check if other dominates obj
                if all(o <= ob for o, ob in zip(other, obj)) and any(o < ob for o, ob in zip(other, obj)):
                    dominated = True
                    break
        if not dominated and obj not in pareto_front:
            pareto_front.append(obj)
    
    elapsed = time.time() - start_time
    
    return ComparisonResult(
        algorithm="EMO+A* (Weight Sampling)",
        runtime_seconds=elapsed,
        nodes_expanded=total_expansions,
        solutions_found=len(pareto_front),
        pareto_front=pareto_front
    )


# =============================================================================
# Comparison Runner
# =============================================================================

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
    
    # Generate synthetic elevation
    np.random.seed(42)
    elevation = np.random.uniform(0, 500, raster.shape).astype(np.float32)
    
    results = {}
    
    # 1. Single-objective baseline
    print(f"\n{'='*50}")
    print("Running: Single-Objective A*")
    print(f"{'='*50}")
    result_single = single_objective_astar(raster, source, target)
    print(f"  Runtime: {result_single.runtime_seconds:.2f}s")
    print(f"  Expansions: {result_single.nodes_expanded:,}")
    print(f"  Solutions: {result_single.solutions_found}")
    results["single_objective"] = result_single
    
    # 2. EMO + A*
    print(f"\n{'='*50}")
    print("Running: EMO+A* (Weight Sampling)")
    print(f"{'='*50}")
    result_emo = emo_astar(raster, elevation, source, target, n_samples=10)
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
