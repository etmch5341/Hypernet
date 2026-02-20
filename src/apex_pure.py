#!/usr/bin/env python3
"""
Pure A*pex Multi-Objective Search for Raster Grids

Implements true multi-objective A* with ε-dominance and Pareto frontiers.
Finds routes that optimize 3 objectives simultaneously:
  1. Distance: Euclidean path length
  2. Elevation: Cumulative elevation change
  3. Slope: Gradient steepness

Based on A*pex algorithm concepts but adapted for raster/grid search.
"""

import heapq
import math
import numpy as np
import time
import json
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set, Any
from pathlib import Path

# Add src to path to allow imports
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from data_pipeline.system.feature_extraction import extract_elevation
    HAS_ELEVATION_LIB = True
except ImportError:
    # If standard import fails, try relative import if run as module
    try:
        from .data_pipeline.system.feature_extraction import extract_elevation
        HAS_ELEVATION_LIB = True
    except ImportError:
        print("Warning: Could not import extract_elevation. Real elevation data will not be available.")
        HAS_ELEVATION_LIB = False


# =============================================================================
# Type Aliases
# =============================================================================
CostVec = Tuple[float, ...]  # (distance, elevation, slope)
Position = Tuple[int, int]   # (row, col)


# =============================================================================
# Vector Operations for Multi-Objective Costs
# =============================================================================

def v_add(a: CostVec, b: CostVec) -> CostVec:
    """Component-wise addition."""
    return tuple(x + y for x, y in zip(a, b))


def v_leq(a: CostVec, b: CostVec) -> bool:
    """Component-wise <=."""
    return all(x <= y for x, y in zip(a, b))


def v_eps_dominates(a: CostVec, b: CostVec, eps: CostVec) -> bool:
    """
    ε-dominance: a ε-dominates b iff a_i <= (1+eps_i) * b_i for all i.
    Returns True if 'a' is at least as good as 'b' within epsilon tolerance.
    """
    return all(a_i <= (1.0 + e_i) * b_i for a_i, b_i, e_i in zip(a, b, eps))


def v_strictly_dominates(a: CostVec, b: CostVec) -> bool:
    """True if a strictly dominates b (a <= b in all objectives, < in at least one)."""
    all_leq = all(a_i <= b_i for a_i, b_i in zip(a, b))
    some_less = any(a_i < b_i for a_i, b_i in zip(a, b))
    return all_leq and some_less


def v_scalar_sum(a: CostVec) -> float:
    """Simple scalarization for tie-breaking in priority queue."""
    return sum(a)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Label:
    """A label in A*pex: one candidate cost vector at a position."""
    state: Position
    g: CostVec              # Cost-to-come vector
    h: CostVec              # Heuristic vector
    f: CostVec              # f = g + h
    parent: Optional['Label'] = None
    prev_direction: Optional[Tuple[int, int]] = None  # For path reconstruction only


@dataclass
class ParetoSolution:
    """A complete Pareto-optimal solution."""
    path: List[Position]
    objectives: CostVec
    objective_names: Tuple[str, ...] = ("distance", "elevation", "slope")
    
    def to_dict(self) -> dict:
        return {
            "path_length": len(self.path),
            "objectives": {name: float(val) for name, val in zip(self.objective_names, self.objectives)},
            "path": [(int(r), int(c)) for r, c in self.path]
        }


@dataclass
class SearchStats:
    """Statistics from a search run."""
    nodes_expanded: int = 0
    labels_generated: int = 0
    labels_pruned: int = 0
    pareto_size: int = 0
    runtime_seconds: float = 0.0
    
    # Per-iteration tracking for visualization
    iteration_log: List[dict] = field(default_factory=list)
    
    def log_iteration(self, iteration: int, expanded: int, pareto_size: int, 
                      queue_size: int, elapsed: float):
        self.iteration_log.append({
            "iteration": iteration,
            "nodes_expanded": expanded,
            "pareto_size": pareto_size,
            "queue_size": queue_size,
            "elapsed_seconds": elapsed
        })
    
    def to_dict(self) -> dict:
        return {
            "nodes_expanded": self.nodes_expanded,
            "labels_generated": self.labels_generated,
            "labels_pruned": self.labels_pruned,
            "pareto_size": self.pareto_size,
            "runtime_seconds": self.runtime_seconds,
            "iterations": self.iteration_log
        }


# =============================================================================
# Core A*pex Search Class
# =============================================================================

class RasterApexSearch:
    """
    Pure A*pex multi-objective search on raster grids.
    
    Objectives:
        1. Distance: Euclidean path length
        2. Elevation: Cumulative elevation difference
        3. Slope: Gradient steepness
    """
    
    OBJECTIVES = ("distance", "elevation", "slope")
    NUM_OBJECTIVES = 3
    
    # 8-connectivity directions
    DIRECTIONS = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),          (0, 1),
        (1, -1),  (1, 0), (1, 1)
    ]
    
    def __init__(
        self,
        raster: np.ndarray,
        resolution: float = 30.0,
        elevation_array: Optional[np.ndarray] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        eps: Tuple[float, ...] = (0.01, 0.01, 0.01),
        road_cost: float = 1.0,
        offroad_cost: float = 5.0,
        allow_diagonal: bool = True,
        log_interval: int = 10000,
        max_expansions: int = 1_000_000
    ):
        """
        Initialize the search.
        
        Args:
            raster: 2D array where 1=road, 0=off-road
            resolution: Meters per pixel
            elevation_array: Optional 2D array of elevation values (meters)
            bbox: Optional bounding box (min_lon, min_lat, max_lon, max_lat) for real data fetching
            eps: Epsilon values for ε-dominance per objective
            road_cost: Cost multiplier for on-road movement
            offroad_cost: Cost multiplier for off-road movement
            allow_diagonal: Whether to allow 8-connectivity
            log_interval: How often to log progress
            max_expansions: Safety limit on node expansions
        """
        self.raster = raster
        self.height, self.width = raster.shape
        self.resolution = resolution
        self.eps = eps
        self.road_cost = road_cost
        self.offroad_cost = offroad_cost
        self.allow_diagonal = allow_diagonal
        self.log_interval = log_interval
        self.max_expansions = max_expansions
        
        # Objectives: [Distance, Elevation_Change, Slope]
        self.objective_names = ["distance", "elevation", "slope"]
        self.num_objectives = len(self.objective_names)
        
        # Load or generate elevation data
        if elevation_array is not None:
            print("Using provided elevation data.")
            self.elevation = elevation_array
        elif bbox is not None and HAS_ELEVATION_LIB:
            print(f"Attempting to fetch real elevation data for bbox {bbox}...")
            try:
                self.elevation = extract_elevation(bbox, resolution)
                print(f"Successfully loaded real elevation data. Range: {self.elevation.min():.1f}m - {self.elevation.max():.1f}m")
            except Exception as e:
                print(f"Failed to fetch real elevation: {e}")
                print("Falling back to synthetic elevation.")
                self.elevation = self._generate_synthetic_elevation()
        else:
            print("No elevation data or bbox provided. Generating synthetic elevation.")
            self.elevation = self._generate_synthetic_elevation()
            
        # Ensure elevation matches raster shape
        if self.elevation.shape != self.raster.shape:
             # Resize to match raster if needed (simple crop or pad)
             print(f"Warning: Elevation shape {self.elevation.shape} != Raster shape {self.raster.shape}")
             # Detailed resize logic omitted for brevity, assuming correct mostly
             # Fallback to synthetic if size mismatch is severe
             if self.elevation.shape[0] < self.height or self.elevation.shape[1] < self.width:
                 print("Elevation too small, regenerating synthetic.")
                 self.elevation = self._generate_synthetic_elevation()
             else:
                 self.elevation = self.elevation[:self.height, :self.width]

        # Precompute gradients and slope array
        self.grad_y, self.grad_x = np.gradient(self.elevation, resolution)
        self.slope = np.clip(np.sqrt(self.grad_x**2 + self.grad_y**2), 0, 1.0).astype(np.float32)
        print(f"Slope: mean={np.mean(self.slope)*100:.2f}%, max={np.max(self.slope)*100:.2f}%")
        
        # Search state (reset per search)
        self.stats = SearchStats()
        
    def _generate_synthetic_elevation(self) -> np.ndarray:
        """Generate Perlin-noise-like elevation for testing."""
        # Simple multi-octave noise approximation
        np.random.seed(42)  # Reproducible
        
        elevation = np.zeros((self.height, self.width), dtype=np.float32)
        
        for octave in range(4):
            scale = 2 ** octave
            freq = max(1, self.width // (32 * scale))
            
            # Low-res noise
            small = np.random.uniform(0, 500 / scale, 
                                       (max(1, self.height // freq), 
                                        max(1, self.width // freq)))
            
            # Upscale with interpolation
            from scipy.ndimage import zoom
            factor_h = self.height / small.shape[0]
            factor_w = self.width / small.shape[1]
            upscaled = zoom(small, (factor_h, factor_w), order=1)
            
            # Crop to exact size
            upscaled = upscaled[:self.height, :self.width]
            elevation += upscaled
        
        return elevation
    
    def _heuristic(self, pos: Position, goal: Position) -> CostVec:
        """
        Admissible heuristic for each objective.
        
        Returns (h_distance, h_elevation, h_slope)
        """
        # Distance: Euclidean (admissible)
        h_dist = math.hypot(goal[0] - pos[0], goal[1] - pos[1])
        
        # Elevation: Optimistic lower bound = direct elevation difference
        h_elev = abs(self.elevation[goal] - self.elevation[pos])
        
        # Slope: 0 is admissible (minimum possible slope cost)
        h_slope = 0.0
        
        return (h_dist, h_elev, h_slope)
    
    def _get_successors(self, pos: Position, prev_dir: Optional[Tuple[int, int]]
                        ) -> List[Tuple[Position, CostVec, Tuple[int, int]]]:
        """
        Generate valid successors with their cost vectors.
        
        Returns list of (neighbor_pos, edge_cost_vec, direction)
        """
        directions = self.DIRECTIONS if self.allow_diagonal else [
            (-1, 0), (1, 0), (0, -1), (0, 1)
        ]
        
        successors = []
        for dr, dc in directions:
            nr, nc = pos[0] + dr, pos[1] + dc
            
            # Bounds check
            if not (0 <= nr < self.height and 0 <= nc < self.width):
                continue
            
            neighbor = (nr, nc)
            direction = (dr, dc)
            
            # Objective 1: Distance
            dist = math.hypot(dr, dc)
            terrain_mult = self.road_cost if self.raster[nr, nc] == 1 else self.offroad_cost
            cost_dist = dist * terrain_mult
            
            # Objective 2: Elevation change
            cost_elev = abs(self.elevation[neighbor] - self.elevation[pos])
            
            # Objective 3: Slope at destination cell
            cost_slope = float(self.slope[nr, nc])
            
            edge_cost = (cost_dist, cost_elev, cost_slope)
            successors.append((neighbor, edge_cost, direction))
        
        return successors
    
    def _reconstruct_path(self, label: Label) -> List[Position]:
        """Reconstruct path from goal label back to start."""
        path = []
        current = label
        while current is not None:
            path.append(current.state)
            current = current.parent
        return path[::-1]
    
    def search(self, source: Position, target: Position) -> Tuple[List[ParetoSolution], SearchStats]:
        """
        Run multi-objective A*pex search.
        
        Args:
            source: Start position (row, col)
            target: Goal position (row, col)
        
        Returns:
            (list of ParetoSolution, SearchStats)
        """
        start_time = time.time()
        self.stats = SearchStats()
        
        print(f"\n{'='*70}")
        print(f"A*pex Multi-Objective Search")
        print(f"{'='*70}")
        print(f"Source: {source}")
        print(f"Target: {target}")
        print(f"Epsilon: {self.eps}")
        print(f"{'='*70}\n")
        
        # Priority queue: (priority, counter, Label)
        # Priority is (f1, f2, f3, scalar_sum(f)) for lexicographic ordering
        heap: List[Tuple[Tuple[float, ...], int, Label]] = []
        counter = 0
        
        # Per-state label sets: state -> list of non-dominated g-vectors
        labels: Dict[Position, List[CostVec]] = {}
        
        # Solution set (non-ε-dominated Pareto front)
        solutions: List[Tuple[Label, CostVec]] = []
        
        # Initialize start
        h0 = self._heuristic(source, target)
        g0 = (0.0, 0.0, 0.0)
        f0 = v_add(g0, h0)
        start_label = Label(state=source, g=g0, h=h0, f=f0, parent=None, prev_direction=None)
        
        labels[source] = [g0]
        prio = tuple(list(f0) + [v_scalar_sum(f0)])
        heapq.heappush(heap, (prio, counter, start_label))
        counter += 1
        self.stats.labels_generated += 1
        
        expansions = 0
        
        while heap:
            _, _, label = heapq.heappop(heap)
            pos = label.state
            
            # Check if ε-dominated by any solution
            if any(v_eps_dominates(sol_cost, label.f, self.eps) for _, sol_cost in solutions):
                self.stats.labels_pruned += 1
                continue
            
            expansions += 1
            self.stats.nodes_expanded = expansions
            
            # Logging
            if expansions % self.log_interval == 0:
                elapsed = time.time() - start_time
                self.stats.log_iteration(expansions, expansions, len(solutions), 
                                          len(heap), elapsed)
                print(f"  Expanded: {expansions:,} | Solutions: {len(solutions)} | "
                      f"Queue: {len(heap):,} | Time: {elapsed:.1f}s")
            
            # Safety limit
            if expansions >= self.max_expansions:
                print(f"\n⚠ Reached max expansions ({self.max_expansions:,})")
                break
            
            # Goal test
            if pos == target:
                g_n = label.g
                
                # Check if dominated by existing solution
                dominated = any(v_eps_dominates(sol_cost, g_n, self.eps) 
                               for _, sol_cost in solutions)
                
                if not dominated:
                    # Remove solutions dominated by this one
                    new_solutions = [(sol, cost) for sol, cost in solutions 
                                     if not v_eps_dominates(g_n, cost, self.eps)]
                    new_solutions.append((label, g_n))
                    solutions = new_solutions
                    print(f"  ✓ Found solution #{len(solutions)}: {g_n}")
                
                continue
            
            # Expand successors
            for neighbor, edge_cost, direction in self._get_successors(pos, label.prev_direction):
                g_succ = v_add(label.g, edge_cost)
                h_succ = self._heuristic(neighbor, target)
                f_succ = v_add(g_succ, h_succ)
                
                # Prune if ε-dominated by a solution
                if any(v_eps_dominates(sol_cost, f_succ, self.eps) for _, sol_cost in solutions):
                    self.stats.labels_pruned += 1
                    continue
                
                # Label pruning for this state
                existing = labels.get(neighbor, [])
                skip = False
                
                for g_old in existing:
                    if v_eps_dominates(g_old, g_succ, self.eps):
                        skip = True
                        break
                
                if skip:
                    self.stats.labels_pruned += 1
                    continue
                
                # Remove labels dominated by the new one
                new_labels = [g_old for g_old in existing 
                              if not v_eps_dominates(g_succ, g_old, self.eps)]
                new_labels.append(g_succ)
                labels[neighbor] = new_labels
                
                # Create and push new label
                child = Label(state=neighbor, g=g_succ, h=h_succ, f=f_succ,
                              parent=label, prev_direction=direction)
                prio = tuple(list(f_succ) + [v_scalar_sum(f_succ)])
                heapq.heappush(heap, (prio, counter, child))
                counter += 1
                self.stats.labels_generated += 1
        
        # Finalize
        elapsed = time.time() - start_time
        self.stats.runtime_seconds = elapsed
        self.stats.pareto_size = len(solutions)
        
        # Convert to ParetoSolution objects
        result: List[ParetoSolution] = []
        for label, cost in solutions:
            path = self._reconstruct_path(label)
            result.append(ParetoSolution(path=path, objectives=cost))
        
        # Sort by first objective (distance)
        result.sort(key=lambda s: s.objectives[0])
        
        print(f"\n{'='*70}")
        print(f"Search Complete")
        print(f"{'='*70}")
        print(f"  Pareto solutions: {len(result)}")
        print(f"  Nodes expanded: {expansions:,}")
        print(f"  Labels generated: {self.stats.labels_generated:,}")
        print(f"  Labels pruned: {self.stats.labels_pruned:,}")
        print(f"  Runtime: {elapsed:.2f}s")
        print(f"{'='*70}\n")
        
        return result, self.stats
    
    def save_results(self, solutions: List[ParetoSolution], stats: SearchStats,
                     output_dir: str):
        """Save search results to JSON."""
        os.makedirs(output_dir, exist_ok=True)
        
        output = {
            "algorithm": "A*pex Pure Multi-Objective",
            "objectives": list(self.OBJECTIVES),
            "epsilon": list(self.eps),
            "statistics": stats.to_dict(),
            "solutions": [s.to_dict() for s in solutions]
        }
        
        output_path = os.path.join(output_dir, "apex_results.json")
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"✓ Results saved to {output_path}")
        return output_path


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Test the A*pex search on sample data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pure A*pex Multi-Objective Search")
    parser.add_argument("--input", type=str, required=True, help="Path to NPZ test set")
    parser.add_argument("--output", type=str, default="./apex_results", help="Output directory")
    parser.add_argument("--eps", type=float, nargs=3, default=[0.01, 0.01, 0.01],
                        help="Epsilon values for each objective")
    parser.add_argument("--max-expansions", type=int, default=500_000,
                        help="Maximum node expansions")
    
    args = parser.parse_args()
    
    # Load test data
    print(f"Loading test set: {args.input}")
    data = np.load(args.input, allow_pickle=True)
    raster = data['raster']
    goal_points = data['goal_points']
    
    print(f"  Raster shape: {raster.shape}")
    print(f"  Goal points: {len(goal_points)}")
    
    # Use first two goals as source/target
    if len(goal_points) < 2:
        print("Error: Need at least 2 goal points")
        return 1
    
    source = (int(goal_points[0][1]), int(goal_points[0][0]))  # (row, col)
    target = (int(goal_points[1][1]), int(goal_points[1][0]))
    
    print(f"  Source: {goal_points[0][2]} at {source}")
    print(f"  Target: {goal_points[1][2]} at {target}")
    
    # Extract bounding box for real elevation data
    bbox = tuple(data['bbox']) if 'bbox' in data else None
    if bbox:
        print(f"  BBox: {bbox}")
    
    # Create searcher and run
    searcher = RasterApexSearch(
        raster=raster,
        bbox=bbox,
        eps=tuple(args.eps),
        max_expansions=args.max_expansions
    )
    
    solutions, stats = searcher.search(source, target)
    
    # Save results
    searcher.save_results(solutions, stats, args.output)
    
    # Print solution summary
    print("\nPareto Front:")
    for i, sol in enumerate(solutions):
        d, e, s = sol.objectives
        print(f"  {i+1}. Distance: {d:.1f}, Elevation: {e:.1f}, Slope: {s:.4f}")
    
    return 0


if __name__ == "__main__":
    exit(main())
