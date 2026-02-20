#!/usr/bin/env python3
"""
A*pex Search Animation Generator

Creates an animated GIF showing the multi-objective A*pex search expanding
across a raster grid. Each frame shows:
  - The road network (background)
  - Explored nodes colored by expansion order (cool→warm gradient)
  - Current Pareto-optimal solution paths (as they are found)
  - HUD overlay with live statistics

Usage:
    python3 src/apex_animate.py [--input path/to/test.npz] [--output path/to/output.gif]
"""

import heapq
import math
import numpy as np
import time
import os
import sys
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

try:
    from apex_pure import (
        CostVec, Position, Label, ParetoSolution,
        v_add, v_eps_dominates, v_scalar_sum
    )
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    from apex_pure import (
        CostVec, Position, Label, ParetoSolution,
        v_add, v_eps_dominates, v_scalar_sum
    )


# =============================================================================
# Animated A*pex Search
# =============================================================================

class AnimatedApexSearch:
    """
    A*pex search with frame capture for GIF generation.
    
    Captures snapshots of the search state at regular intervals to build
    an animation showing how the multi-objective search explores the grid.
    """
    
    DIRECTIONS = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    def __init__(
        self,
        raster: np.ndarray,
        resolution: float = 30.0,
        elevation_array: Optional[np.ndarray] = None,
        eps: Tuple[float, ...] = (0.1, 0.1, 0.1),
        road_cost: float = 1.0,
        offroad_cost: float = 5.0,
        max_expansions: int = 500_000,
        frame_interval: int = 200,
    ):
        self.raster = raster
        self.height, self.width = raster.shape
        self.resolution = resolution
        self.eps = eps
        self.road_cost = road_cost
        self.offroad_cost = offroad_cost
        self.max_expansions = max_expansions
        self.frame_interval = frame_interval
        
        # Elevation
        if elevation_array is not None:
            self.elevation = elevation_array
        else:
            self.elevation = self._generate_synthetic_elevation()
        
        # Ensure shapes match
        if self.elevation.shape != self.raster.shape:
            if (self.elevation.shape[0] < self.height or 
                self.elevation.shape[1] < self.width):
                self.elevation = self._generate_synthetic_elevation()
            else:
                self.elevation = self.elevation[:self.height, :self.width]
        
        # Precompute slope
        grad_y, grad_x = np.gradient(self.elevation, resolution)
        self.slope = np.clip(
            np.sqrt(grad_x**2 + grad_y**2), 0, 1.0
        ).astype(np.float32)
        
        # Frame storage
        self.frames: List[np.ndarray] = []
    
    def _generate_synthetic_elevation(self) -> np.ndarray:
        np.random.seed(42)
        elevation = np.zeros((self.height, self.width), dtype=np.float32)
        for octave in range(4):
            scale = 2 ** octave
            freq = max(1, self.width // (32 * scale))
            small = np.random.uniform(
                0, 500 / scale,
                (max(1, self.height // freq), max(1, self.width // freq))
            )
            from scipy.ndimage import zoom
            factor_h = self.height / small.shape[0]
            factor_w = self.width / small.shape[1]
            upscaled = zoom(small, (factor_h, factor_w), order=1)
            upscaled = upscaled[:self.height, :self.width]
            elevation += upscaled
        return elevation
    
    def _heuristic(self, pos: Position, goal: Position) -> CostVec:
        h_dist = math.hypot(goal[0] - pos[0], goal[1] - pos[1])
        h_elev = abs(self.elevation[goal] - self.elevation[pos])
        h_slope = 0.0
        return (h_dist, h_elev, h_slope)
    
    def _get_successors(self, pos: Position):
        successors = []
        for dr, dc in self.DIRECTIONS:
            nr, nc = pos[0] + dr, pos[1] + dc
            if not (0 <= nr < self.height and 0 <= nc < self.width):
                continue
            neighbor = (nr, nc)
            
            # Distance
            dist = math.hypot(dr, dc)
            terrain_mult = (self.road_cost if self.raster[nr, nc] == 1 
                           else self.offroad_cost)
            cost_dist = dist * terrain_mult
            
            # Elevation
            cost_elev = abs(self.elevation[neighbor] - self.elevation[pos])
            
            # Slope
            cost_slope = float(self.slope[nr, nc])
            
            edge_cost = (cost_dist, cost_elev, cost_slope)
            successors.append((neighbor, edge_cost, (dr, dc)))
        return successors
    
    def _reconstruct_path(self, label: Label) -> List[Position]:
        path = []
        current = label
        while current is not None:
            path.append(current.state)
            current = current.parent
        return path[::-1]
    
    def _render_frame(
        self,
        expansion_map: np.ndarray,
        expansion_count: int,
        solutions: List[Tuple[Label, CostVec]],
        queue_size: int,
        source: Position,
        target: Position,
        elapsed: float,
        fig, ax
    ):
        """Render one animation frame."""
        ax.clear()
        
        # Background: dark base
        bg = np.full((self.height, self.width, 4), [0.06, 0.06, 0.12, 1.0])
        
        # Roads: subtle blue-gray
        road_mask = self.raster > 0
        bg[road_mask] = [0.15, 0.20, 0.28, 1.0]
        
        ax.imshow(bg, interpolation='nearest', origin='upper')
        
        # Explored nodes: color by expansion order
        explored = expansion_map > 0
        if np.any(explored):
            max_exp = expansion_map.max()
            # Normalize to [0, 1]
            norm_map = np.zeros_like(expansion_map, dtype=float)
            norm_map[explored] = expansion_map[explored] / max_exp
            
            # Create colored overlay
            cmap = plt.cm.viridis
            overlay = np.zeros((self.height, self.width, 4))
            overlay[explored] = cmap(norm_map[explored])
            overlay[explored, 3] = 0.7  # Semi-transparent
            
            ax.imshow(overlay, interpolation='nearest', origin='upper')
        
        # Solution paths
        solution_colors = [
            '#FF4444', '#44FF44', '#4488FF', '#FFAA00', '#FF44FF', '#44FFFF'
        ]
        for i, (label, cost) in enumerate(solutions):
            path = self._reconstruct_path(label)
            rows = [p[0] for p in path]
            cols = [p[1] for p in path]
            color = solution_colors[i % len(solution_colors)]
            ax.plot(cols, rows, color=color, linewidth=2.5, alpha=0.9,
                    label=f'Sol {i+1}: d={cost[0]:.0f} e={cost[1]:.0f} s={cost[2]:.2f}')
        
        # Source and Target markers
        ax.scatter(source[1], source[0], c='#00FF88', s=250, marker='*',
                   edgecolors='white', linewidths=1.5, zorder=20, label='Source')
        ax.scatter(target[1], target[0], c='#FF3366', s=250, marker='*',
                   edgecolors='white', linewidths=1.5, zorder=20, label='Target')
        
        # HUD
        hud_text = (
            f"Expansions: {expansion_count:,}\n"
            f"Queue: {queue_size:,}\n"
            f"Solutions: {len(solutions)}\n"
            f"Time: {elapsed:.2f}s"
        )
        ax.text(
            0.02, 0.98, hud_text,
            transform=ax.transAxes,
            fontsize=9, fontfamily='monospace',
            color='white', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7)
        )
        
        # Title
        ax.set_title('A*pex Multi-Objective Search', color='white',
                     fontsize=14, fontweight='bold', pad=10)
        ax.tick_params(colors='#555555', labelsize=7)
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(self.height - 0.5, -0.5)
        
        # Legend for solutions
        if solutions:
            ax.legend(loc='lower right', fontsize=7, framealpha=0.7,
                     facecolor='black', edgecolor='gray', labelcolor='white')
        
        # Capture frame
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        image = np.asarray(buf)[:, :, :3].copy()  # RGBA -> RGB
        self.frames.append(image)
    
    def search_animated(
        self,
        source: Position,
        target: Position
    ) -> Tuple[List[ParetoSolution], List[np.ndarray]]:
        """
        Run A*pex search with frame capture for animation.
        
        Returns:
            (list of ParetoSolution, list of frame images)
        """
        start_time = time.time()
        self.frames = []
        
        print(f"\n{'='*60}")
        print(f"A*pex Animated Search")
        print(f"{'='*60}")
        print(f"Source: {source}")
        print(f"Target: {target}")
        print(f"Grid: {self.height}×{self.width}")
        print(f"Epsilon: {self.eps}")
        print(f"Frame interval: every {self.frame_interval} expansions")
        print(f"{'='*60}\n")
        
        # Setup figure (once, reused for all frames)
        fig, ax = plt.subplots(figsize=(10, 7), facecolor='#101020')
        ax.set_facecolor('#101020')
        plt.tight_layout(pad=2)
        
        # Expansion tracking
        expansion_map = np.zeros((self.height, self.width), dtype=np.int32)
        
        # Search state
        heap = []
        counter = 0
        labels: Dict[Position, List[CostVec]] = {}
        solutions: List[Tuple[Label, CostVec]] = []
        
        # Initialize
        h0 = self._heuristic(source, target)
        g0 = (0.0, 0.0, 0.0)
        f0 = v_add(g0, h0)
        start_label = Label(
            state=source, g=g0, h=h0, f=f0, parent=None, prev_direction=None
        )
        labels[source] = [g0]
        prio = tuple(list(f0) + [v_scalar_sum(f0)])
        heapq.heappush(heap, (prio, counter, start_label))
        counter += 1
        
        expansions = 0
        
        # Capture initial frame
        self._render_frame(
            expansion_map, 0, solutions, len(heap),
            source, target, 0.0, fig, ax
        )
        
        while heap:
            _, _, label = heapq.heappop(heap)
            pos = label.state
            
            # Prune by solutions
            if any(v_eps_dominates(sc, label.f, self.eps) for _, sc in solutions):
                continue
            
            expansions += 1
            expansion_map[pos[0], pos[1]] = expansions
            
            if expansions >= self.max_expansions:
                print(f"⚠ Max expansions reached ({self.max_expansions:,})")
                break
            
            # Goal test
            if pos == target:
                g_n = label.g
                dominated = any(
                    v_eps_dominates(sc, g_n, self.eps) for _, sc in solutions
                )
                if not dominated:
                    new_solutions = [
                        (s, c) for s, c in solutions
                        if not v_eps_dominates(g_n, c, self.eps)
                    ]
                    new_solutions.append((label, g_n))
                    solutions = new_solutions
                    print(f"  ✓ Solution #{len(solutions)}: {g_n}")
                    
                    # Capture frame on solution discovery
                    elapsed = time.time() - start_time
                    self._render_frame(
                        expansion_map, expansions, solutions, len(heap),
                        source, target, elapsed, fig, ax
                    )
                continue
            
            # Expand successors
            for neighbor, edge_cost, direction in self._get_successors(pos):
                g_succ = v_add(label.g, edge_cost)
                h_succ = self._heuristic(neighbor, target)
                f_succ = v_add(g_succ, h_succ)
                
                if any(v_eps_dominates(sc, f_succ, self.eps) for _, sc in solutions):
                    continue
                
                existing = labels.get(neighbor, [])
                skip = False
                for g_old in existing:
                    if v_eps_dominates(g_old, g_succ, self.eps):
                        skip = True
                        break
                if skip:
                    continue
                
                new_labels = [
                    g_old for g_old in existing
                    if not v_eps_dominates(g_succ, g_old, self.eps)
                ]
                new_labels.append(g_succ)
                labels[neighbor] = new_labels
                
                child = Label(
                    state=neighbor, g=g_succ, h=h_succ, f=f_succ,
                    parent=label, prev_direction=direction
                )
                prio = tuple(list(f_succ) + [v_scalar_sum(f_succ)])
                heapq.heappush(heap, (prio, counter, child))
                counter += 1
            
            # Capture frame at regular intervals
            if expansions % self.frame_interval == 0:
                elapsed = time.time() - start_time
                print(f"  [{expansions:,}] queue={len(heap):,} "
                      f"solutions={len(solutions)} t={elapsed:.1f}s")
                self._render_frame(
                    expansion_map, expansions, solutions, len(heap),
                    source, target, elapsed, fig, ax
                )
        
        # Final frame
        elapsed = time.time() - start_time
        self._render_frame(
            expansion_map, expansions, solutions, len(heap),
            source, target, elapsed, fig, ax
        )
        # Hold on final frame for a few extra copies so GIF pauses at end
        for _ in range(5):
            self.frames.append(self.frames[-1].copy())
        
        plt.close(fig)
        
        # Convert to ParetoSolution objects
        result = []
        for label, cost in solutions:
            path = self._reconstruct_path(label)
            result.append(ParetoSolution(path=path, objectives=cost))
        result.sort(key=lambda s: s.objectives[0])
        
        print(f"\n{'='*60}")
        print(f"Search Complete")
        print(f"  Solutions: {len(result)}")
        print(f"  Expansions: {expansions:,}")
        print(f"  Frames captured: {len(self.frames)}")
        print(f"  Runtime: {elapsed:.2f}s")
        print(f"{'='*60}\n")
        
        return result, self.frames
    
    def save_gif(self, output_path: str, fps: int = 4):
        """Save captured frames as an animated GIF."""
        if not self.frames:
            print("No frames to save!")
            return
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        images = [Image.fromarray(frame) for frame in self.frames]
        duration = int(1000 / fps)
        
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0,
            optimize=True
        )
        
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✓ GIF saved to {output_path} ({len(images)} frames, {size_mb:.1f} MB)")


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Animated A*pex Search")
    parser.add_argument(
        "--input", type=str,
        default="src/sample-test-set/seattle_test_raster.npz",
        help="Path to NPZ test set"
    )
    parser.add_argument(
        "--output", type=str,
        default="data/output/apex_animation.gif",
        help="Output GIF path"
    )
    parser.add_argument(
        "--eps", type=float, default=0.1,
        help="Epsilon for ε-dominance (applied to all objectives)"
    )
    parser.add_argument(
        "--max-expansions", type=int, default=500_000,
        help="Maximum node expansions"
    )
    parser.add_argument(
        "--frame-interval", type=int, default=500,
        help="Capture a frame every N expansions"
    )
    parser.add_argument(
        "--fps", type=int, default=4,
        help="Frames per second in output GIF"
    )
    
    args = parser.parse_args()
    
    # Load test data
    print(f"Loading: {args.input}")
    data = np.load(args.input, allow_pickle=True)
    raster = data['raster']
    goal_points = data['goal_points']
    
    source = (int(goal_points[0][1]), int(goal_points[0][0]))  # (row, col)
    target = (int(goal_points[1][1]), int(goal_points[1][0]))
    
    print(f"  Raster: {raster.shape}")
    print(f"  Source: {source}  Target: {target}")
    
    # Run animated search
    searcher = AnimatedApexSearch(
        raster=raster,
        eps=(args.eps, args.eps, args.eps),
        max_expansions=args.max_expansions,
        frame_interval=args.frame_interval
    )
    
    solutions, frames = searcher.search_animated(source, target)
    
    # Print results
    print("Pareto Front:")
    for i, sol in enumerate(solutions):
        d, e, s = sol.objectives
        print(f"  {i+1}. Distance={d:.1f}  Elevation={e:.1f}  Slope={s:.4f}")
    
    # Save GIF
    searcher.save_gif(args.output, fps=args.fps)


if __name__ == "__main__":
    main()
