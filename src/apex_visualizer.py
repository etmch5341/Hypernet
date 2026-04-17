#!/usr/bin/env python3
"""
Visualization for A*pex Multi-Objective Search Results

Creates:
  1. Pareto front plots (3D and 2D projections)
  2. Route overlays on raster map
  3. Iteration/progress charts
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import json
import os
import math
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class ApexVisualizer:
    """Visualize A*pex search results."""
    
    def __init__(self, results_path: str = None, raster: np.ndarray = None):
        """
        Initialize visualizer.
        
        Args:
            results_path: Path to apex_results.json
            raster: Optional raster array for route overlay
        """
        self.results = None
        self.raster = raster
        
        if results_path and os.path.exists(results_path):
            with open(results_path) as f:
                self.results = json.load(f)

    def _add_config_hud(self, ax, config: dict):
        """Add a configuration summary text box to a plot."""
        if not config: return
        
        lines = ["--- A*pex Optimization HUD ---"]
        if 'heuristic' in config: lines.append(f"Heuristic: {config['heuristic']}x")
        if 'epsilon' in config: 
            eps = config['epsilon']
            if isinstance(eps, (list, tuple)):
                lines.append(f"Epsilon: ({', '.join([str(v) for v in eps])})")
            else:
                lines.append(f"Epsilon: {eps}")
        if 'turn_stiffness' in config: lines.append(f"Turn Stiffness: {config['turn_stiffness']}x")
        if 'road_discount' in config: lines.append(f"Road Snapping: {int(config['road_discount']*100)}%")
        
        text = "\n".join(lines)
        # Position at bottom-right with high z-order
        ax.text(0.98, 0.02, text, transform=ax.transAxes, color='white',
                fontsize=7, fontfamily='monospace', va='bottom', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='#555555'),
                zorder=100)
    
    def plot_pareto_3d(self, output_path: str = None, figsize: Tuple[int, int] = (12, 10)):
        """
        Create 3D Pareto front visualization.
        """
        if not self.results or not self.results.get('solutions'):
            print("No solutions to visualize")
            return
        
        solutions = self.results['solutions']
        objectives = self.results.get('objectives', ['Obj 1', 'Obj 2', 'Obj 3'])
        
        # Extract objective values
        obj_values = []
        for sol in solutions:
            if isinstance(sol['objectives'], dict):
                obj_values.append([sol['objectives'].get(o, 0) for o in objectives])
            else:
                obj_values.append(sol['objectives'])
        
        obj_values = np.array(obj_values)
        
        fig = plt.figure(figsize=figsize, facecolor='#0a0a0a')
        ax = fig.add_subplot(111, projection='3d', facecolor='#0a0a0a')
        
        # Scatter plot
        scatter = ax.scatter(
            obj_values[:, 0], obj_values[:, 1], obj_values[:, 2],
            c=np.arange(len(obj_values)),
            cmap='plasma',
            s=100,
            alpha=0.9,
            edgecolors='white',
            linewidths=0.5
        )
        
        # Labels
        ax.set_xlabel(objectives[0].capitalize(), color='white', fontsize=11)
        ax.set_ylabel(objectives[1].capitalize(), color='white', fontsize=11)
        ax.set_zlabel(objectives[2].capitalize(), color='white', fontsize=11)
        ax.set_title('Pareto Front (3D)', color='white', fontsize=14, fontweight='bold')
        
        # Style
        ax.tick_params(colors='white')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        plt.colorbar(scatter, ax=ax, label='Solution Index', pad=0.1)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, facecolor='#0a0a0a', bbox_inches='tight')
            print(f"✓ Saved 3D Pareto plot to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_pareto_2d_projections(self, output_path: str = None, 
                                     figsize: Tuple[int, int] = (18, 5)):
        """
        Create 2D projections of the Pareto front.
        """
        if not self.results or not self.results.get('solutions'):
            print("No solutions to visualize")
            return
        
        solutions = self.results['solutions']
        objectives = self.results.get('objectives', ['distance', 'elevation', 'slope', 'turn_angle'])
        
        # Extract objective values
        obj_values = []
        for sol in solutions:
            if isinstance(sol['objectives'], dict):
                obj_values.append([sol['objectives'].get(o, 0) for o in objectives])
            else:
                obj_values.append(sol['objectives'])
        
        obj_values = np.array(obj_values)
        
        # Generate all unique objective pairs
        from itertools import combinations
        pairs = list(combinations(range(len(objectives)), 2))
        n_pairs = len(pairs)
        n_cols = min(3, n_pairs)
        n_rows = (n_pairs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), facecolor='#111111')
        if n_pairs == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        for ax, (i, j) in zip(axes, pairs):
            ax.set_facecolor('#1a1a1a')
            
            scatter = ax.scatter(
                obj_values[:, i], obj_values[:, j],
                c=np.arange(len(obj_values)),
                cmap='viridis',
                s=120,
                alpha=0.85,
                edgecolors='white',
                linewidths=1
            )
            
            ax.set_xlabel(objectives[i].capitalize(), color='white', fontsize=11)
            ax.set_ylabel(objectives[j].capitalize(), color='white', fontsize=11)
            ax.set_title(f'{objectives[i].capitalize()} vs {objectives[j].capitalize()}', 
                        color='white', fontsize=12)
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.2, color='gray')
            
            # Annotate points
            for idx, (x, y) in enumerate(zip(obj_values[:, i], obj_values[:, j])):
                ax.annotate(str(idx+1), (x, y), fontsize=8, color='yellow', 
                           alpha=0.8, ha='center', va='bottom',
                           xytext=(0, 5), textcoords='offset points')
        
        # Hide unused axes
        for idx in range(n_pairs, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Pareto Front - 2D Projections', color='white', fontsize=14, 
                    fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, facecolor='#111111', bbox_inches='tight')
            print(f"✓ Saved 2D projections to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_routes_on_raster(self, raster: np.ndarray = None, 
                               output_path: str = None,
                               figsize: Tuple[int, int] = (14, 12),
                               max_routes: int = 10):
        """
        Overlay Pareto-optimal routes on the raster map.
        """
        if raster is None:
            raster = self.raster
        
        if raster is None:
            print("No raster provided for route overlay")
            return
        
        if not self.results or not self.results.get('solutions'):
            print("No solutions to visualize")
            return
        
        solutions = self.results['solutions'][:max_routes]
        
        fig, ax = plt.subplots(figsize=figsize, facecolor='#0a0a0a')
        ax.set_facecolor('#111111')
        
        # Display raster
        cmap = ListedColormap(['#1a1a2e', '#00d4ff'])  # Dark blue / cyan
        ax.imshow(raster, cmap=cmap, alpha=0.6, origin='upper')
        
        # Plot each route with different color
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(solutions)))
        
        for i, sol in enumerate(solutions):
            path = sol.get('path', [])
            if not path:
                continue
            
            ys = [p[0] for p in path]
            xs = [p[1] for p in path]
            
            try:
                from scipy.interpolate import splprep, splev
                if len(xs) > 3:
                    # Remove duplicate consecutive points which break splprep curvature algorithms
                    pts = np.array([xs, ys])
                    diffs = np.sum(np.diff(pts, axis=1)**2, axis=0)
                    keep = np.concatenate(([True], diffs > 0))
                    xs_sub = pts[0, keep]
                    ys_sub = pts[1, keep]
                    
                    if len(xs_sub) > 3:
                        tck, u = splprep([xs_sub, ys_sub], s=200.0) # Soft smoothing factor
                        u_new = np.linspace(0, 1, max(len(xs), 200)) # Resample points
                        x_smooth, y_smooth = splev(u_new, tck)
                        ax.plot(x_smooth, y_smooth, color=colors[i], linewidth=2.5, alpha=0.9, label=f"Route {i+1}")
                    else:
                        ax.plot(xs, ys, color=colors[i], linewidth=2, alpha=0.8, label=f"Route {i+1}")
                else:
                    ax.plot(xs, ys, color=colors[i], linewidth=2, alpha=0.8, label=f"Route {i+1}")
            except Exception:
                ax.plot(xs, ys, color=colors[i], linewidth=2, alpha=0.8, label=f"Route {i+1}")
        
        # Mark start/end
        if solutions and solutions[0].get('path'):
            first_path = solutions[0]['path']
            start = first_path[0]
            end = first_path[-1]
            
            ax.scatter(start[1], start[0], c='#00ff00', s=300, marker='*', 
                      zorder=10, edgecolors='white', linewidths=2, label='Start')
            ax.scatter(end[1], end[0], c='#ff00ff', s=300, marker='*', 
                      zorder=10, edgecolors='white', linewidths=2, label='Goal')
        
        ax.set_title('Pareto-Optimal Routes', color='white', fontsize=14, fontweight='bold')
        ax.tick_params(colors='white')
        ax.legend(loc='upper right', fontsize=9, facecolor='#222222', 
                 edgecolor='white', labelcolor='white')
        
        if hasattr(self, '_current_config'):
             self._add_config_hud(ax, self._current_config)
             
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, facecolor='#0a0a0a', bbox_inches='tight')
            print(f"✓ Saved route overlay to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_search_progress(self, output_path: str = None,
                              figsize: Tuple[int, int] = (14, 5)):
        """
        Plot search progress over iterations.
        """
        if not self.results:
            print("No results to visualize")
            return
        
        stats = self.results.get('statistics', {})
        iterations = stats.get('iterations', [])
        
        if not iterations:
            print("No iteration data to plot")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor='#111111')
        
        iters = [it['iteration'] for it in iterations]
        expanded = [it['nodes_expanded'] for it in iterations]
        pareto = [it['pareto_size'] for it in iterations]
        queue = [it['queue_size'] for it in iterations]
        elapsed = [it['elapsed_seconds'] for it in iterations]
        
        # Plot 1: Nodes expanded over time
        axes[0].set_facecolor('#1a1a1a')
        axes[0].plot(elapsed, expanded, color='#00d4ff', linewidth=2)
        axes[0].set_xlabel('Time (s)', color='white')
        axes[0].set_ylabel('Nodes Expanded', color='white')
        axes[0].set_title('Expansion Progress', color='white')
        axes[0].tick_params(colors='white')
        axes[0].grid(True, alpha=0.2)
        
        # Plot 2: Pareto front size
        axes[1].set_facecolor('#1a1a1a')
        axes[1].plot(iters, pareto, color='#ff6b6b', linewidth=2, marker='o', markersize=4)
        axes[1].set_xlabel('Iterations', color='white')
        axes[1].set_ylabel('Pareto Front Size', color='white')
        axes[1].set_title('Solutions Found', color='white')
        axes[1].tick_params(colors='white')
        axes[1].grid(True, alpha=0.2)
        
        # Plot 3: Queue size
        axes[2].set_facecolor('#1a1a1a')
        axes[2].plot(iters, queue, color='#ffd93d', linewidth=2)
        axes[2].set_xlabel('Iterations', color='white')
        axes[2].set_ylabel('Queue Size', color='white')
        axes[2].set_title('Open Set Size', color='white')
        axes[2].tick_params(colors='white')
        axes[2].grid(True, alpha=0.2)
        
        plt.suptitle('Search Progress', color='white', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, facecolor='#111111', bbox_inches='tight')
            print(f"✓ Saved progress plot to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_composite_heatmap(self, custom_costmaps: dict, output_path: str = None, figsize: Tuple[int, int] = (14, 12)):
        """
        Plot a heatmap of the composite objective weights.
        """
        if not custom_costmaps:
            return
            
        shape = list(custom_costmaps.values())[0].shape
        composite = np.zeros(shape)
        
        # We must normalize each map to [0,1] visually before summing them.
        # Otherwise, the raw dollar value of the Construction map (in the billions)
        # literally overwrites the visual variance of Environmental/Geometry maps!
        for name, arr in custom_costmaps.items():
            arr_min = np.min(arr)
            arr_max = np.max(arr)
            if arr_max > arr_min:
                normalized = (arr - arr_min) / (arr_max - arr_min)
            else:
                normalized = np.zeros_like(arr)
            composite += normalized
            
        fig, ax = plt.subplots(figsize=figsize, facecolor='#0a0a0a')
        ax.set_facecolor('#111111')
        
        # Plot heatmap
        im = ax.imshow(composite, cmap='hot', alpha=0.9, origin='upper')
        plt.colorbar(im, ax=ax, label='Composite Cell Penalty (Base + 3 Maps)', shrink=0.8)
        
        # Plot routes ON TOP of heatmap
        if self.results and self.results.get('solutions'):
            solutions = self.results['solutions'][:10]
            colors = plt.cm.cool(np.linspace(0.1, 0.9, len(solutions)))
            
            for i, sol in enumerate(solutions):
                path = sol.get('path', [])
                if not path:
                    continue
                ys = [p[0] for p in path]
                xs = [p[1] for p in path]
                ax.plot(xs, ys, color=colors[i], linewidth=2.5, alpha=0.9, label=f"Route {i+1}")
                
            if solutions and solutions[0].get('path'):
                first_path = solutions[0]['path']
                start = first_path[0]
                end = first_path[-1]
                ax.scatter(start[1], start[0], c='#00ff00', s=300, marker='*', zorder=10, edgecolors='black', linewidths=1.5)
                ax.scatter(end[1], end[0], c='#ff00ff', s=300, marker='*', zorder=10, edgecolors='black', linewidths=1.5)
                
            ax.legend(loc='upper right', fontsize=9, facecolor='#222222', edgecolor='white', labelcolor='white')
            
        ax.set_title('Composite Multi-Objective Heatmap & Route Analysis', color='white', fontsize=14, fontweight='bold')
        ax.tick_params(colors='white')
        
        # Add HUD
        if hasattr(self, '_current_config'):
             self._add_config_hud(ax, self._current_config)
             
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, facecolor='#0a0a0a', bbox_inches='tight')
            print(f"✓ Saved composite heatmap to {output_path}")
        else:
            plt.show()
            
        plt.close()

    def plot_search_intensity(self, diagnostics: dict, output_path: str = None, figsize: Tuple[int, int] = (14, 12)):
        """
        Plot a heatmap of node expansions (search intensity).
        """
        expansion_map = diagnostics.get('expansion_map')
        if expansion_map is None:
            return
            
        fig, ax = plt.subplots(figsize=figsize, facecolor='#0a0a0a')
        ax.set_facecolor('#111111')
        
        # We use a log scale for intensity because some areas are explored 1000x more than others
        intensity = np.log1p(expansion_map)
        
        im = ax.imshow(intensity, cmap='magma', origin='upper')
        plt.colorbar(im, ax=ax, label='Search Intensity (Log Expansions)', shrink=0.8)
        
        # Overlay routes
        if self.results and self.results.get('solutions'):
            for i, sol in enumerate(self.results['solutions'][:5]):
                path = sol.get('path', [])
                if not path: continue
                ys, xs = zip(*path)
                ax.plot(xs, ys, color='cyan', linewidth=1.5, alpha=0.7, label=f"Route {i+1}" if i==0 else "")
        
        ax.set_title('Search Intensity Heatmap (A*pex Exploration Effort)', color='white', fontsize=14, fontweight='bold')
        ax.tick_params(colors='white')
        
        if hasattr(self, '_current_config'):
             self._add_config_hud(ax, self._current_config)
             
        if output_path:
            plt.savefig(output_path, dpi=150, facecolor='#0a0a0a', bbox_inches='tight')
            print(f"✓ Saved search intensity map to {output_path}")
        plt.close()

    def plot_cost_composition(self, custom_costmaps: dict, output_path: str = None, figsize: Tuple[int, int] = (12, 7)):
        """
        Plot how each cost component accumulates over the distance of the best route.
        """
        if not self.results or not self.results.get('solutions') or not custom_costmaps:
            return
            
        # Analyze the first (usually best geometry) solution
        sol = self.results['solutions'][0]
        path = sol.get('path', [])
        if len(path) < 2: return
        
        # Extract individual cost components along the path
        # Note: We re-calculate based on the costmaps to ensure the breakdown is visible
        dist_accum = [0.0]
        costs_accum = {name: [0.0] for name in custom_costmaps.keys()}
        turn_accum = [0.0]
        
        total_dist = 0.0
        prev_dir = None
        
        for i in range(1, len(path)):
            r1, c1 = path[i-1]
            r2, c2 = path[i]
            dr, dc = r2 - r1, c2 - c1
            step_dist = math.hypot(dr, dc)
            total_dist += step_dist
            dist_accum.append(total_dist)
            
            # Components
            for name, arr in custom_costmaps.items():
                val = (1.0 + float(arr[r2, c2]) * 9.0) * step_dist
                costs_accum[name].append(costs_accum[name][-1] + val)
            
            # Turn
            curr_dir = (dr, dc)
            if prev_dir:
                dot = prev_dir[0] * dr + prev_dir[1] * dc
                cross = prev_dir[0] * dc - prev_dir[1] * dr
                angle = abs(math.atan2(cross, dot))
                turn_accum.append(turn_accum[-1] + angle)
            else:
                turn_accum.append(0.0)
            prev_dir = curr_dir

        fig, ax = plt.subplots(figsize=figsize, facecolor='#111111')
        ax.set_facecolor('#1a1a1a')
        
        # Plot stacked area for costs (normalized to % of total for comparison or raw)
        # We'll plot raw values on secondary axes if scales differ wildly, 
        # but let's try a stacked plot first.
        
        colors = ['#ff6b6b', '#4ecdc4', '#ffe66d', '#1a535c']
        names = list(costs_accum.keys())
        y_values = [np.array(costs_accum[n]) for n in names]
        
        # Normalize for visualization if needed, but let's show raw complexity
        ax.stackplot(dist_accum, *y_values, labels=[n.capitalize() for n in names], 
                     colors=colors, alpha=0.8)
        
        ax.set_xlabel('Path Distance (pixels)', color='white')
        ax.set_ylabel('Cumulative Cost Multiplier', color='white')
        ax.set_title(f"Route Cost Composition Analysis (Solution #1)", color='white', fontweight='bold')
        ax.legend(loc='upper left', facecolor='#222222', edgecolor='white', labelcolor='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2)
        
        if hasattr(self, '_current_config'):
            self._add_config_hud(ax, self._current_config)
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, facecolor='#111111', bbox_inches='tight')
            print(f"✓ Saved cost composition chart to {output_path}")
        plt.close()

    def create_search_gif(self, diagnostics: dict, output_path: str, fps: int = 5):
        """
        Stitch diagnostic frames into an animated GIF showing the search process.
        """
        frames_data = diagnostics.get('diagnostic_frames', [])
        if not frames_data:
            return
            
        print(f"Generating search animation from {len(frames_data)} frames...")
        
        from PIL import Image, ImageDraw
        images = []
        
        # Get config for HUD
        config = diagnostics.get('config', {})
        hud_lines = ["--- A*pex Search ---"]
        if 'scenario' in config: hud_lines.append(f"Scenario: {config['scenario']}")
        if 'heuristic' in config: hud_lines.append(f"Heuristic: {config['heuristic']}x")
        if 'turn_stiffness' in config: hud_lines.append(f"Stiffness: {config['turn_stiffness']}x")
        hud_text = "\n".join(hud_lines)

        for i, frame in enumerate(frames_data):
            # frame is a dict with 'expansion_map_sample' (ndarray) and 'frontier_nodes' (list of pos)
            exp_map = frame.get('expansion_map_sample')
            frontier = frame.get('frontier_nodes', [])
            
            # Create RGB image
            h, w = self.raster.shape
            img_data = np.zeros((h, w, 3), dtype=np.uint8)
            
            # 1. Expansions (Red channel, mapped by log intensity)
            if exp_map is not None:
                # Log normalize for visualization
                intensity = np.log1p(exp_map)
                max_int = intensity.max()
                if max_int > 0:
                    red_layer = (intensity / max_int * 200).astype(np.uint8)
                    img_data[:, :, 0] = red_layer
                
            # 2. Frontier/Open set (Cyan)
            if len(frontier) > 0:
                rows, cols = zip(*frontier)
                # Filtering out-of-bounds just in case
                rows = np.array(rows)
                cols = np.array(cols)
                valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
                img_data[rows[valid], cols[valid], 1] = 255 # Green
                img_data[rows[valid], cols[valid], 2] = 255 # Blue
            
            img = Image.fromarray(img_data)
            
            # 3. Add HUD overlay
            draw = ImageDraw.Draw(img)
            # Rectangle background for text
            draw.rectangle([w-130, h-65, w-5, h-5], fill=(0,0,0,180))
            draw.text((w-125, h-60), hud_text, fill=(255,255,255))
            
            images.append(img)
            
            if i % 25 == 0:
                print(f"  Processed frame {i+1}/{len(frames_data)}")

        # Save GIF
        if images:
            images[0].save(
                output_path,
                save_all=True,
                append_images=images[1:],
                duration=int(1000/fps),
                loop=0,
                optimize=True
            )
            print(f"✓ Saved search animation to {output_path}")

    def create_all_visualizations(self, output_dir: str, raster: np.ndarray = None, diagnostics: dict = None):
        """
        Create all visualizations and save to output directory.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nGenerating visualizations in {output_dir}...")
        
        # Store config for use in internal plot methods
        self._current_config = diagnostics.get('config') if diagnostics else None
        
        self.plot_pareto_3d(os.path.join(output_dir, 'pareto_3d.png'))
        self.plot_pareto_2d_projections(os.path.join(output_dir, 'pareto_2d.png'))
        self.plot_search_progress(os.path.join(output_dir, 'search_progress.png'))
        
        if raster is not None:
            self.plot_routes_on_raster(raster, os.path.join(output_dir, 'routes_overlay.png'))
            
        if diagnostics is not None:
            custom_costmaps = diagnostics.get('custom_costmaps')
            if custom_costmaps:
                self.plot_composite_heatmap(custom_costmaps, os.path.join(output_dir, 'composite_heatmap.png'))
                self.plot_cost_composition(custom_costmaps, os.path.join(output_dir, 'cost_composition.png'))
            
            if 'expansion_map' in diagnostics:
                self.plot_search_intensity(diagnostics, os.path.join(output_dir, 'search_intensity.png'))
                
            if 'diagnostic_frames' in diagnostics:
                self.create_search_gif(diagnostics, os.path.join(output_dir, 'search_process.gif'))
        
        print(f"\n✓ All visualizations saved to {output_dir}")


def main():
    """Test visualization on sample results."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize A*pex Results")
    parser.add_argument("--results", type=str, required=True, help="Path to apex_results.json")
    parser.add_argument("--raster", type=str, help="Path to NPZ with raster data")
    parser.add_argument("--output", type=str, default="./apex_viz", help="Output directory")
    
    args = parser.parse_args()
    
    raster = None
    if args.raster:
        data = np.load(args.raster, allow_pickle=True)
        raster = data['raster']
    
    viz = ApexVisualizer(args.results, raster)
    viz.create_all_visualizations(args.output, raster)


if __name__ == "__main__":
    main()
