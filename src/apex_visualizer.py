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
        objectives = self.results.get('objectives', ['distance', 'turns', 'elevation'])
        
        # Extract objective values
        obj_values = []
        for sol in solutions:
            if isinstance(sol['objectives'], dict):
                obj_values.append([sol['objectives'].get(o, 0) for o in objectives])
            else:
                obj_values.append(sol['objectives'])
        
        obj_values = np.array(obj_values)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor='#111111')
        
        pairs = [(0, 1), (0, 2), (1, 2)]
        
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
            
            ax.plot(xs, ys, color=colors[i], linewidth=2, alpha=0.8, 
                   label=f"Route {i+1}")
        
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
    
    def create_all_visualizations(self, output_dir: str, raster: np.ndarray = None):
        """
        Create all visualizations and save to output directory.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nGenerating visualizations in {output_dir}...")
        
        self.plot_pareto_3d(os.path.join(output_dir, 'pareto_3d.png'))
        self.plot_pareto_2d_projections(os.path.join(output_dir, 'pareto_2d.png'))
        self.plot_search_progress(os.path.join(output_dir, 'search_progress.png'))
        
        if raster is not None:
            self.plot_routes_on_raster(raster, os.path.join(output_dir, 'routes_overlay.png'))
        
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
