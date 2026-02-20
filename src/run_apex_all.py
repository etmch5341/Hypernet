#!/usr/bin/env python3
"""
Run A*pex Pure Multi-Objective Search on all test rasters.

Runs the search on Austin, Portland, and Seattle test sets,
generates results JSON, static visualizations, and animated GIFs.

Output structure:
  src/sample-test-set/apex_results/
    ├── seattle/
    │   ├── apex_results.json
    │   ├── routes_overlay.png
    │   ├── pareto_3d.png
    │   ├── pareto_2d.png
    │   ├── search_progress.png
    │   └── apex_animation.gif
    ├── austin/
    │   └── ...
    └── portland/
        └── ...

Usage:
    python3 src/run_apex_all.py
    python3 src/run_apex_all.py --cities seattle austin
    python3 src/run_apex_all.py --no-animation
"""

import numpy as np
import sys
import os
import time
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.getcwd(), 'src'))

from apex_pure import RasterApexSearch, ParetoSolution
from apex_visualizer import ApexVisualizer
from apex_animate import AnimatedApexSearch


# =============================================================================
# City configurations: tuned per raster size
# =============================================================================
CITY_CONFIGS = {
    'seattle': {
        'npz': 'src/sample-test-set/seattle_test_raster.npz',
        'eps': (0.1, 0.1, 0.1),
        'max_expansions': 500_000,
        'frame_interval': 5000,
    },
    'austin': {
        'npz': 'src/sample-test-set/austin_test_raster.npz',
        'eps': (0.1, 0.1, 0.1),
        'max_expansions': 5_000_000,
        'frame_interval': 50000,
    },
    'portland': {
        'npz': 'src/sample-test-set/portland_test_raster.npz',
        'eps': (0.1, 0.1, 0.1),
        'max_expansions': 5_000_000,
        'frame_interval': 50000,
    },
}


def run_city(city: str, config: dict, output_base: str, animate: bool = True):
    """Run A*pex on one city and generate all outputs."""
    
    npz_path = config['npz']
    output_dir = os.path.join(output_base, city)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"  {city.upper()}")
    print(f"{'='*70}")
    
    # Load data
    print(f"Loading: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    raster = data['raster']
    goal_points = data['goal_points']
    bbox = tuple(data['bbox']) if 'bbox' in data else None
    
    source = (int(goal_points[0][1]), int(goal_points[0][0]))  # (row, col)
    target = (int(goal_points[1][1]), int(goal_points[1][0]))
    
    print(f"  Raster: {raster.shape}")
    print(f"  Source: {source} ({goal_points[0][2]})")
    print(f"  Target: {target} ({goal_points[1][2]})")
    
    # =========================================================================
    # 1. Run Pure A*pex Search
    # =========================================================================
    print(f"\n--- Running Pure A*pex ---")
    
    searcher = RasterApexSearch(
        raster=raster,
        bbox=bbox,
        eps=config['eps'],
        max_expansions=config['max_expansions'],
        log_interval=10000,
    )
    
    solutions, stats = searcher.search(source, target)
    
    # Save results JSON
    results_path = searcher.save_results(solutions, stats, output_dir)
    
    # Print summary
    print(f"\n  Pareto Front ({len(solutions)} solutions):")
    for i, sol in enumerate(solutions):
        d, e, s = sol.objectives
        print(f"    {i+1}. Distance={d:.1f}  Elevation={e:.1f}  Slope={s:.4f}")
    
    # =========================================================================
    # 2. Generate Static Visualizations
    # =========================================================================
    print(f"\n--- Generating Visualizations ---")
    
    viz = ApexVisualizer(results_path, raster)
    viz.create_all_visualizations(output_dir, raster)
    
    # =========================================================================
    # 3. Generate Animated GIF (optional)
    # =========================================================================
    if animate:
        print(f"\n--- Generating Animation ---")
        
        anim_searcher = AnimatedApexSearch(
            raster=raster,
            elevation_array=searcher.elevation,  # Reuse computed elevation
            eps=config['eps'],
            max_expansions=config['max_expansions'],
            frame_interval=config['frame_interval'],
        )
        
        anim_solutions, frames = anim_searcher.search_animated(source, target)
        gif_path = os.path.join(output_dir, 'apex_animation.gif')
        anim_searcher.save_gif(gif_path, fps=4)
    
    print(f"\n✓ {city.upper()} complete → {output_dir}/")
    return len(solutions)


def main():
    parser = argparse.ArgumentParser(description="Run A*pex on all test rasters")
    parser.add_argument(
        "--cities", nargs="+", 
        choices=list(CITY_CONFIGS.keys()),
        default=list(CITY_CONFIGS.keys()),
        help="Which cities to run (default: all)"
    )
    parser.add_argument(
        "--output", type=str,
        default="src/sample-test-set/apex_results",
        help="Base output directory"
    )
    parser.add_argument(
        "--no-animation", action="store_true",
        help="Skip GIF animation generation (faster)"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("  A*pex Batch Runner — Pure Multi-Objective Search")
    print(f"  Cities: {', '.join(args.cities)}")
    print(f"  Output: {args.output}")
    print(f"  Animation: {'No' if args.no_animation else 'Yes'}")
    print("="*70)
    
    overall_start = time.time()
    results_summary = {}
    
    for city in args.cities:
        city_start = time.time()
        config = CITY_CONFIGS[city]
        n_solutions = run_city(
            city, config, args.output, 
            animate=not args.no_animation
        )
        city_elapsed = time.time() - city_start
        results_summary[city] = {
            'solutions': n_solutions,
            'runtime': city_elapsed
        }
    
    overall_elapsed = time.time() - overall_start
    
    # Final summary
    print(f"\n\n{'='*70}")
    print("  BATCH COMPLETE")
    print(f"{'='*70}")
    for city, info in results_summary.items():
        print(f"  {city:12s}: {info['solutions']} solutions in {info['runtime']:.1f}s")
    print(f"  {'Total':12s}: {overall_elapsed:.1f}s")
    print(f"  Output: {args.output}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
