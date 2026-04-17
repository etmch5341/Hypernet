#!/usr/bin/env python3
"""
Run A*pex Custom Multi-Objective Search

Runs the A*pex algorithm specifically tuned for arbitrary 2D custom cost maps,
for example: Geometry align, Construction, and Environmental impact.
"""

import numpy as np
import sys
import os
import time
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from apex_pure import RasterApexSearch
from apex_visualizer import ApexVisualizer

def main():
    parser = argparse.ArgumentParser(description="Run A*pex on Custom Cost Maps")
    parser.add_argument(
        "--output", type=str,
        default="src/sample-test-set/apex_custom_results",
        help="Base output directory"
    )
    args = parser.parse_args()
    
    print("="*70)
    print("  A*pex Runner — Custom Cost Maps")
    print("="*70)
    
    # Define file paths
    geo_path = "geometry_cost_map.npz"
    cons_path = "austin_construction_cost.npz"
    env_path = "austin_environmental_impact.npz"
    
    for p in [geo_path, cons_path, env_path]:
        if not os.path.exists(p):
            print(f"Error: Missing cost map file {p}.")
            print("Please ensure you are running this from the repository root.")
            return

    # Load data
    print("Loading custom cost maps...")
    try:
        geo_data = np.load(geo_path, allow_pickle=True)
        cons_data = np.load(cons_path, allow_pickle=True)
        env_data = np.load(env_path, allow_pickle=True)
    except Exception as e:
        print(f"Failed to read npz files: {e}")
        return

    # Extract goal points and verify
    if 'goal_points' not in geo_data:
        print("Error: 'goal_points' not found in geometry_cost_map.npz. We need start and end coordinates!")
        return

    goal_points = geo_data['goal_points']
    
    if len(goal_points) < 2:
        print("Error: Not enough goal points defined in geometry_cost_map.npz.")
        return
    # goal_points format: ['Name', x(col), y(row), lon, lat]
    source = (int(goal_points[0][2]), int(goal_points[0][1]))  # (row, col)
    target = (int(goal_points[1][2]), int(goal_points[1][1]))

    # Load the original road raster to manipulate the existing corridor weights natively
    try:
        original_raster = np.load('src/sample-test-set/austin_test_raster.npz', allow_pickle=True)['raster']
        geo_map = geo_data['cost_map'].copy()
        
        # The user requested an even more aggressive tendency to follow roads. 
        # By dropping the road cost to just 30% of the median terrain (0.30 multiplier),
        # the route will trace highway corridors wherever possible without breaking physics.
        off_road_median = np.median(geo_map[original_raster == 0])
        geo_map[original_raster == 1] = off_road_median * 0.30
        
    except Exception as e:
        print("Could not load original road raster for dampening, falling back to pure geometry.")
        original_raster = np.ones(geo_data['cost_map'].shape, dtype=np.int8)
        geo_map = geo_data['cost_map']

    # Package custom costmaps
    custom_costmaps = {
        'geometry': geo_map,
        'construction': cons_data['cost_map'],
        'environmental': env_data['env_map'] # Note: key is env_map for environmental
    }

    print(f"  Map shape: {original_raster.shape}")
    print(f"  Source: {source} ({goal_points[0][2]})")
    print(f"  Target: {target} ({goal_points[1][2]})")
    
    # =========================================================================
    # SCENARIOS: The Tuning Dashboard
    # =========================================================================
    # You can add or modify scenarios here to see how different weights perform.
    # Road Discount: 0.1 (high snap) to 1.0 (no snap)
    # Stiffness: 1.0 (raw jitter) to 50.0 (smooth straights)
    # Heuristic: 1.2 (wide search) to 1.5 (fast beam)
    
    # =========================================================================
    # THE SWEET SPOT: User-Identified Optimal Weights
    # =========================================================================
    scenarios = [
        {"name": "Infra_Purist",   "stiff": 5.0,  "h": 1.3,  "road": 0.1, "eps": 0.3},
        {"name": "Geometric_Ideal", "stiff": 40.0, "h": 1.3,  "road": 1.0, "eps": 0.3},
        {"name": "Discovery_Sweep", "stiff": 10.0, "h": 1.25, "road": 0.7, "eps": 0.1}
    ]
    
    for scenario in scenarios:
        s_stiff = scenario['stiff']
        s_h     = scenario['h']
        s_road  = scenario['road']
        s_eps   = scenario['eps']
        s_name  = scenario['name']
        
        # Exact Naming Convention Requested by User
        folder_name = f"ApexHUD_H{s_h}_E{s_eps}_T{int(s_stiff)}_S{int(s_road*100)}_{s_name}"
        scenario_output = os.path.join(args.output, folder_name)
        os.makedirs(scenario_output, exist_ok=True)
        
        eps_vector = (s_eps, s_eps, s_eps, s_eps)
        max_expansions = 7_000_000
        
        print(f"\n{'='*80}")
        print(f" GENERATING ANALYSIS: {folder_name}")
        print(f"{'='*80}")
        
        searcher = RasterApexSearch(
            raster=original_raster,
            eps=eps_vector,
            max_expansions=max_expansions,
            log_interval=100000,
            custom_costmaps=custom_costmaps,
            turn_stiffness=s_stiff,
            h_weight=s_h,
            road_discount=s_road
        )
        
        start_time = time.time()
        solutions, stats = searcher.search(source, target)
        elapsed = time.time() - start_time
        
        results_path = searcher.save_results(solutions, stats, scenario_output)
        np.savez_compressed(
            os.path.join(scenario_output, "apex_diagnostics.npz"),
            expansion_map=stats.expansion_map,
            prune_map=stats.prune_map
        )
        
        print(f"\n--- Exporting Analysis: {folder_name} ---")
        viz = ApexVisualizer(results_path, original_raster)
        
        diagnostics = {
            'expansion_map': stats.expansion_map,
            'prune_map': stats.prune_map,
            'diagnostic_frames': stats.diagnostic_frames,
            'custom_costmaps': custom_costmaps,
            'config': {
                'heuristic': s_h,
                'epsilon': s_eps,
                'turn_stiffness': s_stiff,
                'road_discount': s_road,
                'scenario': scenario['name']
            }
        }
        viz.create_all_visualizations(scenario_output, original_raster, diagnostics)

if __name__ == "__main__":
    main()
