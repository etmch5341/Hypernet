"""
A*pex EMO Optimization for Hypernet
Uses NSGA-II to optimize A* weights (alpha, beta, gamma) to find
diverse and optimal hyperloop routes.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import time
import os
import sys

# Pymoo imports
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.decomposition.asf import ASF

# Add src to path to import from apex_emo_test
sys.path.append(os.path.join(os.path.dirname(__file__)))
from apex_emo_test import APexSeeder, load_hypernet_data


class HyperloopWeightProblem(ElementwiseProblem):
    """
    Optimization problem for Hyperloop route weights.
    
    Variables:
        x[0] = alpha (Time weight)
        x[1] = beta (Turn weight)
        x[2] = gamma (Jerk weight)
        
    Objectives (Minimize):
        f[0] = Total Time
        f[1] = Total Turn
        f[2] = Total Jerk
        f[3] = Distance
    """
    
    def __init__(self, seeder):
        # 3 variables (weights), 4 objectives, 0 constraints
        # Weights are in range [0, 1]
        super().__init__(n_var=3, n_obj=4, n_ieq_constr=0, xl=0.0, xu=1.0)
        self.seeder = seeder
        self.eval_count = 0
        
    def _evaluate(self, x, out, *args, **kwargs):
        self.eval_count += 1
        
        # Prepare weights dict
        weights = {
            'alpha': x[0],
            'beta': x[1], 
            'gamma': x[2],
            'name': f'gen_{self.eval_count}'
        }
        
        # Run A* with these weights
        # Note: This is computationally expensive!
        path, objectives = self.seeder.run_astar_with_weights(weights)
        
        if path is None:
            # Penalty for invalid paths
            out["F"] = [1e9, 1e9, 1e9, 1e9]
        else:
            out["F"] = [
                objectives['total_time'],
                objectives['total_turn'],
                objectives['total_jerk'],
                objectives['distance']
            ]


def run_optimization(n_gen=20, pop_size=20):
    print("="*70)
    print("HYPERNET A*PEX EVOLUTIONARY OPTIMIZATION")
    print(f"Algorithm: NSGA-II | Generations: {n_gen} | Population: {pop_size}")
    print("="*70)

    # 1. Load Data & Initialize Seeder
    road_bitmap, protected_bitmap, transform, start_pos, goal_pos = \
        load_hypernet_data(use_liechtenstein=True)
        
    seeder = APexSeeder(
        road_bitmap, 
        protected_bitmap, 
        transform,
        start_pos, 
        goal_pos
    )
    
    # 2. Generate Initial Seeds (Warm Start)
    print("\n[Phase 1] Generating Initial Seeds (A*pex)...")
    seeds = seeder.generate_seeds(n_seeds=pop_size)
    
    # Convert seeds to initial population X
    # We normalized weights in current A*pex implementation to be whatever
    # But for optimization we want them in [0, 1]
    # We'll just take the alpha, beta, gamma from seeds
    X_seed = np.zeros((len(seeds), 3))
    for i, seed in enumerate(seeds):
        w = seed['weights']
        X_seed[i, 0] = w.get('alpha', 0.5)
        X_seed[i, 1] = w.get('beta', 0.5)
        X_seed[i, 2] = w.get('gamma', 0.5)
        
    # 3. Define Problem
    problem = HyperloopWeightProblem(seeder)
    
    # 4. Setup Algorithm
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=X_seed,  # Inject seeds as initial population
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.3, eta=20),
        eliminate_duplicates=True
    )
    
    # 5. Run Optimization
    print(f"\n[Phase 2] Running Evolutionary Search...")
    start_time = time.time()
    
    res = minimize(
        problem,
        algorithm,
        get_termination("n_gen", n_gen),
        seed=1,
        save_history=True,
        verbose=True
    )
    
    elapsed = time.time() - start_time
    print(f"\n✓ Optimization completed in {elapsed:.2f}s")
    
    # 6. Process Results
    save_results(res, seeder, seeds)


def save_results(res, seeder, initial_seeds):
    output_dir = 'data/output/apex_nsga3_results'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract optimized solutions
    opt_solutions = []
    
    # Recalculate paths for the best found weights to get the coordinates
    # (res.X only contains the weights)
    print("\n[Phase 3] Reconstructing Best Paths...")
    
    # Filter unique solutions
    unique_X = np.unique(res.X, axis=0)
    
    for i, x in enumerate(unique_X):
        weights = {
            'alpha': x[0],
            'beta': x[1],
            'gamma': x[2],
            'name': f'opt_{i}'
        }
        
        path, objectives = seeder.run_astar_with_weights(weights)
        
        if path:
            opt_solutions.append({
                'id': i,
                'weights': weights,
                'path': path,
                'objectives': objectives
            })
            
    # Visualize
    visualize_comparison(initial_seeds, opt_solutions, output_dir)
    
    # Save JSON
    serializable = []
    for sol in opt_solutions:
        serializable.append({
            'weights': sol['weights'],
            'objectives': sol['objectives'],
            'path_length': len(sol['path'])
        })
        
    with open(f'{output_dir}/optimized_solutions.json', 'w') as f:
        json.dump(serializable, f, indent=2)
        
    print(f"✓ Results saved to {output_dir}")


def visualize_comparison(seeds, optimized, output_dir):
    """Compare initial seeds vs optimized solutions"""
    
    # 1. Objective Space Projections
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    obj_keys = ['total_time', 'total_turn', 'total_jerk', 'distance']
    labels = ['Time (s)', 'Total Turn (rad)', 'Total Jerk', 'Distance (m)']
    
    pairs = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3), (2, 3)
    ]
    
    for ax, (i, j) in zip(axes.flat, pairs):
        # Plot Initial Seeds
        seed_x = [s['objectives'][obj_keys[i]] for s in seeds]
        seed_y = [s['objectives'][obj_keys[j]] for s in seeds]
        ax.scatter(seed_x, seed_y, c='gray', alpha=0.5, label='Initial A* Seeds')
        
        # Plot Optimized Solutions
        opt_x = [s['objectives'][obj_keys[i]] for s in optimized]
        opt_y = [s['objectives'][obj_keys[j]] for s in optimized]
        ax.scatter(opt_x, opt_y, c='red', s=50, label='Evolutionary Optimized')
        
        ax.set_xlabel(labels[i])
        ax.set_ylabel(labels[j])
        ax.grid(True, alpha=0.3)
        if i == 0 and j == 1:
            ax.legend()
            
    plt.suptitle('A*pex: Initial Seeds vs Optimized Solutions', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_pareto.png', dpi=150)
    
    # 2. Path Visualization (Best Solutions)
    # We can't plot ALL of them, maybe just the non-dominated ones?
    # For now, let's plot all optimized ones
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot paths
    # We can't plot ALL of them, maybe just the non-dominated ones?
    # For now, let's plot all optimized ones
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for sol in optimized:
        path = sol['path']
        xs = [p[1] for p in path]
        ys = [p[0] for p in path]
        ax.plot(xs, ys, color='red', alpha=0.3, linewidth=1)
        
    ax.invert_yaxis() # Match image coords
    ax.set_title("Optimized Routes Overlay")
    plt.savefig(f'{output_dir}/optimized_paths.png')


if __name__ == '__main__':
    run_optimization()
