"""
A*pex → EMO Test for Hypernet
Generates diverse A* seed solutions with different weight configurations
for multi-objective hyperloop route optimization.

Based on your PG/OE specification document.
"""

import json
import os
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
import heapq
from pathlib import Path
import time
from typing import List, Tuple, Dict
import geopandas as gpd


class HyperloopObjectives:
    """Calculate multi-objective costs for hyperloop routes"""
    
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.05, 
                 v_nom=50.0, a_lat_max=0.5):
        """
        Initialize with weights from your PG/OE spec
        
        Args:
            alpha: Time weight
            beta: Turn angle weight
            gamma: Jerk (heading smoothness) weight
            v_nom: Nominal speed (m/s) - using conservative 50 m/s for testing
            a_lat_max: Lateral acceleration limit (m/s²)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.v_nom = v_nom
        self.a_lat_max = a_lat_max
        self.prev_theta = 0.0  # Track previous turn angle for jerk
        
    def compute_edge_cost(self, p_prev, p_curr, p_next, 
                         road_bitmap, protected_bitmap):
        """
        Compute edge cost using your PG/OE formulation:
        c_k^PG = α*t_k + β*θ_k + γ*J_k
        
        Args:
            p_prev: Previous position (x, y)
            p_curr: Current position (x, y)
            p_next: Next position (x, y)
            road_bitmap: Road network raster
            protected_bitmap: Protected areas raster
            
        Returns:
            Edge cost value
        """
        # Step vectors
        delta_k = np.array(p_curr) - np.array(p_prev)
        delta_k_plus_1 = np.array(p_next) - np.array(p_curr)
        
        # Edge length
        edge_length = np.linalg.norm(delta_k_plus_1)
        
        if edge_length < 1e-6:
            return float('inf')
        
        # Turn angle θ_k
        dot_product = np.dot(delta_k, delta_k_plus_1)
        norm_product = np.linalg.norm(delta_k) * np.linalg.norm(delta_k_plus_1)
        
        if norm_product < 1e-6:
            theta_k = 0.0
        else:
            cos_theta = np.clip(dot_product / norm_product, -1.0, 1.0)
            theta_k = np.arccos(cos_theta)
        
        # Curvature κ_k ≈ θ_k / ||Δ_{k+1}||
        kappa_k = theta_k / (edge_length + 1e-6)
        
        # Allowed speed from lateral acceleration
        v_max_k = np.sqrt(self.a_lat_max / (max(kappa_k, 1e-6)))
        v_max_k = min(v_max_k, self.v_nom)
        
        # Time on edge: t_k = ||Δ_{k+1}|| / min(v_nom, v_max_k)
        t_k = edge_length / max(v_max_k, 1.0)
        
        # Jerk proxy: J_k = |θ_k - θ_{k-1}|
        J_k = abs(theta_k - self.prev_theta)
        self.prev_theta = theta_k
        
        # Combined cost from your spec
        cost = self.alpha * t_k + self.beta * theta_k + self.gamma * J_k
        
        # Terrain penalties
        if protected_bitmap[p_next[0], p_next[1]] == 1:
            cost *= 50.0  # Protected areas: very expensive but possible
        elif road_bitmap[p_next[0], p_next[1]] == 0:
            cost *= 5.0  # Off-road: more expensive but allows routing anywhere
        
        return cost


class APexSeeder:
    """Generate diverse A* seeds with different objective weights"""
    
    def __init__(self, road_bitmap, protected_bitmap, transform, 
                 start_pos, goal_pos):
        """
        Initialize seeder
        
        Args:
            road_bitmap: Road network raster
            protected_bitmap: Protected areas raster
            transform: Rasterio transform for coordinate conversion
            start_pos: Starting position (row, col)
            goal_pos: Goal position (row, col)
        """
        self.road_bitmap = road_bitmap
        self.protected_bitmap = protected_bitmap
        self.transform = transform
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.seeds = []
        
    def generate_weight_configs(self, n_configs=15):
        """
        Generate diverse weight combinations for α, β, γ
        
        Returns:
            List of weight dictionaries
        """
        configs = []
        
        # Pure objectives (corners of objective space)
        configs.append({'alpha': 1.0, 'beta': 0.0, 'gamma': 0.0, 'name': 'pure_time'})
        configs.append({'alpha': 0.0, 'beta': 1.0, 'gamma': 0.0, 'name': 'pure_turn'})
        configs.append({'alpha': 0.0, 'beta': 0.0, 'gamma': 1.0, 'name': 'pure_jerk'})
        
        # Pairwise combinations
        configs.append({'alpha': 0.7, 'beta': 0.3, 'gamma': 0.0, 'name': 'time_turn'})
        configs.append({'alpha': 0.7, 'beta': 0.0, 'gamma': 0.3, 'name': 'time_jerk'})
        configs.append({'alpha': 0.0, 'beta': 0.7, 'gamma': 0.3, 'name': 'turn_jerk'})
        
        # Balanced combinations (from your spec)
        configs.append({'alpha': 1.0, 'beta': 0.1, 'gamma': 0.05, 'name': 'default_balanced'})
        configs.append({'alpha': 1.0, 'beta': 0.05, 'gamma': 0.0, 'name': 'beta_0.05'})
        configs.append({'alpha': 1.0, 'beta': 0.2, 'gamma': 0.0, 'name': 'beta_0.2'})
        configs.append({'alpha': 1.0, 'beta': 0.1, 'gamma': 0.1, 'name': 'gamma_0.1'})
        
        # Random samples to increase diversity
        np.random.seed(42)
        while len(configs) < n_configs:
            weights = np.random.dirichlet([1, 1, 1])
            configs.append({
                'alpha': float(weights[0]),
                'beta': float(weights[1]),
                'gamma': float(weights[2]),
                'name': f'random_{len(configs)}'
            })
        
        return configs[:n_configs]
    
    def run_astar_with_weights(self, weights):
        """
        Run A* with specific objective weights
        
        Args:
            weights: Dictionary with 'alpha', 'beta', 'gamma', 'name'
            
        Returns:
            (path, objectives_dict) or (None, None) if no path found
        """
        objectives_calc = HyperloopObjectives(
            alpha=weights['alpha'],
            beta=weights['beta'],
            gamma=weights['gamma']
        )
        
        # A* implementation
        visited = set()
        came_from = {}
        g_score = {self.start_pos: 0.0}
        
        # Priority queue: (f_score, position)
        open_set = [(self._heuristic(self.start_pos), self.start_pos)]
        
        path_found = False
        iterations = 0
        max_iterations = 50000
        
        while open_set and iterations < max_iterations:
            iterations += 1
            _, current = heapq.heappop(open_set)
            
            if current == self.goal_pos:
                path_found = True
                break
            
            if current in visited:
                continue
                
            visited.add(current)
            
            # Get neighbors
            for neighbor in self._get_neighbors(current):
                if neighbor in visited:
                    continue
                
                # Get previous position for edge cost calculation
                if current in came_from:
                    prev = came_from[current]
                else:
                    prev = current
                
                # Calculate edge cost using PG/OE formulation
                edge_cost = objectives_calc.compute_edge_cost(
                    prev, current, neighbor,
                    self.road_bitmap, self.protected_bitmap
                )
                
                tentative_g = g_score[current] + edge_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor)
                    heapq.heappush(open_set, (f_score, neighbor))
        
        if not path_found:
            return None, None
        
        # Reconstruct path
        path = []
        current = self.goal_pos
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(self.start_pos)
        path = path[::-1]
        
        # Calculate final objectives
        objectives = self._calculate_objectives(path)
        objectives['iterations'] = iterations
        objectives['nodes_expanded'] = len(visited)
        
        return path, objectives
    
    def _get_neighbors(self, pos):
        """Get valid neighboring positions"""
        directions = [(0,1), (1,0), (0,-1), (-1,0), 
                     (1,1), (-1,-1), (1,-1), (-1,1)]
        neighbors = []
        
        for dx, dy in directions:
            new_pos = (pos[0] + dx, pos[1] + dy)
            
            # Check bounds
            if (0 <= new_pos[0] < self.road_bitmap.shape[0] and 
                0 <= new_pos[1] < self.road_bitmap.shape[1]):
                neighbors.append(new_pos)
        
        return neighbors
    
    def _heuristic(self, pos):
        """Euclidean distance heuristic"""
        return np.sqrt((pos[0] - self.goal_pos[0])**2 + 
                      (pos[1] - self.goal_pos[1])**2)
    
    def _calculate_objectives(self, path):
        """Calculate all objectives for a complete path"""
        if len(path) < 3:
            return {
                'distance': 0.0,
                'total_time': 0.0,
                'total_turn': 0.0,
                'total_jerk': 0.0
            }
        
        distance = 0.0
        total_time = 0.0
        total_turn = 0.0
        total_jerk = 0.0
        prev_theta = 0.0
        
        for i in range(1, len(path) - 1):
            p_prev = path[i-1]
            p_curr = path[i]
            p_next = path[i+1]
            
            # Distance
            edge_length = np.linalg.norm(np.array(p_next) - np.array(p_curr))
            distance += edge_length
            
            # Turn angle
            delta_k = np.array(p_curr) - np.array(p_prev)
            delta_k_plus_1 = np.array(p_next) - np.array(p_curr)
            
            dot_product = np.dot(delta_k, delta_k_plus_1)
            norm_product = np.linalg.norm(delta_k) * np.linalg.norm(delta_k_plus_1)
            
            if norm_product > 1e-6:
                cos_theta = np.clip(dot_product / norm_product, -1.0, 1.0)
                theta_k = np.arccos(cos_theta)
                total_turn += theta_k
                
                # Jerk
                jerk = abs(theta_k - prev_theta)
                total_jerk += jerk
                prev_theta = theta_k
                
                # Time (simplified)
                kappa_k = theta_k / (edge_length + 1e-6)
                v_max = np.sqrt(0.5 / max(kappa_k, 1e-6))
                v_max = min(v_max, 50.0)
                total_time += edge_length / max(v_max, 1.0)
        
        return {
            'distance': distance,
            'total_time': total_time,
            'total_turn': total_turn,
            'total_jerk': total_jerk
        }
    
    def generate_seeds(self, n_seeds=15):
        """Generate all seed solutions"""
        print(f"\n{'='*70}")
        print(f"GENERATING {n_seeds} A*PEX SEED SOLUTIONS")
        print(f"{'='*70}\n")
        
        configs = self.generate_weight_configs(n_seeds)
        results = []
        
        for i, weights in enumerate(configs):
            print(f"\nSeed {i+1}/{n_seeds}: {weights['name']}")
            print(f"  Weights: α={weights['alpha']:.3f}, β={weights['beta']:.3f}, γ={weights['gamma']:.3f}")
            
            start_time = time.time()
            path, objectives = self.run_astar_with_weights(weights)
            elapsed = time.time() - start_time
            
            if path is None:
                print(f"  ❌ No path found")
                continue
            
            print(f"  ✓ Path found ({len(path)} nodes)")
            print(f"  Distance: {objectives['distance']:.2f}")
            print(f"  Time: {objectives['total_time']:.2f}s")
            print(f"  Total Turn: {objectives['total_turn']:.2f} rad")
            print(f"  Total Jerk: {objectives['total_jerk']:.2f}")
            print(f"  Nodes Expanded: {objectives['nodes_expanded']}")
            print(f"  Computed in {elapsed:.3f}s")
            
            results.append({
                'seed_id': i,
                'weights': weights,
                'path': path,
                'objectives': objectives,
                'compute_time': elapsed
            })
        
        self.seeds = results
        return results
    
    def visualize_seeds(self, output_dir='data/output/apex_results'):
        """Visualize all seed paths with improved aesthetics"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        if not self.seeds:
            print("No seeds to visualize!")
            return
        
        # Plot all paths
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_facecolor('#111111') # Dark background
        
        # Show base map (Darkened)
        # Using a slight blue tint for roads to look sci-fi
        road_layer = np.zeros((*self.road_bitmap.shape, 4)) # RGBA
        road_layer[self.road_bitmap > 0] = [0.2, 0.3, 0.4, 0.5] # Dark Blue-ish roads
        ax.imshow(road_layer, origin='upper')
        
        # Show protected areas (Red tint overlay)
        if np.any(self.protected_bitmap):
            protected_layer = np.zeros((*self.protected_bitmap.shape, 4))
            protected_layer[self.protected_bitmap > 0] = [0.8, 0.2, 0.2, 0.3]
            ax.imshow(protected_layer, origin='upper')
        
        # Plot each seed path
        # Color based on Strategy: Red = FAST (High Alpha), Blue = SMOOTH (Low Alpha/High Beta)
        for seed in self.seeds:
            path = seed['path']
            weights = seed['weights']
            
            # Simple color interpolation
            # Alpha 1.0 -> Red
            # Alpha 0.0 -> Blue
            alpha_val = weights.get('alpha', 0.5)
            color = plt.cm.coolwarm(alpha_val)
            
            xs = [p[1] for p in path]
            ys = [p[0] for p in path]
            
            # Draw line
            ax.plot(xs, ys, color=color, alpha=0.6, linewidth=1.5)
            
        # Draw Start/Goal
        ax.scatter(self.start_pos[1], self.start_pos[0], c='#00FF00', s=200, marker='*', label='Start', zorder=10)
        ax.scatter(self.goal_pos[1], self.goal_pos[0], c='#FF00FF', s=200, marker='*', label='Goal', zorder=10)
        
        # Custom Legend
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color=plt.cm.coolwarm(1.0), lw=2),
            Line2D([0], [0], color=plt.cm.coolwarm(0.0), lw=2)
        ]
        ax.legend(custom_lines, ['Fast (High Speed)', 'Smooth (High Comfort)'], loc='upper right', fontsize=10)
        
        ax.set_title('A*pex Seed Solutions: Hyperloop Corridors\nColor indicates Strategy', color='white', fontsize=14)
        ax.tick_params(colors='white')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/all_seed_paths.png', dpi=150, facecolor='#111111')
        print(f"\n✓ Saved enhanced path visualization to {output_dir}/all_seed_paths.png")
        
        # Plot Pareto front
        self._plot_pareto_front(output_dir)
    
    def _plot_pareto_front(self, output_dir):
        """Plot 2D projections of Pareto front"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        objectives_pairs = [
            ('total_time', 'total_turn', 'Time (s)', 'Total Turn (rad)'),
            ('total_time', 'total_jerk', 'Time (s)', 'Total Jerk'),
            ('total_time', 'distance', 'Time (s)', 'Distance'),
            ('total_turn', 'total_jerk', 'Total Turn (rad)', 'Total Jerk'),
            ('total_turn', 'distance', 'Total Turn (rad)', 'Distance'),
            ('total_jerk', 'distance', 'Total Jerk', 'Distance'),
        ]
        
        for ax, (obj1, obj2, label1, label2) in zip(axes.flat, objectives_pairs):
            x = [s['objectives'][obj1] for s in self.seeds]
            y = [s['objectives'][obj2] for s in self.seeds]
            
            ax.scatter(x, y, s=100, alpha=0.6)
            ax.set_xlabel(label1, fontsize=12)
            ax.set_ylabel(label2, fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Annotate points
            for i, (xi, yi) in enumerate(zip(x, y)):
                ax.annotate(f"{i+1}", (xi, yi), fontsize=8, alpha=0.7)
        
        plt.suptitle('A*pex Seeds - Objective Space Projections', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/pareto_projections.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved Pareto projections to {output_dir}/pareto_projections.png")
    
    def save_seeds(self, output_path='data/output/apex_results/apex_seeds.json'):
        """Save seeds to JSON"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        serializable = []
        for seed in self.seeds:
            serializable.append({
                'seed_id': seed['seed_id'],
                'weights': seed['weights'],
                'objectives': seed['objectives'],
                'compute_time': seed['compute_time'],
                'path_length': len(seed['path'])
            })
        
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        print(f"\n✓ Seeds saved to {output_path}")


def load_hypernet_data(use_liechtenstein=True, npz_path=None):
    """
    Load data from Hypernet repository.
    Supports loading from cached .npz files for speed.
    
    Args:
        use_liechtenstein: If True, use Liechtenstein test data.
        npz_path: Path to cached .npz file (optional). 
                 If provided and exists -> load from it.
                 If provided and missing -> generate and save to it.
    
    Returns:
        (road_bitmap, protected_bitmap, transform, start_pos, goal_pos)
    """
    
    # 1. Try loading from NPZ if it exists
    if npz_path and os.path.exists(npz_path):
        print(f"Loading cached map data from {npz_path}...")
        try:
            data = np.load(npz_path, allow_pickle=True)
            road_bitmap = data['road_bitmap']
            protected_bitmap = data['protected_bitmap']
            transform = data['transform'] # Rasterio transform might need reconstruction depending on how numpy saves it
            # Transform is usually an Affine object, need to ensure it restores correctly or is saved as array
            # Simple hack: assume it saves/loads as object array or similar. 
            # Reconstructing Affine from tuple/list if needed:
            if isinstance(transform, np.ndarray):
               from affine import Affine
               transform = Affine(*transform)

            start_pos = tuple(data['start_pos'])
            goal_pos = tuple(data['goal_pos'])
            
            print(f"  ✓ Loaded road network: {road_bitmap.shape} raster (Cached)")
            return road_bitmap, protected_bitmap, transform, start_pos, goal_pos
        except Exception as e:
            print(f"  ✗ Failed to load cache: {e}. Regenerating...")

    # 2. Fallback to Generation (Existing Logic)
    if use_liechtenstein:
        # Liechtenstein test data (smaller, faster)
        road_path = "./data/output/LI/LI_roads.geojson"
        protected_path = "./data/output/LI/protected_areas.geojson"
        
        # Goal points from your existing code
        point_map = {
            "Eschen": [9.518061467906222, 47.21094022078131],
            "Balzers": [9.497583388657375, 47.066238854375655]
        }
        
        print("Loading Liechtenstein data (Generating from GeoJSON)...")
    else:
        # Texas data
        road_path = "./data/output/TX/austin_dallas.geojson"
        protected_path = "./data/austin-dallas/envr_austin_dallas.geojson"
        
        point_map = {
            "Austin": [-97.73967335952074, 30.277001707654122],
            "Dallas": [-96.7970, 32.7767]
        }
        
        print("Loading Texas data (Generating from GeoJSON)...")
    
    # Load road network
    with open(road_path) as f:
        geojson = json.load(f)
    
    geometries = [(feature["geometry"], 1) for feature in geojson["features"]]
    
    # Define raster dimensions
    width, height = 500, 500
    minx, miny, maxx, maxy = rasterio.features.bounds(geojson)
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    
    # Rasterize roads
    road_bitmap = rasterize(
        geometries, 
        out_shape=(height, width), 
        transform=transform, 
        fill=0, 
        dtype=np.uint8
    )
    
    # Load and rasterize protected areas
    try:
        protected_gdf = gpd.read_file(protected_path)
        protected_bitmap = rasterize(
            ((geom, 1) for geom in protected_gdf.geometry),
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )
    except Exception as e:
        print(f"  ! Warning: Could not load protected areas ({e})")
        print("    Proceeding with empty protection map.")
        protected_bitmap = np.zeros((height, width), dtype=np.uint8)
    
    # Convert start and goal to pixel coordinates
    start_name = list(point_map.keys())[0]
    goal_name = list(point_map.keys())[1]
    
    start_lon, start_lat = point_map[start_name]
    goal_lon, goal_lat = point_map[goal_name]
    
    start_col, start_row = ~transform * (start_lon, start_lat)
    start_pos = (int(round(start_row)), int(round(start_col)))
    
    goal_col, goal_row = ~transform * (goal_lon, goal_lat)
    goal_pos = (int(round(goal_row)), int(round(goal_col)))
    
    print(f"  ✓ Loaded road network: {height}x{width} raster")
    print(f"  ✓ Loaded protected areas")
    print(f"  ✓ Start: {start_name} {start_pos}")
    print(f"  ✓ Goal: {goal_name} {goal_pos}")

    # 3. Save to NPZ Cache if requested
    if npz_path:
        # Convert Affine to tuple for saving
        # transform is an Affine object, we can save its coefficients
        # But wait, numpy won't save Affine objects easily in savez without pickle allow=True which IS allowed above but ideally avoid.
        # Let's save as list of 6 or 9 coefficients or just use pickle.
        try:
            np.savez(npz_path, 
                     road_bitmap=road_bitmap, 
                     protected_bitmap=protected_bitmap, 
                     transform=np.array(transform), 
                     start_pos=np.array(start_pos), 
                     goal_pos=np.array(goal_pos))
            print(f"  ✓ Saved cache to {npz_path}")
        except Exception as e:
            print(f"  ! Warning: Could not save cache: {e}")
    
    return road_bitmap, protected_bitmap, transform, start_pos, goal_pos


def main():
    """Main execution"""
    print("="*70)
    print("HYPERNET A*PEX SEEDING TEST")
    print("Multi-Objective Hyperloop Route Optimization")
    print("="*70)
    
    # Load data
    road_bitmap, protected_bitmap, transform, start_pos, goal_pos = \
        load_hypernet_data(use_liechtenstein=True)
    
    # Initialize seeder
    seeder = APexSeeder(
        road_bitmap, 
        protected_bitmap, 
        transform,
        start_pos, 
        goal_pos
    )
    
    # Generate seeds
    seeds = seeder.generate_seeds(n_seeds=10)
    
    # Visualize
    seeder.visualize_seeds()
    
    # Save
    seeder.save_seeds()
    
    print("\n" + "="*70)
    print("A*PEX SEEDING COMPLETE!")
    print("="*70)
    print(f"\nGenerated {len(seeds)} diverse seed solutions")
    print("\nNext steps:")
    print("1. Use these seeds to initialize EMO (NSGA-III)")
    print("2. Compare with random initialization")
    print("3. Analyze convergence speed and final Pareto front quality")


if __name__ == '__main__':
    main()