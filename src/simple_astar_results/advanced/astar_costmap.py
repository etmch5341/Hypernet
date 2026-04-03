#!/usr/bin/env python3
"""
A* Pathfinding using real cost maps (construction, environmental, geometry).

Loads pre-computed cost maps from npz-files/ and the Austin road raster,
then runs A* where the step cost is a weighted combination of the three
cost layers instead of the simple on-road/off-road binary.

Usage:
    python astar_costmap.py
    python astar_costmap.py --output ./my_output --no-animation
"""

import argparse
import numpy as np
import heapq
import math
import time
import os
import pickle
from astar_animator import AStarAnimator


# -------------------- CONFIG --------------------
ALLOW_DIAGONAL = True
FRAME_RECORD_INTERVAL = 5000
MAX_EXPANSIONS = 5_000_000

# Weights for combining cost maps (sum to 1.0)
W_CONSTRUCTION = 0.40
W_ENVIRONMENTAL = 0.35
W_GEOMETRY = 0.25

# Base cost multiplier so cells aren't free
BASE_COST = 0.1

# Animation settings
ANIMATION_FPS = 15
ANIMATION_FRAMES = 150
# ------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
NPZ_DIR = os.path.join(PROJECT_ROOT, "npz-files")
RASTER_PATH = os.path.join(PROJECT_ROOT, "src", "sample-test-set", "austin_test_raster.npz")


def load_cost_maps():
    """Load and normalize the three cost maps."""
    print("Loading cost maps...")

    # Construction cost (use normalized version, already 0-1)
    d = np.load(os.path.join(NPZ_DIR, "austin_construction_cost.npz"), allow_pickle=True)
    construction = d['cost_map_normalized'].astype(np.float64)
    print(f"  Construction: shape={construction.shape}, range=[{construction.min():.4f}, {construction.max():.4f}]")

    # Environmental impact (use normalized version, already 0-1)
    d = np.load(os.path.join(NPZ_DIR, "austin_environmental_impact.npz"), allow_pickle=True)
    environmental = d['env_map_normalized'].astype(np.float64)
    print(f"  Environmental: shape={environmental.shape}, range=[{environmental.min():.4f}, {environmental.max():.4f}]")

    # Geometry cost (already 0-1)
    d = np.load(os.path.join(NPZ_DIR, "geometry_cost_map.npz"), allow_pickle=True)
    geometry = d['cost_map'].astype(np.float64)
    print(f"  Geometry: shape={geometry.shape}, range=[{geometry.min():.4f}, {geometry.max():.4f}]")

    return construction, environmental, geometry


def build_combined_cost_map(construction, environmental, geometry):
    """Build a single combined cost map from weighted layers."""
    combined = (W_CONSTRUCTION * construction +
                W_ENVIRONMENTAL * environmental +
                W_GEOMETRY * geometry +
                BASE_COST)
    print(f"\n  Combined cost map: range=[{combined.min():.4f}, {combined.max():.4f}]")
    return combined


def load_raster_and_goals():
    """Load road raster and goal points."""
    print(f"\nLoading road raster from: {RASTER_PATH}")
    data = np.load(RASTER_PATH, allow_pickle=True)
    raster = data['raster']
    goal_points = data['goal_points']

    metadata = {
        'width': int(data['width']),
        'height': int(data['height']),
        'bbox': data['bbox'].tolist() if 'bbox' in data else None,
        'target_crs': str(data['target_crs']) if 'target_crs' in data else None,
    }

    print(f"  Raster shape: {raster.shape}")
    goals = []
    for x, y, name in goal_points:
        goals.append((int(y), int(x)))
        print(f"  Goal '{name}': pixel ({x},{y}) -> (row={int(y)}, col={int(x)})")

    return raster, goals, metadata


def mst_cost(points):
    if not points:
        return 0.0
    pts = list(points)
    n = len(pts)
    used = [False] * n
    dist = [float('inf')] * n
    dist[0] = 0.0
    total = 0.0
    for _ in range(n):
        u = -1
        best = float('inf')
        for i in range(n):
            if not used[i] and dist[i] < best:
                best = dist[i]
                u = i
        used[u] = True
        total += dist[u]
        for v in range(n):
            if not used[v]:
                d = math.hypot(pts[u][0] - pts[v][0], pts[u][1] - pts[v][1])
                if d < dist[v]:
                    dist[v] = d
    return total


def heuristic(curr, goals, visited_mask):
    remaining = [g for i, g in enumerate(goals) if not (visited_mask & (1 << i))]
    if not remaining:
        return 0.0
    min_to_goal = min(math.hypot(g[0] - curr[0], g[1] - curr[1]) for g in remaining)
    mst = mst_cost(remaining)
    return (min_to_goal + mst) * BASE_COST  # scale heuristic by minimum possible cost


def update_goal_bitmask(pos, bitmask, goal_positions):
    for i, g in enumerate(goal_positions):
        if pos == g:
            bitmask |= (1 << i)
    return bitmask


def is_goal_state(bitmask, total_goals):
    return bitmask == (1 << total_goals) - 1


def get_successors(pos, combined_cost_map, allow_diagonal=True):
    """Generate successors using the combined cost map for step costs."""
    if allow_diagonal:
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                      (1, 1), (-1, -1), (-1, 1), (1, -1)]
    else:
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    h, w = combined_cost_map.shape
    for dx, dy in directions:
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < h and 0 <= ny < w:
            base_distance = math.hypot(dx, dy)
            terrain_cost = combined_cost_map[nx, ny]
            yield ((nx, ny), base_distance * terrain_cost)


def reconstruct_path(came_from, current_state, start_state):
    path = []
    s = current_state
    while s != start_state:
        path.append(s[0])
        s = came_from.get(s)
        if s is None:
            return None
    path.append(start_state[0])
    return path[::-1]


def a_star_costmap(start, goals, combined_cost_map,
                   frame_interval=FRAME_RECORD_INTERVAL,
                   max_expansions=MAX_EXPANSIONS):
    """A* using combined cost map."""
    start_mask = update_goal_bitmask(start, 0, goals)
    total_goals = len(goals)

    heap = []
    start_h = heuristic(start, goals, start_mask)
    heapq.heappush(heap, (start_h, 0.0, start, start_mask))

    best_g = {}
    came_from = {}
    frames_sparse = []
    visited_snapshot = []
    expansions = 0
    start_time = time.time()
    start_state = (start, start_mask)

    print("\nRunning A* with cost-map terrain costs...")

    while heap:
        f, g, current, mask = heapq.heappop(heap)
        state = (current, mask)

        if best_g.get(state, float('inf')) <= g:
            continue
        best_g[state] = g

        visited_snapshot.append(current)
        expansions += 1

        if frame_interval and (expansions % frame_interval == 0):
            frames_sparse.append((expansions, visited_snapshot.copy()))
            visited_snapshot.clear()
            print(f"  Expansions: {expansions:,} | Queue size: {len(heap):,}")

        if max_expansions is not None and expansions > max_expansions:
            print(f"\nReached max expansions ({max_expansions:,}).")
            break

        if is_goal_state(mask, total_goals):
            elapsed = time.time() - start_time
            if visited_snapshot:
                frames_sparse.append((expansions, visited_snapshot.copy()))
                visited_snapshot.clear()
            path = reconstruct_path(came_from, (current, mask), start_state)
            print(f"\nPath found!")
            print(f"  Expansions: {expansions:,}")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Path length: {len(path)} waypoints")
            print(f"  Total cost: {g:.2f}")
            return path, frames_sparse

        for neighbor, step_cost in get_successors(current, combined_cost_map):
            new_g = g + step_cost
            new_mask = update_goal_bitmask(neighbor, mask, goals)
            neigh_state = (neighbor, new_mask)
            if best_g.get(neigh_state, float('inf')) <= new_g:
                continue
            h = heuristic(neighbor, goals, new_mask)
            heapq.heappush(heap, (new_g + h, new_g, neighbor, new_mask))
            came_from[neigh_state] = state

    print("\nNo path found (or search aborted).")
    if visited_snapshot:
        frames_sparse.append((expansions, visited_snapshot.copy()))
    return None, frames_sparse


def save_output_data(output_dir, road_bitmap, path, frames_sparse, metadata):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving data to {output_dir}...")

    np.savez_compressed(os.path.join(output_dir, "road_bitmap.npz"),
                        road_bitmap=np.asarray(road_bitmap, dtype=np.uint8))
    protected_bitmap = np.zeros_like(road_bitmap, dtype=np.uint8)
    np.savez_compressed(os.path.join(output_dir, "protected_bitmap.npz"),
                        protected_bitmap=protected_bitmap)

    from rasterio.transform import from_bounds
    h, w = road_bitmap.shape
    transform = from_bounds(0, 0, w, h, w, h)
    bounds = (0, 0, w, h)
    meta = {
        "transform": transform,
        "bounds": bounds,
        "original_metadata": metadata,
        "algorithm": "A* (cost-map weighted)",
    }
    with open(os.path.join(output_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    with open(os.path.join(output_dir, "astar_sparse_frames.pkl"), "wb") as f:
        pickle.dump(frames_sparse, f, protocol=pickle.HIGHEST_PROTOCOL)

    if path:
        path_arr = np.array(path, dtype=np.int32)
    else:
        path_arr = np.zeros((0, 2), dtype=np.int32)
    np.save(os.path.join(output_dir, "astar_final_path.npy"), path_arr)
    print("Data saved successfully!")


def save_cost_map_visualization(output_dir, construction, environmental, geometry, combined, path):
    """Save a 4-panel visualization of cost layers + path overlay."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    titles = ['Construction Cost', 'Environmental Impact', 'Geometry Cost', 'Combined Cost + Path']
    maps = [construction, environmental, geometry, combined]
    cmaps_list = ['YlOrRd', 'Greens', 'Blues', 'hot']

    for ax, title, data, cmap in zip(axes.flat, titles, maps, cmaps_list):
        im = ax.imshow(data, cmap=cmap, origin='upper')
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Overlay path on combined panel
    if path:
        path_arr = np.array(path)
        axes[1, 1].plot(path_arr[:, 1], path_arr[:, 0], 'c-', linewidth=1.5, alpha=0.9, label='A* path')
        axes[1, 1].plot(path_arr[0, 1], path_arr[0, 0], 'go', markersize=10, label='Start')
        axes[1, 1].plot(path_arr[-1, 1], path_arr[-1, 0], 'r*', markersize=12, label='End')
        axes[1, 1].legend(loc='upper right', fontsize=9)

    plt.suptitle(f'A* Pathfinding with Real Cost Maps\n'
                 f'Weights: construction={W_CONSTRUCTION}, environmental={W_ENVIRONMENTAL}, geometry={W_GEOMETRY}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_dir, "astar_costmap_layers.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Cost map visualization: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="A* pathfinding with real cost maps")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: ./astar_costmap_austin)")
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--fps", type=int, default=ANIMATION_FPS)
    parser.add_argument("--frames", type=int, default=ANIMATION_FRAMES)
    parser.add_argument("--max-expansions", type=int, default=MAX_EXPANSIONS)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(__file__), "astar_costmap_austin")

    print("=" * 70)
    print("A* Pathfinding with Real Cost Maps")
    print("=" * 70)

    construction, environmental, geometry = load_cost_maps()
    combined = build_combined_cost_map(construction, environmental, geometry)
    raster, goals, metadata = load_raster_and_goals()

    start = goals[0]
    print(f"\nStart: {start}, Goals: {goals}")
    print(f"Weights: construction={W_CONSTRUCTION}, environmental={W_ENVIRONMENTAL}, geometry={W_GEOMETRY}")

    print("\n" + "=" * 70)
    path, frames_sparse = a_star_costmap(start, goals, combined,
                                         max_expansions=args.max_expansions)
    print("=" * 70)

    save_output_data(args.output, raster, path, frames_sparse, metadata)
    save_cost_map_visualization(args.output, construction, environmental, geometry, combined, path)

    if not args.no_animation and path is not None:
        print("\nCreating animations...")
        try:
            animator = AStarAnimator(output_dir=args.output)
            animator.create_animation(output_file="astar_costmap_animation.gif",
                                      fps=args.fps, target_frames=args.frames,
                                      show_path=True, figsize=(12, 10), dpi=100)
            animator.create_static_comparison(output_file="astar_costmap_comparison.png",
                                              figsize=(18, 6), dpi=150)
            print(f"  Animation: {args.output}/astar_costmap_animation.gif")
            print(f"  Comparison: {args.output}/astar_costmap_comparison.png")
        except Exception as e:
            print(f"Error creating animations: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
