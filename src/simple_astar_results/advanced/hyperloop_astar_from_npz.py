#!/usr/bin/env python3
"""
A* Pathfinding from NPZ Test Sets
Loads raster data from NPZ files, runs A* pathfinding, and generates animations.

Usage:
    python astar_from_npz.py --input austin_test_raster.npz
    python astar_from_npz.py --input seattle_test_raster.npz --output ./output/seattle
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
# Whether to allow diagonal moves
ALLOW_DIAGONAL = True

# How often to record a sparse "frame" of visited points
FRAME_RECORD_INTERVAL = 5000

# Cost multipliers for different terrain
ON_ROAD_COST = 1.0      # Cost when on road (value = 1 in raster)
OFF_ROAD_COST = 5.0     # Cost when off road (possible but expensive)

# Maximum nodes to expand as a safety cap
MAX_EXPANSIONS = 5_000_000

# Animation settings
CREATE_ANIMATION = True
ANIMATION_FPS = 15
ANIMATION_FRAMES = 150
# ------------------------------------------------


def load_npz_test_set(npz_path):
    """
    Load test set from NPZ file.
    
    Returns:
        raster: 2D numpy array with road network (1=road, 0=off-road)
        goal_points: list of (x, y, name) tuples in pixel coordinates
        metadata: dict with width, height, bbox, target_crs
    """
    print(f"Loading test set from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    
    raster = data['raster']
    goal_points = data['goal_points']
    
    metadata = {
        'width': int(data['width']),
        'height': int(data['height']),
        'bbox': data['bbox'].tolist() if 'bbox' in data else None,
        'target_crs': str(data['target_crs']) if 'target_crs' in data else None
    }
    
    print(f"  Raster shape: {raster.shape}")
    print(f"  Goal points: {len(goal_points)}")
    for x, y, name in goal_points:
        print(f"    {name}: pixel ({x}, {y})")
    
    return raster, goal_points, metadata


def mst_cost(points):
    """Calculate minimum spanning tree cost for heuristic."""
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
    """
    Multi-goal heuristic using MST.
    
    Args:
        curr: current position (row, col)
        goals: list of goal positions
        visited_mask: bitmask of visited goals
    """
    remaining = [g for i, g in enumerate(goals) if not (visited_mask & (1 << i))]
    if not remaining:
        return 0.0
    min_to_goal = min(math.hypot(g[0] - curr[0], g[1] - curr[1]) for g in remaining)
    mst = mst_cost(remaining)
    return min_to_goal + mst


def update_goal_bitmask(pos, bitmask, goal_positions):
    """Update bitmask when reaching a goal."""
    for i, g in enumerate(goal_positions):
        if pos == g:
            bitmask |= (1 << i)
    return bitmask


def is_goal_state(bitmask, total_goals):
    """Check if all goals have been visited."""
    return bitmask == (1 << total_goals) - 1


def get_successors(pos, road_bitmap, allow_diagonal=ALLOW_DIAGONAL):
    """
    Generate valid successor positions with costs.
    
    Args:
        pos: current position (row, col)
        road_bitmap: 2D array where 1=road, 0=off-road
        allow_diagonal: whether to allow diagonal moves
    
    Yields:
        (neighbor_pos, step_cost) tuples
    """
    if allow_diagonal:
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), 
                     (1, 1), (-1, -1), (-1, 1), (1, -1)]
    else:
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    h, w = road_bitmap.shape
    for dx, dy in directions:
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < h and 0 <= ny < w:
            # Calculate distance (Euclidean for diagonals)
            base_distance = math.hypot(dx, dy)
            
            # Apply terrain cost multiplier
            terrain_cost = ON_ROAD_COST if road_bitmap[nx, ny] == 1 else OFF_ROAD_COST
            
            yield ((nx, ny), base_distance * terrain_cost)


def reconstruct_path(came_from, current_state, start_state):
    """Reconstruct path from start to current state."""
    path = []
    s = current_state
    while s != start_state:
        path.append(s[0])
        s = came_from.get(s)
        if s is None:
            return None
    path.append(start_state[0])
    return path[::-1]


def a_star_multi(start, goals, road_bitmap, frame_interval=FRAME_RECORD_INTERVAL, 
                 max_expansions=MAX_EXPANSIONS):
    """
    Multi-goal A* pathfinding with animation frame collection.
    
    Args:
        start: starting position (row, col)
        goals: list of goal positions [(row, col), ...]
        road_bitmap: 2D array where 1=road, 0=off-road
        frame_interval: how often to save animation frames
        max_expansions: maximum nodes to expand before aborting
    
    Returns:
        path: list of waypoints from start to all goals (or None)
        frames_sparse: list of (expansion_count, visited_positions) for animation
    """
    # Initialize start state
    start_mask = update_goal_bitmask(start, 0, goals)
    total_goals = len(goals)

    # Priority queue: (f_score, g_score, position, goal_mask)
    heap = []
    start_h = heuristic(start, goals, start_mask)
    heapq.heappush(heap, (start_h, 0.0, start, start_mask))

    # Track best g-scores and parent pointers
    best_g = {}
    came_from = {}
    
    # Animation data
    frames_sparse = []
    visited_snapshot = []

    expansions = 0
    start_time = time.time()
    start_state = (start, start_mask)

    print("\nRunning A* search...")
    
    while heap:
        f, g, current, mask = heapq.heappop(heap)
        state = (current, mask)

        # Skip if we've found a better path to this state
        if best_g.get(state, float('inf')) <= g:
            continue

        best_g[state] = g

        # Record visited position for animation
        visited_snapshot.append(current)
        expansions += 1
        
        if frame_interval and (expansions % frame_interval == 0):
            frames_sparse.append((expansions, visited_snapshot.copy()))
            visited_snapshot.clear()
            print(f"  Expansions: {expansions:,} | Queue size: {len(heap):,}")

        # Check expansion limit
        if max_expansions is not None and expansions > max_expansions:
            print(f"\nReached max expansions ({max_expansions:,}). Aborting search.")
            break

        # Check if we've visited all goals
        if is_goal_state(mask, total_goals):
            elapsed = time.time() - start_time
            if visited_snapshot:
                frames_sparse.append((expansions, visited_snapshot.copy()))
                visited_snapshot.clear()
            
            final_state = (current, mask)
            path = reconstruct_path(came_from, final_state, start_state)
            
            print(f"\n✓ Path found!")
            print(f"  Expansions: {expansions:,}")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Path length: {len(path)} waypoints")
            
            return path, frames_sparse

        # Expand neighbors
        for neighbor, step_cost in get_successors(current, road_bitmap):
            new_g = g + step_cost
            new_mask = update_goal_bitmask(neighbor, mask, goals)
            neigh_state = (neighbor, new_mask)

            # Skip if we've found a better path to this neighbor state
            if best_g.get(neigh_state, float('inf')) <= new_g:
                continue

            # Add to queue
            h = heuristic(neighbor, goals, new_mask)
            heapq.heappush(heap, (new_g + h, new_g, neighbor, new_mask))
            came_from[neigh_state] = state

    # No path found
    print("\n✗ No path found (or search aborted).")
    if visited_snapshot:
        frames_sparse.append((expansions, visited_snapshot.copy()))
    
    return None, frames_sparse


def save_output_data(output_dir, road_bitmap, path, frames_sparse, metadata):
    """Save all data needed for animation."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving data to {output_dir}...")
    
    # Save road bitmap (protected bitmap not needed for basic version)
    np.savez_compressed(
        os.path.join(output_dir, "road_bitmap.npz"),
        road_bitmap=np.asarray(road_bitmap, dtype=np.uint8)
    )
    
    # Create empty protected bitmap (for compatibility with animator)
    protected_bitmap = np.zeros_like(road_bitmap, dtype=np.uint8)
    np.savez_compressed(
        os.path.join(output_dir, "protected_bitmap.npz"),
        protected_bitmap=protected_bitmap
    )
    
    # Save metadata (dummy transform and bounds for animator compatibility)
    from rasterio.transform import from_bounds
    h, w = road_bitmap.shape
    transform = from_bounds(0, 0, w, h, w, h)
    bounds = (0, 0, w, h)
    
    meta = {
        "transform": transform,
        "bounds": bounds,
        "original_metadata": metadata
    }
    with open(os.path.join(output_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    
    # Save sparse frames
    with open(os.path.join(output_dir, "astar_sparse_frames.pkl"), "wb") as f:
        pickle.dump(frames_sparse, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save final path
    if path:
        path_arr = np.array(path, dtype=np.int32)
    else:
        path_arr = np.zeros((0, 2), dtype=np.int32)
    np.save(os.path.join(output_dir, "astar_final_path.npy"), path_arr)
    
    print("✓ Data saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Run A* pathfinding on NPZ test sets with animation"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to NPZ test set file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: ./astar_output_<testname>)"
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="Skip creating animations"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=ANIMATION_FPS,
        help=f"Animation FPS (default: {ANIMATION_FPS})"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=ANIMATION_FRAMES,
        help=f"Animation frame count (default: {ANIMATION_FRAMES})"
    )
    parser.add_argument(
        "--max-expansions",
        type=int,
        default=MAX_EXPANSIONS,
        help=f"Maximum node expansions (default: {MAX_EXPANSIONS:,})"
    )
    parser.add_argument(
        "--no-diagonal",
        action="store_true",
        help="Disable diagonal movement"
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output is None:
        test_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"./astar_output_{test_name}"
    
    print("=" * 70)
    print("A* Pathfinding from NPZ Test Set")
    print("=" * 70)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Max expansions: {args.max_expansions:,}")
    print(f"Diagonal movement: {'disabled' if args.no_diagonal else 'enabled'}")
    print("=" * 70)
    
    # Load test set
    raster, goal_points, metadata = load_npz_test_set(args.input)
    
    # Convert goal points from (x, y, name) to (row, col)
    # Note: x is column, y is row in image coordinates
    goals = []
    for x, y, name in goal_points:
        goals.append((int(y), int(x)))  # Convert to (row, col)
    
    if len(goals) < 2:
        print("\n✗ Error: Need at least 2 goals for multi-goal routing")
        return 1
    
    # Use first goal as start
    start = goals[0]
    print(f"\nStarting from: {start}")
    print(f"Total goals: {len(goals)}")
    
    # Run A* search
    print("\n" + "=" * 70)
    allow_diagonal = not args.no_diagonal
    path, frames_sparse = a_star_multi(
        start, goals, raster,
        frame_interval=FRAME_RECORD_INTERVAL,
        max_expansions=args.max_expansions
    )
    print("=" * 70)
    
    # Save output data
    save_output_data(args.output, raster, path, frames_sparse, metadata)
    
    # Create animations
    if not args.no_animation and path is not None:
        print("\n" + "=" * 70)
        print("Creating animations...")
        print("=" * 70)
        
        try:
            animator = AStarAnimator(output_dir=args.output)
            
            # Create animated GIF
            animator.create_animation(
                output_file="astar_animation.gif",
                fps=args.fps,
                target_frames=args.frames,
                show_path=True,
                figsize=(12, 10),
                dpi=100
            )
            
            # Create static comparison
            animator.create_static_comparison(
                output_file="astar_comparison.png",
                figsize=(18, 6),
                dpi=150
            )
            
            print("\n✓ Animations created successfully!")
            print(f"  - Animation: {args.output}/astar_animation.gif")
            print(f"  - Comparison: {args.output}/astar_comparison.png")
            
        except Exception as e:
            print(f"\n✗ Error creating animations: {e}")
            import traceback
            traceback.print_exc()
            print("  (Data was still saved successfully)")
    
    print("\n" + "=" * 70)
    print("Process complete!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())