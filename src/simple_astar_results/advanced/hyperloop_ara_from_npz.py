#!/usr/bin/env python3
"""
ARA* (Anytime Repairing A*) Pathfinding from NPZ Test Sets
Loads raster data from NPZ files, runs ARA* pathfinding, and generates animations.

ARA* is an anytime algorithm that:
1. Quickly finds an initial suboptimal solution using inflated heuristics
2. Iteratively improves the solution by decreasing the inflation factor
3. Provides bounded suboptimality guarantees at any interruption point

Usage:
    python hyperloop_ara_from_npz.py --input austin_test_raster.npz
    python hyperloop_ara_from_npz.py --input seattle_test_raster.npz --output ./output/seattle
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
OFF_ROAD_COST = 5.0     # Cost when off road (value = 0 in raster)

# Maximum nodes to expand as a safety cap (per iteration)
MAX_EXPANSIONS_PER_ITER = 2_000_000

# Animation settings
CREATE_ANIMATION = True
ANIMATION_FPS = 15
ANIMATION_FRAMES = 150

# ARA* specific settings
INITIAL_EPSILON = 3.0      # Starting inflation factor (higher = faster initial solution)
EPSILON_DECREMENT = 0.5    # How much to reduce epsilon each iteration
FINAL_EPSILON = 1.0        # Stop when epsilon reaches this (1.0 = optimal)
MAX_ITERATIONS = 10        # Maximum refinement iterations
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


def compute_path_cost(path, road_bitmap):
    """Compute the total cost of a path."""
    if path is None or len(path) < 2:
        return float('inf')

    total_cost = 0.0
    for i in range(len(path) - 1):
        curr = path[i]
        next_pos = path[i + 1]
        dx = next_pos[0] - curr[0]
        dy = next_pos[1] - curr[1]
        base_distance = math.hypot(dx, dy)
        terrain_cost = ON_ROAD_COST if road_bitmap[next_pos[0], next_pos[1]] == 1 else OFF_ROAD_COST
        total_cost += base_distance * terrain_cost

    return total_cost


def improve_path(start, goals, road_bitmap, epsilon, best_g, came_from,
                 open_set, incons, closed_set, incumbent_cost,
                 frame_interval, max_expansions, visited_snapshot, total_expansions):
    """
    Run one iteration of weighted A* search with given epsilon.

    This is the core of ARA* - it performs weighted A* but:
    1. Reuses g-values from previous iterations
    2. Tracks inconsistent states in INCONS
    3. Prunes based on incumbent solution cost

    Args:
        start: starting position (row, col)
        goals: list of goal positions
        road_bitmap: 2D array where 1=road, 0=off-road
        epsilon: current inflation factor
        best_g: dictionary of best g-scores (persists across iterations)
        came_from: parent pointers (persists across iterations)
        open_set: priority queue of states to explore
        incons: set of inconsistent states
        closed_set: set of expanded states this iteration
        incumbent_cost: cost of best solution found so far
        frame_interval: how often to save animation frames
        max_expansions: maximum nodes to expand this iteration
        visited_snapshot: list to accumulate visited positions
        total_expansions: total expansions across all iterations

    Returns:
        path: path if goal found, None otherwise
        frames_sparse: list of (expansion_count, visited_positions)
        total_expansions: updated expansion count
        goal_state: the goal state if found
    """
    total_goals = len(goals)
    start_mask = update_goal_bitmask(start, 0, goals)
    start_state = (start, start_mask)

    frames_sparse = []
    expansions_this_iter = 0

    while open_set:
        f, g, current, mask = heapq.heappop(open_set)
        state = (current, mask)

        # Skip if already expanded this iteration
        if state in closed_set:
            continue

        # Pruning: skip if f-score exceeds incumbent
        if incumbent_cost < float('inf'):
            h = heuristic(current, goals, mask)
            if g + h >= incumbent_cost:
                continue

        closed_set.add(state)

        # Record visited position for animation
        visited_snapshot.append(current)
        expansions_this_iter += 1
        total_expansions += 1

        if frame_interval and (total_expansions % frame_interval == 0):
            frames_sparse.append((total_expansions, visited_snapshot.copy()))
            visited_snapshot.clear()
            print(f"    Expansions: {total_expansions:,} | Queue size: {len(open_set):,}")

        # Check expansion limit
        if max_expansions is not None and expansions_this_iter > max_expansions:
            print(f"    Reached max expansions for this iteration ({max_expansions:,})")
            break

        # Check if we've visited all goals
        if is_goal_state(mask, total_goals):
            if visited_snapshot:
                frames_sparse.append((total_expansions, visited_snapshot.copy()))
                visited_snapshot.clear()

            path = reconstruct_path(came_from, state, start_state)
            return path, frames_sparse, total_expansions, state

        # Expand neighbors
        for neighbor, step_cost in get_successors(current, road_bitmap):
            new_g = g + step_cost
            new_mask = update_goal_bitmask(neighbor, mask, goals)
            neigh_state = (neighbor, new_mask)

            old_g = best_g.get(neigh_state, float('inf'))

            if new_g < old_g:
                # Found better path to this state
                best_g[neigh_state] = new_g
                came_from[neigh_state] = state

                h = heuristic(neighbor, goals, new_mask)
                f_score = new_g + epsilon * h

                if neigh_state in closed_set:
                    # State was already expanded - add to INCONS
                    incons.add(neigh_state)
                else:
                    # Add to OPEN with weighted f-score
                    heapq.heappush(open_set, (f_score, new_g, neighbor, new_mask))

    # No path found this iteration
    if visited_snapshot:
        frames_sparse.append((total_expansions, visited_snapshot.copy()))

    return None, frames_sparse, total_expansions, None


def ara_star_multi(start, goals, road_bitmap, frame_interval=FRAME_RECORD_INTERVAL,
                   max_expansions=MAX_EXPANSIONS_PER_ITER,
                   initial_epsilon=INITIAL_EPSILON,
                   epsilon_decrement=EPSILON_DECREMENT,
                   final_epsilon=FINAL_EPSILON,
                   max_iterations=MAX_ITERATIONS):
    """
    ARA* (Anytime Repairing A*) multi-goal pathfinding.

    ARA* works by:
    1. Running weighted A* with high epsilon to find initial solution quickly
    2. Decreasing epsilon and re-searching to improve solution
    3. Continuing until epsilon = 1.0 (optimal) or time/iteration limit

    Args:
        start: starting position (row, col)
        goals: list of goal positions [(row, col), ...]
        road_bitmap: 2D array where 1=road, 0=off-road
        frame_interval: how often to save animation frames
        max_expansions: maximum nodes to expand per iteration
        initial_epsilon: starting inflation factor
        epsilon_decrement: how much to reduce epsilon each iteration
        final_epsilon: stop when epsilon reaches this value
        max_iterations: maximum number of improvement iterations

    Returns:
        path: list of waypoints from start to all goals (or None)
        frames_sparse: list of (expansion_count, visited_positions) for animation
        solution_history: list of (epsilon, path_cost, path) for each solution found
    """
    start_time = time.time()
    total_goals = len(goals)

    # Initialize start state
    start_mask = update_goal_bitmask(start, 0, goals)
    start_state = (start, start_mask)

    # Persistent data structures across iterations
    best_g = {start_state: 0.0}
    came_from = {}

    # Track best solution
    best_path = None
    incumbent_cost = float('inf')
    solution_history = []

    # Animation data
    all_frames_sparse = []
    visited_snapshot = []
    total_expansions = 0

    epsilon = initial_epsilon
    iteration = 0

    print(f"\nRunning ARA* search...")
    print(f"  Initial epsilon: {initial_epsilon}")
    print(f"  Final epsilon: {final_epsilon}")
    print(f"  Epsilon decrement: {epsilon_decrement}")

    while epsilon >= final_epsilon and iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration} (epsilon = {epsilon:.2f}) ---")

        # Initialize OPEN with start state (or states from INCONS)
        open_set = []
        incons = set()
        closed_set = set()

        # Add start state to OPEN
        h_start = heuristic(start, goals, start_mask)
        heapq.heappush(open_set, (epsilon * h_start, 0.0, start, start_mask))

        # For subsequent iterations, also add states that need re-expansion
        # These are states whose g-values improved but were already expanded
        if iteration > 1:
            for state in best_g:
                if state != start_state:
                    pos, mask = state
                    g = best_g[state]
                    h = heuristic(pos, goals, mask)
                    f_score = g + epsilon * h
                    # Only add if potentially useful (pruning)
                    if f_score < incumbent_cost:
                        heapq.heappush(open_set, (f_score, g, pos, mask))

        # Run weighted A* for this iteration
        path, frames, total_expansions, goal_state = improve_path(
            start, goals, road_bitmap, epsilon,
            best_g, came_from, open_set, incons, closed_set,
            incumbent_cost, frame_interval, max_expansions,
            visited_snapshot, total_expansions
        )

        all_frames_sparse.extend(frames)

        if path is not None:
            path_cost = compute_path_cost(path, road_bitmap)

            if path_cost < incumbent_cost:
                incumbent_cost = path_cost
                best_path = path
                solution_history.append((epsilon, path_cost, len(path)))

                print(f"  Found solution! Cost: {path_cost:.2f}, Path length: {len(path)}")
                print(f"  Suboptimality bound: {epsilon:.2f}x optimal")
        else:
            print(f"  No improvement found this iteration")

        # Decrease epsilon for next iteration
        epsilon -= epsilon_decrement
        epsilon = max(epsilon, final_epsilon)

        # Early termination if epsilon is already at minimum
        if epsilon < final_epsilon:
            break

    elapsed = time.time() - start_time

    if best_path is not None:
        print(f"\n{'='*50}")
        print(f"ARA* Search Complete!")
        print(f"  Total iterations: {iteration}")
        print(f"  Total expansions: {total_expansions:,}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Final path cost: {incumbent_cost:.2f}")
        print(f"  Final path length: {len(best_path)} waypoints")
        print(f"\nSolution improvement history:")
        for eps, cost, length in solution_history:
            print(f"    epsilon={eps:.2f}: cost={cost:.2f}, length={length}")
    else:
        print(f"\nNo path found after {iteration} iterations.")

    return best_path, all_frames_sparse, solution_history


def save_output_data(output_dir, road_bitmap, path, frames_sparse, metadata, solution_history=None):
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
        "original_metadata": metadata,
        "algorithm": "ARA*",
        "solution_history": solution_history
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

    print("Data saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Run ARA* pathfinding on NPZ test sets with animation"
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
        help="Output directory (default: ./ara_output_<testname>)"
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
        default=MAX_EXPANSIONS_PER_ITER,
        help=f"Maximum node expansions per iteration (default: {MAX_EXPANSIONS_PER_ITER:,})"
    )
    parser.add_argument(
        "--no-diagonal",
        action="store_true",
        help="Disable diagonal movement"
    )
    parser.add_argument(
        "--initial-epsilon",
        type=float,
        default=INITIAL_EPSILON,
        help=f"Initial epsilon/inflation factor (default: {INITIAL_EPSILON})"
    )
    parser.add_argument(
        "--epsilon-decrement",
        type=float,
        default=EPSILON_DECREMENT,
        help=f"Epsilon decrement per iteration (default: {EPSILON_DECREMENT})"
    )
    parser.add_argument(
        "--final-epsilon",
        type=float,
        default=FINAL_EPSILON,
        help=f"Final epsilon value (default: {FINAL_EPSILON})"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=MAX_ITERATIONS,
        help=f"Maximum improvement iterations (default: {MAX_ITERATIONS})"
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output is None:
        test_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"./ara_output_{test_name}"

    print("=" * 70)
    print("ARA* (Anytime Repairing A*) Pathfinding from NPZ Test Set")
    print("=" * 70)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Max expansions/iter: {args.max_expansions:,}")
    print(f"Diagonal movement: {'disabled' if args.no_diagonal else 'enabled'}")
    print(f"Epsilon: {args.initial_epsilon} -> {args.final_epsilon} (decrement: {args.epsilon_decrement})")
    print(f"Max iterations: {args.max_iterations}")
    print("=" * 70)

    # Load test set
    raster, goal_points, metadata = load_npz_test_set(args.input)

    # Convert goal points from (x, y, name) to (row, col)
    # Note: x is column, y is row in image coordinates
    goals = []
    for x, y, name in goal_points:
        goals.append((int(y), int(x)))  # Convert to (row, col)

    if len(goals) < 2:
        print("\nError: Need at least 2 goals for multi-goal routing")
        return 1

    # Use first goal as start
    start = goals[0]
    print(f"\nStarting from: {start}")
    print(f"Total goals: {len(goals)}")

    # Update global diagonal setting
    global ALLOW_DIAGONAL
    ALLOW_DIAGONAL = not args.no_diagonal

    # Run ARA* search
    print("\n" + "=" * 70)
    path, frames_sparse, solution_history = ara_star_multi(
        start, goals, raster,
        frame_interval=FRAME_RECORD_INTERVAL,
        max_expansions=args.max_expansions,
        initial_epsilon=args.initial_epsilon,
        epsilon_decrement=args.epsilon_decrement,
        final_epsilon=args.final_epsilon,
        max_iterations=args.max_iterations
    )
    print("=" * 70)

    # Save output data
    save_output_data(args.output, raster, path, frames_sparse, metadata, solution_history)

    # Create animations
    if not args.no_animation and path is not None:
        print("\n" + "=" * 70)
        print("Creating animations...")
        print("=" * 70)

        try:
            animator = AStarAnimator(output_dir=args.output)

            # Create animated GIF
            animator.create_animation(
                output_file="ara_animation.gif",
                fps=args.fps,
                target_frames=args.frames,
                show_path=True,
                figsize=(12, 10),
                dpi=100
            )

            # Create static comparison
            animator.create_static_comparison(
                output_file="ara_comparison.png",
                figsize=(18, 6),
                dpi=150
            )

            print("\nAnimations created successfully!")
            print(f"  - Animation: {args.output}/ara_animation.gif")
            print(f"  - Comparison: {args.output}/ara_comparison.png")

        except Exception as e:
            print(f"\nError creating animations: {e}")
            import traceback
            traceback.print_exc()
            print("  (Data was still saved successfully)")

    print("\n" + "=" * 70)
    print("Process complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
