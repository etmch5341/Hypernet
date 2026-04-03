#!/usr/bin/env python3
"""
MHA* (Multi-Heuristic A*) Pathfinding from NPZ Test Sets
Loads raster data from NPZ files, runs MHA* pathfinding, and generates animations.

MHA* uses multiple heuristic functions to explore different parts of the state space:
- Anchor heuristic (admissible): guarantees bounded suboptimality
- Inadmissible heuristics: guide search aggressively in different directions
States expanded by any heuristic are shared across all searches.

Based on: Aine et al., "Multi-Heuristic A*" (IJRR 2016)

Usage:
    python hyperloop_mha_from_npz.py --input austin_test_raster.npz
    python hyperloop_mha_from_npz.py --input seattle_test_raster.npz --output ./output/seattle
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
ON_ROAD_COST = 1.0
OFF_ROAD_COST = 5.0
MAX_EXPANSIONS = 5_000_000
CREATE_ANIMATION = True
ANIMATION_FPS = 15
ANIMATION_FRAMES = 150

# MHA* specific settings
W1 = 2.0    # Anchor suboptimality bound
W2 = 1.2    # Inadmissible expansion bound
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
    """Calculate minimum spanning tree cost using Euclidean distance (Prim's algorithm)."""
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


def mst_cost_manhattan(points):
    """Calculate MST cost using Manhattan distance."""
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
                d = abs(pts[u][0] - pts[v][0]) + abs(pts[u][1] - pts[v][1])
                if d < dist[v]:
                    dist[v] = d
    return total


# --- Heuristic functions ---

def h_anchor(curr, goals, visited_mask):
    """h0: Euclidean + MST heuristic (admissible). Same as standard A*."""
    remaining = [g for i, g in enumerate(goals) if not (visited_mask & (1 << i))]
    if not remaining:
        return 0.0
    min_to_goal = min(math.hypot(g[0] - curr[0], g[1] - curr[1]) for g in remaining)
    mst = mst_cost(remaining)
    return min_to_goal + mst


def h_manhattan(curr, goals, visited_mask):
    """h1: Manhattan distance + Manhattan MST (inadmissible on 8-connected grid)."""
    remaining = [g for i, g in enumerate(goals) if not (visited_mask & (1 << i))]
    if not remaining:
        return 0.0
    min_to_goal = min(abs(g[0] - curr[0]) + abs(g[1] - curr[1]) for g in remaining)
    mst = mst_cost_manhattan(remaining)
    return min_to_goal + mst


def h_greedy_road(curr, goals, visited_mask, road_bitmap):
    """h2: 2x Euclidean + MST with road bias (inadmissible, aggressive).

    Inflates the heuristic by 2x and gives a 20% discount when on a road cell,
    encouraging the search to stay on roads while aggressively pursuing goals.
    """
    remaining = [g for i, g in enumerate(goals) if not (visited_mask & (1 << i))]
    if not remaining:
        return 0.0
    min_to_goal = min(math.hypot(g[0] - curr[0], g[1] - curr[1]) for g in remaining)
    mst = mst_cost(remaining)
    base_h = 2.0 * (min_to_goal + mst)

    row, col = curr
    h, w = road_bitmap.shape
    if 0 <= row < h and 0 <= col < w and road_bitmap[row, col] == 1:
        base_h *= 0.8  # 20% discount for being on road
    return base_h


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
            base_distance = math.hypot(dx, dy)
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


def mha_star_multi(start, goals, road_bitmap, w1=W1, w2=W2,
                   frame_interval=FRAME_RECORD_INTERVAL,
                   max_expansions=MAX_EXPANSIONS):
    """
    Multi-Heuristic A* (MHA*) multi-goal pathfinding.

    Maintains separate OPEN lists for the anchor heuristic and each
    inadmissible heuristic. States are shared across all searches.
    The anchor search guarantees bounded suboptimality (w1 * optimal).

    Args:
        start: starting position (row, col)
        goals: list of goal positions [(row, col), ...]
        road_bitmap: 2D array where 1=road, 0=off-road
        w1: anchor suboptimality bound
        w2: inadmissible expansion bound
        frame_interval: how often to save animation frames
        max_expansions: maximum total node expansions

    Returns:
        path: list of waypoints (or None)
        frames_sparse: list of (expansion_count, visited_positions) for animation
        search_stats: dict with per-heuristic expansion counts and metadata
    """
    start_time = time.time()
    total_goals = len(goals)
    num_heuristics = 3  # 0=anchor, 1=manhattan, 2=greedy_road

    # Initialize start state
    start_mask = update_goal_bitmask(start, 0, goals)
    start_state = (start, start_mask)

    # Shared data structures
    best_g = {start_state: 0.0}
    came_from = {}
    closed_anchor = set()
    closed_inad = set()

    # Per-heuristic OPEN lists
    open_lists = [[] for _ in range(num_heuristics)]

    # Helper to compute key for a state in heuristic i
    # Standard MHA*: keys use raw g + h (no inflation).
    # w2 is only used in the expansion condition (key_i <= w2 * anchor_min_key).
    def compute_key(pos, mask, g, i):
        if i == 0:
            return g + h_anchor(pos, goals, mask)
        elif i == 1:
            return g + h_manhattan(pos, goals, mask)
        else:
            return g + h_greedy_road(pos, goals, mask, road_bitmap)

    # Seed all OPEN lists with start state
    for i in range(num_heuristics):
        key = compute_key(start, start_mask, 0.0, i)
        heapq.heappush(open_lists[i], (key, 0.0, start, start_mask))

    # Animation & stats
    frames_sparse = []
    visited_snapshot = []
    expansions = 0
    heuristic_expansions = [0] * num_heuristics
    heuristic_names = ['Anchor (Euclidean+MST)', 'Manhattan+MST', 'Greedy Road-Biased']

    def expand_state(state, g, heuristic_idx):
        """Expand a state and insert successors into all OPEN lists."""
        nonlocal expansions
        pos, mask = state

        visited_snapshot.append(pos)
        expansions += 1
        heuristic_expansions[heuristic_idx] += 1

        if frame_interval and (expansions % frame_interval == 0):
            frames_sparse.append((expansions, visited_snapshot.copy()))
            visited_snapshot.clear()
            print(f"  Expansions: {expansions:,} | "
                  f"Anchor: {heuristic_expansions[0]:,} | "
                  f"Manhattan: {heuristic_expansions[1]:,} | "
                  f"GreedyRoad: {heuristic_expansions[2]:,}")

        for neighbor, step_cost in get_successors(pos, road_bitmap):
            new_g = g + step_cost
            new_mask = update_goal_bitmask(neighbor, mask, goals)
            neigh_state = (neighbor, new_mask)

            old_g = best_g.get(neigh_state, float('inf'))
            if new_g < old_g:
                best_g[neigh_state] = new_g
                came_from[neigh_state] = state

                if neigh_state not in closed_anchor:
                    for i in range(num_heuristics):
                        key_i = compute_key(neighbor, new_mask, new_g, i)
                        heapq.heappush(open_lists[i], (key_i, new_g, neighbor, new_mask))

    print("\nRunning MHA* search...")
    print(f"  w1={w1}, w2={w2}")
    print(f"  Heuristics: {', '.join(heuristic_names)}")

    while open_lists[0]:
        if max_expansions is not None and expansions > max_expansions:
            print(f"\nReached max expansions ({max_expansions:,}). Aborting.")
            break

        # Clean stale entries from anchor OPEN and get min key
        anchor_min_key = None
        while open_lists[0]:
            key0, g0, pos0, mask0 = open_lists[0][0]
            state0 = (pos0, mask0)
            if state0 in closed_anchor or best_g.get(state0, float('inf')) < g0:
                heapq.heappop(open_lists[0])
                continue
            anchor_min_key = key0
            break

        if anchor_min_key is None:
            break

        # Try each inadmissible heuristic
        expanded = False
        for i in range(1, num_heuristics):
            # Clean stale entries
            while open_lists[i]:
                key_i, g_i, pos_i, mask_i = open_lists[i][0]
                state_i = (pos_i, mask_i)
                if state_i in closed_inad or best_g.get(state_i, float('inf')) < g_i:
                    heapq.heappop(open_lists[i])
                    continue
                break
            else:
                continue

            key_i, g_i, pos_i, mask_i = open_lists[i][0]
            state_i = (pos_i, mask_i)

            if key_i <= w2 * anchor_min_key:
                heapq.heappop(open_lists[i])

                # Check if goal
                if is_goal_state(mask_i, total_goals):
                    elapsed = time.time() - start_time
                    if visited_snapshot:
                        frames_sparse.append((expansions, visited_snapshot.copy()))
                    path = reconstruct_path(came_from, state_i, start_state)

                    search_stats = {
                        'heuristic_expansions': heuristic_expansions,
                        'heuristic_names': heuristic_names,
                        'total_expansions': expansions,
                        'time': elapsed,
                        'w1': w1, 'w2': w2,
                        'found_by_heuristic': i
                    }

                    print(f"\nPath found! (by {heuristic_names[i]})")
                    print(f"  Expansions: {expansions:,}")
                    print(f"  Time: {elapsed:.2f}s")
                    if path:
                        print(f"  Path length: {len(path)} waypoints")
                    return path, frames_sparse, search_stats

                closed_inad.add(state_i)
                expand_state(state_i, g_i, i)
                expanded = True
                break

        if not expanded:
            # Expand from anchor
            key0, g0, pos0, mask0 = heapq.heappop(open_lists[0])
            state0 = (pos0, mask0)

            if state0 in closed_anchor or best_g.get(state0, float('inf')) < g0:
                continue

            # Check if goal
            if is_goal_state(mask0, total_goals):
                elapsed = time.time() - start_time
                if visited_snapshot:
                    frames_sparse.append((expansions, visited_snapshot.copy()))
                path = reconstruct_path(came_from, state0, start_state)

                search_stats = {
                    'heuristic_expansions': heuristic_expansions,
                    'heuristic_names': heuristic_names,
                    'total_expansions': expansions,
                    'time': elapsed,
                    'w1': w1, 'w2': w2,
                    'found_by_heuristic': 0
                }

                print(f"\nPath found! (by {heuristic_names[0]})")
                print(f"  Expansions: {expansions:,}")
                print(f"  Time: {elapsed:.2f}s")
                if path:
                    print(f"  Path length: {len(path)} waypoints")
                return path, frames_sparse, search_stats

            closed_anchor.add(state0)
            expand_state(state0, g0, 0)

    # No path found
    elapsed = time.time() - start_time
    print(f"\nNo path found (or search aborted). Time: {elapsed:.2f}s")
    if visited_snapshot:
        frames_sparse.append((expansions, visited_snapshot.copy()))

    search_stats = {
        'heuristic_expansions': heuristic_expansions,
        'heuristic_names': heuristic_names,
        'total_expansions': expansions,
        'time': elapsed,
        'w1': w1, 'w2': w2,
        'found_by_heuristic': None
    }
    return None, frames_sparse, search_stats


def save_output_data(output_dir, road_bitmap, path, frames_sparse, metadata, search_stats=None):
    """Save all data needed for animation."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving data to {output_dir}...")

    np.savez_compressed(
        os.path.join(output_dir, "road_bitmap.npz"),
        road_bitmap=np.asarray(road_bitmap, dtype=np.uint8)
    )

    protected_bitmap = np.zeros_like(road_bitmap, dtype=np.uint8)
    np.savez_compressed(
        os.path.join(output_dir, "protected_bitmap.npz"),
        protected_bitmap=protected_bitmap
    )

    from rasterio.transform import from_bounds
    h, w = road_bitmap.shape
    transform = from_bounds(0, 0, w, h, w, h)
    bounds = (0, 0, w, h)

    meta = {
        "transform": transform,
        "bounds": bounds,
        "original_metadata": metadata,
        "algorithm": "MHA*",
        "search_stats": search_stats
    }
    with open(os.path.join(output_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    with open(os.path.join(output_dir, "astar_sparse_frames.pkl"), "wb") as f:
        pickle.dump(frames_sparse, f, protocol=pickle.HIGHEST_PROTOCOL)

    if search_stats:
        with open(os.path.join(output_dir, "mha_search_stats.pkl"), "wb") as f:
            pickle.dump(search_stats, f, protocol=pickle.HIGHEST_PROTOCOL)

    if path:
        path_arr = np.array(path, dtype=np.int32)
    else:
        path_arr = np.zeros((0, 2), dtype=np.int32)
    np.save(os.path.join(output_dir, "astar_final_path.npy"), path_arr)

    print("Data saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Run MHA* pathfinding on NPZ test sets with animation"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to NPZ test set file"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: ./mha_output_<testname>)"
    )
    parser.add_argument(
        "--no-animation", action="store_true",
        help="Skip creating animations"
    )
    parser.add_argument(
        "--fps", type=int, default=ANIMATION_FPS,
        help=f"Animation FPS (default: {ANIMATION_FPS})"
    )
    parser.add_argument(
        "--frames", type=int, default=ANIMATION_FRAMES,
        help=f"Animation frame count (default: {ANIMATION_FRAMES})"
    )
    parser.add_argument(
        "--max-expansions", type=int, default=MAX_EXPANSIONS,
        help=f"Maximum node expansions (default: {MAX_EXPANSIONS:,})"
    )
    parser.add_argument(
        "--no-diagonal", action="store_true",
        help="Disable diagonal movement"
    )
    parser.add_argument(
        "--w1", type=float, default=W1,
        help=f"Anchor suboptimality bound (default: {W1})"
    )
    parser.add_argument(
        "--w2", type=float, default=W2,
        help=f"Inadmissible expansion bound (default: {W2})"
    )

    args = parser.parse_args()

    if args.output is None:
        test_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"./mha_output_{test_name}"

    print("=" * 70)
    print("MHA* (Multi-Heuristic A*) Pathfinding from NPZ Test Set")
    print("=" * 70)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Max expansions: {args.max_expansions:,}")
    print(f"Diagonal movement: {'disabled' if args.no_diagonal else 'enabled'}")
    print(f"w1={args.w1}, w2={args.w2}")
    print("=" * 70)

    raster, goal_points, metadata = load_npz_test_set(args.input)

    goals = []
    for x, y, name in goal_points:
        goals.append((int(y), int(x)))

    if len(goals) < 2:
        print("\nError: Need at least 2 goals for multi-goal routing")
        return 1

    start = goals[0]
    print(f"\nStarting from: {start}")
    print(f"Total goals: {len(goals)}")

    global ALLOW_DIAGONAL
    ALLOW_DIAGONAL = not args.no_diagonal

    print("\n" + "=" * 70)
    path, frames_sparse, search_stats = mha_star_multi(
        start, goals, raster,
        w1=args.w1, w2=args.w2,
        frame_interval=FRAME_RECORD_INTERVAL,
        max_expansions=args.max_expansions
    )
    print("=" * 70)

    save_output_data(args.output, raster, path, frames_sparse, metadata, search_stats)

    if not args.no_animation and path is not None:
        print("\n" + "=" * 70)
        print("Creating animations...")
        print("=" * 70)

        try:
            animator = AStarAnimator(output_dir=args.output)
            animator.create_animation(
                output_file="mha_animation.gif",
                fps=args.fps,
                target_frames=args.frames,
                show_path=True,
                figsize=(12, 10),
                dpi=100
            )
            animator.create_static_comparison(
                output_file="mha_comparison.png",
                figsize=(18, 6),
                dpi=150
            )
            print(f"\nAnimations created successfully!")
            print(f"  - Animation: {args.output}/mha_animation.gif")
            print(f"  - Comparison: {args.output}/mha_comparison.png")
        except Exception as e:
            print(f"\nError creating animations: {e}")
            import traceback
            traceback.print_exc()
            print("  (Data was still saved successfully)")

    # Print final stats
    if search_stats:
        print("\n" + "=" * 70)
        print("MHA* Search Statistics")
        print("=" * 70)
        for i, name in enumerate(search_stats['heuristic_names']):
            print(f"  {name}: {search_stats['heuristic_expansions'][i]:,} expansions")
        print(f"  Total: {search_stats['total_expansions']:,} expansions")
        print(f"  Time: {search_stats['time']:.2f}s")
        if search_stats['found_by_heuristic'] is not None:
            print(f"  Found by: {search_stats['heuristic_names'][search_stats['found_by_heuristic']]}")
        print("=" * 70)

    print("\nProcess complete!")
    return 0


if __name__ == "__main__":
    exit(main())
