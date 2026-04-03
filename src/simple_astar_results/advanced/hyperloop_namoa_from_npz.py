#!/usr/bin/env python3
"""
NAMOA*-dr (Non-dominated Archive Multi-Objective A* with Delayed Reopening)
Pathfinding from NPZ Test Sets.

NAMOA*-dr is a multi-objective search algorithm that:
1. Finds the complete Pareto front of non-dominated solutions
2. Uses 3 objectives: travel distance, off-road exposure, terrain roughness
3. Employs delayed reopening for efficiency
4. Selects a "balanced" path from the Pareto front for visualization

Three Objectives:
- Distance: pure Euclidean path length (hypot(dx, dy))
- Off-Road Exposure: count of off-road cells traversed
- Terrain Roughness: precomputed local road density inverse

Usage:
    python hyperloop_namoa_from_npz.py --input austin_test_raster.npz
    python hyperloop_namoa_from_npz.py --input seattle_test_raster.npz --output ./output/seattle
"""

import argparse
import csv
import json
import numpy as np
import heapq
import math
import time
import os
import pickle
from collections import defaultdict, deque
from astar_animator import AStarAnimator


# -------------------- CONFIG --------------------
ALLOW_DIAGONAL = True
FRAME_RECORD_INTERVAL = 25000
MAX_EXPANSIONS = 1_500_000
CREATE_ANIMATION = True
ANIMATION_FPS = 15
ANIMATION_FRAMES = 150

# NAMOA*-dr specific settings
ROUGHNESS_WINDOW = 5  # Window size for terrain roughness computation
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


def compute_roughness_map(road_bitmap, window_size=ROUGHNESS_WINDOW):
    """
    Precompute terrain roughness for each cell.

    Roughness = 1 - local_road_density, where local_road_density is the
    fraction of road cells in a window_size x window_size neighborhood.
    High roughness means isolated from road infrastructure.

    Args:
        road_bitmap: 2D array where 1=road, 0=off-road
        window_size: size of the convolution window

    Returns:
        roughness_map: 2D float array, values in [0.0, 1.0]
    """
    try:
        from scipy.ndimage import uniform_filter
        road_float = road_bitmap.astype(np.float64)
        local_density = uniform_filter(road_float, size=window_size, mode='constant', cval=0.0)
    except ImportError:
        # Fallback: pure numpy box filter using cumulative sums
        print("  Warning: scipy not available, using numpy fallback for roughness computation")
        road_float = road_bitmap.astype(np.float64)
        h, w = road_float.shape
        pad = window_size // 2
        padded = np.pad(road_float, pad, mode='constant', constant_values=0.0)
        cum = np.cumsum(np.cumsum(padded, axis=0), axis=1)
        local_sum = (cum[window_size:, window_size:] -
                     cum[:-window_size, window_size:] -
                     cum[window_size:, :-window_size] +
                     cum[:-window_size, :-window_size])
        # Crop to original size if needed
        local_sum = local_sum[:h, :w]
        local_density = local_sum / (window_size * window_size)

    roughness_map = 1.0 - local_density
    return roughness_map


def mst_cost(points):
    """Calculate minimum spanning tree cost using Euclidean distance."""
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


def precompute_heuristic_maps(goals, road_bitmap, roughness_map, allow_diagonal=True):
    """
    Precompute per-goal minimum-cost maps for off-road and roughness objectives
    via reverse search (0-1 BFS for off-road, Dijkstra for roughness).

    Returns:
        offroad_maps: list of 2D arrays, one per goal (min off-road cells to reach goal)
        roughness_maps_precomp: list of 2D arrays, one per goal (min roughness sum to reach goal)
    """
    h, w = road_bitmap.shape

    if allow_diagonal:
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                      (1, 1), (-1, -1), (-1, 1), (1, -1)]
    else:
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    offroad_maps = []
    roughness_maps_precomp = []

    for gi, goal in enumerate(goals):
        print(f"    Goal {gi} at {goal}...")

        # Off-road objective: 0-1 BFS (cost 0 for road, 1 for off-road)
        offroad_map = np.full((h, w), float('inf'))
        offroad_map[goal[0], goal[1]] = 0.0
        dq = deque()
        dq.append((goal[0], goal[1]))

        while dq:
            r, c = dq.popleft()
            d = offroad_map[r, c]
            for dx, dy in directions:
                nr, nc = r + dx, c + dy
                if 0 <= nr < h and 0 <= nc < w:
                    cost = 0.0 if road_bitmap[nr, nc] == 1 else 1.0
                    new_d = d + cost
                    if new_d < offroad_map[nr, nc]:
                        offroad_map[nr, nc] = new_d
                        if cost == 0.0:
                            dq.appendleft((nr, nc))
                        else:
                            dq.append((nr, nc))

        offroad_maps.append(offroad_map)

        # Roughness objective: Dijkstra with roughness edge costs
        rough_map = np.full((h, w), float('inf'))
        rough_map[goal[0], goal[1]] = 0.0
        heap = [(0.0, goal[0], goal[1])]

        while heap:
            d, r, c = heapq.heappop(heap)
            if d > rough_map[r, c]:
                continue
            for dx, dy in directions:
                nr, nc = r + dx, c + dy
                if 0 <= nr < h and 0 <= nc < w:
                    cost = roughness_map[nr, nc]
                    new_d = d + cost
                    if new_d < rough_map[nr, nc]:
                        rough_map[nr, nc] = new_d
                        heapq.heappush(heap, (new_d, nr, nc))

        roughness_maps_precomp.append(rough_map)

    return offroad_maps, roughness_maps_precomp


def heuristic_vector(curr, goals, visited_mask, offroad_maps=None, roughness_maps_precomp=None):
    """
    Component-wise admissible heuristic vector for NAMOA*-dr.

    Returns:
        (h_distance, h_offroad, h_roughness) tuple

    h_distance: Euclidean + MST (admissible for distance objective)
    h_offroad: precomputed min off-road cells to nearest remaining goal (admissible)
    h_roughness: precomputed min roughness sum to nearest remaining goal (admissible)
    """
    remaining_idx = [i for i in range(len(goals)) if not (visited_mask & (1 << i))]
    if not remaining_idx:
        return (0.0, 0.0, 0.0)

    remaining_goals = [goals[i] for i in remaining_idx]

    # Distance: Euclidean + MST (same as before)
    min_to_goal = min(math.hypot(g[0] - curr[0], g[1] - curr[1]) for g in remaining_goals)
    mst = mst_cost(remaining_goals)
    h_dist = min_to_goal + mst

    # Off-road: min precomputed cost to nearest remaining goal
    r, c = curr
    if offroad_maps is not None:
        h_offroad = min(offroad_maps[i][r, c] for i in remaining_idx)
    else:
        h_offroad = 0.0

    # Roughness: min precomputed cost to nearest remaining goal
    if roughness_maps_precomp is not None:
        h_rough = min(roughness_maps_precomp[i][r, c] for i in remaining_idx)
    else:
        h_rough = 0.0

    return (h_dist, h_offroad, h_rough)


def update_goal_bitmask(pos, bitmask, goal_positions):
    """Update bitmask when reaching a goal."""
    for i, g in enumerate(goal_positions):
        if pos == g:
            bitmask |= (1 << i)
    return bitmask


def is_goal_state(bitmask, total_goals):
    """Check if all goals have been visited."""
    return bitmask == (1 << total_goals) - 1


def get_successors_multiobjective(pos, road_bitmap, roughness_map, allow_diagonal=True):
    """
    Generate valid successors with 3-objective cost vectors.

    Yields:
        (neighbor_pos, cost_vector) where cost_vector = (distance, offroad, roughness)
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
            cost_distance = math.hypot(dx, dy)
            cost_offroad = 0.0 if road_bitmap[nx, ny] == 1 else 1.0
            cost_roughness = roughness_map[nx, ny]
            yield ((nx, ny), (cost_distance, cost_offroad, cost_roughness))


def dominates(g1, g2):
    """
    Check if cost vector g1 dominates g2.
    g1 dominates g2 iff g1[i] <= g2[i] for all i AND g1[j] < g2[j] for some j.
    """
    at_least_as_good = True
    strictly_better = False
    for a, b in zip(g1, g2):
        if a > b:
            at_least_as_good = False
            break
        if a < b:
            strictly_better = True
    return at_least_as_good and strictly_better


def eps_dominates(g1, g2, epsilon):
    """
    Check if g1 epsilon-dominates g2.
    g1 ε-dominates g2 iff g1[i] <= (1+ε) * g2[i] for all i.
    This is a relaxed dominance that merges "similar" labels.
    """
    for a, b in zip(g1, g2):
        if a > (1.0 + epsilon) * b:
            return False
    return True


def is_dominated_by_set(g_new, g_set, epsilon=0.0):
    """Check if g_new is dominated by any vector in g_set."""
    if epsilon > 0.0:
        for g_existing in g_set:
            if eps_dominates(g_existing, g_new, epsilon):
                return True
    else:
        for g_existing in g_set:
            if dominates(g_existing, g_new):
                return True
    return False


def add_to_nondominated_set(g_new, g_set, epsilon=0.0):
    """
    Add g_new to the non-dominated set, removing any vectors it dominates.

    Returns:
        True if g_new was added (i.e., it is non-dominated)
    """
    if is_dominated_by_set(g_new, g_set, epsilon):
        return False

    if epsilon > 0.0:
        to_remove = [g for g in g_set if eps_dominates(g_new, g, epsilon)]
    else:
        to_remove = [g for g in g_set if dominates(g_new, g)]
    for g in to_remove:
        g_set.remove(g)

    g_set.add(g_new)
    return True


def reconstruct_pareto_path(came_from, goal_state, goal_g, start_state, start_g):
    """Reconstruct path by following (state, g_vector) parent pointers."""
    path = []
    current = (goal_state, goal_g)
    start_label = (start_state, start_g)

    max_steps = 10_000_000  # safety limit
    steps = 0
    while current != start_label:
        state, g_vec = current
        path.append(state[0])
        parent = came_from.get(current)
        if parent is None:
            return None
        current = parent
        steps += 1
        if steps > max_steps:
            return None

    path.append(start_state[0])
    return path[::-1]


def select_balanced_path(pareto_paths):
    """
    Select the Pareto path closest to the normalized ideal point.

    Normalizes each objective to [0,1] range and picks the path with
    minimum Euclidean distance to the ideal point (0, 0, 0).
    """
    if not pareto_paths:
        return None
    if len(pareto_paths) == 1:
        return pareto_paths[0]['path']

    distances = [p['distance'] for p in pareto_paths]
    offroads = [p['offroad_count'] for p in pareto_paths]
    roughnesses = [p['roughness_sum'] for p in pareto_paths]

    min_d, max_d = min(distances), max(distances)
    min_o, max_o = min(offroads), max(offroads)
    min_r, max_r = min(roughnesses), max(roughnesses)

    best_idx = 0
    best_dist = float('inf')

    for i, p in enumerate(pareto_paths):
        nd = (p['distance'] - min_d) / (max_d - min_d + 1e-9)
        no = (p['offroad_count'] - min_o) / (max_o - min_o + 1e-9)
        nr = (p['roughness_sum'] - min_r) / (max_r - min_r + 1e-9)
        dist_to_ideal = math.sqrt(nd**2 + no**2 + nr**2)
        if dist_to_ideal < best_dist:
            best_dist = dist_to_ideal
            best_idx = i

    return pareto_paths[best_idx]['path']


def namoa_dr_multi(start, goals, road_bitmap, roughness_map,
                   frame_interval=FRAME_RECORD_INTERVAL,
                   max_expansions=MAX_EXPANSIONS,
                   allow_diagonal=ALLOW_DIAGONAL,
                   offroad_maps=None, roughness_maps_precomp=None,
                   epsilon=0.0):
    """
    NAMOA*-dr (Non-dominated Archive Multi-Objective A* with Delayed Reopening).

    Each state can have multiple non-dominated cost vectors (labels).
    The algorithm finds the complete Pareto front of solutions.

    Args:
        start: (row, col)
        goals: list of (row, col)
        road_bitmap: 2D array where 1=road, 0=off-road
        roughness_map: 2D float array with roughness values
        frame_interval: sparse frame recording interval
        max_expansions: safety cap

    Returns:
        selected_path: the "balanced" Pareto-optimal path
        frames_sparse: for animation
        pareto_paths: list of all Pareto-optimal paths with cost vectors
    """
    start_time = time.time()
    total_goals = len(goals)

    start_mask = update_goal_bitmask(start, 0, goals)
    start_state = (start, start_mask)

    # Per-state non-dominated g-vector sets
    G_open = defaultdict(set)
    G_closed = defaultdict(set)

    # Parent pointers: (state, g_vector) -> (parent_state, parent_g_vector)
    came_from = {}

    # Priority queue: (f_scalar, g_vector, position, mask)
    # f_scalar = sum of f-vector components for scalar ordering
    heap = []
    tie_breaker = 0  # for deterministic ordering when f_scalar is equal

    # Delayed reopening set
    delayed = set()

    # Initialize start
    g_start = (0.0, 0.0, 0.0)
    h_start = heuristic_vector(start, goals, start_mask, offroad_maps, roughness_maps_precomp)
    f_scalar = sum(g + h for g, h in zip(g_start, h_start))

    heapq.heappush(heap, (f_scalar, tie_breaker, g_start, start, start_mask))
    tie_breaker += 1
    G_open[start_state].add(g_start)

    # Goal archive
    goal_solutions = []
    goal_cost_set = set()

    # Animation
    frames_sparse = []
    visited_snapshot = []
    expansions = 0

    print("\nRunning NAMOA*-dr search...")
    print(f"  Objectives: distance, off-road exposure, terrain roughness")
    print(f"  Roughness window: {ROUGHNESS_WINDOW}x{ROUGHNESS_WINDOW}")
    if epsilon > 0:
        print(f"  Epsilon-dominance: {epsilon} ({epsilon*100:.0f}% approximate)")

    while heap:
        f_scalar, _tb, g_vec, current, mask = heapq.heappop(heap)
        state = (current, mask)

        # Skip if this g-vector is no longer in G_open
        if g_vec not in G_open[state]:
            continue

        # Skip if dominated by closed labels
        if is_dominated_by_set(g_vec, G_closed[state], epsilon):
            G_open[state].discard(g_vec)
            continue

        # Skip if dominated by known goal solutions
        if is_dominated_by_set(g_vec, goal_cost_set, epsilon):
            G_open[state].discard(g_vec)
            continue

        # Move from OPEN to CLOSED
        G_open[state].discard(g_vec)
        G_closed[state].add(g_vec)

        visited_snapshot.append(current)
        expansions += 1

        if frame_interval and (expansions % frame_interval == 0):
            frames_sparse.append((expansions, visited_snapshot.copy()))
            visited_snapshot.clear()
            print(f"  Expansions: {expansions:,} | Queue: {len(heap):,} | "
                  f"Pareto solutions: {len(goal_solutions)}")

        if max_expansions is not None and expansions > max_expansions:
            print(f"\nReached max expansions ({max_expansions:,}).")
            break

        # Check if goal
        if is_goal_state(mask, total_goals):
            if add_to_nondominated_set(g_vec, goal_cost_set):
                goal_solutions.append((g_vec, state))
                print(f"  Pareto solution #{len(goal_solutions)}: "
                      f"dist={g_vec[0]:.1f}, offroad={g_vec[1]:.0f}, rough={g_vec[2]:.2f}")
            continue

        # Expand neighbors
        for neighbor, edge_cost in get_successors_multiobjective(
                current, road_bitmap, roughness_map, allow_diagonal):
            new_g = tuple(g + c for g, c in zip(g_vec, edge_cost))
            new_mask = update_goal_bitmask(neighbor, mask, goals)
            neigh_state = (neighbor, new_mask)

            # Prune if dominated by goal solutions
            if is_dominated_by_set(new_g, goal_cost_set, epsilon):
                continue

            # Check domination by existing OPEN labels
            if is_dominated_by_set(new_g, G_open[neigh_state], epsilon):
                continue

            # Check domination by CLOSED labels (delayed reopening)
            if is_dominated_by_set(new_g, G_closed[neigh_state], epsilon):
                delayed.add((neigh_state, new_g))
                came_from[(neigh_state, new_g)] = (state, g_vec)
                continue

            # Remove labels in G_open that new_g eps-dominates
            if epsilon > 0:
                dominated_in_open = {g for g in G_open[neigh_state] if eps_dominates(new_g, g, epsilon)}
            else:
                dominated_in_open = {g for g in G_open[neigh_state] if dominates(new_g, g)}
            G_open[neigh_state] -= dominated_in_open

            # Add new label
            G_open[neigh_state].add(new_g)
            h_vec = heuristic_vector(neighbor, goals, new_mask, offroad_maps, roughness_maps_precomp)
            f_scalar = sum(g + h for g, h in zip(new_g, h_vec))
            heapq.heappush(heap, (f_scalar, tie_breaker, new_g, neighbor, new_mask))
            tie_breaker += 1
            came_from[(neigh_state, new_g)] = (state, g_vec)

        # Process delayed reopening periodically
        if expansions % 10000 == 0 and delayed:
            reopen = []
            for (d_state, d_g) in delayed:
                if not is_dominated_by_set(d_g, G_closed[d_state], epsilon):
                    if not is_dominated_by_set(d_g, goal_cost_set, epsilon):
                        reopen.append((d_state, d_g))

            for (d_state, d_g) in reopen:
                delayed.discard((d_state, d_g))
                if not is_dominated_by_set(d_g, G_open[d_state], epsilon):
                    G_open[d_state].add(d_g)
                    d_pos, d_mask = d_state
                    h_vec = heuristic_vector(d_pos, goals, d_mask, offroad_maps, roughness_maps_precomp)
                    f_scalar = sum(g + h for g, h in zip(d_g, h_vec))
                    heapq.heappush(heap, (f_scalar, tie_breaker, d_g, d_pos, d_mask))
                    tie_breaker += 1

    elapsed = time.time() - start_time

    if visited_snapshot:
        frames_sparse.append((expansions, visited_snapshot.copy()))

    # Reconstruct all Pareto-optimal paths
    pareto_paths = []
    for g_vec, goal_state in goal_solutions:
        path = reconstruct_pareto_path(
            came_from, goal_state, g_vec, start_state, g_start)
        if path:
            pareto_paths.append({
                'path': path,
                'cost_vector': g_vec,
                'distance': g_vec[0],
                'offroad_count': g_vec[1],
                'roughness_sum': g_vec[2]
            })

    selected_path = select_balanced_path(pareto_paths)

    print(f"\nNAMOA*-dr Complete!")
    print(f"  Expansions: {expansions:,}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Pareto-optimal solutions: {len(pareto_paths)}")
    if pareto_paths:
        print(f"\n  Pareto front:")
        for i, p in enumerate(pareto_paths):
            marker = " <-- selected" if p['path'] == selected_path else ""
            print(f"    #{i+1}: dist={p['distance']:.1f}, "
                  f"offroad={p['offroad_count']:.0f}, "
                  f"rough={p['roughness_sum']:.2f}{marker}")
    if selected_path:
        print(f"\n  Selected balanced path: {len(selected_path)} waypoints")

    return selected_path, frames_sparse, pareto_paths


def save_output_data(output_dir, road_bitmap, path, frames_sparse, metadata, pareto_paths=None):
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
        "algorithm": "NAMOA*-dr",
        "num_pareto_solutions": len(pareto_paths) if pareto_paths else 0
    }
    with open(os.path.join(output_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    with open(os.path.join(output_dir, "astar_sparse_frames.pkl"), "wb") as f:
        pickle.dump(frames_sparse, f, protocol=pickle.HIGHEST_PROTOCOL)

    if pareto_paths:
        with open(os.path.join(output_dir, "namoa_pareto_paths.pkl"), "wb") as f:
            pickle.dump(pareto_paths, f, protocol=pickle.HIGHEST_PROTOCOL)
        export_pareto_front(output_dir, pareto_paths, selected_path=path)

    if path:
        path_arr = np.array(path, dtype=np.int32)
    else:
        path_arr = np.zeros((0, 2), dtype=np.int32)
    np.save(os.path.join(output_dir, "astar_final_path.npy"), path_arr)

    print("Data saved successfully!")


def export_pareto_front(output_dir, pareto_paths, selected_path=None):
    """Export Pareto solutions in readable formats (JSON, CSV, and per-path NPY files)."""
    solutions = []
    selected_as_tuples = [tuple(p) for p in selected_path] if selected_path else []

    for idx, p in enumerate(pareto_paths, start=1):
        path_points = [tuple(pt) for pt in p.get("path", [])]
        solution = {
            "solution_id": idx,
            "distance": float(p["distance"]),
            "offroad_count": float(p["offroad_count"]),
            "roughness_sum": float(p["roughness_sum"]),
            "num_waypoints": len(path_points),
            "is_selected_balanced": path_points == selected_as_tuples,
            "path_row_col": [[int(r), int(c)] for r, c in path_points]
        }
        solutions.append(solution)

    json_path = os.path.join(output_dir, "namoa_pareto_front.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "algorithm": "NAMOA*-dr",
                "num_solutions": len(solutions),
                "solutions": solutions
            },
            f,
            indent=2
        )

    csv_path = os.path.join(output_dir, "namoa_pareto_summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "solution_id",
                "distance",
                "offroad_count",
                "roughness_sum",
                "num_waypoints",
                "is_selected_balanced",
            ],
        )
        writer.writeheader()
        for s in solutions:
            writer.writerow({k: s[k] for k in writer.fieldnames})

    pareto_dir = os.path.join(output_dir, "pareto_paths")
    os.makedirs(pareto_dir, exist_ok=True)
    for s in solutions:
        arr = np.array(s["path_row_col"], dtype=np.int32)
        np.save(
            os.path.join(pareto_dir, f"pareto_path_{s['solution_id']:02d}.npy"),
            arr,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run NAMOA*-dr multi-objective pathfinding on NPZ test sets"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to NPZ test set file"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: ./namoa_output_<testname>)"
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
        help=f"Maximum label expansions (default: {MAX_EXPANSIONS:,})"
    )
    parser.add_argument(
        "--no-diagonal", action="store_true",
        help="Disable diagonal movement"
    )
    parser.add_argument(
        "--roughness-window", type=int, default=ROUGHNESS_WINDOW,
        help=f"Window size for roughness computation (default: {ROUGHNESS_WINDOW})"
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.1,
        help="Epsilon for approximate Pareto dominance (default: 0.1 = 10%%)"
    )

    args = parser.parse_args()

    if args.output is None:
        test_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"./namoa_output_{test_name}"

    print("=" * 70)
    print("NAMOA*-dr Multi-Objective Pathfinding from NPZ Test Set")
    print("=" * 70)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Max expansions: {args.max_expansions:,}")
    print(f"Diagonal movement: {'disabled' if args.no_diagonal else 'enabled'}")
    print(f"Roughness window: {args.roughness_window}x{args.roughness_window}")
    print(f"Objectives: distance, off-road exposure, terrain roughness")
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

    allow_diagonal = not args.no_diagonal

    # Precompute roughness map
    print("\nComputing terrain roughness map...")
    roughness_map = compute_roughness_map(raster, args.roughness_window)
    print(f"  Roughness range: [{roughness_map.min():.3f}, {roughness_map.max():.3f}]")

    # Precompute heuristic maps for off-road and roughness objectives
    print("\nPrecomputing objective heuristic maps (reverse BFS/Dijkstra from each goal)...")
    offroad_maps, roughness_maps_precomp = precompute_heuristic_maps(
        goals, raster, roughness_map, allow_diagonal=allow_diagonal)
    print("  Heuristic maps ready.")

    print("\n" + "=" * 70)
    path, frames_sparse, pareto_paths = namoa_dr_multi(
        start, goals, raster, roughness_map,
        frame_interval=FRAME_RECORD_INTERVAL,
        max_expansions=args.max_expansions,
        allow_diagonal=allow_diagonal,
        offroad_maps=offroad_maps,
        roughness_maps_precomp=roughness_maps_precomp,
        epsilon=args.epsilon
    )
    print("=" * 70)

    save_output_data(args.output, raster, path, frames_sparse, metadata, pareto_paths)

    if not args.no_animation and path is not None:
        print("\n" + "=" * 70)
        print("Creating animations...")
        print("=" * 70)

        try:
            animator = AStarAnimator(output_dir=args.output)
            animator.create_animation(
                output_file="namoa_animation.gif",
                fps=args.fps,
                target_frames=args.frames,
                show_path=True,
                figsize=(12, 10),
                dpi=100
            )
            animator.create_static_comparison(
                output_file="namoa_comparison.png",
                figsize=(18, 6),
                dpi=150
            )
            print(f"\nAnimations created successfully!")
            print(f"  - Animation: {args.output}/namoa_animation.gif")
            print(f"  - Comparison: {args.output}/namoa_comparison.png")
        except Exception as e:
            print(f"\nError creating animations: {e}")
            import traceback
            traceback.print_exc()
            print("  (Data was still saved successfully)")

    print("\nProcess complete!")
    return 0


if __name__ == "__main__":
    exit(main())
