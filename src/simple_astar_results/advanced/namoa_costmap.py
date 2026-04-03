#!/usr/bin/env python3
"""
NAMOA*-dr Pathfinding using real cost maps (construction, environmental, geometry).

Instead of the original objectives (distance, off-road exposure, roughness),
this version uses the three actual cost map layers as separate objectives:
  Objective 1: Construction cost (normalized)
  Objective 2: Environmental impact (normalized)
  Objective 3: Geometry cost (normalized)

This produces a Pareto front where each solution represents a different
trade-off between construction expense, environmental damage, and geometric
difficulty.

Usage:
    python namoa_costmap.py
    python namoa_costmap.py --output ./my_output --epsilon 0.15
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
ANIMATION_FPS = 15
ANIMATION_FRAMES = 150
# ------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
NPZ_DIR = os.path.join(PROJECT_ROOT, "npz-files")
RASTER_PATH = os.path.join(PROJECT_ROOT, "src", "sample-test-set", "austin_test_raster.npz")


def load_cost_maps():
    """Load the three cost map layers."""
    print("Loading cost maps...")

    d = np.load(os.path.join(NPZ_DIR, "austin_construction_cost.npz"), allow_pickle=True)
    construction = d['cost_map_normalized'].astype(np.float64)
    print(f"  Construction: shape={construction.shape}, range=[{construction.min():.4f}, {construction.max():.4f}]")

    d = np.load(os.path.join(NPZ_DIR, "austin_environmental_impact.npz"), allow_pickle=True)
    environmental = d['env_map_normalized'].astype(np.float64)
    print(f"  Environmental: shape={environmental.shape}, range=[{environmental.min():.4f}, {environmental.max():.4f}]")

    d = np.load(os.path.join(NPZ_DIR, "geometry_cost_map.npz"), allow_pickle=True)
    geometry = d['cost_map'].astype(np.float64)
    print(f"  Geometry: shape={geometry.shape}, range=[{geometry.min():.4f}, {geometry.max():.4f}]")

    return construction, environmental, geometry


def load_raster_and_goals():
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


def precompute_heuristic_maps(goals, cost_maps, allow_diagonal=True):
    """
    Precompute per-goal minimum-cost maps for each objective via Dijkstra
    from each goal backwards.

    Args:
        goals: list of (row, col)
        cost_maps: tuple of (construction, environmental, geometry) arrays

    Returns:
        List of 3 lists, each containing per-goal heuristic maps.
        heuristic_maps[obj_idx][goal_idx] = 2D array of min cost-to-go
    """
    construction, environmental, geometry = cost_maps
    h, w = construction.shape

    if allow_diagonal:
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                      (1, 1), (-1, -1), (-1, 1), (1, -1)]
    else:
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    objective_maps = [construction, environmental, geometry]
    obj_names = ["construction", "environmental", "geometry"]
    all_heuristic_maps = []

    for obj_idx, (obj_map, obj_name) in enumerate(zip(objective_maps, obj_names)):
        goal_maps = []
        for gi, goal in enumerate(goals):
            print(f"    {obj_name} -> Goal {gi} at {goal}...")
            h_map = np.full((h, w), float('inf'))
            h_map[goal[0], goal[1]] = 0.0
            heap = [(0.0, goal[0], goal[1])]

            while heap:
                d, r, c = heapq.heappop(heap)
                if d > h_map[r, c]:
                    continue
                for dx, dy in directions:
                    nr, nc = r + dx, c + dy
                    if 0 <= nr < h and 0 <= nc < w:
                        # Edge cost = Euclidean distance * cost at destination
                        step = math.hypot(dx, dy) * obj_map[nr, nc]
                        new_d = d + step
                        if new_d < h_map[nr, nc]:
                            h_map[nr, nc] = new_d
                            heapq.heappush(heap, (new_d, nr, nc))

            goal_maps.append(h_map)
        all_heuristic_maps.append(goal_maps)

    return all_heuristic_maps


def heuristic_vector(curr, goals, visited_mask, heuristic_maps):
    """
    Component-wise admissible heuristic for 3 objectives.
    For each objective: min(precomputed cost to nearest unvisited goal).
    """
    remaining_idx = [i for i in range(len(goals)) if not (visited_mask & (1 << i))]
    if not remaining_idx:
        return (0.0, 0.0, 0.0)

    r, c = curr
    h_vals = []
    for obj_idx in range(3):
        h_obj = min(heuristic_maps[obj_idx][gi][r, c] for gi in remaining_idx)
        h_vals.append(h_obj)

    return tuple(h_vals)


def update_goal_bitmask(pos, bitmask, goal_positions):
    for i, g in enumerate(goal_positions):
        if pos == g:
            bitmask |= (1 << i)
    return bitmask


def is_goal_state(bitmask, total_goals):
    return bitmask == (1 << total_goals) - 1


def get_successors_multiobjective(pos, cost_maps, allow_diagonal=True):
    """
    Generate successors with 3-objective cost vectors from real cost maps.

    Yields:
        (neighbor_pos, (construction_cost, environmental_cost, geometry_cost))
    """
    construction, environmental, geometry = cost_maps

    if allow_diagonal:
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                      (1, 1), (-1, -1), (-1, 1), (1, -1)]
    else:
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    h, w = construction.shape
    for dx, dy in directions:
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < h and 0 <= ny < w:
            dist = math.hypot(dx, dy)
            c_constr = dist * construction[nx, ny]
            c_env = dist * environmental[nx, ny]
            c_geo = dist * geometry[nx, ny]
            yield ((nx, ny), (c_constr, c_env, c_geo))


def dominates(g1, g2):
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
    for a, b in zip(g1, g2):
        if a > (1.0 + epsilon) * b:
            return False
    return True


def is_dominated_by_set(g_new, g_set, epsilon=0.0):
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
    path = []
    current = (goal_state, goal_g)
    start_label = (start_state, start_g)
    max_steps = 10_000_000
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
    if not pareto_paths:
        return None
    if len(pareto_paths) == 1:
        return pareto_paths[0]['path']

    constr_vals = [p['construction'] for p in pareto_paths]
    env_vals = [p['environmental'] for p in pareto_paths]
    geo_vals = [p['geometry'] for p in pareto_paths]

    min_c, max_c = min(constr_vals), max(constr_vals)
    min_e, max_e = min(env_vals), max(env_vals)
    min_g, max_g = min(geo_vals), max(geo_vals)

    best_idx = 0
    best_dist = float('inf')
    for i, p in enumerate(pareto_paths):
        nc = (p['construction'] - min_c) / (max_c - min_c + 1e-9)
        ne = (p['environmental'] - min_e) / (max_e - min_e + 1e-9)
        ng = (p['geometry'] - min_g) / (max_g - min_g + 1e-9)
        dist_to_ideal = math.sqrt(nc ** 2 + ne ** 2 + ng ** 2)
        if dist_to_ideal < best_dist:
            best_dist = dist_to_ideal
            best_idx = i
    return pareto_paths[best_idx]['path']


def namoa_dr_costmap(start, goals, cost_maps, heuristic_maps,
                     frame_interval=FRAME_RECORD_INTERVAL,
                     max_expansions=MAX_EXPANSIONS,
                     allow_diagonal=ALLOW_DIAGONAL,
                     epsilon=0.0):
    """NAMOA*-dr with real cost map objectives."""
    start_time = time.time()
    total_goals = len(goals)

    start_mask = update_goal_bitmask(start, 0, goals)
    start_state = (start, start_mask)

    G_open = defaultdict(set)
    G_closed = defaultdict(set)
    came_from = {}

    heap = []
    tie_breaker = 0
    delayed = set()

    g_start = (0.0, 0.0, 0.0)
    h_start = heuristic_vector(start, goals, start_mask, heuristic_maps)
    f_scalar = sum(g + h for g, h in zip(g_start, h_start))

    heapq.heappush(heap, (f_scalar, tie_breaker, g_start, start, start_mask))
    tie_breaker += 1
    G_open[start_state].add(g_start)

    goal_solutions = []
    goal_cost_set = set()

    frames_sparse = []
    visited_snapshot = []
    expansions = 0

    print("\nRunning NAMOA*-dr with real cost map objectives...")
    print(f"  Objectives: construction, environmental, geometry")
    if epsilon > 0:
        print(f"  Epsilon-dominance: {epsilon} ({epsilon * 100:.0f}% approximate)")

    while heap:
        f_scalar, _tb, g_vec, current, mask = heapq.heappop(heap)
        state = (current, mask)

        if g_vec not in G_open[state]:
            continue
        if is_dominated_by_set(g_vec, G_closed[state], epsilon):
            G_open[state].discard(g_vec)
            continue
        if is_dominated_by_set(g_vec, goal_cost_set, epsilon):
            G_open[state].discard(g_vec)
            continue

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

        if is_goal_state(mask, total_goals):
            if add_to_nondominated_set(g_vec, goal_cost_set):
                goal_solutions.append((g_vec, state))
                print(f"  Pareto solution #{len(goal_solutions)}: "
                      f"constr={g_vec[0]:.1f}, env={g_vec[1]:.1f}, geo={g_vec[2]:.1f}")
            continue

        for neighbor, edge_cost in get_successors_multiobjective(current, cost_maps, allow_diagonal):
            new_g = tuple(g + c for g, c in zip(g_vec, edge_cost))
            new_mask = update_goal_bitmask(neighbor, mask, goals)
            neigh_state = (neighbor, new_mask)

            if is_dominated_by_set(new_g, goal_cost_set, epsilon):
                continue
            if is_dominated_by_set(new_g, G_open[neigh_state], epsilon):
                continue
            if is_dominated_by_set(new_g, G_closed[neigh_state], epsilon):
                delayed.add((neigh_state, new_g))
                came_from[(neigh_state, new_g)] = (state, g_vec)
                continue

            if epsilon > 0:
                dominated_in_open = {g for g in G_open[neigh_state] if eps_dominates(new_g, g, epsilon)}
            else:
                dominated_in_open = {g for g in G_open[neigh_state] if dominates(new_g, g)}
            G_open[neigh_state] -= dominated_in_open

            G_open[neigh_state].add(new_g)
            h_vec = heuristic_vector(neighbor, goals, new_mask, heuristic_maps)
            f_scalar = sum(g + h for g, h in zip(new_g, h_vec))
            heapq.heappush(heap, (f_scalar, tie_breaker, new_g, neighbor, new_mask))
            tie_breaker += 1
            came_from[(neigh_state, new_g)] = (state, g_vec)

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
                    h_vec = heuristic_vector(d_pos, goals, d_mask, heuristic_maps)
                    f_scalar = sum(g + h for g, h in zip(d_g, h_vec))
                    heapq.heappush(heap, (f_scalar, tie_breaker, d_g, d_pos, d_mask))
                    tie_breaker += 1

    elapsed = time.time() - start_time
    if visited_snapshot:
        frames_sparse.append((expansions, visited_snapshot.copy()))

    pareto_paths = []
    for g_vec, goal_state in goal_solutions:
        path = reconstruct_pareto_path(came_from, goal_state, g_vec, start_state, g_start)
        if path:
            pareto_paths.append({
                'path': path,
                'cost_vector': g_vec,
                'construction': g_vec[0],
                'environmental': g_vec[1],
                'geometry': g_vec[2],
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
            print(f"    #{i + 1}: constr={p['construction']:.1f}, "
                  f"env={p['environmental']:.1f}, geo={p['geometry']:.1f}{marker}")
    if selected_path:
        print(f"\n  Selected balanced path: {len(selected_path)} waypoints")

    return selected_path, frames_sparse, pareto_paths


def save_output_data(output_dir, road_bitmap, path, frames_sparse, metadata, pareto_paths=None):
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
        "algorithm": "NAMOA*-dr (cost-map objectives)",
        "num_pareto_solutions": len(pareto_paths) if pareto_paths else 0,
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
    solutions = []
    selected_as_tuples = [tuple(p) for p in selected_path] if selected_path else []

    for idx, p in enumerate(pareto_paths, start=1):
        path_points = [tuple(pt) for pt in p.get("path", [])]
        solution = {
            "solution_id": idx,
            "construction": float(p["construction"]),
            "environmental": float(p["environmental"]),
            "geometry": float(p["geometry"]),
            "num_waypoints": len(path_points),
            "is_selected_balanced": path_points == selected_as_tuples,
            "path_row_col": [[int(r), int(c)] for r, c in path_points],
        }
        solutions.append(solution)

    json_path = os.path.join(output_dir, "namoa_pareto_front.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"algorithm": "NAMOA*-dr (cost-map)",
                    "objectives": ["construction", "environmental", "geometry"],
                    "num_solutions": len(solutions),
                    "solutions": solutions}, f, indent=2)

    csv_path = os.path.join(output_dir, "namoa_pareto_summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "solution_id", "construction", "environmental", "geometry",
            "num_waypoints", "is_selected_balanced"])
        writer.writeheader()
        for s in solutions:
            writer.writerow({k: s[k] for k in writer.fieldnames})

    pareto_dir = os.path.join(output_dir, "pareto_paths")
    os.makedirs(pareto_dir, exist_ok=True)
    for s in solutions:
        arr = np.array(s["path_row_col"], dtype=np.int32)
        np.save(os.path.join(pareto_dir, f"pareto_path_{s['solution_id']:02d}.npy"), arr)


def save_pareto_visualization(output_dir, construction_map, environmental_map, geometry_map,
                              pareto_paths, selected_path):
    """Save multi-panel visualization: cost layers + all Pareto paths."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Panel 1-3: cost layers
    layers = [(construction_map, 'Construction Cost', 'YlOrRd'),
              (environmental_map, 'Environmental Impact', 'Greens'),
              (geometry_map, 'Geometry Cost', 'Blues')]

    for ax, (data, title, cmap) in zip(axes.flat[:3], layers):
        im = ax.imshow(data, cmap=cmap, origin='upper')
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # Overlay all pareto paths
        colors = plt.cm.Set1(np.linspace(0, 1, max(len(pareto_paths), 1)))
        for i, p in enumerate(pareto_paths):
            pa = np.array(p['path'])
            lw = 2.5 if p['path'] == selected_path else 1.0
            ax.plot(pa[:, 1], pa[:, 0], color=colors[i % len(colors)],
                    linewidth=lw, alpha=0.8)

    # Panel 4: Pareto front scatter (3D projected to 2D pairs)
    ax4 = axes[1, 1]
    if pareto_paths:
        c_vals = [p['construction'] for p in pareto_paths]
        e_vals = [p['environmental'] for p in pareto_paths]
        g_vals = [p['geometry'] for p in pareto_paths]
        sc = ax4.scatter(c_vals, e_vals, c=g_vals, cmap='coolwarm', s=80, edgecolors='k', zorder=5)
        plt.colorbar(sc, ax=ax4, label='Geometry Cost')
        for i, p in enumerate(pareto_paths):
            marker = '*' if p['path'] == selected_path else ''
            ax4.annotate(f"#{i + 1}{marker}", (c_vals[i], e_vals[i]),
                         textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax4.set_xlabel('Construction Cost')
    ax4.set_ylabel('Environmental Cost')
    ax4.set_title('Pareto Front (color = geometry)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'NAMOA*-dr Multi-Objective Pathfinding with Real Cost Maps\n'
                 f'{len(pareto_paths)} Pareto-optimal solutions found',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(output_dir, "namoa_costmap_pareto.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Pareto visualization: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="NAMOA*-dr with real cost map objectives")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--fps", type=int, default=ANIMATION_FPS)
    parser.add_argument("--frames", type=int, default=ANIMATION_FRAMES)
    parser.add_argument("--max-expansions", type=int, default=MAX_EXPANSIONS)
    parser.add_argument("--epsilon", type=float, default=0.15,
                        help="Epsilon for approximate Pareto dominance (default: 0.15)")
    parser.add_argument("--no-diagonal", action="store_true")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(__file__), "namoa_costmap_austin")

    print("=" * 70)
    print("NAMOA*-dr Multi-Objective Pathfinding with Real Cost Maps")
    print("=" * 70)

    construction, environmental, geometry = load_cost_maps()
    raster, goals, metadata = load_raster_and_goals()
    cost_maps = (construction, environmental, geometry)

    start = goals[0]
    allow_diagonal = not args.no_diagonal

    print(f"\nStart: {start}, Goals: {goals}")
    print(f"Objectives: construction, environmental, geometry")
    print(f"Epsilon: {args.epsilon}")

    print("\nPrecomputing objective heuristic maps (Dijkstra from each goal)...")
    heuristic_maps = precompute_heuristic_maps(goals, cost_maps, allow_diagonal)
    print("  Heuristic maps ready.")

    print("\n" + "=" * 70)
    path, frames_sparse, pareto_paths = namoa_dr_costmap(
        start, goals, cost_maps, heuristic_maps,
        frame_interval=FRAME_RECORD_INTERVAL,
        max_expansions=args.max_expansions,
        allow_diagonal=allow_diagonal,
        epsilon=args.epsilon)
    print("=" * 70)

    save_output_data(args.output, raster, path, frames_sparse, metadata, pareto_paths)
    save_pareto_visualization(args.output, construction, environmental, geometry,
                              pareto_paths, path)

    if not args.no_animation and path is not None:
        print("\nCreating animations...")
        try:
            animator = AStarAnimator(output_dir=args.output)
            animator.create_animation(output_file="namoa_costmap_animation.gif",
                                      fps=args.fps, target_frames=args.frames,
                                      show_path=True, figsize=(12, 10), dpi=100)
            animator.create_static_comparison(output_file="namoa_costmap_comparison.png",
                                              figsize=(18, 6), dpi=150)
            print(f"  Animation: {args.output}/namoa_costmap_animation.gif")
            print(f"  Comparison: {args.output}/namoa_costmap_comparison.png")
        except Exception as e:
            print(f"Error creating animations: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
