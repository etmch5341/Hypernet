# hyperloop_astar_with_animation.py
"""
Enhanced A* pathfinding with built-in animation support.
This version automatically generates animations after finding a path.
"""

import json
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import heapq
import geopandas as gpd
import math
import time
import os
import pickle
from astar_animator import AStarAnimator

# -------------------- CONFIG --------------------
# File paths
ROADS_GEOJSON = "./LI_roads.geojson"
PROTECTED_GEOJSON = "./protected_areas.geojson"

# Raster resolution
WIDTH = 300
HEIGHT = 300

# Whether to allow diagonal moves
ALLOW_DIAGONAL = True

# How often to record a sparse "frame" of visited points
FRAME_RECORD_INTERVAL = 5000

# Cost on-road vs off-road
ON_ROAD_COST = 1
OFF_ROAD_COST = 5

# Maximum nodes to expand as a safety cap
MAX_EXPANSIONS = 5_000_000

# Animation settings
CREATE_ANIMATION = True  # Set to False to skip animation
ANIMATION_FPS = 15
ANIMATION_FRAMES = 150
# ------------------------------------------------

def load_and_rasterize(width=WIDTH, height=HEIGHT):
    with open(ROADS_GEOJSON) as f:
        geojson = json.load(f)
    geometries = [(feature["geometry"], 1) for feature in geojson["features"]]
    minx, miny, maxx, maxy = rasterio.features.bounds(geojson)
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    road_bitmap = rasterize(geometries, out_shape=(height, width), transform=transform, fill=0, dtype=np.uint8)

    protected_gdf = gpd.read_file(PROTECTED_GEOJSON)
    protected_bitmap = rasterize(
        ((geom, 1) for geom in protected_gdf.geometry),
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )

    return road_bitmap, protected_bitmap, transform, (minx, miny, maxx, maxy)

def lonlat_to_cell(transform, lon, lat, width, height):
    col, row = ~transform * (lon, lat)
    col, row = int(round(col)), int(round(row))
    if 0 <= row < height and 0 <= col < width:
        return (row, col)
    else:
        return None

def mst_cost(points):
    if not points:
        return 0.0
    pts = list(points)
    n = len(pts)
    used = [False]*n
    dist = [float('inf')]*n
    dist[0] = 0.0
    total = 0.0
    for _ in range(n):
        u = -1
        best = float('inf')
        for i in range(n):
            if not used[i] and dist[i] < best:
                best = dist[i]; u = i
        used[u] = True
        total += dist[u]
        for v in range(n):
            if not used[v]:
                d = math.hypot(pts[u][0]-pts[v][0], pts[u][1]-pts[v][1])
                if d < dist[v]:
                    dist[v] = d
    return total

def heuristic(curr, goals, visited_mask):
    remaining = [g for i,g in enumerate(goals) if not (visited_mask & (1<<i))]
    if not remaining:
        return 0.0
    min_to_goal = min(math.hypot(g[0]-curr[0], g[1]-curr[1]) for g in remaining)
    mst = mst_cost(remaining)
    return min_to_goal + mst

def update_goal_bitmask(pos, bitmask, goal_positions):
    for i, g in enumerate(goal_positions):
        if pos == g:
            bitmask |= (1 << i)
    return bitmask

def is_goal_state(bitmask, total_goals):
    return bitmask == (1 << total_goals) - 1

def get_successors(pos, road_bitmap, protected_bitmap, allow_diagonal=ALLOW_DIAGONAL):
    if allow_diagonal:
        directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (-1,1), (1,-1)]
    else:
        directions = [(1,0), (-1,0), (0,1), (0,-1)]
    h, w = road_bitmap.shape
    for dx, dy in directions:
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < h and 0 <= ny < w:
            if protected_bitmap[nx, ny]:
                continue
            base = math.hypot(dx, dy)
            terrain_cost = ON_ROAD_COST if road_bitmap[nx, ny] == 1 else OFF_ROAD_COST
            yield ( (nx, ny), base * terrain_cost )

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

def a_star_multi(start, goals, road_bitmap, protected_bitmap, frame_interval=FRAME_RECORD_INTERVAL, max_expansions=MAX_EXPANSIONS):
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

        if max_expansions is not None and expansions > max_expansions:
            print(f"[a_star_multi] Reached max expansions ({max_expansions}). Aborting search.")
            break

        if is_goal_state(mask, total_goals):
            elapsed = time.time() - start_time
            if visited_snapshot:
                frames_sparse.append((expansions, visited_snapshot.copy()))
                visited_snapshot.clear()
            final_state = (current, mask)
            path = reconstruct_path(came_from, final_state, start_state)
            print(f"[a_star_multi] Found path in {expansions} expansions, time {elapsed:.2f}s")
            return path, frames_sparse

        for neighbor, step_cost in get_successors(current, road_bitmap, protected_bitmap):
            new_g = g + step_cost
            new_mask = update_goal_bitmask(neighbor, mask, goals)
            neigh_state = (neighbor, new_mask)

            if best_g.get(neigh_state, float('inf')) <= new_g:
                continue

            h = heuristic(neighbor, goals, new_mask)
            heapq.heappush(heap, (new_g + h, new_g, neighbor, new_mask))
            came_from[neigh_state] = state

    print("[a_star_multi] No path found (or search aborted).")
    if visited_snapshot:
        frames_sparse.append((expansions, visited_snapshot.copy()))
    return None, frames_sparse

def main():
    print("=" * 60)
    print("A* Pathfinding with Animation")
    print("=" * 60)
    
    road_bitmap, protected_bitmap, transform, bounds = load_and_rasterize(width=WIDTH, height=HEIGHT)
    minx, miny, maxx, maxy = bounds
    print(f"\nBounds: {bounds}")
    print(f"Raster shape: {road_bitmap.shape}")

    # Define points
    point_map = {
        "Eschen": [9.518061467906222, 47.21094022078131],
        "Malbun": [9.607783557623662, 47.10286302280241],
        "Balzers": [9.497583388657375, 47.066238854375655]
    }

    goals = []
    for name, (lon, lat) in point_map.items():
        cell = lonlat_to_cell(transform, lon, lat, WIDTH, HEIGHT)
        if cell:
            goals.append(cell)
            print(f"  {name}: {cell}")
        else:
            raise ValueError(f"Goal {name} at {lon,lat} is outside raster bounds!")

    if len(goals) < 2:
        raise ValueError("Need at least 2 goals for multi-goal routing")

    start = goals[0]
    print(f"\nStarting from: {start}")
    print(f"Total goals: {len(goals)}")
    
    print("\n" + "=" * 60)
    print("Running A* search...")
    print("=" * 60)
    
    path, frames_sparse = a_star_multi(start, goals, road_bitmap, protected_bitmap,
                                       frame_interval=FRAME_RECORD_INTERVAL,
                                       max_expansions=MAX_EXPANSIONS)

    # Save outputs
    out_dir = "./astar_output"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nSaving data to {out_dir}...")
    
    # Save bitmaps
    np.savez_compressed(os.path.join(out_dir, "road_bitmap.npz"), 
                       road_bitmap=np.asarray(road_bitmap, dtype=np.uint8))
    np.savez_compressed(os.path.join(out_dir, "protected_bitmap.npz"), 
                       protected_bitmap=np.asarray(protected_bitmap, dtype=np.uint8))

    # Save metadata
    meta = {"transform": transform, "bounds": bounds}
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    # Save sparse frames
    with open(os.path.join(out_dir, "astar_sparse_frames.pkl"), "wb") as f:
        pickle.dump(frames_sparse, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save final path
    if path:
        path_arr = np.array(path, dtype=np.int32)
        print(f"Path found with {len(path)} waypoints")
    else:
        path_arr = np.zeros((0,2), dtype=np.int32)
        print("No path found!")
    np.save(os.path.join(out_dir, "astar_final_path.npy"), path_arr)

    print("Data saved successfully!")

    # Create animations if requested
    if CREATE_ANIMATION:
        print("\n" + "=" * 60)
        print("Creating animations...")
        print("=" * 60)
        
        try:
            animator = AStarAnimator(output_dir=out_dir)
            
            # Create animated GIF
            animator.create_animation(
                output_file="astar_animation.gif",
                fps=ANIMATION_FPS,
                target_frames=ANIMATION_FRAMES,
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
            print(f"  - Animation: {out_dir}/astar_animation.gif")
            print(f"  - Comparison: {out_dir}/astar_comparison.png")
            
        except Exception as e:
            print(f"\n✗ Error creating animations: {e}")
            print("  (Data was still saved successfully)")
    
    print("\n" + "=" * 60)
    print("Process complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
