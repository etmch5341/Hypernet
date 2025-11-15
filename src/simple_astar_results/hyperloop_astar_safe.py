# hyperloop_astar_safe.py
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

# -------------------- CONFIG --------------------
# File paths (keep your original names)
ROADS_GEOJSON = "./LI_roads.geojson"
PROTECTED_GEOJSON = "./protected_areas.geojson"

# Raster resolution (reduce to reduce memory/time). Change if you need more detail.
WIDTH = 300   # default smaller than 500
HEIGHT = 300

# Whether to allow diagonal moves (increases branching and runtime)
ALLOW_DIAGONAL = True

# How often to record a sparse "frame" of visited points (store lists of coords).
FRAME_RECORD_INTERVAL = 1000  # raise to 10000+ to use less memory; set None to disable

# Cost on-road vs off-road
ON_ROAD_COST = 1
OFF_ROAD_COST = 5

# Maximum nodes to expand as a safety cap (optional)
MAX_EXPANSIONS = 5_000_000  # huge but prevents runaway; set None to disable
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

# convert lon/lat to raster indices safely
def lonlat_to_cell(transform, lon, lat, width, height):
    col, row = ~transform * (lon, lat)
    col, row = int(round(col)), int(round(row))
    if 0 <= row < height and 0 <= col < width:
        return (row, col)
    else:
        return None

# Heuristic: distance to nearest remaining goal + MST cost over remaining goals (Euclidean)
def mst_cost(points):
    # Prim's on small sets (<= 10 goals) is cheap
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
    # goals: list of (row, col)
    remaining = [g for i,g in enumerate(goals) if not (visited_mask & (1<<i))]
    if not remaining:
        return 0.0
    # min distance to any remaining goal
    min_to_goal = min(math.hypot(g[0]-curr[0], g[1]-curr[1]) for g in remaining)
    # MST over remaining goals
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
                # Skip protected cells entirely
                continue
            # diagonal cost is sqrt(2) if diagonal, else 1; then scaled by on/off-road
            base = math.hypot(dx, dy)
            terrain_cost = ON_ROAD_COST if road_bitmap[nx, ny] == 1 else OFF_ROAD_COST
            yield ( (nx, ny), base * terrain_cost )

def reconstruct_path(came_from, current_state, start_state):
    # came_from maps (pos,mask) -> (prev_pos, prev_mask)
    path = []
    s = current_state
    while s != start_state:
        path.append(s[0])
        s = came_from.get(s)
        if s is None:
            # something went wrong
            return None
    path.append(start_state[0])
    return path[::-1]

def a_star_multi(start, goals, road_bitmap, protected_bitmap, frame_interval=FRAME_RECORD_INTERVAL, max_expansions=MAX_EXPANSIONS):
    start_mask = update_goal_bitmask(start, 0, goals)
    total_goals = len(goals)

    # Priority queue: (f = g + h, g, pos, mask)
    heap = []
    start_h = heuristic(start, goals, start_mask)
    heapq.heappush(heap, (start_h, 0.0, start, start_mask))

    # Best g seen for (pos,mask)
    best_g = {}                # dict: (pos,mask) -> g
    came_from = {}             # dict: (pos,mask) -> (prev_pos, prev_mask)
    frames_sparse = []         # list of (expansion_count, list_of_positions) snapshots (sparse)
    visited_snapshot = []      # accumulate visited positions until frame record

    expansions = 0
    start_time = time.time()

    start_state = (start, start_mask)

    while heap:
        f, g, current, mask = heapq.heappop(heap)
        state = (current, mask)

        # Skip stale entries
        if best_g.get(state, float('inf')) <= g:
            continue

        best_g[state] = g

        # record visited point sparsely
        visited_snapshot.append(current)
        expansions += 1
        if frame_interval and (expansions % frame_interval == 0):
            # store a copy of the recent visited points (small)
            frames_sparse.append((expansions, visited_snapshot.copy()))
            visited_snapshot.clear()

        # Safety cap
        if max_expansions is not None and expansions > max_expansions:
            print(f"[a_star_multi] Reached max expansions ({max_expansions}). Aborting search.")
            break

        # Goal check
        if is_goal_state(mask, total_goals):
            elapsed = time.time() - start_time
            # append remaining snapshot
            if visited_snapshot:
                frames_sparse.append((expansions, visited_snapshot.copy()))
                visited_snapshot.clear()
            final_state = (current, mask)
            path = reconstruct_path(came_from, final_state, start_state)
            print(f"[a_star_multi] Found path in {expansions} expansions, time {elapsed:.2f}s")
            return path, frames_sparse

        # Expand neighbors
        for neighbor, step_cost in get_successors(current, road_bitmap, protected_bitmap):
            new_g = g + step_cost
            new_mask = update_goal_bitmask(neighbor, mask, goals)
            neigh_state = (neighbor, new_mask)

            # If we've already got a better g for this exact state, skip
            if best_g.get(neigh_state, float('inf')) <= new_g:
                continue

            h = heuristic(neighbor, goals, new_mask)
            heapq.heappush(heap, (new_g + h, new_g, neighbor, new_mask))
            came_from[neigh_state] = state

    # if we exit the loop without finding all goals
    print("[a_star_multi] No path found (or search aborted).")
    if visited_snapshot:
        frames_sparse.append((expansions, visited_snapshot.copy()))
    return None, frames_sparse

def main():
    road_bitmap, protected_bitmap, transform, bounds = load_and_rasterize(width=WIDTH, height=HEIGHT)
    minx, miny, maxx, maxy = bounds
    print(f"Bounds: {bounds}")
    print(f"Raster shape: {road_bitmap.shape}")

    # your named points (lon, lat)
    point_map = {
        "Eschen": [9.518061467906222, 47.21094022078131],
        "Malbun": [9.607783557623662, 47.10286302280241],
        "Balzers": [9.497583388657375, 47.066238854375655]
    }

    goals = []
    for lon, lat in point_map.values():
        cell = lonlat_to_cell(transform, lon, lat, WIDTH, HEIGHT)
        if cell:
            goals.append(cell)
        else:
            raise ValueError(f"Goal {lon,lat} is outside raster bounds!")

    if len(goals) < 2:
        raise ValueError("Need at least 2 goals for multi-goal routing")

    start = goals[0]

    print("Goals (raster coords):", goals)
    print("Start:", start)
    path, frames_sparse = a_star_multi(start, goals, road_bitmap, protected_bitmap,
                                       frame_interval=FRAME_RECORD_INTERVAL,
                                       max_expansions=MAX_EXPANSIONS)

    # Save outputs
    out_dir = "./astar_output"
    os.makedirs(out_dir, exist_ok=True)

    # Save big bitmaps compressed (safe for large arrays)
    np.savez_compressed(os.path.join(out_dir, "road_bitmap.npz"), road_bitmap=np.asarray(road_bitmap, dtype=np.uint8))
    np.savez_compressed(os.path.join(out_dir, "protected_bitmap.npz"), protected_bitmap=np.asarray(protected_bitmap, dtype=np.uint8))

    # Save transform + bounds (pickle is fine; these are small)
    meta = {"transform": transform, "bounds": bounds}
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    # Save sparse frames (pickle) - frames_sparse is a list of tuples (expansion_count, [positions])
    with open(os.path.join(out_dir, "astar_sparse_frames.pkl"), "wb") as f:
        pickle.dump(frames_sparse, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save final path as compact integer array (or empty array)
    if path:
        path_arr = np.array(path, dtype=np.int32)  # shape (N,2)
    else:
        path_arr = np.zeros((0,2), dtype=np.int32)
    np.save(os.path.join(out_dir, "astar_final_path.npy"), path_arr)

    print(f"Saved outputs to {out_dir}")

# if __name__ == "__main__":
#     main()

main()