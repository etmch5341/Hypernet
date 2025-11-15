# hyperloop_astar.py
import json
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import heapq
import geopandas as gpd
import matplotlib.pyplot as plt

# --- Load and Rasterize Road Network ---
with open("./data/output/LI/LI_roads.geojson") as f:
    geojson = json.load(f)

geometries = [(feature["geometry"], 1) for feature in geojson["features"]]
width, height = 500, 500
minx, miny, maxx, maxy = rasterio.features.bounds(geojson)
transform = from_bounds(minx, miny, maxx, maxy, width, height)
raster = rasterize(geometries, out_shape=(height, width), transform=transform, fill=0, dtype=np.uint8)

# Pixel size in the X direction (longitude/easting)
pixel_width = (maxx - minx) / width

# Pixel size in the Y direction (latitude/northing)  
pixel_height = (maxy - miny) / height

print(f"Bounds: ({minx}, {miny}) to ({maxx}, {maxy})")
print(f"Pixel width: {(maxx - minx) / width} degrees")
print(f"Pixel height: {(maxy - miny) / height} degrees")
print(f"Approximate pixel size: {(maxx - minx) / width * 111000} meters")  # rough conversion at this latitude

# --- Rasterize Protected Areas ---
protected_gdf = gpd.read_file("./data/output/LI/protected_areas.geojson")
protected_bitmap = rasterize(
    ((geom, 1) for geom in protected_gdf.geometry),
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype=np.uint8
)

# --- Define Goals ---
point_map = {
    "Eschen": [9.518061467906222, 47.21094022078131],
    "Malbun": [9.607783557623662, 47.10286302280241],
    "Balzers": [9.497583388657375, 47.066238854375655]
}
goal_points = []
for lon, lat in point_map.values():
    col, row = ~transform * (lon, lat)
    col, row = int(round(col)), int(round(row))
    if 0 <= row < height and 0 <= col < width:
        goal_points.append((row, col))

# --- A* with Bitmask Goal Flags and Animation Recording ---
def bitmask_goals(goal_flags):
    return sum((1 << i) if goal else 0 for i, goal in enumerate(goal_flags))

def update_goal_bitmask(pos, bitmask, goal_positions):
    for i, g in enumerate(goal_positions):
        if pos == g:
            bitmask |= (1 << i)
    return bitmask

def is_goal_state(bitmask, total_goals):
    return bitmask == (1 << total_goals) - 1

def heuristic(curr, goals, visited_mask):
    return max(
        ((g[0] - curr[0])**2 + (g[1] - curr[1])**2)**0.5
        for i, g in enumerate(goals) if not visited_mask & (1 << i)
    ) if visited_mask != (1 << len(goals)) - 1 else 0

def get_successors(pos, bitmask, road_bitmap, protected_bitmap):
    directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (-1,1), (1,-1)]
    successors = []
    for dx, dy in directions:
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < road_bitmap.shape[0] and 0 <= ny < road_bitmap.shape[1]:
            if protected_bitmap[nx, ny]:
                continue
            cost = 1 if road_bitmap[nx, ny] == 1 else 5
            successors.append(((nx, ny), cost))
    return successors

# Main A* function with animation frame logging
def a_star(start, goals, road_bitmap, protected_bitmap):
    total_goals = len(goals)
    visited = {}
    came_from = {}
    frames = []

    initial_mask = update_goal_bitmask(start, 0, goals)
    heap = [(heuristic(start, goals, initial_mask), 0, start, initial_mask)]

    while heap:
        f, g, current, bitmask = heapq.heappop(heap)
        state = (current, bitmask)

        if state in visited and visited[state] <= g:
            continue
        visited[state] = g

        if is_goal_state(bitmask, total_goals):
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1], frames

        for neighbor, step_cost in get_successors(current, bitmask, road_bitmap, protected_bitmap):
            new_bitmask = update_goal_bitmask(neighbor, bitmask, goals)
            heapq.heappush(heap, (
                g + step_cost + heuristic(neighbor, goals, new_bitmask),
                g + step_cost,
                neighbor,
                new_bitmask
            ))
            came_from[neighbor] = current

        if len(frames) == 0 or len(visited) % 100 == 0:
            frame = np.copy(road_bitmap)
            for (p, _) in visited:
                frame[p[0], p[1]] = 7
            frames.append(frame)

    return None, frames

# Run the search
start = goal_points[0]
path, animation_frames = a_star(start, goal_points, raster, protected_bitmap)

# Save animation data
np.save("./data/astar_animation_frames.npy", animation_frames)
np.save("./data/astar_final_path.npy", path)

# Optional preview
if path:
    for (x, y) in path:
        raster[x, y] = 3
    plt.imshow(raster, cmap="hot", origin="upper")
    plt.title("Final A* Path")
    plt.show()
else:
    print("No valid path found.")
