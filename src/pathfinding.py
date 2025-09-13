import json
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
import heapq
import geopandas as gpd

# Load road network
DATA_PATH = "./data/output/LI/LI_roads.geojson"
with open(DATA_PATH) as f:
    geojson = json.load(f)
geometries = [(feature["geometry"], 1) for feature in geojson["features"]]

# Define raster dimensions
width, height = 1000, 1000
minx, miny, maxx, maxy = rasterio.features.bounds(geojson)
transform = from_bounds(minx, miny, maxx, maxy, width, height)

# Rasterize roads
raster = rasterize(geometries, out_shape=(height, width), transform=transform, fill=0, dtype=np.uint8)

# Load and rasterize protected areas
PROTECTED_PATH = "./data/output/LI/protected_areas.geojson"
protected_gdf = gpd.read_file(PROTECTED_PATH)
protected_bitmap = rasterize(
    ((geom, 1) for geom in protected_gdf.geometry),
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype=np.uint8
)

# Define cities and mark goals
point_map = {
    "Eschen": [9.518061467906222, 47.21094022078131],
    "Malbun": [9.607783557623662, 47.10286302280241],
    "Balzers": [9.497583388657375, 47.066238854375655]
}
goal_points = []
for key, (lon, lat) in point_map.items():
    col, row = ~transform * (lon, lat)
    col, row = int(round(col)), int(round(row))
    if 0 <= row < height and 0 <= col < width:
        goal_points.append((row, col))
        raster[row, col] = 2

# Combining protected areas and road/rail paths
combined_raster = np.maximum(raster, protected_bitmap)

# Just roads
plt.imshow(raster, cmap="grey", origin="upper")
plt.title("Raster Bitmap (Only Roads/Rail)")
plt.colorbar(label="Raster Value")
plt.show()

# Including protected areas
plt.imshow(combined_raster, cmap="nipy_spectral", origin="upper")
plt.title("Raster Bitmap (Roads/Rail + Environmentally Protected Areas)")
plt.colorbar(label="Raster Value")
plt.show()

# A* Multi-goal with Protected Area Avoidance
def a_star(heuristic, start_pos, goal_positions, road_bitmap, protected_bitmap):
    visited = set()
    came_from = {}
    goal_flags = tuple([False] * len(goal_positions))
    start_state = updateState(start_pos, 0, goal_flags, goal_positions)
    fringe = [(calculateHeuristicDistance(heuristic, start_pos, goal_flags, goal_positions), start_state)]

    while fringe:
        h_cost, node = heapq.heappop(fringe)
        pos, cost_so_far = node

        if isGoalState(pos[1]):
            path = []
            current = node[0]
            while current in came_from:
                path.append(current[0])  # current[0] is (x, y)
                current = came_from[current]
            path.append(start_pos)  # add the start position
            return path[::-1]

        if pos not in visited:
            visited.add(pos)

            for successor, step_cost in getSuccessors(pos[0], pos[1], road_bitmap, protected_bitmap):
                new_cost = cost_so_far + step_cost
                h = new_cost + calculateHeuristicDistance(heuristic, successor[0], successor[1], goal_positions)
                heapq.heappush(fringe, (h, (successor, new_cost)))
                came_from[successor] = pos

    return None

def getSuccessors(curr_pos, goal_flags, road_bitmap, protected_bitmap):
    directions = [(1,0), (1,1), (-1,0), (-1,-1), (0,1), (0,-1), (-1,1), (1,-1)]
    successors = []
    for dx, dy in directions:
        new_x = curr_pos[0] + dx
        new_y = curr_pos[1] + dy

        if 0 <= new_x < road_bitmap.shape[0] and 0 <= new_y < road_bitmap.shape[1]:
            if protected_bitmap[new_x, new_y] == 1:
                continue  # Skip protected zones (hard constraint)

            cost = 1 if road_bitmap[new_x, new_y] == 1 else 5  # Road preferred
            updated_flags = updateGoalFlags((new_x, new_y), goal_flags, goal_points)
            successors.append((((new_x, new_y), updated_flags), cost))
    return successors

def updateGoalFlags(curr_pos, goal_flags, goal_positions):
    flags = list(goal_flags)
    for i, g in enumerate(goal_positions):
        if curr_pos == g:
            flags[i] = True
    return tuple(flags)

def updateState(curr_pos, cost, goal_flags, goal_positions):
    updated_flags = updateGoalFlags(curr_pos, goal_flags, goal_positions)
    return ((curr_pos, updated_flags), cost)

def isGoalState(goal_flags):
    return all(goal_flags)

def calculateHeuristicDistance(heuristic, curr, visited_flags, goals):
    dists = []
    for i, g in enumerate(goals):
        if not visited_flags[i]:
            if heuristic == "euclidean":
                d = ((g[0] - curr[0])**2 + (g[1] - curr[1])**2)**0.5
            elif heuristic == "manhattan":
                d = abs(g[0] - curr[0]) + abs(g[1] - curr[1])
            else:
                d = 0
            dists.append(d)
    return max(dists) if dists else 0

# Run A*
start = goal_points[0]
path = a_star("euclidean", start, goal_points, raster, protected_bitmap)

# Visualize result
if path:
    path_img = raster.copy()
    for (x, y) in path:
        path_img[x, y] = 3
    plt.imshow(path_img, cmap="hot", origin="upper")
    plt.title("A* Multi-Goal Path with Protected Area Avoidance")
    plt.colorbar(label="Raster Value")
    plt.show()
else:
    print("No valid path found.")
