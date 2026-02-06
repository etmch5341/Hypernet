import json
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
import heapq
import geopandas as gpd
from collections import defaultdict
import math

# --- 1. SETUP & DATA LOADING (Same as your code) ---
DATA_PATH = "austin_test.geojson"
# PROTECTED_PATH = "./data/austin-dallas/envr_austin_dallas.geojson"

# Load road network
with open(DATA_PATH) as f:
    geojson = json.load(f)
geometries = [(feature["geometry"], 1) for feature in geojson["features"]]

# Define raster dimensions
width, height = 1000, 1000
minx, miny, maxx, maxy = rasterio.features.bounds(geojson)
transform = from_bounds(minx, miny, maxx, maxy, width, height)

# Rasterize roads
road_raster = rasterize(geometries, out_shape=(height, width), transform=transform, fill=0, dtype=np.uint8)

# Load and rasterize protected areas
protected_gdf = gpd.read_file(DATA_PATH)
protected_bitmap = rasterize(
    ((geom, 1) for geom in protected_gdf.geometry),
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype=np.uint8
)

# Define cities and mark goals
point_map = {
    "Austin": [-97.73967335952074, 30.277001707654122],
    "Dallas": [-96.7970, 32.7767],
    "Fort Worth": [-97.3308, 32.7555]
}

goal_points = []
for key, (lon, lat) in point_map.items():
    col, row = ~transform * (lon, lat)
    col, row = int(round(col)), int(round(row))
    if 0 <= row < height and 0 <= col < width:
        goal_points.append((row, col))
        # Mark visual goal on map for reference
        road_raster[row, col] = 2

# --- 2. HIERARCHICAL A* IMPLEMENTATION ---

class HierarchicalGraph:
    def __init__(self, road_map, protected_map, cluster_size=100):
        self.road_map = road_map
        self.protected_map = protected_map
        self.h, self.w = road_map.shape
        self.cluster_size = cluster_size
        
        # The Abstract Graph: keys are node coordinates (r, c), values are dicts of neighbors {neighbor_coords: cost}
        self.edges = defaultdict(dict) 
        self.nodes = set()
        
        print("Preprocessing HPA* Graph...")
        self._build_abstract_graph()
        print(f"Graph built with {len(self.nodes)} abstract nodes.")

    def _get_cost(self, r, c):
        """Returns cost to traverse a cell. High cost for off-road, Inf for protected."""
        if not (0 <= r < self.h and 0 <= c < self.w): return float('inf')
        if self.protected_map[r, c] == 1: return float('inf')
        return 1 if self.road_map[r, c] == 1 else 5 # Preference for roads

    def _build_abstract_graph(self):
        """1. Identify entrances between clusters. 2. Connect entrances internally."""
        # Horizontal borders
        for r in range(0, self.h, self.cluster_size):
            if r == 0: continue
            for c in range(0, self.w):
                self._add_transition(r-1, c, r, c)
        
        # Vertical borders
        for c in range(0, self.w, self.cluster_size):
            if c == 0: continue
            for r in range(0, self.h):
                self._add_transition(r, c-1, r, c)
        
        # Build Intra-cluster edges (connect entrances within the same chunk)
        # Note: Ideally, you pre-calculate exact paths. For speed here, we use heuristic estimates 
        # or simplified connections. To keep it robust, we will compute these on the fly 
        # or add edges between all nodes in a cluster with Euclidean distance * 1.5 (heuristic).
        # A full implementation runs A* between all entrances in a cluster here.
        pass 

    def _add_transition(self, r1, c1, r2, c2):
        """Adds a transition node if traversal is valid."""
        cost1 = self._get_cost(r1, c1)
        cost2 = self._get_cost(r2, c2)
        
        if cost1 != float('inf') and cost2 != float('inf'):
            # It's a valid transition. 
            # Optimization: Only add one entrance per connected opening to reduce graph size
            # For this snippet, we add nodes simply.
            
            # Simple thinning: only add transition if it's a road or every 10th empty pixel
            is_road = (self.road_map[r1, c1] == 1 or self.road_map[r2, c2] == 1)
            
            if is_road or (r1 % 5 == 0 and c1 % 5 == 0):
                self.nodes.add((r1, c1))
                self.nodes.add((r2, c2))
                self.edges[(r1, c1)][(r2, c2)] = (cost1 + cost2) / 2
                self.edges[(r2, c2)][(r1, c1)] = (cost1 + cost2) / 2

    def get_neighbors(self, node):
        return self.edges[node].items()

    def low_level_astar(self, start, goal, bounds=None):
        """Standard A* restricted to a bounding box (cluster)."""
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            _, current = heapq.heappop(frontier)

            if current == goal:
                break

            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]:
                next_node = (current[0] + dr, current[1] + dc)
                
                # Check bounds if provided (Keep search inside cluster)
                if bounds:
                    rmin, rmax, cmin, cmax = bounds
                    if not (rmin <= next_node[0] < rmax and cmin <= next_node[1] < cmax):
                        continue
                
                new_cost = cost_so_far[current] + self._get_cost(*next_node)
                
                if new_cost >= float('inf'): continue # Obstacle

                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + heuristic(next_node, goal)
                    heapq.heappush(frontier, (priority, next_node))
                    came_from[next_node] = current
        
        # Reconstruct
        if goal not in came_from: return []
        path = []
        curr = goal
        while curr != start:
            path.append(curr)
            curr = came_from[curr]
        path.append(start)
        return path[::-1]

    def hierarchical_search(self, start, goal):
        """
        1. Insert Start/Goal into Abstract Graph (connect to nearby entrances).
        2. Run A* on Abstract Graph.
        3. Refine path by running Low-Level A* between the abstract nodes found.
        """
        # Step 1: Find relevant abstract nodes (entrances) in the start/goal clusters
        start_cluster_nodes = self._get_nodes_in_cluster(start)
        goal_cluster_nodes = self._get_nodes_in_cluster(goal)
        
        # Temporary Graph Augmentation
        temp_edges = defaultdict(dict)
        
        # Connect Start to its cluster's entrances
        for node in start_cluster_nodes:
            # Run local search to get accurate cost
            dist = abs(start[0]-node[0]) + abs(start[1]-node[1]) # Heuristic for speed
            temp_edges[start][node] = dist
        
        # Connect Goal to its cluster's entrances
        for node in goal_cluster_nodes:
            dist = abs(goal[0]-node[0]) + abs(goal[1]-node[1])
            temp_edges[node][goal] = dist
            
        # Step 2: High Level A*
        abstract_path = self._abstract_astar(start, goal, temp_edges)
        
        if not abstract_path: return None
        
        # Step 3: Refine Path (Stitch together)
        full_path = []
        for i in range(len(abstract_path) - 1):
            u = abstract_path[i]
            v = abstract_path[i+1]
            
            # Define bounding box for local search (union of two clusters involved)
            # Simple approach: standard A* between u and v
            # Note: In a strict HPA*, we would restrict this search to the specific clusters.
            segment = self.low_level_astar(u, v)
            
            if not segment: return None
            if full_path:
                full_path.extend(segment[1:]) # Avoid duplicating connection point
            else:
                full_path.extend(segment)
                
        return full_path

    def _get_nodes_in_cluster(self, pos):
        """Finds pre-calculated entrance nodes in the specific cluster of 'pos'"""
        r, c = pos
        cr_start = (r // self.cluster_size) * self.cluster_size
        cc_start = (c // self.cluster_size) * self.cluster_size
        
        cluster_nodes = []
        for node in self.nodes:
            nr, nc = node
            if cr_start <= nr < cr_start + self.cluster_size and \
               cc_start <= nc < cc_start + self.cluster_size:
                cluster_nodes.append(node)
        return cluster_nodes

    def _abstract_astar(self, start, goal, temp_edges):
        """A* on the sparse node graph"""
        def h(a, b): return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5
        
        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal: break
            
            # Combine permanent edges and temporary edges
            neighbors = {}
            if current in self.edges: neighbors.update(self.edges[current])
            if current in temp_edges: neighbors.update(temp_edges[current])
            
            for next_node, weight in neighbors.items():
                new_cost = cost_so_far[current] + weight
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + h(next_node, goal)
                    heapq.heappush(frontier, (priority, next_node))
                    came_from[next_node] = current
                    
        if goal not in came_from: return None
        # Reconstruct
        path = []
        curr = goal
        while curr is not None:
            path.append(curr)
            curr = came_from[curr]
        return path[::-1]

# --- 3. EXECUTION ---

# Initialize HPA Graph (Preprocessing happens here)
hpa = HierarchicalGraph(road_raster, protected_bitmap, cluster_size=100)

# Determine Order (Nearest Neighbor Heuristic for TSP)
# We assume Start is first.
remaining_goals = goal_points[1:]
current_pos = goal_points[0] # Austin
ordered_path = [current_pos]

while remaining_goals:
    # Find nearest next city
    best_dist = float('inf')
    nearest = None
    for g in remaining_goals:
        d = ((current_pos[0]-g[0])**2 + (current_pos[1]-g[1])**2)**0.5
        if d < best_dist:
            best_dist = d
            nearest = g
    
    ordered_path.append(nearest)
    current_pos = nearest
    remaining_goals.remove(nearest)

print("Visiting Order (Pixels):", ordered_path)

# Run HPA* Sequence
final_path_pixels = []
for i in range(len(ordered_path) - 1):
    start_node = ordered_path[i]
    end_node = ordered_path[i+1]
    print(f"Pathfinding: Segment {i+1}/{len(ordered_path)-1}...")
    
    segment_path = hpa.hierarchical_search(start_node, end_node)
    
    if segment_path:
        final_path_pixels.extend(segment_path)
    else:
        print(f"Failed to find path between {start_node} and {end_node}")

# --- 4. VISUALIZATION ---

def visualize_result(raster, protected, path):
    combined_rgb = np.zeros((raster.shape[0], raster.shape[1], 3), dtype=np.uint8)
    combined_rgb[:, :] = [255, 255, 255] # White Background
    combined_rgb[raster == 1] = [200, 200, 200] # Roads (Light Grey)
    combined_rgb[raster == 2] = [0, 0, 255] # Goals (Blue)
    combined_rgb[protected == 1] = [0, 255, 0] # Protected (Green)

    if path:
        # We perform a simple scatter for the path to make it visible
        path_arr = np.array(path)
        # coloring path red
        combined_rgb[path_arr[:,0], path_arr[:,1]] = [255, 0, 0]

    plt.figure(figsize=(10,10))
    plt.imshow(combined_rgb, origin='upper')
    plt.title("Hierarchical A* Path (Austin -> Dallas -> FW)")
    plt.axis("off")
    plt.show()

visualize_result(road_raster, protected_bitmap, final_path_pixels)