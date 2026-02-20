#!/usr/bin/env python3
"""
Hierarchical (HPA*) A* Pathfinding from NPZ Test Sets

This module builds a cluster/gateway abstraction over a road bitmap,
runs a hierarchical A* (abstract search over gateways + local refinement),
and writes output (bitmaps, sparse frames, final refined path).

FIXES APPLIED:
1. Gateway creation now properly creates TWO nodes (one on each side of boundary)
2. Intra-cluster edges use proper bidirectional A* pathfinding
3. Path refinement reconstructs in correct order
4. Neighbor iteration handles cases where nodes have no edges
5. Added proper cluster boundary validation
"""
from hpastar_animator import HPAStarAnimator
import argparse
import numpy as np
import heapq
import math
import time
import os
import pickle

# -------------------- CONFIG --------------------
DEFAULT_CLUSTER_SIZE = 40      # Default cluster size (pixels)
CLUSTER_SIZE = DEFAULT_CLUSTER_SIZE
ALLOW_DIAGONAL = True
FRAME_RECORD_INTERVAL = 100
ON_ROAD_COST = 1.0
OFF_ROAD_COST = 5.0
MAX_EXPANSIONS = 5_000_000

ANIMATION_FPS = 15
ANIMATION_FRAMES = 150
# ------------------------------------------------


class HierarchicalGraph:
    """Abstracts the raster into clusters and gateways."""

    def __init__(self, road_bitmap, cluster_size=None):
        if cluster_size is None:
            cluster_size = DEFAULT_CLUSTER_SIZE
        self.grid = road_bitmap
        self.c_size = max(1, int(cluster_size))
        self.height, self.width = road_bitmap.shape
        self.nodes = {}       # node_coord -> list of (neighbor_coord, weight)
        self.clusters = {}    # cluster_id -> list of node_coords
        self._build_graph()

    def _get_cluster_id(self, r, c):
        return (r // self.c_size, c // self.c_size)

    def _add_node_if_missing(self, p):
        if p not in self.nodes:
            self.nodes[p] = []

    def _add_edge(self, p1, p2, weight):
        self._add_node_if_missing(p1)
        self._add_node_if_missing(p2)
        if not any(n == p2 for n, _ in self.nodes[p1]):
            self.nodes[p1].append((p2, weight))
        if not any(n == p1 for n, _ in self.nodes[p2]):
            self.nodes[p2].append((p1, weight))

    def _build_graph(self):
        # collect gateways along cluster boundaries
        num_cx = (self.width + self.c_size - 1) // self.c_size
        num_cy = (self.height + self.c_size - 1) // self.c_size

        # vertical boundaries
        for r0 in range(0, self.height, self.c_size):
            for cc in range(self.c_size, self.width, self.c_size):
                self._find_gateways(r0, cc - 1, r0, cc, vertical=True)

        # horizontal boundaries
        for rr in range(self.c_size, self.height, self.c_size):
            for c0 in range(0, self.width, self.c_size):
                self._find_gateways(rr - 1, c0, rr, c0, vertical=False)

        # group nodes into clusters
        for node in list(self.nodes.keys()):
            cid = self._get_cluster_id(*node)
            self.clusters.setdefault(cid, []).append(node)

        # connect all nodes inside a cluster with local costs
        for cid, nodes in self.clusters.items():
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    a = nodes[i]
                    b = nodes[j]
                    # FIX: Check if nodes are actually in the same cluster
                    if self._get_cluster_id(*a) == cid and self._get_cluster_id(*b) == cid:
                        cost = self.local_astar_dist(a, b, cid)
                        if cost < float('inf'):
                            self._add_edge(a, b, cost)

    def _find_gateways(self, r1, c1, r2, c2, vertical):
        """
        FIX: Properly create gateway nodes on BOTH sides of cluster boundary.
        Original bug: Only created edge between boundary cells, didn't add both as nodes.
        """
        segment = []
        length = self.c_size
        for i in range(length):
            r_a = (r1 + i) if vertical else r1
            c_a = c1 if vertical else (c1 + i)
            r_b = (r2 + i) if vertical else r2
            c_b = c2 if vertical else (c2 + i)

            if 0 <= r_a < self.height and 0 <= c_a < self.width and \
               0 <= r_b < self.height and 0 <= c_b < self.width:
                a_is_road = self.grid[r_a, c_a] == 1
                b_is_road = self.grid[r_b, c_b] == 1
                if a_is_road and b_is_road:
                    segment.append(((r_a, c_a), (r_b, c_b)))
                else:
                    if segment:
                        self._create_gateway_from_segment(segment)
                        segment = []
        if segment:
            self._create_gateway_from_segment(segment)

    def _create_gateway_from_segment(self, segment):
        """
        FIX: Create proper gateway nodes on both sides of boundary.
        A gateway should have nodes in both adjacent clusters.
        """
        mid = len(segment) // 2
        p_a, p_b = segment[mid]
        
        # Both points should be added as nodes in the graph
        self._add_node_if_missing(p_a)
        self._add_node_if_missing(p_b)
        
        # Connect them with edge cost (they're adjacent across boundary)
        self._add_edge(p_a, p_b, 1.0)

    def local_astar_dist(self, start, goal, cluster_id=None):
        """
        FIX: Added proper validation that both start and goal are within cluster bounds.
        """
        # restrict to cluster bounds if provided
        if cluster_id:
            cr, cc = cluster_id
            r0 = cr * self.c_size
            c0 = cc * self.c_size
            rmin, rmax = max(0, r0), min(self.height, r0 + self.c_size)
            cmin, cmax = max(0, c0), min(self.width, c0 + self.c_size)
            
            # Validate that start and goal are within cluster bounds
            if not (rmin <= start[0] < rmax and cmin <= start[1] < cmax):
                return float('inf')
            if not (rmin <= goal[0] < rmax and cmin <= goal[1] < cmax):
                return float('inf')
        else:
            rmin, rmax = 0, self.height
            cmin, cmax = 0, self.width

        pq = [(math.hypot(start[0] - goal[0], start[1] - goal[1]), 0.0, start)]
        dist = {start: 0.0}
        visited = set()
        
        while pq:
            f, g, curr = heapq.heappop(pq)
            
            # FIX: Check if already visited to avoid reprocessing
            if curr in visited:
                continue
            visited.add(curr)
            
            if curr == goal:
                return g
                
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
                nr, nc = curr[0] + dr, curr[1] + dc
                if rmin <= nr < rmax and cmin <= nc < cmax:
                    step_cost = ON_ROAD_COST if self.grid[nr, nc] == 1 else OFF_ROAD_COST
                    new_g = g + math.hypot(dr, dc) * step_cost
                    if new_g + 1e-9 < dist.get((nr, nc), float('inf')):
                        dist[(nr, nc)] = new_g
                        h = math.hypot(nr - goal[0], nc - goal[1])
                        heapq.heappush(pq, (new_g + h, new_g, (nr, nc)))
        return float('inf')


# -------------------- SEARCH & UTILS --------------------

def get_mst_heuristic(curr, goals, mask):
    """Calculate heuristic for remaining goals."""
    remaining = [g for i, g in enumerate(goals) if not (mask & (1 << i))]
    if not remaining:
        return 0.0
    return min(math.hypot(g[0] - curr[0], g[1] - curr[1]) for g in remaining)


def refine_path(abstract_path, graph):
    """
    FIX: Proper path reconstruction in correct order.
    Original bug: Path segments were being reversed incorrectly.
    """
    if not abstract_path or len(abstract_path) < 2:
        return abstract_path
    
    full_path = [abstract_path[0]]  # Start with first point
    
    for i in range(len(abstract_path) - 1):
        start = abstract_path[i]
        goal = abstract_path[i + 1]

        pq = [(math.hypot(start[0]-goal[0], start[1]-goal[1]), 0.0, start)]
        came_from = {}
        gscore = {start: 0.0}
        visited = set()
        found = False

        while pq:
            f, g, curr = heapq.heappop(pq)
            
            if curr in visited:
                continue
            visited.add(curr)
            
            if curr == goal:
                found = True
                break
                
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
                nr, nc = curr[0] + dr, curr[1] + dc
                if 0 <= nr < graph.height and 0 <= nc < graph.width:
                    step_cost = ON_ROAD_COST if graph.grid[nr, nc] == 1 else OFF_ROAD_COST
                    tentative_g = g + math.hypot(dr, dc) * step_cost
                    ncoord = (nr, nc)
                    if tentative_g + 1e-9 < gscore.get(ncoord, float('inf')):
                        gscore[ncoord] = tentative_g
                        came_from[ncoord] = curr
                        h = math.hypot(nr - goal[0], nc - goal[1])
                        heapq.heappush(pq, (tentative_g + h, tentative_g, ncoord))

        if found:
            # FIX: Reconstruct path in correct order (from start to goal)
            segment = []
            cur = goal
            while cur != start and cur in came_from:
                segment.append(cur)
                cur = came_from[cur]
            
            # Reverse to get start->goal order, then add to full path (excluding start which is already there)
            segment.reverse()
            full_path.extend(segment)
        else:
            # If no path found between consecutive abstract nodes, just add goal
            full_path.append(goal)

    return full_path


def a_star_hpa(start, goals, road_bitmap):
    """
    Main HPA* search with fixes for proper node connectivity.
    """
    graph = HierarchicalGraph(road_bitmap, cluster_size=CLUSTER_SIZE)
    total_goals = len(goals)

    # connect start/goals to nodes in their cluster
    points_to_connect = [start] + [g for g in goals if g != start]
    for p in points_to_connect:
        cid = graph._get_cluster_id(*p)
        # Prefer cluster-local gateway nodes when available
        candidates = list(graph.clusters.get(cid, []))

        # If cluster has no gateways, expand search to nearby gateways (within 2 clusters radius)
        if not candidates:
            search_radius = max(1, graph.c_size * 2)
            near = []
            for node in graph.nodes.keys():
                dr = abs(node[0] - p[0])
                dc = abs(node[1] - p[1])
                if dr <= search_radius and dc <= search_radius:
                    near.append((dr + dc, node))
            # sort by Manhattan proximity and take a few nearest gateways
            near.sort(key=lambda x: x[0])
            candidates = [n for _, n in near[:8]]

        # Connect to candidate gateways (if reachable within cluster bounds or locally)
        for node in candidates:
            # attempt cluster-restricted local distance if node lies in same cluster, otherwise unrestricted
            node_cid = graph._get_cluster_id(*node)
            cid_for_search = cid if node_cid == cid else None
            d = graph.local_astar_dist(p, node, cid_for_search)
            if d < float('inf'):
                graph._add_edge(p, node, d)

    # start mask
    start_mask = 0
    for i, g in enumerate(goals):
        if start == g:
            start_mask |= (1 << i)

    pq = [(get_mst_heuristic(start, goals, start_mask), 0.0, start, start_mask)]
    best_g = {}
    came_from = {}
    expansions = 0
    frames_sparse = []
    visited_snapshot = []

    print("\nRunning Hierarchical Search...")
    while pq:
        f, g_cost, curr, mask = heapq.heappop(pq)
        state = (curr, mask)
        
        # FIX: Use proper comparison for already-visited states
        if state in best_g and best_g[state] <= g_cost:
            continue
        best_g[state] = g_cost

        expansions += 1
        visited_snapshot.append(curr)
        if FRAME_RECORD_INTERVAL and (expansions % FRAME_RECORD_INTERVAL == 0):
            frames_sparse.append((expansions, visited_snapshot.copy()))
            visited_snapshot.clear()

        if mask == (1 << total_goals) - 1:
            # reconstruct abstract path states
            abs_path = []
            cur_state = state
            while cur_state in came_from:
                abs_path.append(cur_state[0])
                cur_state = came_from[cur_state]
            abs_path.append(start)
            abs_path = abs_path[::-1]
            refined = refine_path(abs_path, graph)
            # flush any remaining visited snapshot
            if visited_snapshot:
                frames_sparse.append((expansions, visited_snapshot.copy()))
            return refined, frames_sparse

        # FIX: Handle nodes with no neighbors gracefully
        neighbors = graph.nodes.get(curr, [])
        for neighbor, weight in neighbors:
            new_mask = mask
            for i, goal_pos in enumerate(goals):
                if neighbor == goal_pos:
                    new_mask |= (1 << i)

            new_g = g_cost + weight
            neigh_state = (neighbor, new_mask)
            if new_g + 1e-9 < best_g.get(neigh_state, float('inf')):
                came_from[neigh_state] = state
                h = get_mst_heuristic(neighbor, goals, new_mask)
                heapq.heappush(pq, (new_g + h, new_g, neighbor, new_mask))

        # safety cap
        if expansions >= MAX_EXPANSIONS:
            print(f"Hit expansion limit ({MAX_EXPANSIONS})")
            break

    # no path found
    if visited_snapshot:
        frames_sparse.append((expansions, visited_snapshot.copy()))
    return None, frames_sparse


# -------------------- DATA HANDLING --------------------

def load_npz_test_set(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    raster = data['raster']
    goal_points = data['goal_points']
    metadata = {
        'width': int(data['width']) if 'width' in data else raster.shape[1],
        'height': int(data['height']) if 'height' in data else raster.shape[0]
    }
    return raster, goal_points, metadata


def save_output_data(output_dir, road_bitmap, path, frames_sparse, metadata):
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(os.path.join(output_dir, "road_bitmap.npz"), road_bitmap=road_bitmap.astype(np.uint8))
    np.savez_compressed(os.path.join(output_dir, "protected_bitmap.npz"), protected_bitmap=np.zeros_like(road_bitmap, dtype=np.uint8))
    with open(os.path.join(output_dir, "astar_sparse_frames.pkl"), "wb") as f:
        pickle.dump(frames_sparse, f, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(os.path.join(output_dir, "astar_final_path.npy"), np.array(path if path else [], dtype=np.int32))
    try:
        with open(os.path.join(output_dir, "meta.pkl"), "wb") as f:
            pickle.dump(metadata, f)
    except Exception:
        pass


# -------------------- MAIN --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--cluster-size", type=int, default=DEFAULT_CLUSTER_SIZE)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"./hpa_output_{os.path.splitext(os.path.basename(args.input))[0]}"

    raster, goal_points, meta = load_npz_test_set(args.input)
    # goal_points expected as [(x,y,name), ...] per reference; convert to (row,col)
    goals = []
    for item in goal_points:
        try:
            x, y, _ = item
        except Exception:
            x, y = item[:2]
        goals.append((int(y), int(x)))
    if not goals:
        raise SystemExit("No goal points found in test set.")
    start = goals[0]

    global CLUSTER_SIZE
    CLUSTER_SIZE = max(1, int(args.cluster_size))

    t0 = time.time()
    path, frames = a_star_hpa(start, goals, raster)
    t1 = time.time()

    print(f"Search time: {t1 - t0:.2f}s  —  expansions recorded: {len(frames)}")
    save_output_data(args.output, raster, path, frames, meta)

    # Always attempt to create animation (animator now robust to empty frames / paths)
    print("Attempting to create animation (HPA)...")
    try:
        animator = HPAStarAnimator(output_dir=args.output, cluster_size=CLUSTER_SIZE)
        animator.create_animation(output_file="hpa_animation.gif", fps=ANIMATION_FPS, target_frames=ANIMATION_FRAMES)
        animator.create_static_comparison(output_file="hpa_comparison.png")
        print(f"Animation / comparison written to: {args.output}")
    except Exception as e:
        print(f"Animation failed (caught): {e}")


if __name__ == "__main__":
    main()