#!/usr/bin/env python3
"""
Hierarchical (HPA*) A* Pathfinding from NPZ Test Sets

This module builds a cluster/gateway abstraction over a road bitmap,
runs a hierarchical A* (abstract search over gateways + local refinement),
and writes output (bitmaps, sparse frames, final refined path).

Reference behavior (animation/data layout) follows the single-level A*
implementation in hyperloop_astar_from_npz.py so animator utilities can
consume the outputs.
"""
import argparse
import numpy as np
import heapq
import math
import time
import os
import pickle
from hpastar_animator import HPAStarAnimator

# -------------------- CONFIG --------------------
DEFAULT_CLUSTER_SIZE = 40      # Default cluster size (pixels)
CLUSTER_SIZE = DEFAULT_CLUSTER_SIZE
ALLOW_DIAGONAL = True
FRAME_RECORD_INTERVAL = 100
ON_ROAD_COST = 1.0
OFF_ROAD_COST = 5.0
MAX_EXPANSIONS = 5_000_000

# Animation settings
ANIMATION_FPS = 15
ANIMATION_FRAMES = 150
# ------------------------------------------------


class HierarchicalGraph:
    """Abstracts the raster into clusters and gateways."""

    def __init__(self, road_bitmap, cluster_size=None):
        # Use DEFAULT_CLUSTER_SIZE as a safe fallback so the default argument
        # doesn't reference a name that may not yet be defined at import time.
        if cluster_size is None:
            cluster_size = DEFAULT_CLUSTER_SIZE
        self.grid = road_bitmap
        self.c_size = max(1, int(cluster_size))
        self.height, self.width = road_bitmap.shape
        # nodes: dict mapping node_coord -> list of (neighbor_coord, weight)
        self.nodes = {}
        # clusters: mapping cluster_id -> list of node_coords inside that cluster
        self.clusters = {}
        self._build_graph()

    def _get_cluster_id(self, r, c):
        return (r // self.c_size, c // self.c_size)

    def _add_node_if_missing(self, p):
        if p not in self.nodes:
            self.nodes[p] = []

    def _add_edge(self, p1, p2, weight):
        self._add_node_if_missing(p1)
        self._add_node_if_missing(p2)
        # Avoid duplicate edges (keeps small lists)
        if not any(n == p2 for n, _ in self.nodes[p1]):
            self.nodes[p1].append((p2, weight))
        if not any(n == p1 for n, _ in self.nodes[p2]):
            self.nodes[p2].append((p1, weight))

    def _build_graph(self):
        # 1) find gateways along cluster boundaries
        num_cx = (self.width + self.c_size - 1) // self.c_size
        num_cy = (self.height + self.c_size - 1) // self.c_size
        print(f"Constructing HPA* Graph (clusters: {num_cy}x{num_cx}, cluster_size={self.c_size})...")

        # vertical boundaries between clusters (columns)
        for cr in range(0, self.height, self.c_size):
            for cc in range(self.c_size, self.width, self.c_size):
                # boundary between column cc-1 and cc for rows cr..cr+c_size-1
                self._find_gateways(cr, cc - 1, cr, cc, vertical=True)

        # horizontal boundaries between clusters (rows)
        for cr in range(self.c_size, self.height, self.c_size):
            for cc in range(0, self.width, self.c_size):
                # boundary between row cr-1 and cr for cols cc..cc+c_size-1
                self._find_gateways(cr - 1, cc, cr, cc, vertical=False)

        # 2) collect nodes into clusters
        for node in list(self.nodes.keys()):
            cid = self._get_cluster_id(*node)
            self.clusters.setdefault(cid, []).append(node)

        # 3) connect all gateway/entrance nodes within a cluster using local A*
        for cid, nodes in self.clusters.items():
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    a = nodes[i]
                    b = nodes[j]
                    cost = self.local_astar_dist(a, b, cid)
                    if cost < float('inf'):
                        self._add_edge(a, b, cost)

    def _find_gateways(self, r1, c1, r2, c2, vertical):
        """
        Scan along a boundary line and collect contiguous road-crossings.
        For contiguous runs of cells where both sides are road (==1), create
        a pair-node (the two adjacent pixels) and link them as a gateway.
        """
        segment = []
        length = self.c_size
        for i in range(length):
            r_a = r1 + i if vertical else r1
            c_a = c1 if vertical else c1 + i
            r_b = r2 + i if vertical else r2
            c_b = c2 if vertical else c2 + i

            if 0 <= r_a < self.height and 0 <= c_a < self.width and \
               0 <= r_b < self.height and 0 <= c_b < self.width:
                a_is_road = self.grid[r_a, c_a] == 1
                b_is_road = self.grid[r_b, c_b] == 1
                if a_is_road and b_is_road:
                    segment.append(((r_a, c_a), (r_b, c_b)))
                else:
                    if segment:
                        self._create_node_from_segment(segment)
                        segment = []
        if segment:
            self._create_node_from_segment(segment)

    def _create_node_from_segment(self, segment):
        """
        Given a contiguous segment of pairs across a boundary, pick a center
        pair and add an undirected gateway edge between the two pixels.
        """
        mid = len(segment) // 2
        p_a, p_b = segment[mid]
        # connect the two pixels as a gateway (short, cheap edge)
        self._add_edge(p_a, p_b, 1.0)

    def local_astar_dist(self, start, goal, cluster_id=None):
        """
        Compute shortest path cost between start and goal restricted to cluster bounds.
        Returns cost (float) or inf if unreachable.
        """
        if cluster_id:
            cr, cc = cluster_id
            r0 = cr * self.c_size
            c0 = cc * self.c_size
            r_lim = (max(0, r0), min(self.height, r0 + self.c_size))
            c_lim = (max(0, c0), min(self.width, c0 + self.c_size))
        else:
            r_lim, c_lim = (0, self.height), (0, self.width)

        pq = [(math.hypot(start[0]-goal[0], start[1]-goal[1]), 0.0, start)]
        dist = {start: 0.0}
        while pq:
            f, g, curr = heapq.heappop(pq)
            if curr == goal:
                return g
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0),
                           (1,1),(1,-1),(-1,1),(-1,-1)]:
                nr, nc = curr[0] + dr, curr[1] + dc
                if r_lim[0] <= nr < r_lim[1] and c_lim[0] <= nc < c_lim[1]:
                    step_cost = ON_ROAD_COST if self.grid[nr, nc] == 1 else OFF_ROAD_COST
                    new_g = g + math.hypot(dr, dc) * step_cost
                    if new_g + 1e-9 < dist.get((nr, nc), float('inf')):
                        dist[(nr, nc)] = new_g
                        h = math.hypot(nr - goal[0], nc - goal[1])
                        heapq.heappush(pq, (new_g + h, new_g, (nr, nc)))
        return float('inf')


# -------------------- SEARCH & UTILS --------------------

def get_mst_heuristic(curr, goals, mask):
    remaining = [g for i, g in enumerate(goals) if not (mask & (1 << i))]
    if not remaining:
        return 0.0
    # admissible: distance to nearest remaining goal
    return min(math.hypot(g[0] - curr[0], g[1] - curr[1]) for g in remaining)


def refine_path(abstract_path, graph):
    """
    Convert abstract node list (pixel coords) into a refined pixel-by-pixel path.
    Runs a short local A* between each consecutive abstract nodes.
    """
    full_path = []
    for i in range(len(abstract_path) - 1):
        start = abstract_path[i]
        goal = abstract_path[i + 1]

        # local A* (unrestricted bounds; but it's typically short)
        pq = [(math.hypot(start[0]-goal[0], start[1]-goal[1]), 0.0, start)]
        came_from = {}
        gscore = {start: 0.0}
        found = False

        while pq:
            f, g, curr = heapq.heappop(pq)
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

        # reconstruct local path
        if found:
            segment = []
            cur = goal
            while cur != start:
                segment.append(cur)
                cur = came_from.get(cur, start)
                if cur is None:
                    break
            segment.append(start)
            full_path.extend(segment[:-1])  # omit last to avoid duplicates
        else:
            # if local refinement fails, fall back to direct pixel append (rare)
            full_path.append(start)

    # append final endpoint
    full_path.append(abstract_path[-1])
    return full_path


def a_star_hpa(start, goals, road_bitmap):
    """
    Hierarchical A*:
      - build HPA graph (gateways + intra-cluster connections)
      - connect start/goals to gateways in their cluster
      - run A* on abstract graph (nodes are pixel coords representing gateways/entrances)
      - when abstract solution found, refine to full pixel path
    """
    graph = HierarchicalGraph(road_bitmap, cluster_size=CLUSTER_SIZE)
    total_goals = len(goals)

    # Connect both start and goals to nearby gateways inside their cluster
    points_to_connect = [start] + [g for g in goals if g != start]
    for p in points_to_connect:
        cid = graph._get_cluster_id(*p)
        if cid in graph.clusters:
            for node in graph.clusters[cid]:
                d = graph.local_astar_dist(p, node, cid)
                if d < float('inf'):
                    graph._add_edge(p, node, d)

    # initial goal-mask
    start_mask = 0
    for i, g in enumerate(goals):
        if start == g:
            start_mask |= (1 << i)

    # priority queue entries: (f, g, position, mask)
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
        if best_g.get(state, float('inf')) <= g_cost:
            continue
        best_g[state] = g_cost

        expansions += 1
        visited_snapshot.append(curr)
        if FRAME_RECORD_INTERVAL and (expansions % FRAME_RECORD_INTERVAL == 0):
            frames_sparse.append((expansions, visited_snapshot.copy()))
            visited_snapshot.clear()

        # check goal completion
        if mask == (1 << total_goals) - 1:
            # reconstruct abstract path
            abs_path = []
            cur_state = state
            while cur_state in came_from:
                abs_path.append(cur_state[0])
                cur_state = came_from[cur_state]
            abs_path.append(start)
            abs_path = abs_path[::-1]
            print(f"✓ Abstract Path Found ({len(abs_path)} abstract nodes). Refining to pixels...")
            refined = refine_path(abs_path, graph)
            return refined, frames_sparse

        # expand neighbors in abstract graph
        for neighbor, weight in graph.nodes.get(curr, []):
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
    # produce a protected bitmap placeholder for compatibility with other animators
    np.savez_compressed(os.path.join(output_dir, "protected_bitmap.npz"), protected_bitmap=np.zeros_like(road_bitmap, dtype=np.uint8))
    # sparse frames
    with open(os.path.join(output_dir, "astar_sparse_frames.pkl"), "wb") as f:
        pickle.dump(frames_sparse, f, protocol=pickle.HIGHEST_PROTOCOL)
    # final refined path as Nx2 array
    np.save(os.path.join(output_dir, "astar_final_path.npy"), np.array(path if path else [], dtype=np.int32))
    # simple meta (kept minimal)
    try:
        from rasterio.transform import from_bounds
        h, w = road_bitmap.shape
        transform = from_bounds(0, 0, w, h, w, h)
        meta = {"transform": transform, "bounds": (0, 0, w, h), "original_metadata": metadata}
        with open(os.path.join(output_dir, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)
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
    goals = [(int(y), int(x)) for x, y, name in goal_points]
    start = goals[0]

    # update module-level CLUSTER_SIZE so other parts of the module see the value
    global CLUSTER_SIZE
    CLUSTER_SIZE = max(1, int(args.cluster_size))

    start_time = time.time()
    path, frames = a_star_hpa(start, goals, raster)
    end_time = time.time()

    print(f"Total Time: {end_time - start_time:.2f}s")
    save_output_data(args.output, raster, path, frames, meta)

    if path:
        print("Creating Animations...")
        try:
            animator = HPAStarAnimator(output_dir=args.output, cluster_size=args.cluster_size)
            animator.create_animation(output_file="hpa_animation.gif", fps=ANIMATION_FPS, target_frames=ANIMATION_FRAMES)
            animator.create_static_comparison(output_file="hpa_comparison.png")
            print(f"Done! Output in {args.output}")
        except Exception as e:
            print(f"Animation failed: {e}")


if __name__ == "__main__":
    main()