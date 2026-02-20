#!/usr/bin/env python3
"""
Hierarchical (HPA*) A* Pathfinding from NPZ Test Sets

ADDITIONAL FIX: Proper visualization recording for small cluster sizes
- Records all cells visited during local pathfinding within clusters
- Records gateway nodes during abstract search
- Ensures animation works even with very small cluster sizes

NPZ SAFETY: Input NPZ files are opened read-only and are NEVER modified.
Output NPZ files use distinct names and are always written to a separate output dir.

LOGGING: Each run appends a timestamped entry to hpa_run_log.txt in the output folder.
"""
from hpastar_animator import HPAStarAnimator
import argparse
import numpy as np
import heapq
import math
import time
import os
import pickle
import datetime

# -------------------- CONFIG --------------------
DEFAULT_CLUSTER_SIZE = 40
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
        # Track visited cells for visualization
        self.all_visited_cells = []
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
                    if self._get_cluster_id(*a) == cid and self._get_cluster_id(*b) == cid:
                        cost, _ = self.local_astar_dist(a, b, cid, record_visited=False)
                        if cost < float('inf'):
                            self._add_edge(a, b, cost)

    def _find_gateways(self, r1, c1, r2, c2, vertical):
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
        mid = len(segment) // 2
        p_a, p_b = segment[mid]

        self._add_node_if_missing(p_a)
        self._add_node_if_missing(p_b)
        self._add_edge(p_a, p_b, 1.0)

    def local_astar_dist(self, start, goal, cluster_id=None, record_visited=False):
        """
        Modified to optionally return visited cells for visualization.
        Returns: (distance, visited_cells_list)
        """
        if cluster_id:
            cr, cc = cluster_id
            r0 = cr * self.c_size
            c0 = cc * self.c_size
            rmin, rmax = max(0, r0), min(self.height, r0 + self.c_size)
            cmin, cmax = max(0, c0), min(self.width, c0 + self.c_size)

            if not (rmin <= start[0] < rmax and cmin <= start[1] < cmax):
                return float('inf'), []
            if not (rmin <= goal[0] < rmax and cmin <= goal[1] < cmax):
                return float('inf'), []
        else:
            rmin, rmax = 0, self.height
            cmin, cmax = 0, self.width

        pq = [(math.hypot(start[0] - goal[0], start[1] - goal[1]), 0.0, start)]
        dist = {start: 0.0}
        visited = set()
        visited_list = []

        while pq:
            f, g, curr = heapq.heappop(pq)

            if curr in visited:
                continue
            visited.add(curr)

            if record_visited:
                visited_list.append(curr)

            if curr == goal:
                return g, visited_list

            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
                nr, nc = curr[0] + dr, curr[1] + dc
                if rmin <= nr < rmax and cmin <= nc < cmax:
                    step_cost = ON_ROAD_COST if self.grid[nr, nc] == 1 else OFF_ROAD_COST
                    new_g = g + math.hypot(dr, dc) * step_cost
                    if new_g + 1e-9 < dist.get((nr, nc), float('inf')):
                        dist[(nr, nc)] = new_g
                        h = math.hypot(nr - goal[0], nc - goal[1])
                        heapq.heappush(pq, (new_g + h, new_g, (nr, nc)))
        return float('inf'), visited_list


# -------------------- SEARCH & UTILS --------------------

def get_mst_heuristic(curr, goals, mask):
    remaining = [g for i, g in enumerate(goals) if not (mask & (1 << i))]
    if not remaining:
        return 0.0
    return min(math.hypot(g[0] - curr[0], g[1] - curr[1]) for g in remaining)


def refine_path(abstract_path, graph, record_visited=False):
    """
    Modified to optionally record all cells visited during refinement.
    Returns: (full_path, visited_cells_during_refinement)
    """
    if not abstract_path or len(abstract_path) < 2:
        return abstract_path, []

    full_path = [abstract_path[0]]
    all_visited = []

    for i in range(len(abstract_path) - 1):
        start = abstract_path[i]
        goal = abstract_path[i + 1]

        pq = [(math.hypot(start[0]-goal[0], start[1]-goal[1]), 0.0, start)]
        came_from = {}
        gscore = {start: 0.0}
        visited = set()
        visited_list = []
        found = False

        while pq:
            f, g, curr = heapq.heappop(pq)

            if curr in visited:
                continue
            visited.add(curr)

            if record_visited:
                visited_list.append(curr)

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

        all_visited.extend(visited_list)

        if found:
            segment = []
            cur = goal
            while cur != start and cur in came_from:
                segment.append(cur)
                cur = came_from[cur]
            segment.reverse()
            full_path.extend(segment)
        else:
            full_path.append(goal)

    return full_path, all_visited


def a_star_hpa(start, goals, road_bitmap):
    graph = HierarchicalGraph(road_bitmap, cluster_size=CLUSTER_SIZE)
    total_goals = len(goals)

    points_to_connect = [start] + [g for g in goals if g != start]
    connection_visited = []

    for p in points_to_connect:
        cid = graph._get_cluster_id(*p)
        candidates = list(graph.clusters.get(cid, []))

        if not candidates:
            search_radius = max(1, graph.c_size * 2)
            near = []
            for node in graph.nodes.keys():
                dr = abs(node[0] - p[0])
                dc = abs(node[1] - p[1])
                if dr <= search_radius and dc <= search_radius:
                    near.append((dr + dc, node))
            near.sort(key=lambda x: x[0])
            candidates = [n for _, n in near[:8]]

        for node in candidates:
            node_cid = graph._get_cluster_id(*node)
            cid_for_search = cid if node_cid == cid else None
            d, visited_cells = graph.local_astar_dist(p, node, cid_for_search, record_visited=True)
            if d < float('inf'):
                graph._add_edge(p, node, d)
                connection_visited.extend(visited_cells)

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

    if connection_visited:
        visited_snapshot.extend(connection_visited)

    print("\nRunning Hierarchical Search...")
    while pq:
        f, g_cost, curr, mask = heapq.heappop(pq)
        state = (curr, mask)

        if state in best_g and best_g[state] <= g_cost:
            continue
        best_g[state] = g_cost

        expansions += 1
        visited_snapshot.append(curr)

        # For small cluster sizes, also record cells around the gateway node
        # This helps visualization by showing "activity" at gateway locations
        if CLUSTER_SIZE <= 20:
            r, c = curr
            for dr, dc in [(0,0), (-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < graph.height and 0 <= nc < graph.width:
                    visited_snapshot.append((nr, nc))

        if FRAME_RECORD_INTERVAL and (expansions % FRAME_RECORD_INTERVAL == 0):
            frames_sparse.append((expansions, visited_snapshot.copy()))
            visited_snapshot.clear()

        if mask == (1 << total_goals) - 1:
            abs_path = []
            cur_state = state
            while cur_state in came_from:
                abs_path.append(cur_state[0])
                cur_state = came_from[cur_state]
            abs_path.append(start)
            abs_path = abs_path[::-1]

            refined, refinement_visited = refine_path(abs_path, graph, record_visited=True)

            if refinement_visited:
                visited_snapshot.extend(refinement_visited)

            if visited_snapshot:
                frames_sparse.append((expansions, visited_snapshot.copy()))

            return refined, frames_sparse, expansions

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

        if expansions >= MAX_EXPANSIONS:
            print(f"Hit expansion limit ({MAX_EXPANSIONS})")
            break

    if visited_snapshot:
        frames_sparse.append((expansions, visited_snapshot.copy()))
    return None, frames_sparse, expansions


# -------------------- DATA HANDLING --------------------

def load_npz_test_set(npz_path):
    """
    Load input NPZ in read-only mode so the file can never be modified.
    Arrays are immediately copied into plain numpy arrays and the file
    handle is explicitly closed so the mmap is fully released.
    """
    data = np.load(npz_path, allow_pickle=True, mmap_mode='r')
    raster      = np.array(data['raster'])
    goal_points = np.array(data['goal_points'], dtype=object)
    metadata = {
        'width':  int(data['width'])  if 'width'  in data else raster.shape[1],
        'height': int(data['height']) if 'height' in data else raster.shape[0],
    }
    data.close()
    return raster, goal_points, metadata


def save_output_data(output_dir, input_path, road_bitmap, path, frames_sparse, metadata):
    """
    Write all outputs to output_dir. Raises RuntimeError if output_dir
    resolves to the same directory as the input NPZ file, preventing any
    accidental overwrite of source data.
    """
    input_abs  = os.path.realpath(os.path.dirname(os.path.abspath(input_path)))
    output_abs = os.path.realpath(os.path.abspath(output_dir))

    if input_abs == output_abs:
        raise RuntimeError(
            f"Output directory '{output_dir}' is the same as the input file's "
            f"directory '{input_abs}'. Choose a different output directory to "
            f"avoid overwriting your NPZ test sets."
        )

    os.makedirs(output_dir, exist_ok=True)

    # Prefixed names so they can never collide with test-set NPZ filenames
    np.savez_compressed(
        os.path.join(output_dir, "hpa_road_bitmap.npz"),
        road_bitmap=road_bitmap.astype(np.uint8),
    )
    np.savez_compressed(
        os.path.join(output_dir, "hpa_protected_bitmap.npz"),
        protected_bitmap=np.zeros_like(road_bitmap, dtype=np.uint8),
    )
    with open(os.path.join(output_dir, "astar_sparse_frames.pkl"), "wb") as f:
        pickle.dump(frames_sparse, f, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(
        os.path.join(output_dir, "astar_final_path.npy"),
        np.array(path if path else [], dtype=np.int32),
    )
    try:
        with open(os.path.join(output_dir, "meta.pkl"), "wb") as f:
            pickle.dump(metadata, f)
    except Exception:
        pass


# -------------------- LOGGING --------------------

def write_run_log(output_dir, input_path, cluster_size, goals, raster_shape,
                  search_time, expansions, frames_recorded, total_visited,
                  path_length, path_found):
    """
    Append a structured, human-readable entry for this run to
    hpa_run_log.txt inside output_dir. The file is created on the first
    run and appended to on every subsequent run, so the full history is
    preserved.
    """
    log_path  = os.path.join(output_dir, "hpa_run_log.txt")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "=" * 60,
        f"Timestamp        : {timestamp}",
        f"Input file       : {os.path.abspath(input_path)}",
        f"Output directory : {os.path.abspath(output_dir)}",
        f"Cluster size     : {cluster_size}",
        f"Raster shape     : {raster_shape[0]} rows x {raster_shape[1]} cols",
        f"Goal points      : {len(goals)}",
        f"Path found       : {'yes' if path_found else 'no'}",
        f"Path length      : {path_length if path_found else 'N/A'} nodes",
        f"Search time      : {search_time:.3f}s",
        f"Expansions       : {expansions}",
        f"Frames recorded  : {frames_recorded}",
        f"Total visited    : {total_visited}",
        "",  # blank line between entries for readability
    ]

    with open(log_path, "a") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Run log updated  : {log_path}")


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
    path, frames, expansions = a_star_hpa(start, goals, raster)
    t1 = time.time()
    search_time = t1 - t0

    total_visited = sum(len(positions) for _, positions in frames)
    path_length   = len(path) if path else 0

    print(f"Search time      : {search_time:.2f}s  —  expansions recorded: {len(frames)}")
    print(f"Cluster size     : {CLUSTER_SIZE}")
    print(f"Total visited cells recorded: {total_visited}")

    # Create output dir early so the log can be written even if save_output_data
    # raises (e.g. the directory-collision guard fires).
    os.makedirs(args.output, exist_ok=True)

    write_run_log(
        output_dir      = args.output,
        input_path      = args.input,
        cluster_size    = CLUSTER_SIZE,
        goals           = goals,
        raster_shape    = raster.shape,
        search_time     = search_time,
        expansions      = expansions,
        frames_recorded = len(frames),
        total_visited   = total_visited,
        path_length     = path_length,
        path_found      = path is not None,
    )

    save_output_data(args.output, args.input, raster, path, frames, meta)

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