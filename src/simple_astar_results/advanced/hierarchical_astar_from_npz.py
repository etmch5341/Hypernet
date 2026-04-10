#!/usr/bin/env python3
"""
Hierarchical (HPA*) A* Pathfinding from NPZ Test Sets

MULTITHREADING: Cluster-internal path costs are computed in parallel using
ProcessPoolExecutor. Each cluster's gateway-node pairs are dispatched as
independent tasks so all CPU cores contribute during graph construction.
The abstract search and path refinement phases remain sequential.

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
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutureTimeoutError

# -------------------- CONFIG --------------------
DEFAULT_CLUSTER_SIZE = 40
CLUSTER_SIZE = DEFAULT_CLUSTER_SIZE
ALLOW_DIAGONAL = True
FRAME_RECORD_INTERVAL = 100
# Traversal cost: step = BASE_COST + cell_cost * COST_RANGE
# cell_cost is normalised to [0.0 (cheapest) … 1.0 (most expensive)]
# → cheapest cell: 1.0 × euclidean dist  (matches old ON_ROAD_COST)
# → costliest cell: 5.0 × euclidean dist (matches old OFF_ROAD_COST)
BASE_COST  = 1.0
COST_RANGE = 4.0
# Cells cheaper than this threshold are eligible for gateway nodes.
GATEWAY_COST_THRESHOLD = 0.5
MAX_EXPANSIONS = 5_000_000

# Priority order for finding the grid array inside an NPZ file.
RASTER_KEY_PRIORITY = [
    'raster',
    'cost_map_normalized',
    'env_map_normalized',
    'cost_map',
    'env_map',
]

ANIMATION_FPS = 15
ANIMATION_FRAMES = 150

# Number of worker processes. None = one per logical CPU core.
NUM_WORKERS = None
# ------------------------------------------------


# ================================================================
#  TOP-LEVEL WORKER FUNCTION
#  Must live at module scope (not inside a class) so that
#  ProcessPoolExecutor can pickle and dispatch it to child processes.
# ================================================================

# Per-search expansion cap inside a worker. Prevents a single huge-cluster
# A* from running forever. Scales with cluster area so small clusters are
# unaffected while pathologically large ones are bounded.
# Formula: cap = cluster_area * 4  (tunable constant)
def _local_astar_cap(c_size):
    return c_size * c_size * 4


def _compute_cluster_edges(args):
    """
    Worker function executed in a child process.

    Receives everything it needs as plain serialisable objects so no
    shared state or locks are required.

    Parameters
    ----------
    args : tuple
        (grid_bytes, height, width, c_size, cluster_id, node_list)

    Returns
    -------
    list of (p1, p2, cost) tuples for edges whose cost < inf
    """
    grid_bytes, height, width, c_size, cluster_id, node_list = args

    # Reconstruct the float32 cost grid from raw bytes
    grid = np.frombuffer(grid_bytes, dtype=np.float32).reshape(height, width)

    cr, cc = cluster_id
    r0 = cr * c_size
    c0 = cc * c_size
    rmin = max(0, r0)
    rmax = min(height, r0 + c_size)
    cmin = max(0, c0)
    cmax = min(width, c0 + c_size)

    # Cap expansions per A* search so one enormous cluster can't hang a worker
    expansion_cap = _local_astar_cap(c_size)

    edges = []
    n = len(node_list)

    for i in range(n):
        for j in range(i + 1, n):
            start = node_list[i]
            goal  = node_list[j]

            # Validate both nodes are inside cluster bounds
            if not (rmin <= start[0] < rmax and cmin <= start[1] < cmax):
                continue
            if not (rmin <= goal[0] < rmax and cmin <= goal[1] < cmax):
                continue

            # ---- local A* restricted to cluster bounds ----
            pq         = [(math.hypot(start[0] - goal[0], start[1] - goal[1]), 0.0, start)]
            dist       = {start: 0.0}
            seen       = set()
            cost       = float('inf')
            expansions = 0

            while pq:
                if expansions >= expansion_cap:
                    break   # give up on this pair; no edge added
                f, g, curr = heapq.heappop(pq)
                if curr in seen:
                    continue
                seen.add(curr)
                expansions += 1
                if curr == goal:
                    cost = g
                    break
                for dr, dc in [(0,1),(0,-1),(1,0),(-1,0),
                               (1,1),(1,-1),(-1,1),(-1,-1)]:
                    nr, nc = curr[0] + dr, curr[1] + dc
                    if rmin <= nr < rmax and cmin <= nc < cmax:
                        step = BASE_COST + grid[nr, nc] * COST_RANGE
                        ng   = g + math.hypot(dr, dc) * step
                        if ng + 1e-9 < dist.get((nr, nc), float('inf')):
                            dist[(nr, nc)] = ng
                            h = math.hypot(nr - goal[0], nc - goal[1])
                            heapq.heappush(pq, (ng + h, ng, (nr, nc)))

            if cost < float('inf'):
                edges.append((start, goal, cost))

    return edges


class HierarchicalGraph:
    """Abstracts the raster into clusters and gateways."""

    def __init__(self, road_bitmap, cluster_size=None):
        if cluster_size is None:
            cluster_size = DEFAULT_CLUSTER_SIZE
        self.grid    = road_bitmap
        self.c_size  = max(1, int(cluster_size))
        self.height, self.width = road_bitmap.shape
        self.nodes   = {}   # node_coord -> [(neighbor_coord, weight), ...]
        self.clusters = {}  # cluster_id -> [node_coord, ...]
        self.all_visited_cells = []
        self._build_graph()

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def _build_graph(self):
        """
        Phase 1 – find gateway nodes along every cluster boundary.
        Phase 2 – connect gateway pairs within each cluster IN PARALLEL.
        """

        # ---- Phase 1: gateway discovery (sequential, fast) ----
        for r0 in range(0, self.height, self.c_size):
            for cc in range(self.c_size, self.width, self.c_size):
                self._find_gateways(r0, cc - 1, r0, cc, vertical=True)

        for rr in range(self.c_size, self.height, self.c_size):
            for c0 in range(0, self.width, self.c_size):
                self._find_gateways(rr - 1, c0, rr, c0, vertical=False)

        # Group nodes into clusters
        for node in list(self.nodes.keys()):
            cid = self._get_cluster_id(*node)
            self.clusters.setdefault(cid, []).append(node)

        # ---- Phase 2: intra-cluster edge costs IN PARALLEL ----
        #
        # We serialise the grid as raw bytes once and share it across
        # all worker tasks – much cheaper than pickling the full array
        # repeatedly.
        grid_bytes = self.grid.astype(np.float32).tobytes()
        height, width, c_size = self.height, self.width, self.c_size

        # Build one task per cluster that has at least 2 nodes
        tasks = [
            (grid_bytes, height, width, c_size, cid, nodes)
            for cid, nodes in self.clusters.items()
            if len(nodes) >= 2
        ]

        if not tasks:
            return


        # Dispatch to worker pool; collect edges as futures complete.
        # Per-future timeout = 60s * cluster_area_factor so that a worker
        # silently killed by the OOM killer (SIGKILL gives no Python exception)
        # is detected instead of hanging forever in waiter.acquire().
        # On timeout or any pool-level failure we fall back to sequential mode.
        worker_timeout = max(60, (self.c_size ** 2) // 500)   # seconds per future

        parallel_ok = True
        try:
            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
                futures = {pool.submit(_compute_cluster_edges, t): t[4] for t in tasks}
                done  = 0
                total = len(futures)
                for future in as_completed(futures, timeout=worker_timeout * total):
                    done += 1
                    if done % max(1, total // 10) == 0:
                        print(f"  [graph build] {done}/{total} clusters done", flush=True)
                    try:
                        # Per-future timeout catches individual stuck workers
                        for p1, p2, cost in future.result(timeout=worker_timeout):
                            self._add_edge(p1, p2, cost)
                    except FutureTimeoutError:
                        cid = futures[future]
                        print(f"  [graph build] cluster {cid} timed out after {worker_timeout}s — skipping")
                    except MemoryError:
                        raise  # bubble up to outer OOM handler
                    except Exception as exc:
                        cid = futures[future]
                        print(f"  [graph build] cluster {cid} raised: {exc}")
        except FutureTimeoutError:
            # as_completed overall timeout — pool is hung/dead
            print(f"\n  [graph build] Pool timed out waiting for workers after {worker_timeout * total}s")
            parallel_ok = False
        except (MemoryError, OSError) as exc:
            print(f"\n  [graph build] OOM in worker pool ({type(exc).__name__}: {exc})")
            parallel_ok = False
        except Exception as exc:
            # BrokenProcessPool and similar land here
            print(f"\n  [graph build] Worker pool failed ({type(exc).__name__}: {exc})")
            parallel_ok = False

        if not parallel_ok:
            # Wipe any partial intra-cluster edges that arrived before the crash
            # so we don't double-count them when we re-run sequentially.
            print("  [graph build] Falling back to single-process mode ...")
            for cid, nodes in self.clusters.items():
                for node in nodes:
                    self.nodes[node] = [
                        (nb, w) for nb, w in self.nodes.get(node, [])
                        if not (self._get_cluster_id(*nb) == cid
                                and self._get_cluster_id(*node) == cid)
                    ]
            # Re-run sequentially: no pickling, no extra RAM copies at all
            done  = 0
            total = len(tasks)
            for task in tasks:
                done += 1
                if done % max(1, total // 10) == 0:
                    print(f"  [graph build] {done}/{total} clusters done (sequential)", flush=True)
                try:
                    for p1, p2, cost in _compute_cluster_edges(task):
                        self._add_edge(p1, p2, cost)
                except Exception as exc:
                    cid = task[4]
                    print(f"  [graph build] cluster {cid} raised: {exc}")

    def _find_gateways(self, r1, c1, r2, c2, vertical):
        segment = []
        for i in range(self.c_size):
            r_a = (r1 + i) if vertical else r1
            c_a = c1        if vertical else (c1 + i)
            r_b = (r2 + i) if vertical else r2
            c_b = c2        if vertical else (c2 + i)

            if (0 <= r_a < self.height and 0 <= c_a < self.width and
                    0 <= r_b < self.height and 0 <= c_b < self.width):
                if self.grid[r_a, c_a] < GATEWAY_COST_THRESHOLD and self.grid[r_b, c_b] < GATEWAY_COST_THRESHOLD:
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

    # ------------------------------------------------------------------
    def local_astar_dist(self, start, goal, cluster_id=None, record_visited=False):
        """
        Single local A* search, optionally restricted to a cluster.
        Used at query-time to connect start/goal nodes to the graph.
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

        pq   = [(math.hypot(start[0] - goal[0], start[1] - goal[1]), 0.0, start)]
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
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0),
                           (1,1),(1,-1),(-1,1),(-1,-1)]:
                nr, nc = curr[0] + dr, curr[1] + dc
                if rmin <= nr < rmax and cmin <= nc < cmax:
                    step = BASE_COST + self.grid[nr, nc] * COST_RANGE
                    ng   = g + math.hypot(dr, dc) * step
                    if ng + 1e-9 < dist.get((nr, nc), float('inf')):
                        dist[(nr, nc)] = ng
                        h = math.hypot(nr - goal[0], nc - goal[1])
                        heapq.heappush(pq, (ng + h, ng, (nr, nc)))
        return float('inf'), visited_list


# -------------------- SEARCH & UTILS --------------------

def get_mst_heuristic(curr, goals, mask):
    remaining = [g for i, g in enumerate(goals) if not (mask & (1 << i))]
    if not remaining:
        return 0.0
    return min(math.hypot(g[0] - curr[0], g[1] - curr[1]) for g in remaining)


def remove_path_loops(path):
    """
    Remove loops from a pixel-level path in O(n).

    Loops arise in HPA* when the unconstrained local A* used during refinement
    routes a segment through cells that belong to a different segment of the
    abstract path (because it searches the whole grid, not just the local cluster).

    Strategy: record the *last* index at which each coordinate appears, then
    walk forward and always jump straight to that last index — skipping the
    loop in between.
    """
    if len(path) < 3:
        return path
    last_seen = {p: i for i, p in enumerate(path)}
    result = []
    i = 0
    while i < len(path):
        p = path[i]
        result.append(p)
        i = last_seen[p] + 1   # skip forward past the last occurrence of p
    return result


def _astar_segment(start, goal, grid):
    """
    Minimal unconstrained A* returning the pixel path from start to goal.
    Used to fill gaps left by incomplete came_from chains at cluster boundaries.
    Returns a list that includes both start and goal.
    """
    h, w = grid.shape
    pq = [(math.hypot(start[0] - goal[0], start[1] - goal[1]), 0.0, start)]
    came_from = {}
    gscore    = {start: 0.0}
    visited   = set()

    while pq:
        _, g, curr = heapq.heappop(pq)
        if curr in visited:
            continue
        visited.add(curr)
        if curr == goal:
            seg, c = [], goal
            while c in came_from:
                seg.append(c)
                c = came_from[c]
            seg.append(start)
            seg.reverse()
            return seg
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            nr, nc = curr[0] + dr, curr[1] + dc
            if 0 <= nr < h and 0 <= nc < w:
                step   = BASE_COST + grid[nr, nc] * COST_RANGE
                ng     = g + math.hypot(dr, dc) * step
                ncoord = (nr, nc)
                if ng + 1e-9 < gscore.get(ncoord, float('inf')):
                    gscore[ncoord]    = ng
                    came_from[ncoord] = curr
                    heapq.heappush(pq, (ng + math.hypot(nr - goal[0], nc - goal[1]), ng, ncoord))

    return [start, goal]   # unreachable on a connected grid; fallback


def stitch_path_gaps(path, graph):
    """
    Scan the refined path for gaps (non-8-adjacent consecutive cells) and fill
    each one with a direct pixel A* path.

    Gaps at cluster boundaries arise when the came_from backtracking chain in
    refine_path is broken: a gscore update overwrites came_from[n] with a new
    predecessor, orphaning the old chain so backtracking terminates early and
    the segment is missing a few cells at the cluster boundary crossing.
    """
    if len(path) < 2:
        return path
    result = [path[0]]
    for nxt in path[1:]:
        prev = result[-1]
        if abs(nxt[0] - prev[0]) <= 1 and abs(nxt[1] - prev[1]) <= 1:
            result.append(nxt)
        else:
            # Gap detected: fill with direct A*
            fill = _astar_segment(prev, nxt, graph.grid)
            result.extend(fill[1:])   # prev already in result
    return result


def refine_path(abstract_path, graph, record_visited=False):
    """
    Refine the abstract gateway path to a pixel-level path.
    Returns: (full_path, visited_cells_during_refinement)
    """
    if not abstract_path or len(abstract_path) < 2:
        return abstract_path, []

    full_path  = [abstract_path[0]]
    all_visited = []

    for i in range(len(abstract_path) - 1):
        start = abstract_path[i]
        goal  = abstract_path[i + 1]

        pq       = [(math.hypot(start[0]-goal[0], start[1]-goal[1]), 0.0, start)]
        came_from = {}
        gscore   = {start: 0.0}
        visited  = set()
        visited_list = []
        found    = False

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
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0),
                           (1,1),(1,-1),(-1,1),(-1,-1)]:
                nr, nc = curr[0] + dr, curr[1] + dc
                if 0 <= nr < graph.height and 0 <= nc < graph.width:
                    step        = BASE_COST + graph.grid[nr, nc] * COST_RANGE
                    tentative_g = g + math.hypot(dr, dc) * step
                    ncoord      = (nr, nc)
                    if tentative_g + 1e-9 < gscore.get(ncoord, float('inf')):
                        gscore[ncoord]    = tentative_g
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

    clean = stitch_path_gaps(remove_path_loops(full_path), graph)
    return clean, all_visited


def a_star_hpa(start, goals, road_bitmap):
    graph       = HierarchicalGraph(road_bitmap, cluster_size=CLUSTER_SIZE)
    total_goals = len(goals)

    # Connect start/goals into the abstract graph
    points_to_connect = [start] + [g for g in goals if g != start]
    connection_visited = []

    for p in points_to_connect:
        cid        = graph._get_cluster_id(*p)
        candidates = list(graph.clusters.get(cid, []))

        if not candidates:
            # Search ALL gateway nodes by Manhattan distance (no radius cap).
            # A fixed radius of c_size*2 is too small when clusters are tiny,
            # leaving start/goal completely disconnected from the abstract graph.
            near = sorted(
                ((abs(node[0] - p[0]) + abs(node[1] - p[1]), node)
                 for node in graph.nodes.keys()),
                key=lambda x: x[0],
            )
            candidates = [n for _, n in near[:8]]

        for node in candidates:
            node_cid      = graph._get_cluster_id(*node)
            cid_for_search = cid if node_cid == cid else None
            d, visited_cells = graph.local_astar_dist(p, node, cid_for_search, record_visited=True)
            if d < float('inf'):
                graph._add_edge(p, node, d)
                connection_visited.extend(visited_cells)

    start_mask = 0
    for i, g in enumerate(goals):
        if start == g:
            start_mask |= (1 << i)

    pq              = [(get_mst_heuristic(start, goals, start_mask), 0.0, start, start_mask)]
    best_g          = {}
    came_from       = {}
    expansions      = 0
    frames_sparse   = []
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

        if CLUSTER_SIZE <= 20:
            r, c = curr
            for dr, dc in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:
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

        for neighbor, weight in graph.nodes.get(curr, []):
            new_mask = mask
            for i, goal_pos in enumerate(goals):
                if neighbor == goal_pos:
                    new_mask |= (1 << i)
            new_g      = g_cost + weight
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

    # Fallback: abstract graph is disconnected (common with small cluster sizes
    # where gateway nodes are sparse or start/goal can't reach them).
    # Chain direct pixel-level A* segments through goals in declaration order.
    print("  [HPA*] Abstract path not found — falling back to direct pixel A*")
    fallback_path = [start]
    for g in goals[1:]:   # goals[0] == start
        seg = _astar_segment(fallback_path[-1], g, road_bitmap)
        fallback_path.extend(seg[1:])
    return fallback_path, frames_sparse, expansions


# -------------------- DATA HANDLING --------------------

def load_npz_test_set(npz_path):
    """
    Load input NPZ read-only, copy arrays into plain numpy objects,
    then close the file handle so it can never be modified.

    Returns a float32 cost grid normalised to [0.0, 1.0] where
    0.0 = cheapest/most traversable and 1.0 = most expensive.

    Supported NPZ layouts (tried in priority order):
      'raster'             – binary uint8 road map (1=road, 0=off-road)
      'cost_map_normalized'– pre-normalised float cost map
      'env_map_normalized' – pre-normalised float environmental map
      'cost_map'           – raw float cost map (will be normalised)
      'env_map'            – raw float environmental map (will be normalised)
      <first 2-D numeric>  – fallback: any other 2-D numeric array
    """
    data = np.load(npz_path, allow_pickle=True, mmap_mode='r')

    # --- locate the grid array ---
    raster_key = None
    for key in RASTER_KEY_PRIORITY:
        if key in data:
            raster_key = key
            break
    if raster_key is None:
        for key in data.keys():
            v = data[key]
            if hasattr(v, 'ndim') and v.ndim == 2 and np.issubdtype(v.dtype, np.number):
                raster_key = key
                break
    if raster_key is None:
        raise ValueError(
            f"No 2-D numeric grid found in {npz_path}. Keys present: {list(data.keys())}"
        )

    raw = np.array(data[raster_key])

    # --- normalise to float32 cost grid [0.0 = cheapest, 1.0 = costliest] ---
    if np.issubdtype(raw.dtype, np.integer):
        # Binary raster: road (1) → 0.0 cost, off-road (0) → 1.0 cost
        cost_grid = np.where(raw == 1, 0.0, 1.0).astype(np.float32)
    else:
        # Float cost map: normalise min→0.0, max→1.0
        lo, hi = float(raw.min()), float(raw.max())
        if hi > lo:
            cost_grid = ((raw - lo) / (hi - lo)).astype(np.float32)
        else:
            cost_grid = np.zeros_like(raw, dtype=np.float32)

    goal_points = np.array(data['goal_points'], dtype=object)
    metadata    = {
        'width':  int(data['width'])  if 'width'  in data else cost_grid.shape[1],
        'height': int(data['height']) if 'height' in data else cost_grid.shape[0],
        'grid_key': raster_key,
    }
    data.close()
    return cost_grid, goal_points, metadata


def save_output_data(output_dir, input_path, road_bitmap, path, frames_sparse, metadata):
    """
    Write all outputs to output_dir. Raises RuntimeError if output_dir
    resolves to the same directory as the input NPZ to prevent overwrites.
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

    # Animator expects uint8 where 1 = "road" (cheap) and 0 = off-road.
    # For float32 cost grids, treat cells cheaper than the gateway threshold as road.
    animator_bitmap = (road_bitmap < GATEWAY_COST_THRESHOLD).astype(np.uint8)
    np.savez_compressed(
        os.path.join(output_dir, "hpa_road_bitmap.npz"),
        road_bitmap=animator_bitmap,
    )
    np.savez_compressed(
        os.path.join(output_dir, "hpa_protected_bitmap.npz"),
        protected_bitmap=np.zeros_like(animator_bitmap, dtype=np.uint8),
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
                  path_length, path_found, available_ram_mb, ram_safe_workers,
                  cpu_cores, error=None):
    """
    Append a structured entry for this run to hpa_run_log.txt.
    File is created on first run and appended to on every subsequent run.
    Always called even when the run crashes, so the log is never skipped.
    """
    log_path    = os.path.join(output_dir, "hpa_run_log.txt")
    timestamp   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    num_workers = NUM_WORKERS if NUM_WORKERS is not None else os.cpu_count()

    lines = [
        "=" * 60,
        f"Timestamp        : {timestamp}",
        f"Input file       : {os.path.abspath(input_path)}",
        f"Output directory : {os.path.abspath(output_dir)}",
        f"Cluster size     : {cluster_size}",
        f"Raster shape     : {raster_shape[0]} rows x {raster_shape[1]} cols",
        f"Available RAM    : {available_ram_mb:.0f} MB",
        f"CPU cores        : {cpu_cores}",
        f"RAM-safe workers : {ram_safe_workers}",
        f"Worker processes : {num_workers}",
        f"Goal points      : {len(goals)}",
        f"Status           : {'ERROR' if error else ('path found' if path_found else 'no path')}",
        f"Path length      : {path_length if path_found else 'N/A'} nodes",
        f"Search time      : {search_time:.3f}s",
        f"Expansions       : {expansions}",
        f"Frames recorded  : {frames_recorded}",
        f"Total visited    : {total_visited}",
    ]
    if error:
        lines.append(f"Error            : {error}")
    lines.append("")   # blank line between entries

    with open(log_path, "a") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Run log updated  : {log_path}")


# -------------------- MAIN --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",        required=True)
    parser.add_argument("--output",       default=None)
    parser.add_argument("--cluster-size", type=int, default=DEFAULT_CLUSTER_SIZE)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes for graph build "
             "(default: one per CPU core)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = f"./hpa_output_{os.path.splitext(os.path.basename(args.input))[0]}"

    global CLUSTER_SIZE, NUM_WORKERS
    CLUSTER_SIZE = max(1, int(args.cluster_size))

    cost_grid, goal_points, meta = load_npz_test_set(args.input)
    print(f"Grid key         : {meta['grid_key']}")

    goals = []
    for item in goal_points:
        # Two formats observed:
        #   [x, y, name]          — austin_test_raster style
        #   [name, y, x, lon, lat] — geometry_cost_map style (name first)
        if isinstance(item[0], str):
            x, y = item[1], item[2]  # format: [name, col_x, row_y, ...]
        else:
            try:
                x, y, _ = item
            except Exception:
                x, y = item[:2]
        goals.append((int(y), int(x)))
    if not goals:
        raise SystemExit("No goal points found in test set.")
    start = goals[0]

    # ---- RAM-aware worker count ----
    # Each worker receives a full copy of the grid as bytes (on Windows/macOS
    # with spawn). We leave at least half of available RAM free for the main
    # process (abstract graph, priority queue, frames, etc.).
    raster_bytes = cost_grid.nbytes
    try:
        import psutil
        available_ram = psutil.virtual_memory().available
    except ImportError:
        # psutil not installed — fall back to a conservative fixed budget.
        # Assume 4 GB available out of 8 GB total (half reserved for OS + main).
        available_ram = 4 * 1024 ** 3

    # Reserve half of available RAM for the main process; split the rest
    # across workers. Each worker needs ~2× raster_bytes (grid copy + working
    # memory for A* dicts/heaps inside the cluster).
    ram_budget      = available_ram // 2
    per_worker_cost = max(1, raster_bytes * 2)
    ram_safe_workers = max(1, ram_budget // per_worker_cost)

    cpu_workers = os.cpu_count() or 1

    if args.workers is not None:
        # User explicitly set --workers; honour it but warn if risky.
        NUM_WORKERS = args.workers
        if NUM_WORKERS > ram_safe_workers:
            print(
                f"WARNING: --workers {NUM_WORKERS} may exceed safe RAM usage. "
                f"Recommended max for this raster: {ram_safe_workers}. "
                f"Consider reducing if you see OOM errors."
            )
    else:
        # Auto-select: whichever is smaller — CPU count or RAM-safe limit.
        NUM_WORKERS = min(cpu_workers, ram_safe_workers)

    print(f"Cluster size     : {CLUSTER_SIZE}")
    print(f"Grid             : {cost_grid.shape[0]} x {cost_grid.shape[1]}  "
          f"({raster_bytes / 1024**2:.1f} MB)")
    print(f"Available RAM    : {available_ram / 1024**2:.0f} MB")
    print(f"RAM-safe workers : {ram_safe_workers}  (CPU cores: {cpu_workers})")
    print(f"Worker processes : {NUM_WORKERS}")

    # Ensure the output directory exists before anything else so the log
    # can always be written, even when a later step crashes.
    os.makedirs(args.output, exist_ok=True)

    # These are populated inside the try block but declared here so the
    # finally block can always reference them safely.
    path          = None
    frames        = []
    expansions    = 0
    search_time   = 0.0
    fatal_error   = None

    try:
        t0 = time.time()
        path, frames, expansions = a_star_hpa(start, goals, cost_grid)
        t1 = time.time()
        search_time = t1 - t0
    except Exception as exc:
        search_time = time.time() - t0
        fatal_error = exc
        print(f"ERROR during search: {exc}")
    finally:
        # --- Log is always written, crash or not ---
        total_visited = sum(len(positions) for _, positions in frames)
        path_length   = len(path) if path else 0

        print(f"Search time      : {search_time:.2f}s")
        print(f"Expansions       : {expansions}")
        print(f"Frames recorded  : {len(frames)}")
        print(f"Total visited    : {total_visited}")

        write_run_log(
            output_dir       = args.output,
            input_path       = args.input,
            cluster_size     = CLUSTER_SIZE,
            goals            = goals,
            raster_shape     = cost_grid.shape,
            search_time      = search_time,
            expansions       = expansions,
            frames_recorded  = len(frames),
            total_visited    = total_visited,
            path_length      = path_length,
            path_found       = path is not None,
            available_ram_mb = available_ram / 1024 ** 2,
            ram_safe_workers = ram_safe_workers,
            cpu_cores        = cpu_workers,
            error            = str(fatal_error) if fatal_error else None,
        )
    # If the search itself crashed, re-raise now that the log is safe.
    if fatal_error is not None:
        raise fatal_error

    save_output_data(args.output, args.input, cost_grid, path, frames, meta)

    print("Attempting to create animation (HPA)...")
    try:
        animator = HPAStarAnimator(output_dir=args.output, cluster_size=CLUSTER_SIZE)
        animator.create_animation(output_file="hpa_animation.gif", fps=ANIMATION_FPS, target_frames=ANIMATION_FRAMES)
        animator.create_static_comparison(output_file="hpa_comparison.png")
        print(f"Animation / comparison written to: {args.output}")
    except Exception as e:
        print(f"Animation failed (caught): {e}")


# ProcessPoolExecutor spawns child processes, so the entry-point guard
# is mandatory on Windows and good practice everywhere.
if __name__ == "__main__":
    main()