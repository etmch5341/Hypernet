#!/usr/bin/env python3
"""Clustered NAMOA*-dr: arvin-dev's HPA* clustering/gateway scheme,
with every scalar A* search replaced by multi-objective NAMOA*-dr over
the three cost-map layers (construction, environmental, geometry).

Design:
  1. Partition the raster into square clusters (same as HPA*).
  2. Find gateway nodes on cluster boundaries using the composite scalar
     cost grid (cheap cells only), identical to HPA*.
  3. For every pair of gateways inside a cluster, run NAMOA*-dr locally
     to get a Pareto-optimal set of 3-D cost vectors. Each Pareto point
     becomes a multi-edge in the abstract graph.
  4. Abstract search uses a vector-dominance NAMOA*-dr over the gateway
     graph, producing a Pareto-optimal set of abstract paths from start
     through every goal.
  5. Each abstract path is refined to pixel level by running NAMOA*-dr
     between consecutive gateways (best-compromise single pixel path per
     segment) and stitching the segments together.

The refined pixel paths are returned with their 3-D cost vectors so the
APEX stage can select the best path.
"""

import heapq
import math
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutureTimeoutError

import numpy as np


ALLOW_DIAGONAL = True
DIRS_8 = [(0, 1), (0, -1), (1, 0), (-1, 0),
          (1, 1), (1, -1), (-1, 1), (-1, -1)]
DIRS_4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# A cheap cell for gateway selection: composite cost below this threshold.
GATEWAY_COST_THRESHOLD = 0.5

# Per-segment NAMOA-dr expansion cap. Scales with cluster area.
def _segment_expansion_cap(cluster_size):
    return max(5000, cluster_size * cluster_size * 8)


# ---------------------------------------------------------------------------
# Pareto helpers (3 objectives)
# ---------------------------------------------------------------------------

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


def eps_dominates(g1, g2, eps):
    return all(a <= (1.0 + eps) * b for a, b in zip(g1, g2))


def add_to_nondominated(g_new, g_set, eps=0.0):
    if eps > 0.0:
        for g in g_set:
            if eps_dominates(g, g_new, eps):
                return False
        to_drop = [g for g in g_set if eps_dominates(g_new, g, eps)]
    else:
        for g in g_set:
            if dominates(g, g_new):
                return False
        to_drop = [g for g in g_set if dominates(g_new, g)]
    for g in to_drop:
        g_set.remove(g)
    g_set.add(g_new)
    return True


def is_dominated_by_set(g_new, g_set, eps=0.0):
    if eps > 0.0:
        return any(eps_dominates(g, g_new, eps) for g in g_set)
    return any(dominates(g, g_new) for g in g_set)


# ---------------------------------------------------------------------------
# Pixel-level NAMOA*-dr between two points, constrained to a rectangle.
# ---------------------------------------------------------------------------

def _edge_vec(dr, dc, nr, nc, cost_maps):
    """Return the 3-D edge cost vector for stepping into (nr, nc)."""
    step = math.hypot(dr, dc)
    c, e, g = cost_maps
    return (step * float(c[nr, nc]),
            step * float(e[nr, nc]),
            step * float(g[nr, nc]))


def namoa_dr_segment(start, goal, cost_maps, bounds=None,
                     allow_diagonal=ALLOW_DIAGONAL,
                     eps=0.1, max_expansions=None,
                     return_path=False):
    """Run NAMOA*-dr from ``start`` to ``goal`` constrained to ``bounds``.

    Returns (pareto_costs, pareto_paths_or_none).
    ``pareto_costs`` is a list of 3-tuples on the Pareto frontier; if
    ``return_path=True`` the paths (pixel coordinates) are also returned,
    otherwise the path list holds ``None`` placeholders.
    """
    c_map, e_map, g_map = cost_maps
    height, width = c_map.shape
    if bounds is None:
        rmin, rmax, cmin, cmax = 0, height, 0, width
    else:
        rmin, rmax, cmin, cmax = bounds

    if not (rmin <= start[0] < rmax and cmin <= start[1] < cmax):
        return [], []
    if not (rmin <= goal[0] < rmax and cmin <= goal[1] < cmax):
        return [], []
    if start == goal:
        zero = (0.0, 0.0, 0.0)
        return [zero], [[start] if return_path else None]

    directions = DIRS_8 if allow_diagonal else DIRS_4

    # Scalar A* heuristic (admissible lower bound in each component is 0;
    # we use Euclidean distance to steer the search quickly).
    def h_scalar(pos):
        return math.hypot(pos[0] - goal[0], pos[1] - goal[1])

    g_open = defaultdict(set)
    g_closed = defaultdict(set)
    came_from = {}

    heap = []
    tie = 0
    g0 = (0.0, 0.0, 0.0)
    heapq.heappush(heap, (h_scalar(start), tie, g0, start))
    tie += 1
    g_open[start].add(g0)

    solutions = []        # list of (cost_vec, state)
    solution_set = set()  # dominance bookkeeping
    expansions = 0
    cap = max_expansions if max_expansions is not None else 10_000_000

    while heap:
        _f, _t, g_vec, pos = heapq.heappop(heap)
        if g_vec not in g_open[pos]:
            continue
        if is_dominated_by_set(g_vec, g_closed[pos], eps):
            g_open[pos].discard(g_vec)
            continue
        if is_dominated_by_set(g_vec, solution_set, eps):
            g_open[pos].discard(g_vec)
            continue

        g_open[pos].discard(g_vec)
        g_closed[pos].add(g_vec)
        expansions += 1
        if expansions > cap:
            break

        if pos == goal:
            if add_to_nondominated(g_vec, solution_set, eps):
                solutions.append((g_vec, pos))
            continue

        for dr, dc in directions:
            nr, nc = pos[0] + dr, pos[1] + dc
            if not (rmin <= nr < rmax and cmin <= nc < cmax):
                continue
            edge = _edge_vec(dr, dc, nr, nc, cost_maps)
            new_g = tuple(a + b for a, b in zip(g_vec, edge))

            if is_dominated_by_set(new_g, solution_set, eps):
                continue
            if is_dominated_by_set(new_g, g_open[(nr, nc)], eps):
                continue
            if is_dominated_by_set(new_g, g_closed[(nr, nc)], eps):
                # Delayed-reopen check — if this proves itself later we
                # re-open it via the came_from chain.
                came_from[((nr, nc), new_g)] = (pos, g_vec)
                continue

            if eps > 0.0:
                g_open[(nr, nc)] -= {g for g in g_open[(nr, nc)]
                                     if eps_dominates(new_g, g, eps)}
            else:
                g_open[(nr, nc)] -= {g for g in g_open[(nr, nc)]
                                     if dominates(new_g, g)}
            g_open[(nr, nc)].add(new_g)
            came_from[((nr, nc), new_g)] = (pos, g_vec)
            heapq.heappush(heap,
                           (sum(new_g) + h_scalar((nr, nc)), tie,
                            new_g, (nr, nc)))
            tie += 1

    # Reconstruct paths if requested.
    if not return_path:
        return [gv for gv, _ in solutions], [None] * len(solutions)

    paths = []
    costs = []
    for g_vec, end_state in solutions:
        cur = (end_state, g_vec)
        path = [cur[0]]
        start_key = (start, g0)
        guard = 0
        ok = True
        while cur != start_key:
            parent = came_from.get(cur)
            if parent is None:
                ok = False
                break
            pos, gv = parent
            path.append(pos)
            cur = parent
            guard += 1
            if guard > 5_000_000:
                ok = False
                break
        if ok:
            path.reverse()
            paths.append(path)
            costs.append(g_vec)

    return costs, paths


# ---------------------------------------------------------------------------
# Top-level worker for ProcessPoolExecutor. Must live at module scope so
# that `spawn`-based pools (macOS / Windows) can pickle it.
# ---------------------------------------------------------------------------

def _compute_cluster_namoa_edges(args):
    """Run NAMOA*-dr for every gateway pair inside one cluster.

    Parameters
    ----------
    args : tuple
        (c_bytes, e_bytes, g_bytes, dtype_str, height, width,
         bounds, nodes, cap, segment_eps, allow_diagonal)

    Returns
    -------
    list of (node_a, node_b, list[vector])
    """
    (c_bytes, e_bytes, g_bytes, dtype_str, height, width,
     bounds, nodes, cap, segment_eps, allow_diagonal) = args

    dtype = np.dtype(dtype_str)
    c_map = np.frombuffer(c_bytes, dtype=dtype).reshape(height, width)
    e_map = np.frombuffer(e_bytes, dtype=dtype).reshape(height, width)
    g_map = np.frombuffer(g_bytes, dtype=dtype).reshape(height, width)
    cost_maps = (c_map, e_map, g_map)

    edges = []
    n = len(nodes)
    for i in range(n):
        for j in range(i + 1, n):
            a = nodes[i]
            b = nodes[j]
            costs, _ = namoa_dr_segment(
                a, b, cost_maps,
                bounds=bounds,
                allow_diagonal=allow_diagonal,
                eps=segment_eps,
                max_expansions=cap,
                return_path=False,
            )
            if costs:
                edges.append((a, b, costs))
    return edges


# ---------------------------------------------------------------------------
# Hierarchical graph over NAMOA-dr cluster-internal edges.
# ---------------------------------------------------------------------------

class ClusteredNamoaGraph:
    """Gateway abstract graph whose edges carry Pareto sets of 3-D vectors."""

    def __init__(self, composite_grid, cost_maps, cluster_size=40,
                 segment_eps=0.15, verbose=True, num_workers=None):
        self.composite = composite_grid.astype(np.float32)
        self.cost_maps = cost_maps
        self.c_size = max(1, int(cluster_size))
        self.height, self.width = self.composite.shape
        self.segment_eps = segment_eps
        self.verbose = verbose
        # None => one worker per logical core. 1 => force sequential.
        self.num_workers = num_workers

        # node -> list of (neighbor, list[vector cost])
        self.nodes = defaultdict(list)
        # cluster_id -> list[node]
        self.clusters = defaultdict(list)
        # node -> cluster_id (used by refinement to avoid full-grid NAMOA-dr)
        self.node_cluster = {}

        self._build()

    # ---- cluster helpers ----------------------------------------------------

    def _cluster_id(self, r, c):
        return (r // self.c_size, c // self.c_size)

    def _cluster_bounds(self, cid):
        cr, cc = cid
        r0 = cr * self.c_size
        c0 = cc * self.c_size
        return (max(0, r0), min(self.height, r0 + self.c_size),
                max(0, c0), min(self.width, c0 + self.c_size))

    def _add_edge(self, a, b, vectors):
        # Store multiple vector-weighted edges per pair so downstream
        # search can explore every Pareto option.
        if not vectors:
            return
        self.nodes[a].append((b, vectors))
        self.nodes[b].append((a, vectors))

    # ---- gateway discovery (same logic as HPA*) ----------------------------

    def _find_gateways(self, r1, c1, r2, c2, vertical):
        seg = []
        for i in range(self.c_size):
            ra = (r1 + i) if vertical else r1
            ca = c1 if vertical else (c1 + i)
            rb = (r2 + i) if vertical else r2
            cb = c2 if vertical else (c2 + i)
            if (0 <= ra < self.height and 0 <= ca < self.width and
                    0 <= rb < self.height and 0 <= cb < self.width):
                cheap = (self.composite[ra, ca] < GATEWAY_COST_THRESHOLD and
                         self.composite[rb, cb] < GATEWAY_COST_THRESHOLD)
                if cheap:
                    seg.append(((ra, ca), (rb, cb)))
                elif seg:
                    self._commit_segment(seg)
                    seg = []
        if seg:
            self._commit_segment(seg)

    def _commit_segment(self, segment):
        mid = len(segment) // 2
        a, b = segment[mid]
        # Inter-cluster edge: single step with 3-D vector cost.
        dr = b[0] - a[0]
        dc = b[1] - a[1]
        vec = _edge_vec(dr, dc, b[0], b[1], self.cost_maps)
        self.nodes.setdefault(a, [])
        self.nodes.setdefault(b, [])
        self._add_edge(a, b, [vec])

    def _build(self):
        if self.verbose:
            print(f"Building HPA graph: grid={self.composite.shape} "
                  f"cluster={self.c_size}")

        # Phase 1: gateway discovery.
        for r0 in range(0, self.height, self.c_size):
            for cc in range(self.c_size, self.width, self.c_size):
                self._find_gateways(r0, cc - 1, r0, cc, vertical=True)
        for rr in range(self.c_size, self.height, self.c_size):
            for c0 in range(0, self.width, self.c_size):
                self._find_gateways(rr - 1, c0, rr, c0, vertical=False)

        # Group nodes into clusters.
        for node in list(self.nodes.keys()):
            cid = self._cluster_id(*node)
            self.clusters[cid].append(node)
            self.node_cluster[node] = cid

        if self.verbose:
            print(f"  Gateways: {len(self.nodes)}; "
                  f"clusters touched: {len(self.clusters)}")

        # Phase 2: for every cluster, connect every gateway pair with a
        # local NAMOA*-dr Pareto set. Dispatched to a process pool so
        # every cluster runs in parallel on its own core.
        cap = _segment_expansion_cap(self.c_size)
        tasks = self._build_cluster_tasks(cap)
        total_clusters = len(tasks)
        if total_clusters == 0:
            return

        workers = self.num_workers
        if workers is None:
            workers = os.cpu_count() or 1
        workers = max(1, int(workers))
        # Clamp to number of tasks — no point having more workers than clusters.
        workers = min(workers, total_clusters)

        if self.verbose:
            print(f"  [NAMOA-dr graph] {total_clusters} clusters to process "
                  f"with {workers} worker(s)")

        t0 = time.time()
        if workers == 1:
            self._build_cluster_edges_sequential(tasks)
        else:
            ok = self._build_cluster_edges_parallel(tasks, workers)
            if not ok:
                if self.verbose:
                    print("  [NAMOA-dr graph] parallel build failed, "
                          "falling back to sequential")
                self._build_cluster_edges_sequential(tasks)
        if self.verbose:
            print(f"  [NAMOA-dr graph] phase 2 done in {time.time() - t0:.2f}s")

    def _build_cluster_tasks(self, cap):
        """Materialise one pool task per cluster with >= 2 gateways."""
        dtype = np.dtype(np.float64)  # NAMOA-dr expects float64
        c = self.cost_maps[0].astype(dtype, copy=False)
        e = self.cost_maps[1].astype(dtype, copy=False)
        g = self.cost_maps[2].astype(dtype, copy=False)
        dtype_str = dtype.str  # e.g. "<f8" — round-trips cleanly via np.dtype(...)
        c_bytes = c.tobytes()
        e_bytes = e.tobytes()
        g_bytes = g.tobytes()

        tasks = []
        for cid, nodes in self.clusters.items():
            if len(nodes) < 2:
                continue
            bounds = self._cluster_bounds(cid)
            tasks.append((
                c_bytes, e_bytes, g_bytes, dtype_str,
                self.height, self.width,
                bounds, tuple(nodes), cap,
                self.segment_eps, ALLOW_DIAGONAL,
            ))
        return tasks

    def _build_cluster_edges_sequential(self, tasks):
        total = len(tasks)
        for idx, task in enumerate(tasks, start=1):
            for a, b, costs in _compute_cluster_namoa_edges(task):
                self._add_edge(a, b, costs)
            if self.verbose and total and (idx % max(1, total // 10) == 0):
                print(f"  [NAMOA-dr graph] {idx}/{total} clusters (seq)")

    def _build_cluster_edges_parallel(self, tasks, workers):
        """Returns True if pool completed normally, False on a pool-level error."""
        total = len(tasks)
        # Each NAMOA-dr worker can be slow on a large cluster; scale the
        # timeout with area so pathologically large clusters don't kill
        # the run, but a truly hung worker still trips the guard.
        per_future_timeout = max(120, (self.c_size ** 2) // 50)

        done = 0
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_compute_cluster_namoa_edges, t): t
                           for t in tasks}
                for future in as_completed(futures,
                                           timeout=per_future_timeout * total):
                    done += 1
                    if self.verbose and total and (
                            done % max(1, total // 10) == 0):
                        print(f"  [NAMOA-dr graph] {done}/{total} "
                              f"clusters (par)", flush=True)
                    try:
                        edges = future.result(timeout=per_future_timeout)
                    except FutureTimeoutError:
                        if self.verbose:
                            print("  [NAMOA-dr graph] cluster timed out — "
                                  "skipping")
                        continue
                    except Exception as exc:
                        if self.verbose:
                            print(f"  [NAMOA-dr graph] cluster raised: {exc}")
                        continue
                    for a, b, costs in edges:
                        self._add_edge(a, b, costs)
            return True
        except FutureTimeoutError:
            if self.verbose:
                print("  [NAMOA-dr graph] pool-level timeout")
            return False
        except (MemoryError, OSError) as exc:
            if self.verbose:
                print(f"  [NAMOA-dr graph] pool OOM/OSError: {exc}")
            return False
        except Exception as exc:
            if self.verbose:
                print(f"  [NAMOA-dr graph] pool failed "
                      f"({type(exc).__name__}: {exc})")
            return False

    # ---- start / goal attachment ------------------------------------------

    def attach_point(self, p):
        """Connect external point ``p`` to its cluster's gateways."""
        cid = self._cluster_id(*p)
        self.node_cluster.setdefault(p, cid)
        bounds = self._cluster_bounds(cid)
        candidates = list(self.clusters.get(cid, []))
        if not candidates:
            # Fallback: find nearest 8 gateways by Manhattan distance.
            near = sorted(
                ((abs(n[0] - p[0]) + abs(n[1] - p[1]), n)
                 for n in self.nodes.keys()),
                key=lambda x: x[0],
            )
            candidates = [n for _, n in near[:8]]
            bounds = None
        cap = _segment_expansion_cap(self.c_size)
        attached = 0
        for node in candidates:
            costs, _ = namoa_dr_segment(
                p, node, self.cost_maps, bounds=bounds,
                allow_diagonal=ALLOW_DIAGONAL,
                eps=self.segment_eps, max_expansions=cap,
                return_path=False,
            )
            if costs:
                self._add_edge(p, node, costs)
                attached += 1
        return attached

    def bounds_for_pair(self, a, b):
        """Return the smallest cluster rectangle that contains both a and b,
        or ``None`` if they live in different clusters (caller should skip
        NAMOA-dr for that segment — it's a 1-step inter-cluster edge)."""
        ca = self.node_cluster.get(a)
        cb = self.node_cluster.get(b)
        if ca is not None and ca == cb:
            return self._cluster_bounds(ca)
        return None


# ---------------------------------------------------------------------------
# Multi-objective abstract search (visits every goal).
# ---------------------------------------------------------------------------

def _goal_mask(pos, mask, goals):
    for i, g in enumerate(goals):
        if pos == g:
            mask |= (1 << i)
    return mask


def _abstract_heuristic(pos, mask, goals):
    # Euclidean lower bound to nearest unvisited goal, replicated across
    # the three objectives (every component is admissible since cost
    # layers are non-negative and an Euclidean bound is optimistic).
    remaining = [g for i, g in enumerate(goals) if not (mask & (1 << i))]
    if not remaining:
        return (0.0, 0.0, 0.0)
    h = min(math.hypot(g[0] - pos[0], g[1] - pos[1]) for g in remaining)
    return (h, h, h)


def namoa_dr_abstract(graph, start, goals, eps=0.15,
                      max_expansions=500_000, verbose=True):
    """Abstract vector-search over gateway graph; returns Pareto abstract paths."""
    total_goals = len(goals)
    start_mask = _goal_mask(start, 0, goals)

    heap = []
    tie = 0
    g0 = (0.0, 0.0, 0.0)
    state0 = (start, start_mask)

    heapq.heappush(heap, (0.0, tie, g0, state0))
    tie += 1

    g_open = defaultdict(set)
    g_closed = defaultdict(set)
    g_open[state0].add(g0)
    came_from = {}

    goal_solutions = []   # list of (g_vec, final_state)
    goal_set = set()
    expansions = 0

    while heap:
        _f, _t, g_vec, state = heapq.heappop(heap)
        if g_vec not in g_open[state]:
            continue
        if is_dominated_by_set(g_vec, g_closed[state], eps):
            g_open[state].discard(g_vec)
            continue
        if is_dominated_by_set(g_vec, goal_set, eps):
            g_open[state].discard(g_vec)
            continue
        g_open[state].discard(g_vec)
        g_closed[state].add(g_vec)
        expansions += 1

        if expansions > max_expansions:
            if verbose:
                print(f"  [abstract] expansion cap hit ({max_expansions})")
            break

        pos, mask = state
        if mask == (1 << total_goals) - 1:
            if add_to_nondominated(g_vec, goal_set, eps):
                goal_solutions.append((g_vec, state))
                if verbose:
                    print(f"  [abstract] Pareto #{len(goal_solutions)}: "
                          f"{tuple(round(v, 2) for v in g_vec)}")
            continue

        for neighbor, vector_list in graph.nodes.get(pos, []):
            new_mask = _goal_mask(neighbor, mask, goals)
            nstate = (neighbor, new_mask)
            for edge_vec in vector_list:
                new_g = tuple(a + b for a, b in zip(g_vec, edge_vec))
                if is_dominated_by_set(new_g, goal_set, eps):
                    continue
                if is_dominated_by_set(new_g, g_open[nstate], eps):
                    continue
                if is_dominated_by_set(new_g, g_closed[nstate], eps):
                    came_from[(nstate, new_g)] = (state, g_vec, edge_vec)
                    continue
                if eps > 0:
                    g_open[nstate] -= {g for g in g_open[nstate]
                                       if eps_dominates(new_g, g, eps)}
                else:
                    g_open[nstate] -= {g for g in g_open[nstate]
                                       if dominates(new_g, g)}
                g_open[nstate].add(new_g)
                came_from[(nstate, new_g)] = (state, g_vec, edge_vec)
                h = _abstract_heuristic(neighbor, new_mask, goals)
                f_scalar = sum(a + b for a, b in zip(new_g, h))
                heapq.heappush(heap, (f_scalar, tie, new_g, nstate))
                tie += 1

    # Reconstruct abstract paths: sequence of gateway nodes + the edge
    # vector used at each step (needed so refinement can pick a Pareto
    # pixel path whose cost vector matches the abstract choice).
    abstract_paths = []
    for g_vec, end_state in goal_solutions:
        cur = (end_state, g_vec)
        seq = [cur[0][0]]
        edge_vecs = []
        guard = 0
        ok = True
        while cur[0] != state0 or cur[1] != g0:
            parent = came_from.get(cur)
            if parent is None:
                ok = False
                break
            prev_state, prev_g, edge = parent
            seq.append(prev_state[0])
            edge_vecs.append(edge)
            cur = (prev_state, prev_g)
            guard += 1
            if guard > 1_000_000:
                ok = False
                break
        if ok:
            seq.reverse()
            edge_vecs.reverse()
            abstract_paths.append({
                "sequence": seq,
                "edge_vectors": edge_vecs,
                "cost_vector": g_vec,
            })
    return abstract_paths, expansions


# ---------------------------------------------------------------------------
# Refine abstract gateway path to pixel path by running NAMOA-dr between
# consecutive gateways and picking the best-compromise segment.
# ---------------------------------------------------------------------------

def _balanced_pick(costs):
    """Normalise + pick the entry closest to the ideal corner."""
    if not costs:
        return 0
    arr = np.asarray(costs, dtype=np.float64)
    mn = arr.min(axis=0)
    mx = arr.max(axis=0)
    span = np.where(mx - mn > 1e-9, mx - mn, 1.0)
    norm = (arr - mn) / span
    dist = np.linalg.norm(norm, axis=1)
    return int(np.argmin(dist))


def _closest_pick(costs, target):
    """Return the index of ``costs`` closest to ``target`` in normalised space."""
    if not costs:
        return 0
    arr = np.asarray(costs, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mn = arr.min(axis=0)
    mx = arr.max(axis=0)
    span = np.where(mx - mn > 1e-9, mx - mn, 1.0)
    norm = (arr - mn) / span
    target_norm = np.clip((target - mn) / span, 0.0, 1.0)
    return int(np.argmin(np.linalg.norm(norm - target_norm, axis=1)))


def refine_abstract_path(abstract_seq, cost_maps, graph=None,
                         edge_vectors=None, eps=0.1, memo=None, verbose=False):
    """Return (pixel_path, cost_vector) for the abstract gateway sequence.

    ``graph`` — the ``ClusteredNamoaGraph`` the sequence was produced
    from; used to bound the per-segment NAMOA-dr to a single cluster.

    ``edge_vectors`` — optional list (len == len(abstract_seq) - 1) of
    the abstract edge vectors chosen for this path. When supplied the
    refinement picks the Pareto pixel path whose cost vector is nearest
    to the abstract choice, so paths that favour different trade-offs at
    the abstract level stay distinct at the pixel level. Without this
    every path would collapse to the same balanced pick per segment.

    ``memo`` — shared cache across abstract paths. Keyed by
    ``(a, b, rounded_target)`` so two abstract paths asking for very
    different trade-offs on the same gateway pair don't overwrite each
    other.
    """
    if not abstract_seq:
        return [], (0.0, 0.0, 0.0)
    pixel_path = [abstract_seq[0]]
    total = [0.0, 0.0, 0.0]
    if memo is None:
        memo = {}

    for i in range(len(abstract_seq) - 1):
        a = abstract_seq[i]
        b = abstract_seq[i + 1]
        target = edge_vectors[i] if edge_vectors else None
        # Bucket the target to 1-unit resolution so nearly-identical
        # targets share cache entries.
        tkey = (tuple(round(v, 0) for v in target)
                if target is not None else None)
        key = (a, b, tkey)

        cached = memo.get(key)
        if cached is not None:
            seg, seg_cost = cached
            pixel_path.extend(seg[1:])
            for k in range(3):
                total[k] += seg_cost[k]
            continue

        # Inter-cluster single-step edge (8-adjacent gateway pair) — no
        # NAMOA-dr needed.
        dr = b[0] - a[0]
        dc = b[1] - a[1]
        if abs(dr) <= 1 and abs(dc) <= 1 and (dr or dc):
            vec = _edge_vec(dr, dc, b[0], b[1], cost_maps)
            memo[key] = ([a, b], vec)
            pixel_path.append(b)
            for k in range(3):
                total[k] += vec[k]
            continue

        # Intra-cluster segment — bound the search to the cluster that
        # contains both endpoints.
        bounds = graph.bounds_for_pair(a, b) if graph is not None else None
        cap = _segment_expansion_cap(
            graph.c_size if graph is not None else 64)

        costs, paths = namoa_dr_segment(
            a, b, cost_maps,
            bounds=bounds,
            eps=eps,
            max_expansions=cap,
            return_path=True,
        )

        if not paths:
            vec = _edge_vec(dr, dc, b[0], b[1], cost_maps)
            memo[key] = ([a, b], vec)
            pixel_path.append(b)
            for k in range(3):
                total[k] += vec[k]
            continue

        idx = (_closest_pick(costs, target) if target is not None
               else _balanced_pick(costs))
        seg = paths[idx]
        seg_cost = costs[idx]
        memo[key] = (seg, seg_cost)
        pixel_path.extend(seg[1:])
        for k in range(3):
            total[k] += seg_cost[k]

    return pixel_path, tuple(total)


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def run_clustered_namoa(start, goals, cost_maps, composite_grid,
                        cluster_size=40, segment_eps=0.2, abstract_eps=0.15,
                        abstract_max_expansions=500_000, verbose=True,
                        num_workers=None):
    """End-to-end HPA* + NAMOA-dr: returns a list of candidate paths each
    with a 3-D cost vector (construction, environmental, geometry)."""
    t0 = time.time()
    graph = ClusteredNamoaGraph(composite_grid, cost_maps,
                                cluster_size=cluster_size,
                                segment_eps=segment_eps,
                                verbose=verbose,
                                num_workers=num_workers)
    if verbose:
        print(f"Graph build: {time.time() - t0:.2f}s")

    for p in [start] + [g for g in goals if g != start]:
        graph.attach_point(p)

    t1 = time.time()
    abstract_paths, expansions = namoa_dr_abstract(
        graph, start, goals,
        eps=abstract_eps,
        max_expansions=abstract_max_expansions,
        verbose=verbose,
    )
    if verbose:
        print(f"Abstract search: {time.time() - t1:.2f}s "
              f"({expansions} expansions, {len(abstract_paths)} Pareto paths)")

    candidates = []
    segment_memo = {}  # shared across every abstract path
    t_ref = time.time()
    for i, ap in enumerate(abstract_paths):
        if verbose:
            print(f"  Refining abstract path {i + 1}/{len(abstract_paths)} ...",
                  flush=True)
        pixel_path, cost_vec = refine_abstract_path(
            ap["sequence"], cost_maps,
            graph=graph,
            edge_vectors=ap.get("edge_vectors"),
            eps=segment_eps * 0.5,
            memo=segment_memo,
            verbose=verbose,
        )
        if pixel_path:
            candidates.append({
                "path": pixel_path,
                "abstract_sequence": ap["sequence"],
                "abstract_cost_vector": ap["cost_vector"],
                "cost_vector": cost_vec,
                "construction": cost_vec[0],
                "environmental": cost_vec[1],
                "geometry": cost_vec[2],
                "candidate_id": i + 1,
            })

    if verbose:
        print(f"Refinement: {time.time() - t_ref:.2f}s "
              f"({len(segment_memo)} unique segments cached)")
        print(f"Clustered NAMOA-dr produced {len(candidates)} candidate paths")
    return candidates, graph
