#!/usr/bin/env python3
"""
APEX + HPA* Combined Pipeline

Uses HPA* gateway clustering to build an abstract graph, then runs
APEX (multi-objective A* with ε-dominance) over it with 3 cost objectives:
  1. Construction cost
  2. Environmental impact
  3. Geometry cost

Pipeline:
  1. Build HierarchicalGraph (gateway nodes + cluster boundaries) from road bitmap
  2. Assign 3-D cost vectors to every edge using the three cost-map layers
  3. Run multi-objective APEX search over the abstract graph → Pareto front of paths
  4. Refine each abstract path to pixel level via local A* within clusters
  5. APEX Pareto filter + distance-to-ideal ranking → single best path
"""

import heapq
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
Vec3 = Tuple[float, float, float]   # (construction, environmental, geometry)
Pos  = Tuple[int, int]              # (row, col)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_CLUSTER_SIZE   = 40
GATEWAY_ROAD_REQUIRED  = True    # gateways only on road cells (road bitmap == 1)
ALLOW_DIAGONAL         = True
ON_ROAD_COST           = 1.0
OFF_ROAD_COST          = 5.0
MAX_ABSTRACT_EXPANSIONS = 500_000

DIRS_8 = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]


# ===========================================================================
# Cost-map helpers
# ===========================================================================

def load_cost_maps(project_root: str):
    """Load the 3 normalised cost layers from npz-files/."""
    npz = os.path.join(project_root, "npz-files")

    d = np.load(os.path.join(npz, "austin_construction_cost.npz"), allow_pickle=True)
    construction = d["cost_map_normalized"].astype(np.float64)

    d = np.load(os.path.join(npz, "austin_environmental_impact.npz"), allow_pickle=True)
    environmental = d["env_map_normalized"].astype(np.float64)

    d = np.load(os.path.join(npz, "geometry_cost_map.npz"), allow_pickle=True)
    geometry = d["cost_map"].astype(np.float64)

    return construction, environmental, geometry


def edge_vec(dr: int, dc: int, nr: int, nc: int,
             cost_maps: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Vec3:
    """3-D cost vector for one pixel step into (nr, nc)."""
    step = math.hypot(dr, dc)
    c, e, g = cost_maps
    return (step * float(c[nr, nc]),
            step * float(e[nr, nc]),
            step * float(g[nr, nc]))


def composite_cost(cost_maps: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Equal-weight mean of the 3 layers — used for gateway selection only."""
    c, e, g = cost_maps
    return ((c + e + g) / 3.0).astype(np.float32)


def local_3d_cost(start: Pos, goal: Pos,
                  cost_maps: Tuple[np.ndarray, np.ndarray, np.ndarray],
                  road_bitmap: np.ndarray,
                  bounds: Optional[Tuple[int,int,int,int]] = None) -> Optional[Vec3]:
    """
    Scalar A* inside a cluster to get the cheapest path (by composite cost),
    then accumulate the 3-D cost vector along that path.
    Returns None if unreachable.
    """
    c_map, e_map, g_map = cost_maps
    H, W = c_map.shape
    composite = (c_map + e_map + g_map) / 3.0

    if bounds:
        rmin, rmax, cmin, cmax = bounds
    else:
        rmin, rmax, cmin, cmax = 0, H, 0, W

    if start == goal:
        return (0.0, 0.0, 0.0)

    pq = [(math.hypot(start[0]-goal[0], start[1]-goal[1]), 0.0, start, start)]
    gscore: Dict[Pos, float] = {start: 0.0}
    came_from: Dict[Pos, Pos] = {}
    visited: set = set()

    while pq:
        _, g, curr, prev = heapq.heappop(pq)
        if curr in visited:
            continue
        visited.add(curr)
        came_from[curr] = prev

        if curr == goal:
            # reconstruct path and sum 3-D costs
            path = []
            c = curr
            while c != start:
                path.append(c)
                c = came_from[c]
            path.reverse()
            cv = [0.0, 0.0, 0.0]
            prev_p = start
            for p in path:
                dr = p[0] - prev_p[0]
                dc = p[1] - prev_p[1]
                step = math.hypot(dr, dc)
                cv[0] += step * float(c_map[p[0], p[1]])
                cv[1] += step * float(e_map[p[0], p[1]])
                cv[2] += step * float(g_map[p[0], p[1]])
                prev_p = p
            return tuple(cv)

        for dr, dc in DIRS_8:
            nr, nc = curr[0]+dr, curr[1]+dc
            if not (rmin <= nr < rmax and cmin <= nc < cmax):
                continue
            if GATEWAY_ROAD_REQUIRED and road_bitmap[nr, nc] != 1:
                step_penalty = OFF_ROAD_COST
            else:
                step_penalty = ON_ROAD_COST
            step_g = g + math.hypot(dr, dc) * (composite[nr, nc] + step_penalty * 0.1)
            nb = (nr, nc)
            if step_g < gscore.get(nb, float('inf')):
                gscore[nb] = step_g
                came_from[nb] = curr
                h = math.hypot(nr - goal[0], nc - goal[1])
                heapq.heappush(pq, (step_g + h, step_g, nb, curr))
    return None


# ===========================================================================
# HPA* gateway graph (extended with 3-D cost vectors)
# ===========================================================================

class HierarchicalGraph3D:
    """
    HPA* gateway abstraction where every edge carries a 3-D cost vector
    instead of a scalar weight.
    """

    def __init__(self, road_bitmap: np.ndarray,
                 cost_maps: Tuple[np.ndarray, np.ndarray, np.ndarray],
                 cluster_size: int = DEFAULT_CLUSTER_SIZE,
                 verbose: bool = True):
        self.grid      = road_bitmap
        self.cost_maps = cost_maps
        self.composite = composite_cost(cost_maps)
        self.c_size    = max(1, int(cluster_size))
        self.height, self.width = road_bitmap.shape
        self.verbose   = verbose

        # node -> list of (neighbor, Vec3)
        self.nodes: Dict[Pos, List[Tuple[Pos, Vec3]]] = defaultdict(list)
        # cluster_id -> list of gateway nodes
        self.clusters: Dict[Tuple[int,int], List[Pos]] = defaultdict(list)
        # node -> cluster_id
        self.node_cluster: Dict[Pos, Tuple[int,int]] = {}

        self._build()

    # ---- helpers --------------------------------------------------------

    def _cid(self, r: int, c: int):
        return (r // self.c_size, c // self.c_size)

    def _bounds(self, cid):
        cr, cc = cid
        r0, c0 = cr * self.c_size, cc * self.c_size
        return (max(0, r0), min(self.height, r0 + self.c_size),
                max(0, c0), min(self.width,  c0 + self.c_size))

    def _add_edge(self, a: Pos, b: Pos, vec: Vec3):
        self.nodes[a].append((b, vec))
        self.nodes[b].append((a, vec))

    # ---- gateway discovery (same as pranav-dev HPA*) --------------------

    def _find_gateways(self, r1, c1, r2, c2, vertical: bool):
        segment = []
        for i in range(self.c_size):
            ra = (r1 + i) if vertical else r1
            ca = c1 if vertical else (c1 + i)
            rb = (r2 + i) if vertical else r2
            cb = c2 if vertical else (c2 + i)
            if not (0 <= ra < self.height and 0 <= ca < self.width and
                    0 <= rb < self.height and 0 <= cb < self.width):
                continue
            if self.grid[ra, ca] == 1 and self.grid[rb, cb] == 1:
                segment.append(((ra, ca), (rb, cb)))
            else:
                if segment:
                    self._commit(segment)
                    segment = []
        if segment:
            self._commit(segment)

    def _commit(self, segment):
        mid = len(segment) // 2
        pa, pb = segment[mid]
        # inter-cluster edge: one pixel step
        dr = pb[0] - pa[0]
        dc = pb[1] - pa[1]
        vec = edge_vec(dr, dc, pb[0], pb[1], self.cost_maps)
        # ensure both nodes exist
        if pa not in self.nodes:
            self.nodes[pa] = []
        if pb not in self.nodes:
            self.nodes[pb] = []
        self._add_edge(pa, pb, vec)

    # ---- build ----------------------------------------------------------

    def _build(self):
        if self.verbose:
            print(f"[HPA*] Building gateway graph: grid={self.grid.shape}, "
                  f"cluster={self.c_size}")

        # Phase 1: discover gateways
        for r0 in range(0, self.height, self.c_size):
            for cc in range(self.c_size, self.width, self.c_size):
                self._find_gateways(r0, cc-1, r0, cc, vertical=True)
        for rr in range(self.c_size, self.height, self.c_size):
            for c0 in range(0, self.width, self.c_size):
                self._find_gateways(rr-1, c0, rr, c0, vertical=False)

        # Index nodes into clusters
        for node in list(self.nodes.keys()):
            cid = self._cid(*node)
            self.clusters[cid].append(node)
            self.node_cluster[node] = cid

        if self.verbose:
            print(f"[HPA*] Gateways: {len(self.nodes)}, "
                  f"clusters touched: {len(self.clusters)}")

        # Phase 2: connect gateway pairs inside each cluster with 3-D cost
        t0 = time.time()
        edges_added = 0
        for cid, nodes in self.clusters.items():
            if len(nodes) < 2:
                continue
            bounds = self._bounds(cid)
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    a, b = nodes[i], nodes[j]
                    vec = local_3d_cost(a, b, self.cost_maps,
                                        self.grid, bounds)
                    if vec is not None:
                        self._add_edge(a, b, vec)
                        edges_added += 1
        if self.verbose:
            print(f"[HPA*] Intra-cluster edges: {edges_added}  "
                  f"({time.time()-t0:.1f}s)")

    # ---- attach external points ----------------------------------------

    def attach(self, p: Pos):
        """Connect an external point to its cluster's gateway nodes."""
        cid = self._cid(*p)
        self.node_cluster.setdefault(p, cid)
        candidates = list(self.clusters.get(cid, []))
        if not candidates:
            # fallback: 8 nearest gateways by Manhattan
            near = sorted(self.nodes.keys(),
                          key=lambda n: abs(n[0]-p[0]) + abs(n[1]-p[1]))
            candidates = near[:8]
        bounds = self._bounds(cid) if candidates else None
        for node in candidates:
            vec = local_3d_cost(p, node, self.cost_maps, self.grid, bounds)
            if vec is not None:
                self._add_edge(p, node, vec)


# ===========================================================================
# APEX: Multi-objective A* with ε-dominance over the abstract graph
# ===========================================================================

def _dominates(a: Vec3, b: Vec3) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

def _eps_dominates(a: Vec3, b: Vec3, eps: float) -> bool:
    return all(x <= (1.0 + eps) * y for x, y in zip(a, b))

def _dominated_by_set(v: Vec3, s: set, eps: float) -> bool:
    if eps > 0:
        return any(_eps_dominates(x, v, eps) for x in s)
    return any(_dominates(x, v) for x in s)

def _heuristic(pos: Pos, goals: List[Pos], mask: int) -> Vec3:
    """Admissible: Euclidean lower bound replicated across all 3 objectives."""
    remaining = [g for i, g in enumerate(goals) if not (mask & (1 << i))]
    if not remaining:
        return (0.0, 0.0, 0.0)
    h = min(math.hypot(g[0]-pos[0], g[1]-pos[1]) for g in remaining)
    return (h, h, h)


def apex_search(graph: HierarchicalGraph3D, start: Pos, goals: List[Pos],
                eps: float = 0.1,
                max_expansions: int = MAX_ABSTRACT_EXPANSIONS,
                verbose: bool = True):
    """
    Multi-objective APEX search over the HPA* abstract gateway graph.
    Returns list of abstract paths: [{'sequence': [...], 'cost_vector': Vec3}]
    """
    total = len(goals)
    start_mask = sum((1 << i) for i, g in enumerate(goals) if g == start)

    g_open:   Dict = defaultdict(set)
    g_closed: Dict = defaultdict(set)
    came_from: Dict = {}

    heap = []
    tie = 0
    g0: Vec3 = (0.0, 0.0, 0.0)
    state0 = (start, start_mask)
    heapq.heappush(heap, (0.0, tie, g0, state0))
    g_open[state0].add(g0)
    tie += 1

    solutions = []   # (g_vec, final_state)
    sol_set:  set = set()
    expansions = 0

    while heap:
        _f, _t, g_vec, state = heapq.heappop(heap)
        pos, mask = state

        if g_vec not in g_open[state]:
            continue
        if _dominated_by_set(g_vec, g_closed[state], eps):
            g_open[state].discard(g_vec)
            continue
        if _dominated_by_set(g_vec, sol_set, eps):
            g_open[state].discard(g_vec)
            continue

        g_open[state].discard(g_vec)
        g_closed[state].add(g_vec)
        expansions += 1

        if expansions > max_expansions:
            if verbose:
                print(f"  [APEX] expansion cap hit ({max_expansions})")
            break

        # goal check
        if mask == (1 << total) - 1:
            if not _dominated_by_set(g_vec, sol_set, eps):
                # remove solutions that g_vec eps-dominates
                to_drop = {s for s in sol_set if _eps_dominates(g_vec, s, eps)}
                sol_set -= to_drop
                sol_set.add(g_vec)
                solutions.append((g_vec, state))
                if verbose:
                    print(f"  [APEX] Pareto #{len(solutions)}: "
                          f"{tuple(round(v, 2) for v in g_vec)}")
            continue

        for neighbor, e_vec in graph.nodes.get(pos, []):
            new_mask = mask
            for i, g in enumerate(goals):
                if neighbor == g:
                    new_mask |= (1 << i)
            nstate = (neighbor, new_mask)
            new_g: Vec3 = tuple(a + b for a, b in zip(g_vec, e_vec))

            if _dominated_by_set(new_g, sol_set, eps):
                continue
            if _dominated_by_set(new_g, g_open[nstate], eps):
                continue
            if _dominated_by_set(new_g, g_closed[nstate], eps):
                came_from[(nstate, new_g)] = (state, g_vec)
                continue

            # prune dominated open labels
            g_open[nstate] = {v for v in g_open[nstate]
                              if not _eps_dominates(new_g, v, eps)}
            g_open[nstate].add(new_g)
            came_from[(nstate, new_g)] = (state, g_vec)

            h = _heuristic(neighbor, goals, new_mask)
            f_scalar = sum(a + b for a, b in zip(new_g, h))
            heapq.heappush(heap, (f_scalar, tie, new_g, nstate))
            tie += 1

    if verbose:
        print(f"  [APEX] {expansions} expansions, {len(solutions)} Pareto paths")

    # reconstruct abstract sequences
    abstract_paths = []
    for g_vec, end_state in solutions:
        cur = (end_state, g_vec)
        seq = [end_state[0]]
        ok = True
        guard = 0
        while cur[0] != state0 or cur[1] != g0:
            parent = came_from.get(cur)
            if parent is None:
                ok = False
                break
            prev_state, prev_g = parent
            seq.append(prev_state[0])
            cur = (prev_state, prev_g)
            guard += 1
            if guard > 1_000_000:
                ok = False
                break
        if ok:
            seq.reverse()
            abstract_paths.append({"sequence": seq, "cost_vector": g_vec})

    return abstract_paths


# ===========================================================================
# Refine abstract gateway path → pixel path
# ===========================================================================

def refine_path(abstract_seq: List[Pos],
                graph: HierarchicalGraph3D) -> Tuple[List[Pos], Vec3]:
    """Reconstruct pixel-level path via local A* between consecutive gateways."""
    if not abstract_seq:
        return [], (0.0, 0.0, 0.0)

    full_path = [abstract_seq[0]]
    total_cv = [0.0, 0.0, 0.0]

    for i in range(len(abstract_seq) - 1):
        a, b = abstract_seq[i], abstract_seq[i+1]
        dr, dc = b[0]-a[0], b[1]-a[1]

        # adjacent inter-cluster step
        if abs(dr) <= 1 and abs(dc) <= 1 and (dr or dc):
            vec = edge_vec(dr, dc, b[0], b[1], graph.cost_maps)
            full_path.append(b)
            for k in range(3):
                total_cv[k] += vec[k]
            continue

        # intra-cluster: local A* with path reconstruction
        ca = graph.node_cluster.get(a)
        cb = graph.node_cluster.get(b)
        bounds = graph._bounds(ca) if ca and ca == cb else None

        c_map, e_map, g_map = graph.cost_maps
        composite = graph.composite
        H, W = graph.height, graph.width
        rmin, rmax, cmin, cmax = bounds if bounds else (0, H, 0, W)

        pq = [(math.hypot(dr, dc), 0.0, a)]
        gscore = {a: 0.0}
        came: Dict[Pos, Pos] = {}
        vis: set = set()
        found = False

        while pq:
            _, g, curr = heapq.heappop(pq)
            if curr in vis:
                continue
            vis.add(curr)
            if curr == b:
                found = True
                break
            for ddr, ddc in DIRS_8:
                nr, nc = curr[0]+ddr, curr[1]+ddc
                if not (rmin <= nr < rmax and cmin <= nc < cmax):
                    continue
                step_g = g + math.hypot(ddr, ddc) * float(composite[nr, nc])
                nb = (nr, nc)
                if step_g < gscore.get(nb, float('inf')):
                    gscore[nb] = step_g
                    came[nb] = curr
                    heapq.heappush(pq, (step_g + math.hypot(nr-b[0], nc-b[1]),
                                        step_g, nb))

        if found:
            seg = []
            cur = b
            while cur != a and cur in came:
                seg.append(cur)
                cur = came[cur]
            seg.reverse()
            # accumulate 3-D costs
            prev = a
            for p in seg:
                d = math.hypot(p[0]-prev[0], p[1]-prev[1])
                total_cv[0] += d * float(c_map[p[0], p[1]])
                total_cv[1] += d * float(e_map[p[0], p[1]])
                total_cv[2] += d * float(g_map[p[0], p[1]])
                prev = p
            full_path.extend(seg)
        else:
            vec = edge_vec(dr, dc, b[0], b[1], graph.cost_maps)
            full_path.append(b)
            for k in range(3):
                total_cv[k] += vec[k]

    return full_path, tuple(total_cv)


# ===========================================================================
# APEX Pareto filter + distance-to-ideal ranking
# ===========================================================================

def apex_rank(candidates: List[dict], eps: float = 0.05):
    """Filter to ε-Pareto front, rank by normalised distance-to-ideal."""
    if not candidates:
        return [], [], None, None, None

    pareto, dominated = [], []
    for cand in candidates:
        cv = cand["cost_vector"]
        is_dom = any(
            _eps_dominates(other["cost_vector"], cv, eps) and other is not cand
            for other in candidates
        )
        (dominated if is_dom else pareto).append(cand)

    source = pareto if pareto else candidates
    costs = np.array([c["cost_vector"] for c in source], dtype=np.float64)
    ideal = tuple(costs.min(axis=0).tolist())
    nadir = tuple(costs.max(axis=0).tolist())

    mn = costs.min(axis=0)
    mx = costs.max(axis=0)
    span = np.where(mx - mn > 1e-9, mx - mn, 1.0)
    norm = (costs - mn) / span
    dists = np.linalg.norm(norm, axis=1)

    for cand, dist in zip(source, dists):
        cand["apex_dist"] = float(dist)

    ranked = sorted(source, key=lambda c: c["apex_dist"])
    for rank, cand in enumerate(ranked, 1):
        cand["apex_rank"] = rank

    best = ranked[0] if ranked else None
    return pareto, dominated, ideal, nadir, best


# ===========================================================================
# Top-level runner
# ===========================================================================

def run_apex_hpa(road_bitmap: np.ndarray,
                 cost_maps: Tuple[np.ndarray, np.ndarray, np.ndarray],
                 goals: List[Pos],
                 cluster_size: int = DEFAULT_CLUSTER_SIZE,
                 apex_eps: float = 0.1,
                 filter_eps: float = 0.05,
                 max_expansions: int = MAX_ABSTRACT_EXPANSIONS,
                 verbose: bool = True):
    """
    Full APEX+HPA* pipeline.

    Returns:
        candidates  - all refined paths with cost vectors
        pareto      - Pareto-optimal subset
        best        - single best-compromise path dict
        graph       - the HierarchicalGraph3D (for inspection/viz)
    """
    start = goals[0]
    t0 = time.time()

    # 1. Build HPA* gateway graph with 3-D edge costs
    if verbose:
        print("\n[1/4] Building HPA* graph with 3-D edge costs ...")
    graph = HierarchicalGraph3D(road_bitmap, cost_maps,
                                cluster_size=cluster_size, verbose=verbose)

    # 2. Attach start and all goals
    for p in [start] + [g for g in goals if g != start]:
        graph.attach(p)

    # 3. APEX multi-objective search over abstract graph
    if verbose:
        print("\n[2/4] Running APEX search over abstract graph ...")
    t1 = time.time()
    abstract_paths = apex_search(graph, start, goals,
                                 eps=apex_eps,
                                 max_expansions=max_expansions,
                                 verbose=verbose)
    if verbose:
        print(f"  Abstract search: {time.time()-t1:.2f}s, "
              f"{len(abstract_paths)} Pareto abstract paths")

    if not abstract_paths:
        if verbose:
            print("  No abstract paths found.")
        return [], [], None, graph

    # 4. Refine each abstract path to pixel level
    if verbose:
        print("\n[3/4] Refining abstract paths to pixel level ...")
    candidates = []
    t2 = time.time()
    for i, ap in enumerate(abstract_paths):
        if verbose:
            print(f"  Refining {i+1}/{len(abstract_paths)} ...", flush=True)
        pixel_path, cost_vec = refine_path(ap["sequence"], graph)
        if pixel_path:
            candidates.append({
                "candidate_id":      i + 1,
                "path":              pixel_path,
                "abstract_sequence": ap["sequence"],
                "cost_vector":       cost_vec,
                "construction":      cost_vec[0],
                "environmental":     cost_vec[1],
                "geometry":          cost_vec[2],
            })
    if verbose:
        print(f"  Refinement: {time.time()-t2:.2f}s, "
              f"{len(candidates)} pixel paths produced")

    # 5. APEX Pareto filter + ranking
    if verbose:
        print("\n[4/4] APEX Pareto filter + ranking ...")
    pareto, dominated, ideal, nadir, best = apex_rank(candidates, eps=filter_eps)

    if verbose:
        print(f"  Pareto-optimal: {len(pareto)}  |  Dominated: {len(dominated)}")
        print(f"  Ideal : {tuple(round(v,2) for v in ideal)}")
        print(f"  Nadir : {tuple(round(v,2) for v in nadir)}")
        if best:
            cv = best["cost_vector"]
            print(f"  BEST  : id={best['candidate_id']}  "
                  f"constr={cv[0]:.1f}  env={cv[1]:.1f}  geo={cv[2]:.1f}  "
                  f"waypoints={len(best['path'])}")
        print(f"\nTotal runtime: {time.time()-t0:.2f}s")

    return candidates, pareto, best, graph
