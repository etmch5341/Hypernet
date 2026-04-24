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
DEFAULT_CLUSTER_SIZE   = 0       # 0 = auto-scale to 10% of map
GATEWAY_MAX_COMPOSITE  = 0.6     # Clever bridge: non-road pixels valid if composite cost < 0.6
ALLOW_DIAGONAL         = True
ON_ROAD_COST           = 1.0
OFF_ROAD_COST          = 2.0     # Lowered from 5.0 to allow corner-cutting
MAX_ABSTRACT_EXPANSIONS = 500_000

DIRS_8 = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]


# ===========================================================================
# Cost-map helpers
# ===========================================================================

def load_cost_maps(project_root: str):
    """Load the 3 normalised cost layers from npz-files/."""
    for candidate_dir in [project_root, os.path.join(project_root, "npz-files")]:
        if os.path.exists(os.path.join(candidate_dir, "austin_construction_cost.npz")):
            npz = candidate_dir
            break
    else:
        raise FileNotFoundError(
            f"Cannot find austin_construction_cost.npz under {project_root}")

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


def _run_local_astar(start: Pos, goal: Pos,
                     cost_maps: Tuple[np.ndarray, np.ndarray, np.ndarray],
                     weights: Tuple[float, float, float],
                     road_bitmap: np.ndarray,
                     bounds: Optional[Tuple[int,int,int,int]]) -> Tuple[Optional[List[Pos]], Optional[Vec3]]:
    c_map, e_map, g_map = cost_maps
    H, W = c_map.shape
    w_c, w_e, w_g = weights
    composite = w_c * c_map + w_e * e_map + w_g * g_map

    if bounds:
        rmin, rmax, cmin, cmax = bounds
    else:
        rmin, rmax, cmin, cmax = 0, H, 0, W

    if start == goal:
        return [start], (0.0, 0.0, 0.0)

    pq = [(math.hypot(start[0]-goal[0], start[1]-goal[1]), 0.0, start, start)]
    gscore = {start: 0.0}
    came_from = {}
    visited = set()

    while pq:
        _, g, curr, prev = heapq.heappop(pq)
        if curr in visited: continue
        visited.add(curr)
        came_from[curr] = prev

        if curr == goal:
            path = []
            c = curr
            while c != start:
                path.append(c)
                c = came_from[c]
            path.append(start)
            path.reverse()
            cv = [0.0, 0.0, 0.0]
            prev_p = start
            for p in path[1:]:
                dr, dc = p[0] - prev_p[0], p[1] - prev_p[1]
                step = math.hypot(dr, dc)
                cv[0] += step * float(c_map[p[0], p[1]])
                cv[1] += step * float(e_map[p[0], p[1]])
                cv[2] += step * float(g_map[p[0], p[1]])
                prev_p = p
            return path, tuple(cv)

        for dr, dc in DIRS_8:
            nr, nc = curr[0]+dr, curr[1]+dc
            if not (rmin <= nr < rmax and cmin <= nc < cmax): continue
            # Modified penalty: no strict road requirement, just penalize off-road
            step_penalty = OFF_ROAD_COST if road_bitmap[nr, nc] != 1 else ON_ROAD_COST
            step_g = g + math.hypot(dr, dc) * (composite[nr, nc] + step_penalty * 0.1)
            nb = (nr, nc)
            if step_g < gscore.get(nb, float('inf')):
                gscore[nb] = step_g
                came_from[nb] = curr
                h = math.hypot(nr - goal[0], nc - goal[1])
                heapq.heappush(pq, (step_g + h, step_g, nb, curr))
    return None, None

def local_3d_pareto(start: Pos, goal: Pos,
                    cost_maps: Tuple[np.ndarray, np.ndarray, np.ndarray],
                    road_bitmap: np.ndarray,
                    bounds: Optional[Tuple[int,int,int,int]] = None) -> List[Tuple[List[Pos], Vec3]]:
    """Runs local A* with 5 distinct personality profiles to get up to 5 Pareto paths."""
    results = []
    profiles = [
        (0.33, 0.33, 0.33), # Balanced
        (0.80, 0.10, 0.10), # Construction focus
        (0.10, 0.80, 0.10), # Environment focus
        (0.10, 0.10, 0.80), # Geometry focus
        (0.45, 0.45, 0.10)  # Eco-Build (ignores geometry)
    ]
    
    for weights in profiles:
        p, v = _run_local_astar(start, goal, cost_maps, weights, road_bitmap, bounds)
        if p and v:
            # Only add if it's uniquely valuable (not strongly dominated by an existing profile)
            if not any(v == ev for _, ev in results) and not any(all(x <= (1.0 + 0.01) * y for x, y in zip(ev, v)) for _, ev in results):
                results.append((p, v))
            
    return results

# ===========================================================================
# HPA* gateway graph (extended with 3-D cost vectors)
# ===========================================================================

class HierarchicalGraph3D:
    def __init__(self, road_bitmap: np.ndarray,
                 cost_maps: Tuple[np.ndarray, np.ndarray, np.ndarray],
                 cluster_size: int = DEFAULT_CLUSTER_SIZE,
                 verbose: bool = True):
        self.grid      = road_bitmap
        self.cost_maps = cost_maps
        self.composite = composite_cost(cost_maps)
        self.height, self.width = road_bitmap.shape
        
        if cluster_size <= 0:
            self.c_size = max(10, max(self.height, self.width) // 10)
        else:
            self.c_size = max(1, int(cluster_size))
            
        self.verbose   = verbose

        self.nodes: Dict[Pos, List[Tuple[Pos, Vec3]]] = defaultdict(list)
        self.local_paths: Dict[Tuple[Pos, Pos, Vec3], List[Pos]] = {}
        self.clusters: Dict[Tuple[int,int], List[Pos]] = defaultdict(list)
        self.node_cluster: Dict[Pos, Tuple[int,int]] = {}

        self._build()

    def _cid(self, r: int, c: int):
        return (r // self.c_size, c // self.c_size)

    def _bounds(self, cid):
        cr, cc = cid
        r0, c0 = cr * self.c_size, cc * self.c_size
        return (max(0, r0), min(self.height, r0 + self.c_size),
                max(0, c0), min(self.width,  c0 + self.c_size))

    def _add_edge(self, a: Pos, b: Pos, vec: Vec3, path: Optional[List[Pos]] = None):
        for nbr, e_vec in self.nodes[a]:
            if nbr == b and e_vec == vec:
                return
        self.nodes[a].append((b, vec))
        self.nodes[b].append((a, vec))
        if path:
            self.local_paths[(a, b, vec)] = path
            self.local_paths[(b, a, vec)] = path[::-1]

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
            cost_a = float(self.composite[ra, ca])
            cost_b = float(self.composite[rb, cb])
            
            # Clever Gateway: Valid if it's a road OR if the terrain is cheap enough to bridge across
            if (self.grid[ra, ca] == 1 and self.grid[rb, cb] == 1) or (cost_a < GATEWAY_MAX_COMPOSITE and cost_b < GATEWAY_MAX_COMPOSITE):
                segment.append(((ra, ca), (rb, cb)))
            else:
                if segment:
                    self._commit(segment)
                    segment = []
        if segment:
            self._commit(segment)

    def _commit(self, segment):
        if not segment: return
        gateways = [segment[0]]
        if len(segment) > 2:
            gateways.append(segment[-1])
            
        for pa, pb in gateways:
            dr, dc = pb[0] - pa[0], pb[1] - pa[1]
            vec = edge_vec(dr, dc, pb[0], pb[1], self.cost_maps)
            if pa not in self.nodes: self.nodes[pa] = []
            if pb not in self.nodes: self.nodes[pb] = []
            self._add_edge(pa, pb, vec, path=[pa, pb])

    def _build(self):
        if self.verbose:
            print(f"[HPA*] Building gateway graph: grid={self.grid.shape}, "
                  f"cluster={self.c_size}")

        for r0 in range(0, self.height, self.c_size):
            for cc in range(self.c_size, self.width, self.c_size):
                self._find_gateways(r0, cc-1, r0, cc, vertical=True)
        for rr in range(self.c_size, self.height, self.c_size):
            for c0 in range(0, self.width, self.c_size):
                self._find_gateways(rr-1, c0, rr, c0, vertical=False)

        for node in list(self.nodes.keys()):
            cid = self._cid(*node)
            self.clusters[cid].append(node)
            self.node_cluster[node] = cid

        if self.verbose:
            print(f"[HPA*] Gateways: {len(self.nodes)}, "
                  f"clusters touched: {len(self.clusters)}")

        t0 = time.time()
        edges_added = 0
        for cid, nodes in self.clusters.items():
            if len(nodes) < 2:
                continue
            bounds = self._bounds(cid)
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    a, b = nodes[i], nodes[j]
                    path_vecs = local_3d_pareto(a, b, self.cost_maps, self.grid, bounds)
                    for path, vec in path_vecs:
                        self._add_edge(a, b, vec, path=path)
                        edges_added += 1
        if self.verbose:
            print(f"[HPA*] Intra-cluster edges: {edges_added}  "
                  f"({time.time()-t0:.1f}s)")

    def attach(self, p: Pos):
        cid = self._cid(*p)
        self.node_cluster.setdefault(p, cid)
        candidates = list(self.clusters.get(cid, []))
        if not candidates:
            near = sorted(self.nodes.keys(),
                          key=lambda n: abs(n[0]-p[0]) + abs(n[1]-p[1]))
            candidates = near[:8]
        bounds = self._bounds(cid) if candidates else None
        for node in candidates:
            path_vecs = local_3d_pareto(p, node, self.cost_maps, self.grid, bounds)
            for path, vec in path_vecs:
                self._add_edge(p, node, vec, path=path)

# ===========================================================================
# APEX
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
    remaining = [g for i, g in enumerate(goals) if not (mask & (1 << i))]
    if not remaining:
        return (0.0, 0.0, 0.0)
    h = min(math.hypot(g[0]-pos[0], g[1]-pos[1]) for g in remaining)
    return (h, h, h)


def apex_search(graph: HierarchicalGraph3D, start: Pos, goals: List[Pos],
                eps: float = 0.1,
                max_expansions: int = MAX_ABSTRACT_EXPANSIONS,
                verbose: bool = True):
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

    solutions = []   
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

        if mask == (1 << total) - 1:
            if not _dominated_by_set(g_vec, sol_set, eps):
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
                came_from[(nstate, new_g)] = (state, g_vec, e_vec)
                continue

            g_open[nstate] = {v for v in g_open[nstate]
                              if not _eps_dominates(new_g, v, eps)}
            g_open[nstate].add(new_g)
            came_from[(nstate, new_g)] = (state, g_vec, e_vec)

            h = _heuristic(neighbor, goals, new_mask)
            f_scalar = sum(a + b for a, b in zip(new_g, h))
            heapq.heappush(heap, (f_scalar, tie, new_g, nstate))
            tie += 1

    if verbose:
        print(f"  [APEX] {expansions} expansions, {len(solutions)} Pareto paths")

    abstract_paths = []
    for g_vec, end_state in solutions:
        cur = (end_state, g_vec)
        seq = [{"node": end_state[0], "edge_vec": None}]
        ok = True
        guard = 0
        while cur[0] != state0 or cur[1] != g0:
            parent = came_from.get(cur)
            if parent is None:
                ok = False
                break
            prev_state, prev_g, e_vec = parent
            seq[-1]["edge_vec"] = e_vec
            seq.append({"node": prev_state[0], "edge_vec": None})
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

def refine_path(abstract_seq: List[dict],
                graph: HierarchicalGraph3D) -> Tuple[List[Pos], Vec3]:
    """Reconstruct pixel-level path via O(1) cache lookup."""
    if not abstract_seq:
        return [], (0.0, 0.0, 0.0)

    full_path = [abstract_seq[0]["node"]]
    total_cv = [0.0, 0.0, 0.0]

    for i in range(1, len(abstract_seq)):
        prev_node = abstract_seq[i-1]["node"]
        curr_node = abstract_seq[i]["node"]
        e_vec = abstract_seq[i]["edge_vec"]

        seg = graph.local_paths.get((prev_node, curr_node, e_vec))
        if seg:
            full_path.extend(seg[1:])
            for k in range(3):
                total_cv[k] += e_vec[k]
        else:
            full_path.append(curr_node)

    return full_path, tuple(total_cv)

# ===========================================================================
# APEX Pareto filter + distance-to-ideal ranking
# ===========================================================================

def apex_rank(candidates: List[dict], eps: float = 0.05):
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
    start = goals[0]
    t0 = time.time()

    if verbose:
        print(f"\n[1/4] Building HPA* graph (cluster={cluster_size}, max 2 local paths) ...")
    graph = HierarchicalGraph3D(road_bitmap, cost_maps,
                                cluster_size=cluster_size, verbose=verbose)

    for p in [start] + [g for g in goals if g != start]:
        graph.attach(p)

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

    if verbose:
        print("\n[3/4] Refining abstract paths to pixel level via O(1) cache ...")
    candidates = []
    t2 = time.time()
    for i, ap in enumerate(abstract_paths):
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
        print(f"  Refinement: {time.time()-t2:.4f}s, "
              f"{len(candidates)} pixel paths produced")

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

