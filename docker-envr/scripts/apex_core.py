# scripts/apex_core.py
#
# A*pex-inspired multi-objective search for graphs with vector costs.
# - Uses an Open list (priority queue) and per-state label sets
# - Each label stores a g-cost vector and a pointer to its parent node
# - Uses ε-dominance to prune nodes and approximate the Pareto front
#
# NOTE: This is NOT the official A*pex implementation from the paper,
# but a simplified Python implementation based on the same ideas:
# best-first multi-objective search + ε-approximate dominance.

import heapq
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Callable

import networkx as nx


CostVec = Tuple[float, ...]


def haversine_dist(node_a: Tuple[float, float],
                   node_b: Tuple[float, float]) -> float:
    """Approximate great-circle distance (meters) between two (lon, lat) nodes."""
    lon1, lat1 = node_a
    lon2, lat2 = node_b

    R = 6_371_000.0  # Earth radius (m)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(
        dlambda / 2.0
    ) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# ---------- small vector helpers ----------

def v_add(a: CostVec, b: CostVec) -> CostVec:
    return tuple(x + y for x, y in zip(a, b))


def v_leq(a: CostVec, b: CostVec) -> bool:
    """Component-wise <=."""
    return all(x <= y for x, y in zip(a, b))


def v_eps_dom(a: CostVec, b: CostVec, eps: CostVec) -> bool:
    """
    ε-dominance: a ε-dominates b iff a_i <= (1+eps_i) * b_i for all i.
    (Multi-objective minimization.)
    """
    return all(a_i <= (1.0 + e_i) * b_i for a_i, b_i, e_i in zip(a, b, eps))


def v_scalar_sum(a: CostVec) -> float:
    """Simple scalarization for tie-breaking in Open."""
    return sum(a)


@dataclass
class Node:
    state: Any
    g: CostVec              # cost-to-come
    h: CostVec              # heuristic
    f: CostVec              # g + h (lower bound on solution cost)
    parent: Optional["Node"] = None


class ApexRouter:
    """
    A*pex-style multi-objective search on a NetworkX graph.

    Graph requirements:
      - Each edge (u, v) must have objective fields; by default we expect
        ["length", "env_risk", "tunnel"] but this is configurable.

    Public API:
      - search(source, target) -> list of Pareto-approx paths (path, cost)
      - route(source, target, weights) -> single path picked from the set
        by minimizing a weighted sum of objectives (for EMO seeding).
    """

    def __init__(
    self,
    G: nx.Graph,
    objective_keys=("length", "env_risk", "tunnel"),
    heuristic_func=None,
    eps=(0.01, 0.01, 0.01),
):
        """
        :param G: NetworkX graph
        :param objective_keys: edge attribute names forming the cost vector
        :param heuristic_func: h(state, goal, dim) -> CostVec (optional).
                               If None, uses: [length heuristic, 0, 0, ...]
        :param eps: ε vector for ε-dominance (approximation factor) per objective
        """
        self.G = G
        self.objective_keys = objective_keys
        self.num_obj = len(objective_keys)
        self.eps = eps

        if heuristic_func is None:
            self.heuristic_func = self._default_heuristic
        else:
            self.heuristic_func = heuristic_func

    # ---------- Heuristic ----------

    def _default_heuristic(self, s: Any, goal: Any, dim: int) -> CostVec:
        """
        Default vector heuristic:
          - first objective: haversine distance (meters) between positions
          - others: 0 (admissible lower bounds)
        Requires nodes to have 'pos' attribute as (lon, lat).
        """
        pos_s = self.G.nodes[s].get("pos")
        pos_g = self.G.nodes[goal].get("pos")

        if pos_s is None or pos_g is None:
            # No geometry: just return zeros
            return tuple(0.0 for _ in range(self.num_obj))

        d = haversine_dist(pos_s, pos_g)
        h_vec = [0.0] * self.num_obj
        h_vec[0] = d  # assume first objective is length
        return tuple(h_vec)

    # ---------- Edge cost vector ----------

    def edge_cost_vec(self, u: Any, v: Any) -> CostVec:
        data = self.G[u][v]
        return tuple(float(data.get(k, 0.0)) for k in self.objective_keys)

    # ---------- Core multi-objective search ----------

    def search(
        self,
        source: Any,
        target: Any,
        max_solutions: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run A*pex-style multi-objective best-first search to find an
        ε-approximate Pareto front of paths from source to target.

        Returns a list of dicts:
          { "path": [states...], "objectives": CostVec }
        """

        # Open: priority queue of (priority_tuple, counter, Node)
        # priority_tuple = (f1, f2, ..., fN, scalar_sum(f))
        open_heap: List[Tuple[Tuple[float, ...], int, Node]] = []
        counter = 0

        # per-state label sets: state -> list of g-vectors (non-ε-dominated)
        labels: Dict[Any, List[CostVec]] = {}

        # current solution set (non-ε-dominated)
        solutions: List[Tuple[Node, CostVec]] = []

        # init
        h0 = self.heuristic_func(source, target, self.num_obj)
        g0 = tuple(0.0 for _ in range(self.num_obj))
        f0 = v_add(g0, h0)
        start_node = Node(state=source, g=g0, h=h0, f=f0, parent=None)

        labels[source] = [g0]
        prio = tuple(list(f0) + [v_scalar_sum(f0)])
        heapq.heappush(open_heap, (prio, counter, start_node))
        counter += 1

        while open_heap:
            _, _, node = heapq.heappop(open_heap)
            s = node.state

            # If this node's f is ε-dominated by an existing solution,
            # it cannot lead to a better ε-approx solution set.
            if any(v_eps_dom(sol_cost, node.f, self.eps)
                   for _, sol_cost in solutions):
                continue

            # Goal test
            if s == target:
                # Check if g(node) is ε-dominated by existing solutions
                g_n = node.g
                dominated = False
                for _, sol_cost in solutions:
                    if v_eps_dom(sol_cost, g_n, self.eps):
                        dominated = True
                        break

                if dominated:
                    continue

                # Remove any solutions ε-dominated by this one
                new_solutions: List[Tuple[Node, CostVec]] = []
                for sol_node, sol_cost in solutions:
                    if not v_eps_dom(g_n, sol_cost, self.eps):
                        new_solutions.append((sol_node, sol_cost))
                new_solutions.append((node, g_n))
                solutions = new_solutions

                if max_solutions is not None and len(solutions) >= max_solutions:
                    break

                # Continue: there might be other non-dominated solutions
                continue

            # Expand neighbors
            for succ in self.G.neighbors(s):
                c_vec = self.edge_cost_vec(s, succ)
                g_succ = v_add(node.g, c_vec)
                h_succ = self.heuristic_func(succ, target, self.num_obj)
                f_succ = v_add(g_succ, h_succ)

                # prune if ε-dominated by a solution
                if any(v_eps_dom(sol_cost, f_succ, self.eps)
                       for _, sol_cost in solutions):
                    continue

                # label pruning for succ
                existing = labels.get(succ, [])
                skip = False
                # If any existing label ε-dominates this one -> discard new
                for g_old in existing:
                    if v_eps_dom(g_old, g_succ, self.eps):
                        skip = True
                        break
                if skip:
                    continue

                # Remove labels ε-dominated by the new one
                new_labels = []
                for g_old in existing:
                    if not v_eps_dom(g_succ, g_old, self.eps):
                        new_labels.append(g_old)
                new_labels.append(g_succ)
                labels[succ] = new_labels

                # Push to Open
                child = Node(state=succ, g=g_succ, h=h_succ, f=f_succ, parent=node)
                prio = tuple(list(f_succ) + [v_scalar_sum(f_succ)])
                heapq.heappush(open_heap, (prio, counter, child))
                counter += 1

        # Decode Node -> (path, cost)
        result: List[Dict[str, Any]] = []
        for node, cost in solutions:
            path_states: List[Any] = []
            cur = node
            while cur is not None:
                path_states.append(cur.state)
                cur = cur.parent
            path_states.reverse()
            result.append({"path": path_states, "objectives": cost})

        return result

    # ---------- Compatibility: pick 1 path by weights ----------

    def route(
        self,
        source: Any,
        target: Any,
        weights: Tuple[float, ...],
        max_solutions: Optional[int] = None,
    ) -> Tuple[List[Any], Dict[str, float]]:
        """
        For compatibility with your earlier pipeline:
          - Run multi-objective search once
          - Pick the solution that minimizes Σ w_i * cost_i
        """
        sols = self.search(source, target, max_solutions=max_solutions)

        if not sols:
            raise nx.NetworkXNoPath(f"No (approximate) path from {source} to {target}")

        def score(obj: CostVec) -> float:
            return sum(w * c for w, c in zip(weights, obj))

        best = min(sols, key=lambda s: score(tuple(s["objectives"])))

        cost_dict = {
            k: float(v)
            for k, v in zip(self.objective_keys, best["objectives"])
        }
        return best["path"], cost_dict
