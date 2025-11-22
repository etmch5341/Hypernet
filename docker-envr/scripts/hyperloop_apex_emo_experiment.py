import json
import math
from pathlib import Path

import networkx as nx
from shapely.geometry import shape, LineString

from apex_core import ApexRouter


LI_GEOJSON_PATH = Path("/workspace/data/output/LI/LI_roads.geojson")

# ---------- 1) Build a simple graph from the GeoJSON ----------

def build_graph_from_geojson(path: Path) -> nx.Graph:
    with path.open() as f:
        gj = json.load(f)

    G = nx.Graph()

    for feature in gj["features"]:
        geom = shape(feature["geometry"])
        if not isinstance(geom, LineString):
            continue

        coords = list(geom.coords)
        for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
            n1 = (x1, y1)
            n2 = (x2, y2)

            if n1 not in G:
                G.add_node(n1, pos=n1)
            if n2 not in G:
                G.add_node(n2, pos=n2)

            # distance in *degrees*
            deg_dist = math.dist(n1, n2)
            # convert to meters (~111km per degree)
            seg_len_m = deg_dist * 111_000.0

            # toy env & tunnel placeholders
            env_risk = 0.1 * (abs(x1) + abs(y1))
            tunnel_cost = 0.0

            G.add_edge(
                n1, n2,
                length=seg_len_m,       # meters
                env_risk=env_risk,
                tunnel=tunnel_cost,
            )

    return G


# ---------- 2) A*pex: multi-heuristic A* ----------

def run_apex_seed(apex_router, source, target, w_len=1.0, w_env=0.0, w_tunnel=0.0):
    """
    Call A*pex-style multi-objective search, then choose the path
    that best matches the requested weights (for EMO seeding).
    Normalize keys so EMO always sees: length, env, tunnel.
    """
    weights = (w_len, w_env, w_tunnel)
    path, raw_obj = apex_router.route(source, target, weights)

    # Map env_risk -> env for the EMO layer
    obj_dict = {
        "length": float(raw_obj["length"]),
        "env":    float(raw_obj.get("env_risk", 0.0)),
        "tunnel": float(raw_obj.get("tunnel", 0.0)),
    }

    return {
        "path": path,
        "objectives": obj_dict,
        "weights": weights,
    }



# ---------- 3) Tiny EMO “test harness” seeded by A* ----------

def dominates(a, b):
    """Pareto dominance for minimization: a dominates b?"""
    ao, bo = a["objectives"], b["objectives"]
    better_or_equal = (
        ao["length"] <= bo["length"]
        and ao["env"] <= bo["env"]
        and ao["tunnel"] <= bo["tunnel"]
    )
    strictly_better = (
        ao["length"] < bo["length"]
        or ao["env"] < bo["env"]
        or ao["tunnel"] < bo["tunnel"]
    )
    return better_or_equal and strictly_better


def pareto_front(solutions):
    front = []
    for s in solutions:
        if not any(dominates(other, s) for other in solutions):
            front.append(s)
    return front


def run_experiment():
    if not LI_GEOJSON_PATH.exists():
        raise FileNotFoundError(
            f"{LI_GEOJSON_PATH} not found. "
            "Run lichtenstein_roads.py first inside the container."
        )

    print("Building graph from LI_roads.geojson...")
    G = build_graph_from_geojson(LI_GEOJSON_PATH)

    # A*pex-style multi-objective router
    apex_router = ApexRouter(
        G,
        objective_keys=("length", "env_risk", "tunnel"),
        eps=(0.01, 0.01, 0.01),  # tweak as needed
    )

    # Pick arbitrary source/target for now: first and last nodes
    nodes = list(G.nodes)
    source = nodes[0]
    target = nodes[-1]
    print(f"Using source={source}, target={target}")

    # ----- 3a) Seed population with multiple A*pex runs -----
    seed_weights = [
        (1.0, 0.0, 0.0),   # pure length
        (0.8, 0.2, 0.0),   # slight env penalty
        (0.5, 0.5, 0.0),   # equal length/env
        (0.5, 0.4, 0.1),   # include some tunnel cost
        (0.3, 0.6, 0.1),
    ]

    seeds = []
    for w_len, w_env, w_tunnel in seed_weights:
        try:
            s = run_apex_seed(apex_router, source, target, w_len, w_env, w_tunnel)
            seeds.append(s)
            o = s["objectives"]
            print(
                f"Seed {w_len,w_env,w_tunnel} -> "
                f"len={o['length']/1000:.3f} km, "
                f"env={o['env']:.1f}, "
                f"tun={o['tunnel']:.1f}"
            )
        except nx.NetworkXNoPath:
            print(f"No path found for weights {w_len,w_env,w_tunnel}")

    # ----- 3b) EMO step: compute Pareto front of seeds -----
    front = pareto_front(seeds)
    print("\nApproximate Pareto front (from seeds only):")
    for s in front:
        w = s["weights"]
        o = s["objectives"]
        print(
            f"weights={w} -> "
            f"len={o['length']/1000:.3f} km, env={o['env']:.1f}, tun={o['tunnel']:.1f}"
        )

    # Dump results to /workspace/outputs for analysis
    out_path = Path("/workspace/outputs/apex_emo_seed_results.json")
    with out_path.open("w") as f:
        json.dump(
            {
                "source": source,
                "target": target,
                "seeds": seeds,
                "pareto_front": front,
            },
            f,
            indent=2,
        )
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    run_experiment()
