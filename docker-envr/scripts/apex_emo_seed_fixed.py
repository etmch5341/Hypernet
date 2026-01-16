import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import networkx as nx
import numpy as np
from shapely.geometry import shape, LineString

from apex_core import ApexRouter


# ---------- 1) Build graph from GeoJSON OR NPZ ----------

def build_graph_from_geojson(path: Path) -> nx.Graph:
    """Build graph from GeoJSON file with road network data."""
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


def build_graph_from_npz(npz_path: Path) -> Tuple[nx.Graph, np.ndarray, Dict]:
    """
    Build graph from NPZ raster data.
    Returns: (graph, raster, metadata_dict)
    """
    data = np.load(npz_path, allow_pickle=True)
    raster = data['raster']
    width = int(data['width'])
    height = int(data['height'])
    bbox = data['bbox']
    goal_points = data['goal_points']
    
    metadata = {
        'width': width,
        'height': height,
        'bbox': bbox,
        'goal_points': goal_points,
        'target_crs': str(data.get('target_crs', 'unknown'))
    }
    
    print(f"Loaded raster: {raster.shape}, goals: {len(goal_points)}")
    
    G = nx.Graph()
    
    # Build 8-connected grid graph from raster
    for y in range(height):
        for x in range(width):
            node = (x, y)
            G.add_node(node, pos=node, cost=float(raster[y, x]))
            
            # Connect to neighbors (8-connectivity)
            for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                nx_pos = x + dx
                ny_pos = y + dy
                
                if 0 <= nx_pos < width and 0 <= ny_pos < height:
                    neighbor = (nx_pos, ny_pos)
                    
                    # Edge length: diagonal edges are sqrt(2) longer
                    edge_dist = math.sqrt(dx**2 + dy**2) * 10.0  # 10m resolution
                    
                    # Cost based on average of the two nodes
                    avg_cost = (raster[y, x] + raster[ny_pos, nx_pos]) / 2.0
                    
                    # Environmental risk based on raster values
                    env_risk = avg_cost * 100.0  # Scale appropriately
                    
                    # Tunnel cost: higher for difficult terrain
                    tunnel_cost = max(0, avg_cost - 50.0) * 10.0  # Penalize high-cost areas
                    
                    G.add_edge(
                        node, neighbor,
                        length=edge_dist,
                        env_risk=float(env_risk),
                        tunnel=float(tunnel_cost),
                    )
    
    print(f"Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G, raster, metadata


# ---------- 2) A*pex: multi-heuristic A* ----------

def run_apex_seed(apex_router, source, target, w_len=1.0, w_env=0.0, w_tunnel=0.0):
    """
    Call A*pex-style multi-objective search, then choose the path
    that best matches the requested weights (for EMO seeding).
    Normalize keys so EMO always sees: length, env, tunnel.
    """
    weights = (w_len, w_env, w_tunnel)
    
    try:
        path, raw_obj = apex_router.route(source, target, weights)
    except Exception as e:
        print(f"  Warning: Routing failed for weights {weights}: {e}")
        raise
    
    # Map env_risk -> env for the EMO layer
    obj_dict = {
        "length": float(raw_obj.get("length", 0.0)),
        "env":    float(raw_obj.get("env_risk", 0.0)),
        "tunnel": float(raw_obj.get("tunnel", 0.0)),
    }

    return {
        "path": path,
        "objectives": obj_dict,
        "weights": weights,
    }


# ---------- 3) EMO utilities ----------

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
    """Extract non-dominated solutions."""
    front = []
    for s in solutions:
        if not any(dominates(other, s) for other in solutions):
            front.append(s)
    return front


# ---------- 4) Main experiment ----------

def run_experiment_geojson(geojson_path: Path, source=None, target=None):
    """Run experiment using GeoJSON input."""
    if not geojson_path.exists():
        raise FileNotFoundError(f"{geojson_path} not found")

    print(f"Building graph from {geojson_path}...")
    G = build_graph_from_geojson(geojson_path)

    # Pick source/target
    nodes = list(G.nodes)
    if source is None:
        source = nodes[0]
    if target is None:
        target = nodes[-1]
    
    print(f"Source: {source}")
    print(f"Target: {target}")
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return run_apex_emo_pipeline(G, source, target, "geojson")


def run_experiment_npz(npz_path: Path):
    """Run experiment using NPZ raster input."""
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} not found")

    print(f"Building graph from {npz_path}...")
    G, raster, metadata = build_graph_from_npz(npz_path)
    
    # Extract goal points as source and target
    goal_points = metadata['goal_points']
    if len(goal_points) < 2:
        raise ValueError("Need at least 2 goal points for routing")
    
    # Goal points are (x, y, name) tuples
    source = (int(goal_points[0][0]), int(goal_points[0][1]))
    target = (int(goal_points[1][0]), int(goal_points[1][1]))
    
    print(f"Source: {goal_points[0][2]} at {source}")
    print(f"Target: {goal_points[1][2]} at {target}")
    
    return run_apex_emo_pipeline(G, source, target, "npz", metadata)


def run_apex_emo_pipeline(G: nx.Graph, source, target, input_type: str, metadata: Optional[Dict] = None):
    """Core A*PEX-to-EMO pipeline."""
    
    # A*pex-style multi-objective router
    apex_router = ApexRouter(
        G,
        objective_keys=("length", "env_risk", "tunnel"),
        eps=(0.01, 0.01, 0.01),  # tolerance for epsilon-constraint
    )

    # ----- Seed population with multiple A*pex runs -----
    seed_weights = [
        (1.0, 0.0, 0.0),   # pure length
        (0.9, 0.1, 0.0),   # slight env penalty
        (0.8, 0.2, 0.0),
        (0.7, 0.3, 0.0),
        (0.6, 0.4, 0.0),
        (0.5, 0.5, 0.0),   # equal length/env
        (0.5, 0.4, 0.1),   # include some tunnel cost
        (0.4, 0.5, 0.1),
        (0.3, 0.6, 0.1),
        (0.3, 0.5, 0.2),
        (0.0, 0.0, 1.0),   # pure tunnel minimization
        (0.0, 1.0, 0.0),   # pure env minimization
    ]

    seeds = []
    successful_seeds = 0
    
    print("\n" + "="*60)
    print("SEEDING PHASE: Running A*PEX with different weight vectors")
    print("="*60)
    
    for w_len, w_env, w_tunnel in seed_weights:
        try:
            s = run_apex_seed(apex_router, source, target, w_len, w_env, w_tunnel)
            seeds.append(s)
            successful_seeds += 1
            o = s["objectives"]
            print(
                f"✓ Seed {w_len:.1f},{w_env:.1f},{w_tunnel:.1f} -> "
                f"len={o['length']:.1f}m, "
                f"env={o['env']:.1f}, "
                f"tun={o['tunnel']:.1f}"
            )
        except (nx.NetworkXNoPath, Exception) as e:
            print(f"✗ No path for weights ({w_len:.1f},{w_env:.1f},{w_tunnel:.1f}): {e}")

    print(f"\nSuccessfully generated {successful_seeds}/{len(seed_weights)} seed solutions")

    # ----- EMO step: compute Pareto front of seeds -----
    if not seeds:
        print("\n⚠ ERROR: No valid seed solutions found! Cannot proceed.")
        return None
    
    front = pareto_front(seeds)
    print("\n" + "="*60)
    print(f"PARETO FRONT: {len(front)} non-dominated solutions")
    print("="*60)
    
    for i, s in enumerate(front, 1):
        w = s["weights"]
        o = s["objectives"]
        print(
            f"{i}. weights=({w[0]:.1f},{w[1]:.1f},{w[2]:.1f}) -> "
            f"len={o['length']:.1f}m, env={o['env']:.1f}, tun={o['tunnel']:.1f}"
        )

    # Dump results
    out_path = Path(f"/workspace/outputs/apex_emo_seed_{input_type}_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    result_data = {
        "input_type": input_type,
        "source": list(source) if isinstance(source, tuple) else source,
        "target": list(target) if isinstance(target, tuple) else target,
        "graph_stats": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges()
        },
        "seeds": [{
            "weights": list(s["weights"]),
            "objectives": s["objectives"],
            "path_length": len(s["path"])
        } for s in seeds],
        "pareto_front": [{
            "weights": list(s["weights"]),
            "objectives": s["objectives"],
            "path_length": len(s["path"])
        } for s in front],
    }
    
    if metadata:
        result_data["metadata"] = {
            "width": int(metadata["width"]),
            "height": int(metadata["height"]),
            "bbox": list(metadata["bbox"]),
            "target_crs": metadata["target_crs"]
        }
    
    with out_path.open("w") as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\n✓ Saved results to {out_path}")
    return result_data


# ---------- 5) Entry point ----------

if __name__ == "__main__":
    import sys
    
    # Example usage with different input types
    
    # Option 1: Use LI GeoJSON (original)
    LI_GEOJSON_PATH = Path("/workspace/data/output/LI/LI_roads.geojson")
    
    # Option 2: Use NPZ test sets
    NPZ_TEST_PATHS = {
        "austin": Path("../src/sample-test-set/austin_test_raster.npz"),
        "seattle": Path("../src/sample-test-set/seattle_test_raster.npz"),
        "portland": Path("../src/sample-test-set/portland_test_raster.npz"),
    }
    
    print("="*60)
    print("A*PEX-to-EMO Multi-Objective Routing Experiment")
    print("="*60)
    
    # Determine which dataset to use
    if len(sys.argv) > 1:
        dataset = sys.argv[1].lower()
    else:
        dataset = "austin"  # default
    
    try:
        if dataset == "geojson" and LI_GEOJSON_PATH.exists():
            print(f"\nUsing GeoJSON input: {LI_GEOJSON_PATH}")
            run_experiment_geojson(LI_GEOJSON_PATH)
        elif dataset in NPZ_TEST_PATHS:
            npz_path = NPZ_TEST_PATHS[dataset]
            print(f"\nUsing NPZ input: {npz_path}")
            run_experiment_npz(npz_path)
        else:
            print(f"\n⚠ Dataset '{dataset}' not found. Available: geojson, austin, seattle, portland")
            print("Defaulting to Austin...")
            if NPZ_TEST_PATHS["austin"].exists():
                run_experiment_npz(NPZ_TEST_PATHS["austin"])
            else:
                print("ERROR: No valid input files found!")
                sys.exit(1)
                
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n✓ Experiment complete!")
    