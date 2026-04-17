#!/usr/bin/env python3
"""Shared cost-map + raster loader for the combined pipeline.

Loads the three normalised cost layers (construction, environmental,
geometry) plus the Austin road raster with goal points, and provides a
single scalar "composite" cost grid used for HPA* clustering/gateway
decisions.
"""

import os
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NPZ_DIR = os.path.join(PROJECT_ROOT, "npz-files")
DEFAULT_RASTER = os.path.join(PROJECT_ROOT, "src", "sample-test-set",
                              "austin_test_raster.npz")


def load_cost_maps(npz_dir=NPZ_DIR, verbose=True):
    """Load the three cost-map layers as float64 arrays in [0, 1]."""
    d = np.load(os.path.join(npz_dir, "austin_construction_cost.npz"),
                allow_pickle=True)
    construction = d["cost_map_normalized"].astype(np.float64)

    d = np.load(os.path.join(npz_dir, "austin_environmental_impact.npz"),
                allow_pickle=True)
    environmental = d["env_map_normalized"].astype(np.float64)

    d = np.load(os.path.join(npz_dir, "geometry_cost_map.npz"),
                allow_pickle=True)
    geometry = d["cost_map"].astype(np.float64)

    if verbose:
        for name, arr in (("construction", construction),
                          ("environmental", environmental),
                          ("geometry", geometry)):
            print(f"  {name}: shape={arr.shape} "
                  f"range=[{arr.min():.3f}, {arr.max():.3f}]")

    return construction, environmental, geometry


def load_raster_and_goals(raster_path=DEFAULT_RASTER, verbose=True):
    """Load the binary road raster and goal points (row, col)."""
    data = np.load(raster_path, allow_pickle=True)
    raster = data["raster"]
    goal_points = data["goal_points"]
    metadata = {
        "width": int(data["width"]) if "width" in data.files else raster.shape[1],
        "height": int(data["height"]) if "height" in data.files else raster.shape[0],
        "bbox": data["bbox"].tolist() if "bbox" in data.files else None,
        "target_crs": (str(data["target_crs"])
                       if "target_crs" in data.files else None),
    }

    goals = []
    for x, y, name in goal_points:
        goals.append((int(y), int(x)))
        if verbose:
            print(f"  Goal '{name}': pixel ({x},{y}) -> "
                  f"(row={int(y)}, col={int(x)})")
    return raster, goals, metadata


def composite_cost_grid(cost_maps, weights=(1.0, 1.0, 1.0)):
    """Weighted mean of the three normalised cost layers.

    HPA* uses this scalar grid only for clustering/gateway decisions;
    the downstream NAMOA-dr search still consumes the full 3-vector cost.
    """
    c, e, g = cost_maps
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return (w[0] * c + w[1] * e + w[2] * g).astype(np.float32)
