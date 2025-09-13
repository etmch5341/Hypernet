import json
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
import geopandas as gpd

# --- Load Protected Areas GeoJSON ---
PROTECTED_PATH = "./data/output/LI/protected_areas.geojson"  # Replace if needed
protected_gdf = gpd.read_file(PROTECTED_PATH)

# --- Setup Raster Dimensions (match main map if you already have one) ---
width, height = 1000, 1000
minx, miny, maxx, maxy = protected_gdf.total_bounds  # Auto-bounds from protected areas
transform = from_bounds(minx, miny, maxx, maxy, width, height)

# --- Rasterize Protected Areas ---
protected_bitmap = rasterize(
    ((geom, 1) for geom in protected_gdf.geometry),
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype=np.uint8
)

# --- Optional: Load Roads Raster ---
try:
    with open("./data/output/LI/LI_roads.geojson") as f:
        road_geojson = json.load(f)
    road_geometries = [(feature["geometry"], 1) for feature in road_geojson["features"]]
    road_bitmap = rasterize(road_geometries, out_shape=(height, width), transform=transform, fill=0, dtype=np.uint8)
    overlay = road_bitmap + protected_bitmap * 2  # Different color scale
except FileNotFoundError:
    overlay = protected_bitmap

# --- Display ---
plt.figure(figsize=(8, 8))
plt.imshow(overlay, cmap="YlGn", origin="upper")
plt.title("Protected Areas (Green) + Roads (Light)")
plt.colorbar(label="Raster Value")
plt.show()
