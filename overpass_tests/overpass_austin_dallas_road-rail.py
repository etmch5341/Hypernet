import overpy
import geopandas as gpd
from shapely.geometry import LineString
import os

#https://overpass-turbo.eu/

# Create Overpass API instance
api = overpy.Overpass()

# Define your Overpass QL query
query = """
[out:json][timeout:60];
(
  way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential)$"](30.0000, -98.0000, 33.0000, -96.0000);
  way["railway"~"^(rail|light_rail|subway|tram)$"](30.0000, -98.0000, 33.0000, -96.0000);
);
out body;
>;
out skel qt;
"""

# Run the query
result = api.query(query)

# Prepare features
features = []
for way in result.ways:
    try:
        coords = [(float(node.lon), float(node.lat)) for node in way.nodes]
        if len(coords) >= 2:
            line = LineString(coords)
            features.append({"geometry": line, "tags": way.tags})
    except Exception as e:
        print(f"Error with way ID {way.id}: {e}")

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")

# Export to GeoJSON
output_path = "output/highways_and_rail.geojson"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
gdf.to_file(output_path, driver="GeoJSON")

print(f"Exported {len(gdf)} features to {output_path}")
