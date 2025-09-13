import overpy
import geopandas as gpd
from shapely.geometry import Polygon, LineString

# Create Overpass API instance
api = overpy.Overpass()

# Define bounding box for your area (Austin to Dallas example)
bbox = "30.0000,-98.0000,33.0000,-96.0000"  # (south, west, north, east)

# Define Overpass QL query
query = f"""
[out:json][timeout:25];
(
    relation["boundary"="protected_area"]({bbox});
    way["boundary"="protected_area"]({bbox});
    relation["leisure"="nature_reserve"]({bbox});
    way["leisure"="nature_reserve"]({bbox});
    relation["landuse"="conservation"]({bbox});
    way["landuse"="conservation"]({bbox});
);
out body;
>;
out skel qt;
"""

# Run query
result = api.query(query)

# Extract geometries from ways
features = []
for way in result.ways:
    try:
        coords = [(float(node.lon), float(node.lat)) for node in way.nodes]
        if len(coords) >= 3 and coords[0] == coords[-1]:  # Closed loop
            geometry = Polygon(coords)
        else:
            geometry = LineString(coords)
        features.append({"geometry": geometry, "tags": way.tags})
    except Exception as e:
        print(f"Error processing way {way.id}: {e}")

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")

# Export to GeoJSON
output_path = "protected_areas.geojson"
gdf.to_file(output_path, driver="GeoJSON")

print(f"✅ Exported {len(gdf)} protected features to {output_path}")
