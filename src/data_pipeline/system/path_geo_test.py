from feature_extraction import extract_elevation, extract_road_network, extract_slope, extract_existing_corridors
from geometry_cost_map import GeometryCostMap

# extract_road_network(bbox=[-122.705, 45.454, -122.614, 45.545], resolution=10.0, target_crs="EPSG:32610", pbf_input_path= "./data/input/oregon-251121.osm.pbf", output_path="./data/output/OR/portland_test.geojson")

extract_elevation(bbox=[-122.705, 45.454, -122.614, 45.545], resolution=10.0,target_crs="EPSG:32610")

elevation_map = extract_elevation(bbox=[-97.795, 30.238, -97.704, 30.285], resolution=10.0,target_crs="EPSG:3083")
# i think the crs conversion is slightly off because the resulting display is not a complete rectangle -> not crs conversion, download doesn't account fully for rotation, so we added download buffer

slope_map = extract_slope(bbox=[-97.795, 30.238, -97.704, 30.285], resolution=10.0,target_crs="EPSG:3083")

geometry_cost_map = GeometryCostMap(bbox=[-97.795, 30.238, -97.704, 30.285], resolution=10.0)
features_dict = {
    'elevation': elevation_map,
    'existing_corridors': extract_existing_corridors(bbox=[-97.795, 30.238, -97.704, 30.285], geojson_input_path="./src/sample-test-set/austin_test.geojson", resolution=10.0, target_crs="EPSG:3083")
}
cost_map = geometry_cost_map.combine_features(features_dict)
print(cost_map)
import matplotlib.pyplot as plt
plt.imshow(cost_map, cmap="hot", origin="upper")
plt.colorbar(label="Geometry Cost Multiplier")
plt.title("Geometry Cost Map")
plt.show()