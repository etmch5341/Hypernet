from feature_extraction import *

# Test Set 1: Austin

#bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
# extract_road_network(bbox=[-97.795, 30.238, -97.704, 30.285], resolution=10.0, target_crs="EPSG:3083", pbf_input_path= "./data/input/texas-latest.osm.pbf", output_path="./data/output/TX/austin_test.geojson")

# Test Set 2: Seattle
extract_road_network(bbox=[-122.345, 47.656, -122.326, 47.665], resolution=10.0, target_crs="EPSG:32610", pbf_input_path= "./data/input/washington-251121.osm.pbf", output_path="./data/output/WA/seattle_test.geojson")

# Test Set 3: Portland
extract_road_network(bbox=[-122.705, 45.454, -122.614, 45.545], resolution=10.0, target_crs="EPSG:32610", pbf_input_path= "./data/input/oregon-251121.osm.pbf", output_path="./data/output/OR/portland_test.geojson")