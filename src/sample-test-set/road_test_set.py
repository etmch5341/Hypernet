from data_pipeline.system.feature_extraction import *
import matplotlib.pyplot as plt
import numpy as np

# NOTE: Should try to always provide bbox to avoid high memory usage

# Test Set 1: Austin

#bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
test_raster1, width, height = extract_road_network(bbox=[-97.795, 30.238, -97.704, 30.285], resolution=10.0, target_crs="EPSG:3083", geojson_input_path="./src/sample-test-set/austin_test.geojson")

# Define cities and mark goals
point_map1 = {
    "station1": [-97.7077039651338, 30.282005561829678],
    "station2": [-97.7791076803512, 30.240382625014902]
}

# # Save as .npy (binary, efficient)
# np.save('./output/austin_test_raster.npy', test_raster1)
# print("NumPy array saved")

test_raster1, goal_points1 = add_goal_points_to_raster(
    test_raster1, 
    point_map1, 
    width, 
    height, 
    bbox=[-97.795, 30.238, -97.704, 30.285],
    target_crs="EPSG:3083"
)

# Save with metadata
np.savez('./src/sample-test-set/austin_test_raster.npz',
         raster=test_raster1,
         width=width,
         height=height,
         bbox=[-97.795, 30.238, -97.704, 30.285],
         target_crs="EPSG:3083",
         goal_points=np.array(goal_points1, dtype=object))

print("NumPy archive saved with metadata")

# Optional: Show statistics
# print("\n=== Raster Statistics ===")
# print(f"Raster shape: {test_raster1.shape}")
# print(f"Width: {width} pixels, Height: {height} pixels")
# print(f"Value range: {test_raster1.min()} to {test_raster1.max()}")
# print(f"Number of goal points: {len(goal_points1)}")
# print(f"\nGoal points:")
# for x, y, name in goal_points1:
#     print(f"  {name}: pixel ({x}, {y}), value = {test_raster1[y, x]}")

# Test Set 2: Seattle
test_raster2, width, height = extract_road_network(bbox=[-122.345, 47.656, -122.326, 47.665], resolution=10.0, target_crs="EPSG:32610", geojson_input_path="./src/sample-test-set/seattle_test.geojson")
point_map2 = {
    "station1": [-122.32721309032283, 47.66316949769572],
    "station2": [-122.3435757623188, 47.65632853533111]
}

test_raster2, goal_points2 = add_goal_points_to_raster(
    test_raster2, 
    point_map2, 
    width, 
    height, 
    bbox=[-122.345, 47.656, -122.326, 47.665],
    target_crs="EPSG:32610"
)

np.savez('./src/sample-test-set/seattle_test_raster.npz',
         raster=test_raster2,
         width=width,
         height=height,
         bbox=[-122.345, 47.656, -122.326, 47.665],
         target_crs="EPSG:32610",
         goal_points=np.array(goal_points2, dtype=object))

print("NumPy archive saved with metadata")

# Test Set 3: Portland
test_raster3, width, height = extract_road_network(bbox=[-122.705, 45.454, -122.614, 45.545], resolution=10.0, target_crs="EPSG:32610", geojson_input_path="./src/sample-test-set/portland_test.geojson")

point_map3 = {
    "station1": [-122.67929951039493, 45.45902100626367],
    "station2": [-122.6306767172761, 45.53819140440015]
}

test_raster3, goal_points3 = add_goal_points_to_raster(
    test_raster3, 
    point_map3, 
    width, 
    height, 
    bbox=[-122.705, 45.454, -122.614, 45.545],
    target_crs="EPSG:32610"
)

np.savez('./src/sample-test-set/portland_test_raster.npz',
         raster=test_raster3,
         width=width,
         height=height,
         bbox=[-122.705, 45.454, -122.614, 45.545],
         target_crs="EPSG:32610",
         goal_points=np.array(goal_points3, dtype=object))

print("NumPy archive saved with metadata")