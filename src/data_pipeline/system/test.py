from feature_extraction import *



raster_li = rasterize_geometries(10.0, "./data/output/LI/LI_roads.geojson", target_crs="EPSG:2056")

# extract_road_network(bbox=None, resolution=10.0, geojson_input_path="./data/output/LI/LI_roads.geojson")

# #bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
# extract_road_network(bbox=[-97.795, 30.238, -97.704, 30.285], resolution=10.0, pbf_input_path= "./data/input/texas-latest.osm.pbf", output_path="./data/output/TX/austin_test.geojson")


# plt.imshow(raster_li, cmap="gray", origin="upper")
# plt.colorbar(label="Raster LI Value")
# plt.show()

# raster_tx = rasterize_geometries(10.0, "./data/output/TX/austin_dallas.geojson")

raster_tx = rasterize_geometries(10.0, "./src/sample-test-set/austin_test.geojson", bbox = [-97.795, 30.238, -97.704, 30.285])

# plt.imshow(raster_tx, cmap="gray", origin="upper")
# plt.colorbar(label="Raster TX Value")
# plt.show()