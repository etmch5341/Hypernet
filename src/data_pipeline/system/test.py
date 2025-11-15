from feature_extraction import *



raster_li = rasterize_geometries(10.0, "./data/output/LI/LI_roads.geojson")

# plt.imshow(raster_li, cmap="gray", origin="upper")
# plt.colorbar(label="Raster LI Value")
# plt.show()

# raster_tx = rasterize_geometries(10.0, "./data/output/TX/austin_dallas.geojson")

# plt.imshow(raster_tx, cmap="gray", origin="upper")
# plt.colorbar(label="Raster TX Value")
# plt.show()