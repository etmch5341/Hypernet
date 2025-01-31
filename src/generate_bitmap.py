import json
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt

DATA_PATH = "./data/output/LI/LI_roads.geojson"

# Open the GeoJSON file
with open(DATA_PATH) as f:
    geojson = json.load(f)

# Extract geometries and assign a value (e.g., 1 for all features)
geometries = [(feature["geometry"], 1) for feature in geojson["features"]]

# Define raster properties
width, height = 1000, 1000  # Define output raster size
minx, miny, maxx, maxy = rasterio.features.bounds(geojson)  # Get extent from GeoJSON
# print(f"Bounding Box: ({minx}, {miny}) , ({maxx}, {maxy}) ")
transform = from_bounds(minx, miny, maxx, maxy, width, height)  # Define transform

# Rasterize the geometries; creates a numpy array of size width, height
raster = rasterize(geometries, out_shape=(height, width), transform=transform, fill=0, dtype=np.uint8)
# print(f"Raster: {raster}")

# Determine the goal states of the A* algorithm
# Balzers, Malbun, Eschen are the three cities we are working with 
'''
Eschen - 
"coordinates": [
          9.518061467906222,
          47.21094022078131
        ],
        
Malbun - 
"coordinates": [
          9.607783557623662,
          47.10286302280241
        ],
        
Balzers - 
"coordinates": [
          9.497583388657375,
          47.066238854375655
        ],
'''
# Goal points are the center of cities
# Use geojson.io and just place a markeron the center of the city to get the coordinate
point_map = {
    "Eschen" : [9.518061467906222, 47.21094022078131], 
    "Malbun" : [9.607783557623662, 47.10286302280241], 
    "Balzers" : [9.497583388657375, 47.066238854375655]
}

for key in point_map:
    target_lon = point_map[key][0]
    target_lat = point_map[key][1]

    # Convert the coordinate to pixel indices
    col, row = ~transform * (target_lon, target_lat)
    col, row = int(round(col)), int(round(row))

    # Ensure the indices are within bounds
    if 0 <= row < height and 0 <= col < width:
        raster[row, col] = 2

# Display the raster
plt.imshow(raster, cmap="gray", origin="upper")
plt.colorbar(label="Raster Value")
plt.show()


# Define a state as current coordinate,
# then a tuple of size 3 of booleans denoting
# whether or not each city has been reached
# (position, cost, reachedCity1, reachedCity2, reachedCity3)

# 
def a_star(heuristic, startingPosition):
    visited = set()
    if not validStartingPosition(startingPosition):
        return
    
    pass

def validStartingPosition(startingPosition):
    return true

def calculateHeuristicDistance(heuristic, currentLocation, validGoalLocations):
    if heuristic == "manhattan":
        maxDist = -1
        for goalPoint in validGoalLocations:
            currentDist = abs(goalPoint[1]-currentLocation[1]) + abs(goalPoint[0]-currentLocation[0])
            maxDist = max(maxDist, currentDist)
        return maxDist
    elif heuristic == "euclidean":
        maxDist = -1
        for goalPoint in validGoalLocations:
            currentDist = ((goalPoint[1]-currentLocation[1])**2 + (goalPoint[0]-currentLocation[0])**2)**0.5
            maxDist = max(maxDist, currentDist)
        return maxDist
    return 0
