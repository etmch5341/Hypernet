import json
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
import heapq

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

goal_points = []

for key in point_map:
    target_lon = point_map[key][0]
    target_lat = point_map[key][1]

    # Convert the coordinate to pixel indices
    col, row = ~transform * (target_lon, target_lat)
    col, row = int(round(col)), int(round(row))

    # Ensure the indices are within bounds
    if 0 <= row < height and 0 <= col < width:
        raster[row, col] = 2
        goal_points.append((row, col))

# Display the raster
plt.imshow(raster, cmap="gray", origin="upper")
plt.colorbar(label="Raster Value")
plt.show()


# Define a state as current coordinate,
# then a list of size 3 of booleans denoting
# whether or not each city has been reached
# (position, [reachedCity1, reachedCity2, reachedCity3])

# Each node in the heap is
# (h_cost, ((startingPosition, booleans), f_cost))

# True cost for later: https://developers.google.com/maps/documentation/directions/overview

# TODO: Pass in point_map, make sure getSuccessor() checks to make sure the point is valid
# TODO: Make a class instead of triple tuple
def a_star(heuristic, startingPosition, goalPositions, max_x, max_y):
    if not validStartingPosition(startingPosition):
        return None
    visited = set()
    came_from = {} # Map tracking predecessors of each node
    start_state = updateState(startingPosition, 0, [False, False, False], goalPositions)
    fringe = [(calculateHeuristicDistance(heuristic, startingPosition, [False, False, False], goalPositions), start_state)]

    while fringe:
        # Grab smallest node
        h_cost, node = heapq.heappop(fringe)

        # Reached the end
        if isGoalState(node[0][1]):
            path = []
            node = node[0]
            while node:
                path.append(node[0][0])
                node = came_from.get(node)
            return path[::-1]  # Reverse the path to get start -> goal
        
        # Expand state if not visited
        if node[0] not in visited:
            visited.add(node[0])

            # Expand all valid successors
            # Successors are formatted ((newPosition, goalBooleans), true_cost to successor)
            for successor in getSuccessors():
                # Calculate true path cost
                true_cost = successor[1] + node[1]

                # Calculate h
                h = true_cost + calculateHeuristicDistance(heuristic, successor[0][0], successor[0][1], goalPositions)

                # Add to heap
                heapq.heappush(fringe, (h, (successor[0], true_cost)))
                
                # Update predecessors
                came_from[successor[0]] = node[0]

    return None

def getSuccessors(visited, currPos, goal, max_x, max_y):
    dirs = [(1,0), (1,1), (-1,0), (-1,-1), (0,1), (0,-1), (-1, 1), (1, -1)]
    successors = []
    for dir in dirs:
        new_x = currPos[0] + dir[0]
        new_y = currPos[1] + dir[1]
        node = ((new_x, new_y), goal)
        if node not in visited:
            true_cost = truePathCost(currPos, node[0])
            successors.append(node, true_cost)
    return successors

def truePathCost(currPos, newPos):
    return 1

def updateState(currPos, cost, goalBooleans, goalPositions):
    for x in range(len(goalPositions)):
        if currPos == goalPositions[x]:
            goalBooleans[x] = True
    return ((currPos, goalBooleans), cost)

def isGoalState(goalBooleans):
    for goal in goalBooleans:
        if goal == False:
            return False
    return True

def validStartingPosition(startingPosition):
    return True

def calculateHeuristicDistance(heuristic, currentLocation, goalsVisited, validGoalLocations):
    if heuristic == "manhattan":
        maxDist = -1
        for x in range(len(validGoalLocations)):
            if goalsVisited[x]:
                continue
            goalPoint = validGoalLocations[x]
            currentDist = abs(goalPoint[1]-currentLocation[1]) + abs(goalPoint[0]-currentLocation[0])
            maxDist = max(maxDist, currentDist)
        return maxDist
    elif heuristic == "euclidean":
        maxDist = -1
        for x in range(len(validGoalLocations)):
            if goalsVisited[x]:
                continue
            goalPoint = validGoalLocations[x]
            currentDist = ((goalPoint[1]-currentLocation[1])**2 + (goalPoint[0]-currentLocation[0])**2)**0.5
            maxDist = max(maxDist, currentDist)
        return maxDist
    return 0
