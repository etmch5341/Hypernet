'''
For simple tests the base should just be the bounding box w/: 
1. only blocked areas
2. Roads
(simple test for verification)

However, there will be 3 main layers of cost maps:
1. Path Geometry/Operational Efficiency
2. Construction Cost
3. Environmental Impact

These are the 3 cost maps that need to be optimized for.

However, the overall objectives that need to be optimized for this route are:
1. Distance
2. Travel time/operational efficiency
3. Construction cost
4. Environmental impact

other reach objectives
5. Demand/ridership capture
6. Political/social feasibility (includes land aquisition cost)

*NOTE: Distance/Travel time are somewhat correlated, but not perfectly. 
Could be that some situtations that use a slightly longer route may be faster if it optimizes for curvature.

*NOTE: Distance is simply just the distance along the path (which is why it is not a seperate cost map).
Some features need to be calculated seperately without a cost map.
'''

